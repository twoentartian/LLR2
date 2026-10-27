"""Pythia/Pile normalized train-descent/val-ascent misalignment."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))

import misalignment_pythia as mp
from misalignment.gradient_combination import combine_normalized_gradient_tensors
from py_src.language_model.pile import PileTokenDataset
from py_src.language_model.pythia import build_pythia
from test.test_pile_preparation import make_source
from test.test_pythia_pile import tiny_config
from tool.prepare_pile import prepare

REPO = Path(__file__).resolve().parents[1]
TIMING = {"tokens_per_second", "eval_seconds"}


def corpus(tmp_path):
    docs = [[(i * 7 + j * 3) % 31 + 1 for j in range(5 + i % 6)] for i in range(40)]
    source = make_source(tmp_path / "source", docs)
    output = tmp_path / "data"
    prepare(source, output, block_tokens=16, vocabulary_size=32)
    return output


def run(root, config, output, *, world=1, max_steps=None, resume=False, directions="both", extra=()):
    command = [sys.executable]
    if world > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}"]
    command += [
        "tool/misalignment_pythia.py", "--data", str(root), "--output", str(output),
        "--model-config", str(config), "--sequence-length", "5", "--cpu", "--precision", "fp32",
        "--micro-batch-size", str(4 // world), "--gradient-accumulation", "2", "--no-compile",
        "--log-interval", "1", "--eval-interval", "2", "--eval-sequences", "6", "--eval-batch-size", "4",
        "--gap-top-k", "2", "--directions", directions, *extra,
    ]
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
    if resume:
        command += ["--resume"]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    result = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True,
                            timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def checkpoint(output, name):
    return torch.load(output / f"training_checkpoint_{name}.pt", map_location="cpu", weights_only=False)


def test_first_update_matches_manual_normalized_combination(tmp_path):
    root = corpus(tmp_path)
    config_path = tmp_path / "tiny.json"
    tiny_config(config_path)
    output = run(root, config_path, tmp_path / "run", max_steps=1, directions="a_as_train_b_as_val")
    actual = checkpoint(output, "a_as_train_b_as_val")
    assert actual["progress"]["step"] == 1

    from transformers import GPTNeoXConfig

    mp.seed_everything(42)
    model = build_pythia(GPTNeoXConfig.from_json_file(str(config_path)))
    initial = torch.load(output / "initial_a_as_train_b_as_val.model.pt", weights_only=True)["state_dict"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    datasets = {name: PileTokenDataset(root, partitions=name, sequence_length=5) for name in "AB"}
    loader_args = SimpleNamespace(micro_batch_size=4, workers=0)
    budget = datasets["A"].target_tokens
    device = torch.device("cpu")
    train = mp.PartitionStream(datasets["A"], args=loader_args, device=device, seed=42 + mp.ORDER_SEED_OFFSET["A"],
                               epoch=0, next_micro=0, epochs=1, token_limit=lambda: budget).take(2)
    val = mp.PartitionStream(datasets["B"], args=loader_args, device=device, seed=42 + mp.ORDER_SEED_OFFSET["B"],
                             epoch=0, next_micro=0).take(2)
    parameters = list(model.parameters())
    totals = torch.zeros(3, dtype=torch.float64)
    train_gradients, train_tokens = mp.window_gradients(model, parameters, train, device=device,
                                                        precision="fp32", totals=totals)
    val_gradients, val_tokens = mp.window_gradients(model, parameters, val, device=device,
                                                    precision="fp32", totals=totals.clone())
    train_gradients = [g / train_tokens for g in train_gradients]
    val_gradients = [g / val_tokens for g in val_gradients]
    combined, _ = combine_normalized_gradient_tensors(parameters, train_gradients, val_gradients, (0.8, 0.2), True)
    for parameter, train_gradient, gradient in zip(parameters, train_gradients, combined):
        # Train descent plus val ascent keeps each tensor's train-gradient norm.
        torch.testing.assert_close(gradient.norm(), train_gradient.norm())
        parameter.grad = gradient
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer = mp.build_optimizer(model, SimpleNamespace(weight_decay=0.1, learning_rate=1e-3))
    for group in optimizer.param_groups:
        group["lr"] = mp.scheduled_lr(train_tokens, budget, peak=1e-3, minimum=1e-4, warmup_fraction=0.01)
    optimizer.step()
    assert actual["progress"]["train_tokens"] == train_tokens and actual["progress"]["val_tokens"] == val_tokens
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, actual["model"][name], rtol=0, atol=1e-7)


def test_both_directions_complete_with_outputs(tmp_path):
    root = corpus(tmp_path)
    config = tmp_path / "tiny.json"
    tiny_config(config)
    output = run(root, config, tmp_path / "run")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["run_order"] == ["a_as_train_b_as_val", "b_as_train_a_as_val"]
    data = {name: PileTokenDataset(root, partitions=name, sequence_length=5) for name in "AB"}
    initial = {}
    for name, (train_name, val_name) in mp.DIRECTIONS.items():
        run_summary = summary["runs"][name]
        assert run_summary["complete"] and run_summary["train_partition"] == train_name
        assert run_summary["progress"]["train_tokens"] == data[train_name].target_tokens
        rows = read_csv(output / f"optimization_{name}.csv")
        steps = [int(float(row["step"])) for row in rows]
        final = run_summary["progress"]["step"]
        assert steps == sorted(set(list(range(0, final, 2)) + [final]))
        assert float(rows[0]["learning_rate"]) == 0.0
        assert all(float(row["train_gradient_weight"]) == 0.8 for row in rows)
        assert float(rows[-1]["abs_accuracy_gap"]) == pytest.approx(
            abs(float(rows[-1]["train_accuracy"]) - float(rows[-1]["val_accuracy"])))
        geometry = read_csv(output / f"gradient_geometry_{name}.csv")
        assert {int(float(row["step"])) for row in geometry} == set(steps[1:])
        assert len(read_csv(output / f"weight_variance_{name}.csv")) == len(rows)
        selection = json.loads((output / f"gap_checkpoints_{name}" / "selection.json").read_text())
        saved = {path.name for path in (output / f"gap_checkpoints_{name}").glob("*.model.pt")}
        assert saved == {Path(point["checkpoint"]).name for point in selection["union"]}
        assert 1 <= len(saved) <= 4
        assert (output / f"final_{name}.model.pt").exists()
        initial[name] = torch.load(output / f"initial_{name}.model.pt", weights_only=True)["state_dict"]
    for key, value in initial["a_as_train_b_as_val"].items():
        torch.testing.assert_close(value, initial["b_as_train_a_as_val"][key], rtol=0, atol=0)


def test_resume_matches_uninterrupted_and_rejects_changed_settings(tmp_path):
    root = corpus(tmp_path)
    config = tmp_path / "tiny.json"
    tiny_config(config)
    baseline = run(root, config, tmp_path / "baseline")
    # Step 3 is inside an evaluation interval: its partial sums must resume.
    resumed = run(root, config, tmp_path / "resumed", max_steps=3)
    assert not (resumed / "summary.json").exists()
    assert checkpoint(resumed, "a_as_train_b_as_val")["progress"]["step"] == 3
    run(root, config, resumed, resume=True)
    for name in mp.DIRECTIONS:
        expected, actual = checkpoint(baseline, name), checkpoint(resumed, name)
        assert expected["progress"] == actual["progress"]
        for key, value in expected["model"].items():
            torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
        for stem in ("optimization", "gradient_geometry", "weight_variance", "train"):
            first, second = read_csv(baseline / f"{stem}_{name}.csv"), read_csv(resumed / f"{stem}_{name}.csv")
            assert len(first) == len(second)
            for row_a, row_b in zip(first, second):
                assert {k: v for k, v in row_a.items() if k not in TIMING} == {
                    k: v for k, v in row_b.items() if k not in TIMING}
        assert (json.loads((baseline / f"gap_checkpoints_{name}" / "selection.json").read_text())
                == json.loads((resumed / f"gap_checkpoints_{name}" / "selection.json").read_text()))
    with pytest.raises(AssertionError, match="different settings"):
        run(root, config, resumed, resume=True, extra=("--val-gradient-weight", "0.3"))


def test_two_ranks_match_single_process(tmp_path):
    root = corpus(tmp_path)
    config = tmp_path / "tiny.json"
    tiny_config(config)
    single = run(root, config, tmp_path / "single", directions="a_as_train_b_as_val")
    parallel = run(root, config, tmp_path / "parallel", world=2, directions="a_as_train_b_as_val")
    name = "a_as_train_b_as_val"
    expected, actual = checkpoint(single, name), checkpoint(parallel, name)
    assert expected["progress"] == actual["progress"]
    for key, value in expected["model"].items():
        if key.endswith("attention.query_key_value.bias"):
            # As in test_pythia_pile: the non-rotary key bias is a softmax-
            # invariant direction whose ~1e-10 reduction-order gradients Adam
            # amplifies; only that slice gets the looser tolerance.
            value, other = value.view(4, 3, 8), actual["model"][key].view(4, 3, 8)
            torch.testing.assert_close(value[:, 0], other[:, 0], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(value[:, 2], other[:, 2], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(value[:, 1, :2], other[:, 1, :2], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(value[:, 1, 2:], other[:, 1, 2:], rtol=0, atol=1e-4)
            continue
        torch.testing.assert_close(value, actual["model"][key], rtol=1e-4, atol=1e-5)
    for row_a, row_b in zip(read_csv(single / f"optimization_{name}.csv"),
                            read_csv(parallel / f"optimization_{name}.csv")):
        for key in ("loss_a", "loss_b", "train_accuracy", "val_accuracy", "running_train_loss"):
            if row_a[key] != "nan":
                assert float(row_a[key]) == pytest.approx(float(row_b[key]), rel=1e-4, abs=1e-6)
