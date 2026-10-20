"""Causal target coverage, random initialization, DDP normalization and resume."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from py_src.language_model.pile import ExactTokenBatchSampler, PileTokenDataset
from py_src.language_model.pythia import build_pythia, pythia_14m_config
from test.test_pile_preparation import make_source
from tool.prepare_pile import prepare


def corpus(tmp_path):
    docs = [[(i * 7 + j) % 31 + 1 for j in range(7)] for i in range(11)]
    source = make_source(tmp_path / "source", docs)
    output = tmp_path / "data"
    prepare(source, output, block_tokens=8, vocabulary_size=32)
    return output


def test_causal_targets_tail_and_partition_boundary(tmp_path):
    root = corpus(tmp_path)
    dataset = PileTokenDataset(root, sequence_length=5)
    targets = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels = item["labels"]
        count = int(labels.ne(-100).sum())
        assert count == dataset.valid_tokens(i)
        assert torch.equal(item["input_ids"][1:count], labels[: count - 1])
        targets.extend(labels[labels != -100].tolist())
    expected = []
    for part in ("A", "B"):
        expected.extend(np.fromfile(root / f"{part}.bin", dtype="<u2")[1:].tolist())
    assert targets == expected
    assert len(targets) == dataset.target_tokens
    assert not dataset[(-1, 0)]["labels"].ne(-100).any()


@pytest.mark.parametrize("size", [1, 2, 4])
def test_exact_shards_and_token_cap(tmp_path, size):
    dataset = PileTokenDataset(corpus(tmp_path), sequence_length=5)
    total = 0
    seen = []
    shards = []
    for rank in range(size):
        sampler = ExactTokenBatchSampler(
            dataset,
            rank=rank,
            world_size=size,
            micro_batch_size=2,
            token_limit=53,
            seed=72,
        )
        batches = list(sampler)
        shards.append(len(batches))
        for batch in batches:
            for index, limit in batch:
                if limit:
                    seen.append(index)
                total += int(dataset[(index, limit)]["labels"].ne(-100).sum())
    assert len(set(shards)) == 1
    assert total == 53
    assert len(seen) == len(set(seen))
    all_targets = 0
    all_indices = []
    for rank in range(size):
        sampler = ExactTokenBatchSampler(
            dataset, rank=rank, world_size=size, micro_batch_size=2
        )
        for batch in sampler:
            for index, limit in batch:
                if index >= 0:
                    all_indices.append(index)
                    all_targets += limit
    assert sorted(all_indices) == list(range(len(dataset)))
    assert all_targets == dataset.target_tokens


def test_official_architecture_random_initialization():
    config = pythia_14m_config()
    assert (config.hidden_size, config.num_hidden_layers, config.vocab_size) == (
        128,
        6,
        50304,
    )
    torch.manual_seed(42)
    first = build_pythia(config)
    assert sum(p.numel() for p in first.parameters()) == 14_067_712
    torch.manual_seed(42)
    second = build_pythia(pythia_14m_config())
    assert torch.equal(
        first.model.gpt_neox.embed_in.weight, second.model.gpt_neox.embed_in.weight
    )
    assert (
        first.model.gpt_neox.embed_in.weight.data_ptr()
        != first.model.lm_head.weight.data_ptr()
    )


def tiny_config(path):
    config = pythia_14m_config()
    config.vocab_size = 32
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.max_position_embeddings = 16
    config.to_json_file(path)


def run_training(
    root, config, output, *, world=1, accumulation=2, max_steps=None, resume=False
):
    command = [sys.executable]
    if world > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
        ]
    command += [
        "tool/train_pythia.py",
        "--data",
        str(root),
        "--output",
        str(output),
        "--model-config",
        str(config),
        "--sequence-length",
        "5",
        "--cpu",
        "--precision",
        "fp32",
        "--micro-batch-size",
        str(4 // world),
        "--gradient-accumulation",
        str(accumulation),
        "--max-tokens",
        "53",
        "--no-compile",
        "--log-interval",
        "1",
        "--save-interval",
        "1",
        "--eval-interval",
        "20",
        "--eval-sequences",
        "4",
    ]
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
    if resume:
        command += ["--resume", str(output / "training_checkpoint.pt")]
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return torch.load(
        output / "training_checkpoint.pt", map_location="cpu", weights_only=False
    )


def test_resume_matches_uninterrupted_and_rejects_changed_data(tmp_path):
    root = corpus(tmp_path)
    config = tmp_path / "tiny.json"
    tiny_config(config)
    baseline = run_training(root, config, tmp_path / "baseline")
    stopped = run_training(root, config, tmp_path / "resumed", max_steps=1)
    assert stopped["progress"]["tokens"] < 53
    resumed = run_training(root, config, tmp_path / "resumed", resume=True)
    assert resumed["progress"] == baseline["progress"]
    assert resumed["progress"]["tokens"] == 53
    for name, value in baseline["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
    summary = json.loads((tmp_path / "resumed" / "summary.json").read_text())
    assert summary["complete"] and summary["evaluation_kind"] == "training_probe"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seed"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="Checkpoint configuration/data differs"):
        run_training(root, config, tmp_path / "resumed", resume=True)


@pytest.mark.parametrize("world", [2, 4])
def test_ddp_matches_single_process_with_partial_token_window(tmp_path, world):
    root = corpus(tmp_path)
    config = tmp_path / "tiny.json"
    tiny_config(config)
    baseline = run_training(root, config, tmp_path / "single")
    parallel = run_training(root, config, tmp_path / "parallel", world=world)
    assert parallel["progress"] == baseline["progress"]
    for name, value in baseline["model"].items():
        if name.endswith("attention.query_key_value.bias"):
            # Non-rotary key bias is an exact softmax-invariant direction.
            # FP32 batch/reduction order leaves ~1e-10 numerical gradients;
            # Adam's eps=1e-8 magnifies those into ~1e-5 parameter differences.
            actual = parallel["model"][name].view(4, 3, 8)
            expected = value.view(4, 3, 8)
            torch.testing.assert_close(
                expected[:, 0], actual[:, 0], rtol=2e-5, atol=2e-6
            )
            torch.testing.assert_close(
                expected[:, 2], actual[:, 2], rtol=2e-5, atol=2e-6
            )
            torch.testing.assert_close(
                expected[:, 1, :2], actual[:, 1, :2], rtol=2e-5, atol=2e-6
            )
            torch.testing.assert_close(
                expected[:, 1, 2:], actual[:, 1, 2:], rtol=0, atol=2e-5
            )
        else:
            torch.testing.assert_close(
                value, parallel["model"][name], rtol=2e-5, atol=2e-6
            )
    # Check optimizer gradient moments too, including the invariant key biases;
    # this detects incorrect rank/token normalization without Adam amplification.
    for key, state in baseline["optimizer"]["state"].items():
        torch.testing.assert_close(
            state["exp_avg"],
            parallel["optimizer"]["state"][key]["exp_avg"],
            rtol=3e-5,
            atol=3e-8,
        )
