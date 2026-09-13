"""Check pair enumeration, matching recovery and function-preserving permutations."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from py_src.model_opti_save_load import save_model_state
from result_processing_tool.calculate_pairwise_layer_cosine import main as cosine_main, pairwise_cosines
from result_processing_tool.model_weight_utils import load_checkpoint
from result_processing_tool.permute_models import (
    apply_permutations, builtin_spec, main as permute_main, make_model,
    match_weights, read_spec, validate_spec, verify_outputs,
)


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_cosines_match_direct_calculation():
    matrix = np.random.default_rng(1).normal(size=(7, 29))
    cosines = pairwise_cosines(matrix, 3)
    for i in range(7):
        for j in range(7):
            expected = np.dot(matrix[i], matrix[j]) / np.linalg.norm(matrix[i]) / np.linalg.norm(matrix[j])
            assert cosines[i, j] == pytest.approx(expected, abs=1e-14)


def test_zero_norm_and_opposite_vectors():
    actual = pairwise_cosines(np.array([[1, 2], [-1, -2], [0, 0.0]]), 1)
    assert actual[0, 1] == pytest.approx(-1)
    assert np.isnan(actual[2]).all()
    assert np.isnan(actual[:, 2]).all()


def test_pairwise_cli_means_and_pair_counts(tmp_path):
    for i, values in enumerate([[1, 0], [0, 1], [1, 1]]):
        save_model_state(str(tmp_path / f"{i}.model.pt"), {
            "fc.weight": torch.tensor([values], dtype=torch.float32),
            "fc.bias": torch.tensor([0.0]), "zero.weight": torch.zeros(1),
            "bn.running_mean": torch.ones(2),
        }, "toy", "toy_data")
    pairs = tmp_path / "pairs.csv"
    cosine_main([str(tmp_path), "--pairs-file", str(pairs), "--block-size", "1"])
    with (tmp_path / "layer_cosine_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["layer"] for row in rows] == ["fc", "zero"]
    assert int(rows[0]["pair_count"]) == 3
    assert float(rows[0]["mean_cosine"]) == pytest.approx(np.sqrt(2) / 3)
    assert int(rows[1]["valid_pair_count"]) == 0
    assert int(rows[1]["undefined_pair_count"]) == 3
    assert np.isnan(float(rows[1]["mean_cosine"]))
    with pairs.open() as handle:
        detail = list(csv.DictReader(handle))
    assert len(detail) == 6
    fc_pairs = [(Path(row["model_a"]).name, Path(row["model_b"]).name)
                for row in detail if row["layer"] == "fc"]
    assert fc_pairs == [("0.model.pt", "1.model.pt"), ("0.model.pt", "2.model.pt"), ("1.model.pt", "2.model.pt")]


def toy_state():
    rng = torch.Generator().manual_seed(7)
    return {"fc1.weight": torch.randn(6, 3, generator=rng),
            "fc1.bias": torch.randn(6, generator=rng),
            "fc2.weight": torch.randn(2, 6, generator=rng), "fc2.bias": torch.randn(2, generator=rng)}


def toy_spec():
    return {"fc1.weight": [("hidden", 1), None], "fc1.bias": [("hidden", 1)],
            "fc2.weight": [None, ("hidden", 1)], "fc2.bias": [None]}


def test_matching_recovers_known_permutation_and_does_not_mutate_inputs():
    a, spec = toy_state(), toy_spec()
    b = apply_permutations(a, spec, {"hidden": torch.tensor([4, 1, 0, 5, 2, 3])})
    original = {k: v.clone() for k, v in b.items()}
    c, report = match_weights(a, b, spec, max_iter=10)
    for key in a:
        torch.testing.assert_close(c[key], a[key], rtol=0, atol=0)
        torch.testing.assert_close(b[key], original[key], rtol=0, atol=0)
    assert report["parameter_cosine_after"] == pytest.approx(1)
    assert report["converged"]


def test_matching_identity_and_monotonicity():
    a, spec = toy_state(), toy_spec()
    c, report = match_weights(a, a, spec)
    assert report["iterations"] == 1
    for key in a:
        assert torch.equal(c[key], a[key])
    torch.manual_seed(12)
    b = {k: torch.randn_like(v) for k, v in a.items()}
    _, report = match_weights(a, b, spec)
    assert report["parameter_cosine_after"] >= report["parameter_cosine_before"]


@pytest.mark.parametrize("model_type", ["bnn", "bnn_floating", "lenet4", "lenet5", "lenet5_large_fc"])
def test_builtin_permutations_preserve_outputs(model_type):
    torch.manual_seed(5)
    model, _ = make_model(model_type)
    state = model.state_dict()
    # Nontrivial BN buffers are needed to catch forgotten buffer permutations.
    for key, value in state.items():
        if key.endswith("running_mean"):
            value.copy_(torch.randn_like(value) * 0.1)
        elif key.endswith("running_var"):
            value.copy_(torch.rand_like(value) + 0.5)
    spec = builtin_spec(model_type, state)
    sizes = validate_spec(spec, state)
    permutations = {group: torch.randperm(size) for group, size in sizes.items()}
    c = apply_permutations(state, spec, permutations)
    assert verify_outputs(state, c, model_type, seed=42, samples=2)["passed"]
    if model_type == "bnn":
        assert spec["fc1.weight"][1] == ("conv6", 9)
        assert torch.equal(c["bn9.running_mean"], state["bn9.running_mean"])
        torch.testing.assert_close(c["bn1.running_mean"], state["bn1.running_mean"][permutations["conv1"]])


def test_invalid_specs_and_permutations():
    a, spec = toy_state(), toy_spec()
    with pytest.raises(ValueError, match="explicitly cover"):
        validate_spec({"fc1.weight": spec["fc1.weight"]}, a)
    bad = dict(spec, **{"fc1.weight": [("hidden", 1), ("hidden", 1)]})
    with pytest.raises(ValueError, match="multiple axes"):
        validate_spec(bad, a)
    with pytest.raises(ValueError, match="bijection"):
        apply_permutations(a, spec, {"hidden": torch.zeros(6, dtype=torch.long)})
    b = dict(a, **{"fc1.weight": torch.zeros(4, 3)})
    with pytest.raises(ValueError, match="shape/dtype"):
        match_weights(a, b, spec)


def test_batch_cli_saves_multiple_models_with_metadata(tmp_path):
    a, spec = toy_state(), toy_spec()
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    reference = source / "a.model.pt"
    save_model_state(str(reference), a, "toy", "toy_data")
    for index in range(2):
        b = apply_permutations(a, spec, {"hidden": torch.roll(torch.arange(6), index + 1)})
        save_model_state(str(source / f"b{index}.model.pt"), b, "toy", "toy_data")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"axes": {key: [entry[0] if entry else None for entry in axes]
                                                for key, axes in spec.items()}}))
    assert read_spec(spec_path) == spec
    permute_main(["-a", str(reference), "-b", str(source), "-o", str(output), "--spec", str(spec_path)])
    files = list(output.glob("*.model.pt"))
    assert len(files) == 2
    for path in files:
        state, metadata = load_checkpoint(path)
        assert metadata == ("toy", "toy_data")
        for key in a:
            assert torch.equal(state[key], a[key])
        report = json.loads(path.with_suffix(".json").read_text())
        assert report["parameter_cosine_after"] == pytest.approx(1)
        assert report["verification"] is None  # Custom graph unavailable.
    with pytest.raises(SystemExit):
        permute_main(["-a", str(reference), "-b", str(source), "-o", str(output), "--spec", str(spec_path)])
