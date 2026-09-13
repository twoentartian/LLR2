"""Numerical and checkpoint integration tests for layer-weight PCA."""

import csv
import json

import numpy as np
import pytest
import torch

from py_src.model_opti_save_load import save_model_state
from result_processing_tool.plot_layer_weight_pca import main, pca_2d, select_layers


@pytest.mark.parametrize("shape", [(8, 31), (12, 3), (2, 7), (4, 1)])
def test_pca_matches_centered_svd(shape):
    matrix = np.random.default_rng(42).normal(size=shape) + 12
    scores, ratios, rank = pca_2d(matrix, block_size=3)
    centered = matrix - matrix.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    count = min(2, shape[0] - 1, shape[1])
    expected = u[:, :count] * s[:count]
    # Signs are arbitrary; the projected pairwise geometry must agree.
    np.testing.assert_allclose(scores @ scores.T, expected @ expected.T, atol=1e-11)
    np.testing.assert_allclose(ratios[:count], s[:count] ** 2 / (s ** 2).sum())
    np.testing.assert_allclose(scores.mean(axis=0), 0, atol=1e-12)
    assert rank == min(shape[0] - 1, shape[1])
    np.testing.assert_allclose(scores, pca_2d(matrix, block_size=17)[0], atol=1e-11)


def test_degenerate_inputs():
    scores, ratios, rank = pca_2d(np.full((20, 3), 0.1))
    assert rank == 0
    assert not scores.any()
    assert not ratios.any()
    scores, ratios, rank = pca_2d(np.arange(5)[:, None] * np.array([[1, 2, 3]]))
    assert rank == 1
    assert not scores[:, 1].any()
    np.testing.assert_allclose(ratios, [1, 0])


@pytest.mark.parametrize("matrix", [np.ones((1, 4)), np.ones((2, 0)),
                                        np.array([[0], [np.nan]]), np.array([[0], [np.inf]])])
def test_invalid_pca_inputs(matrix):
    with pytest.raises(ValueError):
        pca_2d(matrix)


def test_parameter_selection():
    state = {
        "block.conv.weight": torch.ones(2, 3), "block.conv.bias": torch.ones(2),
        "block.bn.weight": torch.ones(2), "block.bn.bias": torch.ones(2),
        "block.bn.running_mean": torch.ones(2), "block.bn.running_var": torch.ones(2),
        "block.bn.num_batches_tracked": torch.tensor(1),
        "pos_embedding": torch.ones(1, 5),
    }
    layers = select_layers(state)
    assert list(layers) == ["block.conv", "block.bn", "pos_embedding"]
    assert layers["block.bn"] == ["block.bn.weight", "block.bn.bias"]
    assert select_layers(state, key_regex="conv", exclude_bias=True) == {
        "block.conv": ["block.conv.weight"]}
    assert select_layers(state, key_regex="block", exclude_regex="bn", layer_level=1) == {
        "block": ["block.conv.weight", "block.conv.bias"]}


def write_checkpoints(folder, problem=None):
    folder.mkdir()
    for index in [10, 2, 1]:
        state = {"conv.weight": torch.tensor([[index, index * 2.0]]),
                 "bn.weight": torch.tensor([index * 0.1]), "bn.bias": torch.tensor([0.0]),
                 "bn.running_mean": torch.tensor([100.0]),
                 "bn.num_batches_tracked": torch.tensor(100)}
        if index == 10:
            if problem == "shape":
                state["conv.weight"] = torch.ones(3)
            elif problem == "keys":
                del state["conv.weight"]
            elif problem == "nan":
                state["conv.weight"][0, 0] = float("nan")
            # Both training wrappers must normalize to the same layer keys.
            state = {"_orig_mod.module." + key: value for key, value in state.items()}
        save_model_state(str(folder / f"{index}.model.pt"), state, "test_model",
                         "other" if problem == "metadata" and index == 10 else "test_dataset")
    (folder / "1.optimizer.pt").write_text("must not be loaded")


def test_cli_outputs_and_cleanup(tmp_path):
    source, output, cache = tmp_path / "models", tmp_path / "out", tmp_path / "cache"
    cache.mkdir()
    write_checkpoints(source)
    main([str(source), "-o", str(output), "--cache-dir", str(cache), "--block-size", "1"])
    assert not list(cache.iterdir())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["model_count"] == 3
    assert len(summary["layers"]) == 2
    for layer in summary["layers"]:
        assert (output / layer["image"]).read_bytes().startswith(b"\x89PNG")
        with (output / layer["coordinates"]).open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row["model_id"] for row in rows] == ["1", "2", "10"]
        assert all(float(row["PC2"]) == 0 for row in rows)
    with pytest.raises(SystemExit):
        main([str(source), "-o", str(output)])


@pytest.mark.parametrize("problem", ["metadata", "shape", "keys", "nan"])
def test_incompatible_models_fail_before_plotting(tmp_path, problem):
    source, output, cache = tmp_path / "models", tmp_path / "out", tmp_path / "cache"
    cache.mkdir()
    write_checkpoints(source, problem)
    with pytest.raises(SystemExit):
        main([str(source), "-o", str(output), "--cache-dir", str(cache)])
    assert not list(cache.iterdir())
    assert not list(output.iterdir())
