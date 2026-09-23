import pytest
import torch

from py_src.model_opti_save_load import load_model_state_file, save_model_state
from result_processing_tool.override_model_pt_info import main


def test_print_and_in_place_override_preserves_state(tmp_path, capsys, monkeypatch):
    state = {"layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    save_model_state(str(tmp_path / "02.model.pt"), state, "old_removed_model", "old_removed_dataset")
    save_model_state(str(tmp_path / "01.model.pt"), state, "old_removed_model", "old_removed_dataset")

    answers = iter(["n", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    main([str(tmp_path)])
    listed = capsys.readouterr().out
    assert "old_removed_model" in listed
    assert "old_removed_dataset" in listed
    assert "01.model.pt" in listed
    assert "02.model.pt" in listed

    main([str(tmp_path), "--model-name", "bnn_floating", "--dataset-name", "cifar10", "--in-place"])
    for path in sorted(tmp_path.glob("*.model.pt")):
        loaded_state, model_name, dataset_name = load_model_state_file(str(path), map_location="cpu")
        assert model_name == "bnn_floating"
        assert dataset_name == "cifar10"
        torch.testing.assert_close(loaded_state["layer.weight"], state["layer.weight"])


def test_interactive_override_and_output_directory(tmp_path, monkeypatch):
    state = {"weight": torch.arange(4, dtype=torch.float32)}
    save_model_state(str(tmp_path / "model.model.pt"), state, "old_removed_model", "old_removed_dataset")
    output = tmp_path / "interactive-retagged"
    answers = iter(["y", "y", "bnn_floating", "cifar10", "n", str(output)])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    main([str(tmp_path)])

    destination = output / "model.model.pt"
    loaded_state, model_name, dataset_name = load_model_state_file(str(destination), map_location="cpu")
    assert model_name == "bnn_floating"
    assert dataset_name == "cifar10"
    torch.testing.assert_close(loaded_state["weight"], state["weight"])
    _, original_model, original_dataset = load_model_state_file(
        str(tmp_path / "model.model.pt"), map_location="cpu")
    assert original_model == "old_removed_model"
    assert original_dataset == "old_removed_dataset"


def test_no_arguments_prompts_for_input_directory(tmp_path, capsys, monkeypatch):
    state = {"weight": torch.ones(1)}
    save_model_state(str(tmp_path / "model.model.pt"), state, "old_model", "old_dataset")
    answers = iter([str(tmp_path), "n", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    main([])

    output = capsys.readouterr().out
    assert "Input directory" not in output  # prompts are supplied by the patched input
    assert "old_model" in output
    assert "old_dataset" in output


def test_output_directory_and_recursive_discovery(tmp_path, capsys):
    nested = tmp_path / "nested"
    nested.mkdir()
    state = {"weight": torch.ones(1)}
    save_model_state(str(nested / "model.model.pt"), state, "bnn", "cifar10")
    output = tmp_path / "retagged"
    main([str(tmp_path), "--recursive", "--model-name", "binary_attention_cct_7_3x1_32",
          "--output-dir", str(output)])
    destination = output / "nested" / "model.model.pt"
    assert destination.exists()
    _, model_name, dataset_name = load_model_state_file(str(destination), map_location="cpu")
    assert model_name == "binary_attention_cct_7_3x1_32"
    assert dataset_name == "cifar10"
    original_state, original_model, original_dataset = load_model_state_file(
        str(nested / "model.model.pt"), map_location="cpu")
    assert original_model == "bnn"
    assert original_dataset == "cifar10"
    torch.testing.assert_close(original_state["weight"], state["weight"])
    assert "Rewrote" in capsys.readouterr().out


def test_requires_explicit_write_target(tmp_path):
    state = {"weight": torch.ones(1)}
    save_model_state(str(tmp_path / "model.model.pt"), state, "bnn", "cifar10")
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--model-name", "bnn_floating"])
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--model-name", "bnn_floating", "--in-place", "--output-dir", str(tmp_path / "out")])


def test_rejects_unknown_target_names(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--model-name", "model_removed_from_code", "--in-place"])
