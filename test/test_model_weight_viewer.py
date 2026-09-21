"""Checkpoint parsing and raw-value paging for the local weight viewer."""

from pathlib import Path

import pytest
import torch

from py_src.model_opti_save_load import save_model_state
from result_processing_tool.model_weight_viewer import (
    checkpoint_overview,
    load_checkpoint_file,
    tensor_statistics,
    tensor_value_page,
)


def test_loads_llr2_checkpoint_and_reports_all_tensor_metadata(tmp_path: Path):
    path = tmp_path / "sample.model.pt"
    state = {
        "layer.weight": torch.tensor([[1.25, -2.5], [3.0, 4.5]], dtype=torch.float32),
        "counter": torch.tensor(7, dtype=torch.int64),
    }
    save_model_state(str(path), state, "any_model", "any_dataset")

    checkpoint = load_checkpoint_file(path)
    overview = checkpoint_overview("test-id", checkpoint)

    assert overview["model_type"] == "any_model"
    assert overview["dataset_type"] == "any_dataset"
    assert overview["source_format"] == "LLR2 state_dict"
    assert overview["tensor_count"] == 2
    assert overview["total_values"] == 5
    weight = next(item for item in overview["entries"] if item["name"] == "layer.weight")
    assert weight["shape"] == [2, 2]
    assert weight["ndim"] == 2
    assert weight["dtype"] == "float32"
    assert weight["numel"] == 4
    assert weight["bytes"] == 16


def test_loads_plain_state_dict_and_pages_exact_values(tmp_path: Path):
    path = tmp_path / "raw.pt"
    torch.save({"values": torch.arange(12, dtype=torch.float64).reshape(3, 4)}, path)
    checkpoint = load_checkpoint_file(path)

    assert checkpoint.source_format == "raw state_dict"
    assert checkpoint.model_type is None
    page = tensor_value_page(checkpoint.state_dict["values"], offset=5, limit=4)
    assert page == {
        "offset": 5,
        "limit": 4,
        "returned": 4,
        "total": 12,
        "values": ["5.0", "6.0", "7.0", "8.0"],
    }


def test_statistics_handle_nonfinite_and_empty_tensors():
    stats = tensor_statistics(torch.tensor([0.0, 2.0, float("nan"), float("inf"), -float("inf")]))
    assert stats["count"] == 5
    assert stats["finite_count"] == 2
    assert stats["zero_count"] == 1
    assert stats["nan_count"] == 1
    assert stats["positive_infinity_count"] == 1
    assert stats["negative_infinity_count"] == 1
    assert stats["minimum"] == "0.0"
    assert stats["maximum"] == "2.0"
    assert tensor_statistics(torch.empty(0)) == {"count": 0}


def test_page_validation():
    value = torch.arange(3)
    with pytest.raises(ValueError, match="offset"):
        tensor_value_page(value, -1, 2)
    with pytest.raises(ValueError, match="limit"):
        tensor_value_page(value, 0, 0)
