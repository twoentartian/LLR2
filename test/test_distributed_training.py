"""Numerical multi-process checks; run on CPU without a cluster allocation."""

from __future__ import annotations

import copy
import logging
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import TensorDataset

from py_src.adapters import StandardAdapter
from py_src.engine import Device, train, val
from py_src.ml_setup.dataloader_util import DataloaderConfig, build_dataloader
from py_src import distributed


class _RankStateLoader:
    def __init__(self, rank):
        self.value = {"reader_rank": rank, "position": rank + 7}

    def state_dict(self):
        return dict(self.value)

    def load_state_dict(self, state):
        self.value = dict(state)

    def runtime_info(self):
        return {"backend": "test"}


def _engine_worker(rank, size, directory):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    original = nn.Linear(3, 2)
    original.register_buffer("test_buffer", torch.zeros(1))
    data = TensorDataset(torch.randn(12, 3), torch.arange(12) % 2)
    validation = TensorDataset(torch.randn(5, 3), torch.arange(5) % 2)
    expected = copy.deepcopy(original)
    expected_adapter = StandardAdapter(expected, nn.CrossEntropyLoss())
    expected_result = train(
        expected_adapter, build_dataloader(data, 4, config=DataloaderConfig(shuffle=False)),
        torch.optim.SGD(expected.parameters(), lr=0.1), device=Device.cpu(),
        gradient_accumulate_every=2,
    )
    expected_val = val(expected_adapter, build_dataloader(validation, 4, is_train=False), device=Device.cpu())

    dist.init_process_group("gloo", rank=rank, world_size=size,
                            init_method=f"file://{directory}/group", timeout=timedelta(seconds=90))
    try:
        adapter = StandardAdapter(Device.cpu().wrap_ddp(original), nn.CrossEntropyLoss())
        actual = train(
            adapter, build_dataloader(data, 4, config=DataloaderConfig(shuffle=False)),
            torch.optim.SGD(original.parameters(), lr=0.1), device=Device.cpu(),
            gradient_accumulate_every=2,
        )
        actual_val = val(adapter, build_dataloader(validation, 4, is_train=False), device=Device.cpu())
        for before, after in zip(expected.parameters(), original.parameters(), strict=True):
            torch.testing.assert_close(before, after)
        assert actual.total_count == expected_result.total_count == 12
        assert actual.avg_loss == pytest.approx(expected_result.avg_loss, rel=1e-6)
        assert actual_val.total_count == 5
        assert actual_val.avg_loss == pytest.approx(expected_val.avg_loss, rel=1e-6)
        assert actual_val.total_correct == expected_val.total_correct

        # Cover the rank-zero-only file write and every rank's saved RNG state.
        from tool.generate_high_accuracy_model import save_training_checkpoint, load_training_checkpoint, restore_dataloader_states
        path = f"{directory}/checkpoint.pt"
        loader = _RankStateLoader(rank)
        save_training_checkpoint(
            path, model=original, optimizer=None, lr_scheduler=None, scaler=None,
            completed_epoch=0, total_epochs=1, index=0, number_of_models=1,
            output_folder=directory, run_config={"model_type": "test", "dataset_type": "test"},
            dataloader=loader,
        )
        distributed.barrier()
        checkpoint = load_training_checkpoint(path)
        loader.value = {}
        restore_dataloader_states(checkpoint, loader, None, logging.getLogger(__name__))
        assert loader.value == {"reader_rank": rank, "position": rank + 7}
        if rank == 0:
            assert checkpoint["world_size"] == size
            assert len(checkpoint["rng_states_by_rank"]) == size
            assert [states["train"]["reader_rank"] for states in checkpoint["dataloader_states_by_rank"]] == list(range(size))
            assert all(not name.startswith("module.") for name in checkpoint["state_dict"])
    finally:
        distributed.shutdown()


@pytest.mark.parametrize("size", [2, 4])
def test_engine_matches_global_batch_training(size):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_engine_worker, args=(size, directory), nprocs=size, join=True)


def _misalignment_worker(rank, size, directory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
    import misalignment_measurement_2 as measurement
    from py_src.ml_setup import MLSetup
    from py_src.ml_setup_dataset import DatasetType
    from py_src.ml_setup_model import ModelType

    torch.set_num_threads(1)
    torch.manual_seed(123)
    original = nn.Linear(3, 3)
    data_a = TensorDataset(torch.randn(16, 3), torch.arange(16) % 3)
    data_b = TensorDataset(torch.randn(16, 3), (torch.arange(16) + 1) % 3)
    eval_data = TensorDataset(torch.randn(5, 3), torch.arange(5) % 3)
    expected_states, expected_rows = {}, {}

    def run(model, objective, output):
        output.mkdir(exist_ok=True)
        loader_a = measurement._build_loader(data_a, 8, device=torch.device("cpu"), num_workers=0, seed=0)
        loader_b = measurement._build_loader(data_b, 8, device=torch.device("cpu"), num_workers=0, seed=0)
        eval_loader = measurement._build_loader(eval_data, 8, device=torch.device("cpu"),
                                                num_workers=0, seed=0, is_train=False)
        options = dict(
            device=torch.device("cpu"), modular=False, eq_position=None,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1), scheduler=None,
            epochs=2, report_interval=1, csv_path=str(output / "optimization.csv"),
            eval_loader_a=eval_loader, eval_loader_b=eval_loader, progress_enabled=False,
        )
        if objective == "normalized_two_sided":
            return measurement.train_normalized_two_sided(
                model, loader_a, loader_b, nn.CrossEntropyLoss(),
                gradient_geometry_csv_path=str(output / "geometry.csv"), **options,
            )
        return measurement.train_symmetric_objective(
            model, loader_a, loader_b, nn.CrossEntropyLoss(), objective_mode=objective, **options,
        )

    for objective in ("normalized_two_sided", "difference", "mean_relative_gap"):
        model = copy.deepcopy(original)
        expected_rows[objective] = run(model, objective, Path(directory) / f"expected_{rank}_{objective}")
        expected_states[objective] = copy.deepcopy(model.state_dict())

    dist.init_process_group("gloo", rank=rank, world_size=size,
                            init_method=f"file://{directory}/group", timeout=timedelta(seconds=90))
    try:
        for objective in expected_states:
            model = copy.deepcopy(original)
            actual_rows = run(model, objective, Path(directory) / f"distributed_{objective}")
            for name, expected in expected_states[objective].items():
                torch.testing.assert_close(model.state_dict()[name], expected, atol=1e-7, rtol=1e-5)
            for actual, expected in zip(actual_rows, expected_rows[objective], strict=True):
                for name in ("loss_a", "loss_b", "running_train_loss", "running_val_loss"):
                    assert actual[name] == pytest.approx(expected[name], rel=1e-5)

        setup = MLSetup(model=copy.deepcopy(original), model_type=ModelType.lenet5,
                        dataset_type=DatasetType.cifar10, criterion=nn.CrossEntropyLoss(), default_batch_size=8)
        bundle = measurement.DatasetBundle(
            partition_a=data_a, partition_b=data_b, ml_setup=setup, dataset_name="cifar10",
            dataset_type_name="cifar10", source_examples=32, permutation=torch.arange(32),
            excluded_indices=torch.empty(0, dtype=torch.int64), modular=False,
        )
        output = Path(directory) / "full_run"
        argv = ["measurement", "--dataset", "cifar10", "--device", "cpu", "--batch_size", "8",
                "--epochs", "2", "--num_workers", "0", "--no-compile", "--no-amp",
                "--output_folder_name", str(output)]
        # No seed supplied: the entry point must broadcast the seed and split.
        with patch.object(sys, "argv", argv), patch.object(measurement, "load_dataset_bundle", return_value=bundle):
            measurement.main()
        distributed.barrier()
        if rank == 0:
            summary = json.loads((output / "summary.json").read_text())
            assert summary["aggregate"]["mean_misalignment_measure"] is None
            assert len(summary["runs"]) == 2
            for name, run in summary["runs"].items():
                assert run["relative_flatness_status"] == "deferred"
                assert run["runtime"]["world_size"] == size
                assert (output / f"final_{name}.model.pt").is_file()
                assert len(run["gap_checkpoints"]["union"]) == 2
            assert not list(output.glob("relative_flatness*.csv"))
    finally:
        distributed.shutdown()


@pytest.mark.parametrize("size", [2, 4])
def test_misalignment_matches_global_gradients_and_saves_models(size):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_misalignment_worker, args=(size, directory), nprocs=size, join=True)


def _generation_worker(rank, size, directory):
    from py_src.ml_setup import ApplicationType, MLSetup
    from py_src.ml_setup_dataset import DatasetType
    from py_src.ml_setup_model import ModelType
    from tool import generate_high_accuracy_model as generation

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=size,
                            init_method=f"file://{directory}/group", timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(4)
        data = TensorDataset(torch.randn(8, 3), torch.arange(8) % 2)

        def factory(*args, **kwargs):
            model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.1), nn.Linear(5, 2))
            return MLSetup(
                model=model, adapter=StandardAdapter(model, nn.CrossEntropyLoss()),
                model_type=ModelType.lenet5, dataset_type=DatasetType.mnist,
                application_type=ApplicationType.classifier, training_data=data, testing_data=data,
                criterion=nn.CrossEntropyLoss(), default_batch_size=4,
            )

        output = Path(directory) / "generation"
        argv = ["generation", "--cpu", "--no-compile", "--batch_size", "4", "--enable_eval",
                "--random_seed", "8", "--epoch", "1", "--core", str(size), "-o", str(output)]
        with patch.object(sys, "argv", argv), patch.object(generation, "get_ml_setup_from_config", side_effect=factory):
            generation.main()
        distributed.barrier()
        path = output / "0" / "training_checkpoint.pt"
        argv = ["generation", "--load_checkpoint", str(path), "--epoch", "2", "--core", str(size)]
        with patch.object(sys, "argv", argv), patch.object(generation, "get_ml_setup_from_config", side_effect=factory):
            generation.main()
        distributed.barrier()
        if rank == 0:
            checkpoint = generation.load_training_checkpoint(str(path))
            assert checkpoint["completed_epoch"] == 1
            assert checkpoint["run_config"]["batch_size"] == 4
            assert len((output / "0.log.csv").read_text().splitlines()) == 3
            assert (output / "0.model.pt").is_file()
    finally:
        distributed.shutdown()


def test_generation_entry_point_and_resume():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_generation_worker, args=(2, directory), nprocs=2, join=True)
