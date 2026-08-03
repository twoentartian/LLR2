"""Node implementation used by the decentralized learning simulator."""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from py_src import util
from py_src.adapters import clone_adapter_for_model
from py_src.engine import Device
from py_src.ml_setup import MLSetup
from py_src.ml_setup_dataset.dataset_intermediate_layer import (
    DatasetWithFastLabelSelection,
)
from py_src.model_average import ModelAverager


def _infer_dataset_labels(dataset: Any) -> list[int]:
    if dataset is None:
        return []

    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
        if hasattr(base_dataset, "targets"):
            targets = np.asarray(base_dataset.targets)
            return sorted(set(int(targets[index]) for index in indices))

    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if isinstance(labels, set):
            return sorted(int(label) for label in labels)
        try:
            return sorted(int(label) for label in set(labels))
        except TypeError:
            pass

    if hasattr(dataset, "classes"):
        classes = getattr(dataset, "classes")
        try:
            return list(range(len(classes)))
        except TypeError:
            pass

    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if torch.is_tensor(targets):
            return sorted(set(int(value) for value in targets.detach().cpu().view(-1).tolist()))
        if isinstance(targets, np.ndarray):
            return sorted(set(int(value) for value in targets.reshape(-1).tolist()))
        return sorted(set(int(value) for value in targets))

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        labels: set[int] = set()
        for index in range(len(dataset)):
            sample = dataset[index]
            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                continue
            label = sample[1]
            if torch.is_tensor(label):
                if label.numel() == 1:
                    labels.add(int(label.item()))
            elif isinstance(label, np.ndarray):
                if label.size == 1:
                    labels.add(int(label.item()))
            elif isinstance(label, (int, np.integer)):
                labels.add(int(label))
        return sorted(labels)

    return []


def _infer_tensor_size(dataset: Any) -> Optional[torch.Size]:
    if dataset is None:
        return None

    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset

    if hasattr(dataset, "get_first_data_tensor"):
        first = dataset.get_first_data_tensor()
        if torch.is_tensor(first):
            return first.shape

    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__") and len(dataset) > 0:
        sample = dataset[0]
        data = sample[0] if isinstance(sample, (tuple, list)) and sample else sample
        if torch.is_tensor(data):
            return data.shape

    return None


def _assign_names_to_layers(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        if not hasattr(module, "_module_name"):
            module._module_name = name


def _re_initialize_model_for_simulation(model: torch.nn.Module, ml_setup: MLSetup) -> None:
    _assign_names_to_layers(model)

    original_reinitialize = getattr(ml_setup, "_original_re_initialize_model", None)
    if callable(original_reinitialize):
        original_reinitialize(model)
        return

    weights_init_func = getattr(ml_setup, "weights_init_func", None)
    if weights_init_func is not None:
        model.apply(weights_init_func)
        return

    seed = int.from_bytes(os.urandom(4), byteorder="big")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    util.re_initialize_model(model)


def ensure_ml_setup_compatibility(ml_setup: MLSetup) -> MLSetup:
    """Add the legacy attributes expected by simulator configs and services."""
    primary_dataset = (
        ml_setup.training_data
        if ml_setup.training_data is not None
        else ml_setup.testing_data
    )
    if not hasattr(ml_setup, "_original_re_initialize_model"):
        original_reinitialize = getattr(ml_setup, "re_initialize_model", None)
        ml_setup._original_re_initialize_model = (  # type: ignore[attr-defined]
            original_reinitialize if callable(original_reinitialize) else None
        )
    if not hasattr(ml_setup, "model_name"):
        ml_setup.model_name = (  # type: ignore[attr-defined]
            ml_setup.model_type.name
            if ml_setup.model_type is not None
            else type(ml_setup.model).__name__
        )
    if not hasattr(ml_setup, "dataset_name"):
        ml_setup.dataset_name = (  # type: ignore[attr-defined]
            ml_setup.dataset_type.name if ml_setup.dataset_type is not None else "unknown"
        )
    if not hasattr(ml_setup, "training_batch_size"):
        ml_setup.training_batch_size = ml_setup.default_batch_size  # type: ignore[attr-defined]
    if not ml_setup.dataset_label:
        ml_setup.dataset_label = _infer_dataset_labels(primary_dataset)
    if not hasattr(ml_setup, "dataset_tensor_size"):
        ml_setup.dataset_tensor_size = _infer_tensor_size(primary_dataset)  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "collate_fn"):
        ml_setup.collate_fn = ml_setup.default_collate_fn  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "collate_fn_val"):
        ml_setup.collate_fn_val = ml_setup.default_collate_fn_val  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "sampler_fn"):
        ml_setup.sampler_fn = ml_setup.default_sampler_fn  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "override_training_dataset_loader"):
        ml_setup.override_training_dataset_loader = ml_setup.override_train_loader  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "override_testing_dataset_loader"):
        ml_setup.override_testing_dataset_loader = ml_setup.override_test_loader  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "weights_init_func"):
        ml_setup.weights_init_func = None  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "func_handler_post_training"):
        ml_setup.func_handler_post_training = []  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "self_validate"):
        ml_setup.self_validate = lambda: None  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "re_initialize_model"):
        ml_setup.re_initialize_model = (  # type: ignore[attr-defined]
            lambda model: _re_initialize_model_for_simulation(model, ml_setup)
        )
    if not hasattr(ml_setup, "get_brief_description"):
        ml_setup.get_brief_description = (  # type: ignore[attr-defined]
            lambda: f"{ml_setup.model_name}@{ml_setup.dataset_name}"
        )
    return ml_setup


def _move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class Node:
    def __init__(
        self,
        name: int,
        ml_setup: MLSetup,
        *,
        device: Device,
        use_model_stat: bool | None = None,
        allocated_gpu: Any = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        use_cpu: bool = False,
        enable_amp: bool = False,
    ):
        del use_model_stat, allocated_gpu, optimizer

        self.name = name
        self.use_cpu = use_cpu
        self.device_obj = device if not use_cpu else Device.cpu()
        self.device = self.device_obj.device
        self.is_using_model_stat = False

        model = copy.deepcopy(ml_setup.model)
        _re_initialize_model_for_simulation(model, ml_setup)
        self.model = model.to(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.scaler = (
            self.device_obj.make_scaler()
            if enable_amp and self.device.type == "cuda"
            else None
        )
        self.ml_setup = ensure_ml_setup_compatibility(copy.copy(ml_setup))
        self._adapter = clone_adapter_for_model(
            self.ml_setup.adapter,
            self.model,
            criterion=self.ml_setup.criterion,
        )

        self.next_training_tick = 0
        self.normalized_dataset_label_distribution = None
        self.train_loader: Optional[Iterable] = None
        self._dataloader_worker: Optional[int] = None

        self._dataset_label_distribution = None
        self._dataset_with_fast_label: Optional[DatasetWithFastLabelSelection] = None

        self.model_averager: Optional[ModelAverager] = None
        self.model_buffer_size: Optional[int] = None

        self.is_training_this_tick = False
        self.is_averaging_this_tick = False

        self.num_of_batch_per_training = 1
        self.send_model_after_P_training = 1
        self._send_model_counter = 0
        self.most_recent_loss = torch.tensor(0.0, device=self.device)
        self.most_recent_accuracy = 0.0
        self.most_recent_lrs: list[float] = []

        self.enable_receiving = True
        self.enable_training = True
        self.enable_sending = True
        self.enable_averaging = True

    def is_sending_model(self) -> bool:
        self._send_model_counter += 1
        if self._send_model_counter >= self.send_model_after_P_training:
            self._send_model_counter = 0
            return True
        return False

    def reset_status_flags(self) -> None:
        self.is_training_this_tick = False
        self.is_averaging_this_tick = False

    def set_average_algorithm(self, average_algorithm: ModelAverager) -> None:
        self.model_averager = average_algorithm

    def set_average_buffer_size(self, average_buffer_size: int) -> None:
        self.model_buffer_size = average_buffer_size

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def set_lr_scheduler(
        self,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        self.lr_scheduler = lr_scheduler

    def set_ml_setup(self, setup: MLSetup) -> None:
        self.ml_setup = ensure_ml_setup_compatibility(copy.copy(setup))
        self._adapter = clone_adapter_for_model(
            self.ml_setup.adapter,
            self.model,
            criterion=self.ml_setup.criterion,
        )
        if self._dataset_with_fast_label is not None:
            self.set_label_distribution(
                self._dataset_label_distribution,
                self._dataset_with_fast_label,
            )

    def set_batch_size(self, batch_size: int) -> None:
        new_setup = copy.copy(self.ml_setup)
        new_setup.training_batch_size = batch_size
        new_setup.default_batch_size = batch_size
        self.set_ml_setup(new_setup)

    def set_next_training_tick(self, tick: int) -> None:
        self.next_training_tick = tick

    def set_label_distribution(
        self,
        dataset_label_distribution=None,
        dataset_with_fast_label: Optional[DatasetWithFastLabelSelection] = None,
        worker: Optional[int] = None,
    ) -> None:
        self._dataset_label_distribution = dataset_label_distribution
        self._dataloader_worker = worker
        self.train_loader = None
        if dataset_with_fast_label is not None:
            self._dataset_with_fast_label = dataset_with_fast_label
        elif self._dataset_label_distribution is not None and self._dataset_with_fast_label is None:
            raise ValueError(
                "dataset_with_fast_label must be provided on first "
                "set_label_distribution call"
            )

        assert self._dataset_with_fast_label is not None
        if dataset_label_distribution is None:
            self.normalized_dataset_label_distribution = None
        else:
            label_distribution = np.asarray(
                dataset_label_distribution,
                dtype=np.float64,
            )
            total = label_distribution.sum()
            if total <= 0:
                raise ValueError(
                    f"node {self.name} received an invalid label distribution "
                    "with non-positive sum"
                )
            self.normalized_dataset_label_distribution = label_distribution / total

    def set_model_stat(self, model_stat: dict) -> None:
        self.model.load_state_dict(model_stat, strict=True)

    def set_optimizer_stat(self, optimizer_stat: dict) -> None:
        if self.optimizer is None:
            raise ValueError("optimizer is not initialized")
        self.optimizer.load_state_dict(optimizer_stat)
        _move_optimizer_to_device(self.optimizer, self.device)

    def set_lr_scheduler_stat(self, lr_scheduler_stat: dict) -> None:
        if self.lr_scheduler is None:
            raise ValueError("lr scheduler is not initialized")
        self.lr_scheduler.load_state_dict(lr_scheduler_stat)

    def get_dataset_label_distribution(self):
        return self.normalized_dataset_label_distribution

    def get_data_loader(self) -> Iterable:
        """Build a dedicated loader lazily for legacy callers.

        The simulator uses ``sample_training_indices`` and its shared loader,
        so this compatibility path does not create per-node worker pools during
        normal simulation.
        """

        if self._dataset_with_fast_label is None:
            raise ValueError(f"node {self.name} training dataset is not initialized")
        if self.train_loader is None:
            batch_size = self.ml_setup.training_batch_size
            if self.normalized_dataset_label_distribution is None:
                self.train_loader = (
                    self._dataset_with_fast_label.get_train_loader_default(
                        batch_size,
                        worker=self._dataloader_worker,
                    )
                )
            else:
                self.train_loader = (
                    self._dataset_with_fast_label.get_train_loader_by_label_prob(
                        self.normalized_dataset_label_distribution,
                        batch_size,
                        worker=self._dataloader_worker,
                    )
                )
        return self.train_loader

    def sample_training_indices(self, sample_count: int) -> list[int]:
        if self._dataset_with_fast_label is None:
            raise ValueError(f"node {self.name} training dataset is not initialized")
        return self._dataset_with_fast_label.sample_indices(
            self.normalized_dataset_label_distribution,
            sample_count,
        )

    def get_model_stat(self) -> dict:
        return {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

    def get_communication_payload(self) -> dict:
        """Return model state plus any algorithm-specific auxiliary state."""

        model_stat = self.get_model_stat()
        if self.model_averager is None:
            return model_stat
        return self.model_averager.get_communication_payload(model_stat)

    def add_model_to_buffer(self, model_stat: dict) -> None:
        if self.enable_receiving and self.model_averager is not None:
            self.model_averager.add_model(model_stat)

    def check_averaging(self) -> bool:
        if (
            not self.enable_averaging
            or self.model_averager is None
            or self.model_buffer_size is None
        ):
            return False
        received_model_count = self.model_averager.get_model_count()
        if received_model_count == 0:
            return False
        if self.model_buffer_size <= received_model_count:
            averaged_model = self.model_averager.get_model(
                self_model=self.get_model_stat()
            )
            self.set_model_stat(averaged_model)
            self.model_averager.on_after_averaging(self.optimizer)
            self.is_averaging_this_tick = True
            return True
        return False
