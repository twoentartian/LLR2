from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import logging
import os
import pickle
import random
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from py_src import model_average as model_average_module
from py_src import util
from py_src.adapters import clone_adapter_for_model
from py_src.engine import Device, train as engine_train
from py_src.ml_setup import MLSetup
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.model_variance_correct import VarianceCorrectionType, VarianceCorrector
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.special_torch_layers import (
    is_ignored_layer_averaging,
    is_ignored_layer_variance_correction,
)


LOGGER_NAME = "SimulatorBase"
LOG_FILE_NAME = "info.log"
BACKUP_DIR_NAME = "backup"
CONFIG_MODULE_NAME = "config"
REPORT_FINISH_TIME_PER_TICK = 100
DEFAULT_DEVICE = "auto"

SIMULATOR_LOGGER = logging.getLogger(LOGGER_NAME)


def _import_networkx():
    try:
        import networkx as nx  # type: ignore
    except ImportError as exc:
        raise ImportError("networkx is required for simulator topologies. Install it with `pip install networkx`.") from exc
    return nx


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
        if isinstance(sample, (tuple, list)) and sample:
            data = sample[0]
        else:
            data = sample
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

    if hasattr(ml_setup, "weights_init_func") and getattr(ml_setup, "weights_init_func") is not None:
        model.apply(ml_setup.weights_init_func)
        return

    random_data = os.urandom(4)
    seed = int.from_bytes(random_data, byteorder="big")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    util.re_initialize_model(model)


def _ensure_ml_setup_compatibility(ml_setup: MLSetup) -> MLSetup:
    primary_dataset = ml_setup.training_data if ml_setup.training_data is not None else ml_setup.testing_data
    if not hasattr(ml_setup, "_original_re_initialize_model"):
        original_reinitialize = getattr(ml_setup, "re_initialize_model", None)
        ml_setup._original_re_initialize_model = original_reinitialize if callable(original_reinitialize) else None
    if not hasattr(ml_setup, "model_name"):
        ml_setup.model_name = ml_setup.model_type.name if ml_setup.model_type is not None else type(ml_setup.model).__name__
    if not hasattr(ml_setup, "dataset_name"):
        ml_setup.dataset_name = ml_setup.dataset_type.name if ml_setup.dataset_type is not None else "unknown"
    if not hasattr(ml_setup, "training_batch_size"):
        ml_setup.training_batch_size = ml_setup.default_batch_size
    if not hasattr(ml_setup, "dataset_label"):
        ml_setup.dataset_label = _infer_dataset_labels(primary_dataset)
    if not hasattr(ml_setup, "dataset_tensor_size"):
        ml_setup.dataset_tensor_size = _infer_tensor_size(primary_dataset)
    if not hasattr(ml_setup, "collate_fn"):
        ml_setup.collate_fn = ml_setup.default_collate_fn
    if not hasattr(ml_setup, "collate_fn_val"):
        ml_setup.collate_fn_val = ml_setup.default_collate_fn_val
    if not hasattr(ml_setup, "sampler_fn"):
        ml_setup.sampler_fn = ml_setup.default_sampler_fn
    if not hasattr(ml_setup, "override_training_dataset_loader"):
        ml_setup.override_training_dataset_loader = ml_setup.override_train_loader
    if not hasattr(ml_setup, "override_testing_dataset_loader"):
        ml_setup.override_testing_dataset_loader = ml_setup.override_test_loader
    if not hasattr(ml_setup, "weights_init_func"):
        ml_setup.weights_init_func = None
    if not hasattr(ml_setup, "func_handler_post_training"):
        ml_setup.func_handler_post_training = []
    if not hasattr(ml_setup, "self_validate"):
        ml_setup.self_validate = lambda: None  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "re_initialize_model"):
        ml_setup.re_initialize_model = lambda model: _re_initialize_model_for_simulation(model, ml_setup)  # type: ignore[attr-defined]
    if not hasattr(ml_setup, "get_brief_description"):
        ml_setup.get_brief_description = lambda: f"{ml_setup.model_name}@{ml_setup.dataset_name}"  # type: ignore[attr-defined]
    return ml_setup


class LabelProbabilitySampler(Sampler[int]):
    def __init__(self, label_probabilities: np.ndarray, indices_by_label: dict[int, np.ndarray], num_samples: int):
        self.label_probabilities = np.asarray(label_probabilities, dtype=np.float64)
        self.indices_by_label = indices_by_label
        self.num_samples = int(num_samples)
        self.labels_in_order = list(indices_by_label.keys())

    def __iter__(self):
        for _ in range(self.num_samples):
            label_index = np.random.choice(len(self.labels_in_order), p=self.label_probabilities)
            label = self.labels_in_order[label_index]
            yield int(np.random.choice(self.indices_by_label[label]))

    def __len__(self) -> int:
        return self.num_samples


def _resolve_sampler(dataset: Any, sampler_or_factory: Any) -> Optional[Sampler]:
    if sampler_or_factory is None:
        return None
    if isinstance(sampler_or_factory, Sampler):
        return sampler_or_factory
    if callable(sampler_or_factory):
        return sampler_or_factory(dataset)
    return sampler_or_factory


class DatasetWithFastLabelSelection:
    def __init__(self, dataset: Any, ml_setup: MLSetup):
        self.raw_dataset = dataset
        self.ml_setup = ml_setup
        self.labels = np.asarray(_infer_dataset_labels_for_indices(dataset))
        if self.labels.size == 0:
            raise ValueError("Dataset label selection requires a map-style dataset with accessible labels")
        indices = np.arange(len(self.labels))
        self.indices_by_label: dict[int, np.ndarray] = {}
        for label in sorted(set(int(value) for value in self.labels.tolist())):
            self.indices_by_label[label] = indices[self.labels == label]

    def get_train_loader_by_label_prob(self, label_prob: np.ndarray, batch_size: int, worker: Optional[int] = None) -> Iterable:
        if hasattr(self.raw_dataset, "build_dataloader"):
            raise NotImplementedError("label-distribution sampling is not supported for datasets with custom build_dataloader() backends")

        sampler = LabelProbabilitySampler(label_prob, self.indices_by_label, batch_size)
        loader_kwargs = _build_dataloader_kwargs(
            batch_size=batch_size,
            num_workers=worker,
            collate_fn=self.ml_setup.collate_fn,
            sampler=sampler,
            shuffle=False,
        )
        return DataLoader(self.raw_dataset, **loader_kwargs)

    def get_train_loader_default(self, batch_size: int, worker: Optional[int] = None) -> Iterable:
        if self.ml_setup.override_training_dataset_loader is not None:
            return self.ml_setup.override_training_dataset_loader

        if hasattr(self.raw_dataset, "build_dataloader"):
            config = DataloaderConfig(
                batch_size=batch_size,
                num_workers=worker or 0,
                shuffle=True,
                pin_memory=True,
                prefetch_factor=4 if (worker or 0) > 0 else None,
                persistent_workers=(worker or 0) > 0,
            )
            return self.raw_dataset.build_dataloader(default_batch_size=batch_size, config=config, is_train=True)

        sampler = _resolve_sampler(self.raw_dataset, self.ml_setup.sampler_fn)
        loader_kwargs = _build_dataloader_kwargs(
            batch_size=batch_size,
            num_workers=worker,
            collate_fn=self.ml_setup.collate_fn,
            sampler=sampler,
            shuffle=sampler is None,
        )
        return DataLoader(self.raw_dataset, **loader_kwargs)


def _infer_dataset_labels_for_indices(dataset: Any) -> list[int]:
    if dataset is None:
        return []

    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, "targets"):
            targets = np.asarray(base.targets)
            return [int(targets[index]) for index in dataset.indices]

    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if torch.is_tensor(targets):
            return [int(value) for value in targets.detach().cpu().view(-1).tolist()]
        if isinstance(targets, np.ndarray):
            return [int(value) for value in targets.reshape(-1).tolist()]
        return [int(value) for value in targets]

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        labels: list[int] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                label = sample[1]
                if torch.is_tensor(label):
                    labels.append(int(label.item()))
                else:
                    labels.append(int(label))
        return labels

    return []


def _build_dataloader_kwargs(
    *,
    batch_size: int,
    num_workers: Optional[int],
    collate_fn: Any,
    sampler: Optional[Sampler],
    shuffle: bool,
) -> dict[str, Any]:
    worker_count = 0 if num_workers is None else int(num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": worker_count,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if worker_count > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return kwargs


class ModelAverager:
    def __init__(self, variance_corrector: Optional[VarianceCorrector] = None):
        self.variance_corrector = variance_corrector

    def add_model(self, model_stat: dict) -> None:
        raise NotImplementedError

    def get_model(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def get_model_count(self) -> int:
        raise NotImplementedError

    @staticmethod
    def _iadd_two_model(src: dict, addition: dict, *, weight_src: float = 1.0, weight_addition: float = 1.0, check_same_keys: bool = True) -> dict:
        with torch.no_grad():
            assert (not check_same_keys) or (set(src.keys()) == set(addition.keys()))
            for layer_name in src.keys():
                addition_tensor = addition[layer_name]
                if src[layer_name].device != addition_tensor.device:
                    addition_tensor = addition_tensor.to(src[layer_name].device)
                if weight_src == 1.0 and weight_addition == 1.0:
                    src[layer_name] += addition_tensor
                else:
                    src[layer_name] = src[layer_name] * weight_src + addition_tensor * weight_addition
        return src

    @staticmethod
    def _move_state_dict(state_dict: dict, device: torch.device) -> None:
        for key, value in state_dict.items():
            state_dict[key] = value.to(device)

    @staticmethod
    def _get_device_from_model_stat(state_dict: dict) -> torch.device:
        return next(iter(state_dict.values())).device


class StandardModelAverager(ModelAverager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variance_corrector is not None:
            vc_type = self.variance_corrector.variance_correction_type
            assert vc_type == VarianceCorrectionType.FollowOthers
        self.model_buffer: Optional[dict] = None
        self.model_counter = 0

    def add_model(self, model_stat: dict) -> None:
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                for layer_name in list(self.model_buffer.keys()):
                    if is_ignored_layer_averaging(layer_name):
                        del self.model_buffer[layer_name]
            else:
                self.model_buffer = ModelAverager._iadd_two_model(self.model_buffer, model_stat, check_same_keys=False)
            self.model_counter += 1
            if self.variance_corrector is not None:
                self.variance_corrector.add_variance(model_stat)

    def get_model(self, self_model: dict, *args, **kwargs) -> dict:
        assert self.model_buffer is not None
        with torch.no_grad():
            device = ModelAverager._get_device_from_model_stat(self.model_buffer)
            self_model = copy.deepcopy(self_model)
            ModelAverager._move_state_dict(self_model, device)

            output = copy.deepcopy(self_model)
            for layer_name in output:
                if layer_name not in self.model_buffer:
                    continue
                output[layer_name] = self.model_buffer[layer_name] / self.model_counter

            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance()
                for layer_name, single_layer_variance in target_variance.items():
                    if is_ignored_layer_variance_correction(layer_name):
                        continue
                    output[layer_name] = VarianceCorrector.scale_tensor_to_variance(output[layer_name], single_layer_variance)

            self.model_buffer = None
            self.model_counter = 0
            return output

    def get_model_count(self) -> int:
        return self.model_counter


class ConservativeModelAverager(ModelAverager):
    def __init__(self, conservative: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.0 <= conservative <= 1.0
        self.conservative = conservative
        self.model_buffer: Optional[dict] = None
        self.model_counter = 0

    def add_model(self, model_stat: dict) -> None:
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                for layer_name in list(self.model_buffer.keys()):
                    if is_ignored_layer_averaging(layer_name):
                        del self.model_buffer[layer_name]
            else:
                self.model_buffer = ModelAverager._iadd_two_model(self.model_buffer, model_stat, check_same_keys=False)
            self.model_counter += 1
            if self.variance_corrector is not None:
                self.variance_corrector.add_variance(model_stat)

    def get_model(self, self_model: dict, *args, **kwargs) -> dict:
        assert self.model_buffer is not None
        with torch.no_grad():
            device = ModelAverager._get_device_from_model_stat(self.model_buffer)
            self_model = copy.deepcopy(self_model)
            ModelAverager._move_state_dict(self_model, device)

            averaged = copy.deepcopy(self_model)
            for layer_name in averaged:
                if layer_name not in self.model_buffer:
                    continue
                averaged[layer_name] = self.model_buffer[layer_name] / self.model_counter

            output = ModelAverager._iadd_two_model(
                self_model,
                averaged,
                weight_src=self.conservative,
                weight_addition=1 - self.conservative,
            )
            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance(self_model, self.conservative)
                for layer_name, single_layer_variance in target_variance.items():
                    if is_ignored_layer_variance_correction(layer_name):
                        continue
                    output[layer_name] = VarianceCorrector.scale_tensor_to_variance(output[layer_name], single_layer_variance)

            self.model_buffer = None
            self.model_counter = 0
            return output

    def get_model_count(self) -> int:
        return self.model_counter


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
        self.scaler = self.device_obj.make_scaler() if enable_amp and self.device.type == "cuda" else None
        self.ml_setup = _ensure_ml_setup_compatibility(copy.copy(ml_setup))
        self._adapter = clone_adapter_for_model(self.ml_setup.adapter, self.model, criterion=self.ml_setup.criterion)

        self.next_training_tick = 0
        self.normalized_dataset_label_distribution = None
        self.train_loader: Optional[Iterable] = None

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

    def set_lr_scheduler(self, lr_scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def set_ml_setup(self, setup: MLSetup) -> None:
        self.ml_setup = _ensure_ml_setup_compatibility(copy.copy(setup))
        self._adapter = clone_adapter_for_model(self.ml_setup.adapter, self.model, criterion=self.ml_setup.criterion)
        if self._dataset_with_fast_label is not None:
            self.set_label_distribution(self._dataset_label_distribution, self._dataset_with_fast_label)

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
        if dataset_with_fast_label is not None:
            self._dataset_with_fast_label = dataset_with_fast_label
        elif self._dataset_label_distribution is not None and self._dataset_with_fast_label is None:
            raise ValueError("dataset_with_fast_label must be provided on first set_label_distribution call")

        assert self._dataset_with_fast_label is not None
        batch_size = self.ml_setup.training_batch_size
        if dataset_label_distribution is None:
            self.normalized_dataset_label_distribution = None
            self.train_loader = self._dataset_with_fast_label.get_train_loader_default(batch_size, worker=worker)
        else:
            label_distribution = np.asarray(dataset_label_distribution, dtype=np.float64)
            total = label_distribution.sum()
            if total <= 0:
                raise ValueError(f"node {self.name} received an invalid label distribution with non-positive sum")
            self.normalized_dataset_label_distribution = label_distribution / total
            self.train_loader = self._dataset_with_fast_label.get_train_loader_by_label_prob(
                self.normalized_dataset_label_distribution,
                batch_size,
                worker=worker,
            )

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
        if self.train_loader is None:
            raise ValueError(f"node {self.name} training loader is not initialized")
        return self.train_loader

    def get_model_stat(self) -> dict:
        return {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

    def add_model_to_buffer(self, model_stat: dict) -> None:
        if self.enable_receiving and self.model_averager is not None:
            self.model_averager.add_model(model_stat)

    def check_averaging(self) -> bool:
        if not self.enable_averaging or self.model_averager is None or self.model_buffer_size is None:
            return False
        received_model_count = self.model_averager.get_model_count()
        if received_model_count == 0:
            return False
        if self.model_buffer_size <= received_model_count:
            averaged_model = self.model_averager.get_model(self_model=self.get_model_stat())
            self.set_model_stat(averaged_model)
            self.is_averaging_this_tick = True
            return True
        return False


def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def global_broadcast(runtime_parameters: RuntimeParameters, src_node_name: int, mpi_world=None) -> None:
    del mpi_world
    assert src_node_name in runtime_parameters.node_container, (
        f"node {src_node_name} does not exist in the network: {runtime_parameters.node_container.keys()}"
    )
    src_model_stat = runtime_parameters.node_container[src_node_name].get_model_stat()
    for node_name, node_target in runtime_parameters.node_container.items():
        if src_node_name == node_name:
            continue
        node_target.set_model_stat(src_model_stat)


def split_to_equal_size_communities(topology: nx.Graph, num_of_communities: int):
    _import_networkx()
    from networkx.algorithms.community import kernighan_lin_bisection

    communities = [list(component) for component in kernighan_lin_bisection(topology)]
    while len(communities) < num_of_communities:
        new_communities = []
        for community in communities:
            subgraph = topology.subgraph(community)
            if len(subgraph) > 1:
                new_communities.extend([list(component) for component in kernighan_lin_bisection(subgraph)])
            else:
                new_communities.append(community)
        communities = new_communities

    while len(communities) > num_of_communities:
        communities = sorted(communities, key=len)
        merged_community = communities[0] + communities[1]
        communities = [merged_community] + communities[2:]

    while True:
        largest = max(communities, key=len)
        smallest = min(communities, key=len)
        if len(largest) - len(smallest) <= 1:
            break
        smallest.append(largest.pop())

    return communities


def get_inter_community_edges(topology: nx.Graph, communities) -> list[tuple[int, int]]:
    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    inter_community_edges = []
    for source, destination in topology.edges():
        if node_to_community[source] != node_to_community[destination]:
            inter_community_edges.append((source, destination))
    return inter_community_edges


def load_topology_from_edge_list_file(file_path: str) -> nx.Graph:
    nx = _import_networkx()
    graph = nx.Graph()
    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            node_1, node_2 = map(int, line.split())
            graph.add_edge(node_1, node_2)
    return graph


def _install_legacy_compat_modules() -> None:
    import py_src
    import py_src.ml_setup as ml_setup_module
    import py_src.node as node_module

    ml_setup_module.MlSetup = MLSetup  # type: ignore[attr-defined]
    node_module.Node = Node  # type: ignore[attr-defined]

    model_average_module.ModelAverager = ModelAverager  # type: ignore[attr-defined]
    model_average_module.StandardModelAverager = StandardModelAverager  # type: ignore[attr-defined]
    model_average_module.ConservativeModelAverager = ConservativeModelAverager  # type: ignore[attr-defined]

    config_file_util_module = types.ModuleType("py_src.config_file_util")
    config_file_util_module.__path__ = []  # type: ignore[attr-defined]

    label_distribution_module = types.ModuleType("py_src.config_file_util.label_distribution")
    label_distribution_module.label_distribution_default = lambda target_node, parameters: None
    label_distribution_module.label_distribution_iid = (
        lambda target_node, parameters: np.repeat(1, len(parameters.dataset_label))
    )
    label_distribution_module.label_distribution_non_iid_dirichlet = (
        lambda target_node, parameters, alpha: np.random.dirichlet(np.repeat(alpha, len(parameters.dataset_label)))
    )
    label_distribution_module.label_distribution_first_half = _label_distribution_first_half
    label_distribution_module.label_distribution_second_half = _label_distribution_second_half

    node_behavior_control_module = types.ModuleType("py_src.node_behavior_control_lib")
    node_behavior_control_module.global_broadcast = global_broadcast

    nx_lib_module = types.ModuleType("py_src.nx_lib")
    nx_lib_module.split_to_equal_size_communities = split_to_equal_size_communities
    nx_lib_module.get_inter_community_edges = get_inter_community_edges
    nx_lib_module.load_topology_from_edge_list_file = load_topology_from_edge_list_file

    sys.modules["py_src.config_file_util"] = config_file_util_module
    sys.modules["py_src.config_file_util.label_distribution"] = label_distribution_module
    sys.modules["py_src.node_behavior_control_lib"] = node_behavior_control_module
    sys.modules["py_src.nx_lib"] = nx_lib_module

    setattr(config_file_util_module, "label_distribution", label_distribution_module)
    setattr(py_src, "config_file_util", config_file_util_module)
    setattr(py_src, "node_behavior_control_lib", node_behavior_control_module)
    setattr(py_src, "nx_lib", nx_lib_module)


def _label_distribution_first_half(target_node, parameters):
    del target_node
    size = len(parameters.dataset_label) // 2
    return np.concatenate(
        [np.repeat(1, size), np.repeat(0, len(parameters.dataset_label) - size)],
        axis=None,
    )


def _label_distribution_second_half(target_node, parameters):
    del target_node
    size = len(parameters.dataset_label) // 2
    return np.concatenate(
        [np.repeat(0, size), np.repeat(1, len(parameters.dataset_label) - size)],
        axis=None,
    )


def _load_module_from_path(module_name: str, module_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_configuration(config_file_path: str):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"config file ({config_file_path}) does not exist")
    spec = importlib.util.spec_from_file_location(CONFIG_MODULE_NAME, config_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import config file {config_file_path}")
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[CONFIG_MODULE_NAME] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def check_consistent_nodes(topology_generation_function, total_tick: int) -> set[int]:
    parameters = RuntimeParameters()
    parameters.max_tick = total_tick
    parameters.phase = SimulationPhase.INITIALIZING
    topology = topology_generation_function(parameters)
    if topology is None:
        raise ValueError("get_topology() must return a topology during initialization")

    previous_nodes = set(topology.nodes())
    initial_edge_count = len(topology.edges())
    max_edge_count = initial_edge_count

    for tick in range(total_tick + 1):
        parameters.phase = SimulationPhase.START_OF_TICK
        parameters.current_tick = tick
        topology = topology_generation_function(parameters)
        if topology is None:
            continue
        current_nodes = set(topology.nodes())
        max_edge_count = max(max_edge_count, len(topology.edges()))
        if previous_nodes != current_nodes:
            extra_nodes = current_nodes - previous_nodes
            missing_nodes = previous_nodes - current_nodes
            SIMULATOR_LOGGER.critical(
                "nodes (count:%s) at tick %s differ from previous nodes (count:%s). Extra: %s, missing: %s",
                len(current_nodes),
                tick,
                len(previous_nodes),
                extra_nodes,
                missing_nodes,
            )
            raise SystemExit(1)

    SIMULATOR_LOGGER.info(
        "total nodes: %s, initial edges: %s, max edge count: %s",
        len(previous_nodes),
        initial_edge_count,
        max_edge_count,
    )
    return previous_nodes


def save_topology_to_file(topology: nx.Graph, current_tick: int, output_path: str) -> None:
    topology_folder = os.path.join(output_path, "topology")
    os.makedirs(topology_folder, exist_ok=True)
    with open(os.path.join(topology_folder, f"{current_tick}.pickle"), "wb") as topology_file:
        pickle.dump(topology, topology_file)


def _slice_batches(loader: Iterable, batch_limit: int):
    return itertools.islice(iter(loader), max(0, int(batch_limit)))


def _refresh_label_distributions(runtime_parameters: RuntimeParameters, config_file) -> None:
    for single_node in runtime_parameters.node_container.values():
        new_label_distribution = config_file.get_label_distribution(single_node, runtime_parameters)
        if new_label_distribution is not None:
            single_node.set_label_distribution(new_label_distribution)
            SIMULATOR_LOGGER.info(
                "update label distribution to %s for %s.",
                new_label_distribution,
                single_node.name,
            )


def _trigger_services(runtime_parameters: RuntimeParameters) -> None:
    for service_inst in runtime_parameters.service_container.values():
        service_inst.trigger(runtime_parameters)


def simulation_phase_start_of_tick(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.START_OF_TICK
    _trigger_services(runtime_parameters)
    for node_target in runtime_parameters.node_container.values():
        node_target.reset_status_flags()
    SIMULATOR_LOGGER.info("current tick: %s/%s", runtime_parameters.current_tick, runtime_parameters.max_tick)


def simulation_phase_before_training(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.BEFORE_TRAINING
    _trigger_services(runtime_parameters)


def simulation_phase_training(runtime_parameters: RuntimeParameters, config_file) -> None:
    runtime_parameters.phase = SimulationPhase.TRAINING
    _trigger_services(runtime_parameters)

    training_node_names = []
    for node_name, node_target in runtime_parameters.node_container.items():
        if node_target.next_training_tick != runtime_parameters.current_tick:
            continue

        training_node_names.append(node_name)
        train_loader = node_target.get_data_loader()
        batches = _slice_batches(train_loader, node_target.num_of_batch_per_training)
        training_batch_count = 0
        if node_target.enable_training and not getattr(runtime_parameters, "performance_disable_training", False):
            result = engine_train(
                node_target._adapter,
                batches,
                node_target.optimizer,
                node_target.lr_scheduler,
                device=node_target.device_obj,
                scaler=node_target.scaler,
                gradient_accumulate_every=getattr(node_target.ml_setup, "gradient_accumulate_every", 1),
                max_grad_norm=getattr(node_target.ml_setup, "max_grad_norm", None),
            )
            training_batch_count = result.iterations
            node_target.most_recent_loss = torch.tensor(result.avg_loss, device=node_target.device)
            node_target.most_recent_accuracy = float(result.accuracy or 0.0)
        else:
            for _ in batches:
                training_batch_count += 1

        if node_target.optimizer is not None:
            node_target.most_recent_lrs = [group.get("lr", 0.0) for group in node_target.optimizer.param_groups]
        else:
            node_target.most_recent_lrs = []

        if training_batch_count > 0 and node_target.enable_sending:
            node_target.is_training_this_tick = True

        most_recent_lrs_str = [f"{value:.3e}" for value in node_target.most_recent_lrs]
        SIMULATOR_LOGGER.info(
            "tick: %s, training node: %s for %s times, loss=%.4f, lrs=%s",
            runtime_parameters.current_tick,
            node_target.name,
            training_batch_count,
            float(node_target.most_recent_loss.detach().cpu().item()),
            most_recent_lrs_str,
        )

        for handler in getattr(node_target.ml_setup, "func_handler_post_training", []):
            handler(node_target.model)

    for node_name in training_node_names:
        node_target = runtime_parameters.node_container[node_name]
        node_target.next_training_tick = config_file.get_next_training_time(node_target, runtime_parameters)


def simulation_phase_after_training(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.AFTER_TRAINING
    _trigger_services(runtime_parameters)


def simulation_phase_before_averaging(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.BEFORE_AVERAGING
    _trigger_services(runtime_parameters)


def simulation_phase_averaging(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.AVERAGING
    _trigger_services(runtime_parameters)

    if getattr(runtime_parameters, "performance_disable_communication", False):
        return

    nodes_averaged = set()
    for node_target in runtime_parameters.node_container.values():
        if not node_target.is_training_this_tick:
            continue
        if not node_target.is_sending_model():
            continue

        model_stat = node_target.get_model_stat()
        if getattr(runtime_parameters, "average_on_cpu", False):
            model_stat = {
                key: value.detach().cpu()
                if torch.is_tensor(value)
                else value
                for key, value in model_stat.items()
            }

        for neighbor in runtime_parameters.topology.neighbors(node_target.name):
            runtime_parameters.node_container[neighbor].add_model_to_buffer(model_stat)

    for node_name in runtime_parameters.node_container:
        if runtime_parameters.node_container[node_name].check_averaging():
            nodes_averaged.add(node_name)

    if nodes_averaged:
        SIMULATOR_LOGGER.info(
            "tick: %s, averaging on %s nodes: %s",
            runtime_parameters.current_tick,
            len(nodes_averaged),
            nodes_averaged,
        )


def simulation_phase_after_averaging(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.AFTER_AVERAGING
    _trigger_services(runtime_parameters)


def simulation_phase_end_of_tick(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.END_OF_TICK
    _trigger_services(runtime_parameters)


def begin_simulation(runtime_parameters: RuntimeParameters, config_file) -> None:
    timer = datetime.now().timestamp()

    while runtime_parameters.current_tick <= config_file.max_tick:
        runtime_parameters.phase = SimulationPhase.START_OF_TICK

        new_topology = config_file.get_topology(runtime_parameters)
        if new_topology is not None:
            save_topology_to_file(new_topology, runtime_parameters.current_tick, runtime_parameters.output_path)
            runtime_parameters.topology = new_topology
            SIMULATOR_LOGGER.info("topology is updated at tick %s", runtime_parameters.current_tick)

        _refresh_label_distributions(runtime_parameters, config_file)

        if runtime_parameters.current_tick % REPORT_FINISH_TIME_PER_TICK == 0 and runtime_parameters.current_tick != 0:
            now = datetime.now().timestamp()
            time_elapsed = now - timer
            timer = now
            remaining = (config_file.max_tick - runtime_parameters.current_tick) // REPORT_FINISH_TIME_PER_TICK
            finish_time = now + remaining * time_elapsed
            SIMULATOR_LOGGER.info(
                "time taken for %s ticks: %.2fs, expected to finish at %s",
                REPORT_FINISH_TIME_PER_TICK,
                time_elapsed,
                datetime.fromtimestamp(finish_time),
            )

        simulation_phase_start_of_tick(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_before_training(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_training(runtime_parameters, config_file)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_after_training(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_before_averaging(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_averaging(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_after_averaging(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_end_of_tick(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        runtime_parameters.current_tick += 1


def _resolve_output_folder(config_file, output_folder_name: Optional[str]) -> str:
    output_folder_path = getattr(config_file, "save_name", None)
    if output_folder_path is None:
        if output_folder_name is None:
            output_folder_path = os.path.join(os.curdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
        else:
            output_folder_path = os.path.join(os.curdir, output_folder_name)
    if os.path.exists(output_folder_path):
        raise FileExistsError(f"{output_folder_path} exists.")
    os.mkdir(output_folder_path)
    backup_path = os.path.join(output_folder_path, BACKUP_DIR_NAME)
    os.mkdir(backup_path)
    return output_folder_path


def _create_device(device_name: str, force_use_cpu: bool) -> Device:
    if force_use_cpu:
        return Device.cpu()
    if device_name == DEFAULT_DEVICE:
        return Device.auto()
    return Device(device_name)


def main(
    config_file_path: str,
    output_folder_name: Optional[str] = None,
    *,
    device_name: str = DEFAULT_DEVICE,
    enable_amp: bool = False,
) -> None:
    _install_legacy_compat_modules()

    config_file = load_configuration(config_file_path)
    output_folder_path = _resolve_output_folder(config_file, output_folder_name)
    backup_path = os.path.join(output_folder_path, BACKUP_DIR_NAME)

    util.setup_logging(
        SIMULATOR_LOGGER,
        LOGGER_NAME,
        log_file_path=os.path.join(output_folder_path, LOG_FILE_NAME),
        exit_on_critical=True,
    )

    shutil.copy2(config_file_path, backup_path)
    SIMULATOR_LOGGER.info(
        "config file path: (%s), name: (%s).",
        config_file_path,
        getattr(config_file, "config_name", Path(config_file_path).name),
    )

    config_ml_setup = _ensure_ml_setup_compatibility(config_file.get_ml_setup())
    config_ml_setup.self_validate()

    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_ml_setup.dataset_label
    runtime_parameters.phase = SimulationPhase.INITIALIZING
    runtime_parameters.output_path = output_folder_path
    runtime_parameters.mpi_enabled = False
    runtime_parameters.performance_disable_training = getattr(config_file, "performance_disable_training", False)
    runtime_parameters.performance_disable_communication = getattr(config_file, "performance_disable_communication", False)
    runtime_parameters.average_on_cpu = False

    nodes_set = check_consistent_nodes(config_file.get_topology, config_file.max_tick)
    topology = config_file.get_topology(runtime_parameters)
    runtime_parameters.topology = topology
    save_topology_to_file(topology, runtime_parameters.current_tick, runtime_parameters.output_path)
    SIMULATOR_LOGGER.info("topology is updated at tick %s", runtime_parameters.current_tick)

    device = _create_device(device_name, getattr(config_file, "force_use_cpu", False))
    SIMULATOR_LOGGER.info("simulation device: %s", device.device)
    SIMULATOR_LOGGER.info("automatic mixed precision: %s", enable_amp and device.device.type == "cuda")

    training_dataset = DatasetWithFastLabelSelection(config_ml_setup.training_data, config_ml_setup)

    runtime_parameters.node_container = {}
    for single_node in sorted(nodes_set):
        temp_node = Node(
            single_node,
            config_ml_setup,
            device=device,
            use_cpu=getattr(config_file, "force_use_cpu", False),
            enable_amp=enable_amp,
        )
        optimizer, lr_scheduler = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
        temp_node.set_optimizer(optimizer)
        if lr_scheduler is not None:
            temp_node.set_lr_scheduler(lr_scheduler)

        temp_node.set_ml_setup(config_ml_setup)
        temp_node.set_next_training_tick(config_file.get_next_training_time(temp_node, runtime_parameters))
        temp_node.set_average_algorithm(config_file.get_average_algorithm(temp_node, runtime_parameters))
        temp_node.set_average_buffer_size(config_file.get_average_buffer_size(temp_node, runtime_parameters))

        label_distribution = config_file.get_label_distribution(temp_node, runtime_parameters)
        dataloader_worker = getattr(config_file, "preset_training_loader_worker", None)
        temp_node.set_label_distribution(label_distribution, dataset_with_fast_label=training_dataset, worker=dataloader_worker)

        runtime_parameters.node_container[single_node] = temp_node

    config_file.node_behavior_control(runtime_parameters)

    service_list = config_file.get_service_list()
    for service_inst in service_list:
        service_inst.initialize(
            runtime_parameters,
            output_folder_path,
            config_file=config_file,
            ml_setup=config_ml_setup,
            device=device,
            cuda_env=None,
            gpu=None,
        )
        runtime_parameters.service_container[service_inst.get_service_name()] = service_inst

    if hasattr(config_file, "preset_averaging_on_cpu"):
        if config_file.preset_averaging_on_cpu is None:
            runtime_parameters.average_on_cpu = True
        else:
            runtime_parameters.average_on_cpu = config_file.preset_averaging_on_cpu

    begin_simulation(runtime_parameters, config_file)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="DFL simulator ported to LLR2")
    parser.add_argument("--config", type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py"')
    parser.add_argument("-o", "--output_folder_name", default=None, help="specify the output folder name")
    parser.add_argument("-T", "--thread", default=1, type=int, help="specify the number of thread for pytorch")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help='device to use: "auto", "cpu", "cuda", or "cuda:0"')
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision on CUDA")
    args = parser.parse_args()

    torch.set_num_threads(args.thread)
    main(args.config, args.output_folder_name, device_name=args.device, enable_amp=args.amp)
