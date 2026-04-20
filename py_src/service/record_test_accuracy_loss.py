"""Record test accuracy and loss per tick.

Ported from DFL_torch/py_src/service/record_test_accuracy_loss.py.
Uses LLR2's engine.val() instead of the DFL_torch val() function.
The caller provides the evaluation model directly, and the service reuses that
single model on the chosen device for all measurements.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.adapters import ModelAdapter, clone_adapter_for_model
from py_src.engine import Device, val as engine_val
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.model_opti_save_load import save_model_state


class ServiceTestAccuracyLossRecorder(Service):
    """Evaluate model accuracy / loss on a test dataset at configurable ticks.

    Supports:
    * Whole-dataset evaluation (``test_whole_dataset=True``).
    * Fixed balanced subset (``use_fixed_testing_dataset=True``, default).
    * Optional train / val split of the test dataset.
    * Keeping the top-N highest-accuracy model state dicts on disk.
    """

    def __init__(
        self,
        interval: int,
        test_batch_size: int,
        model_name: str,
        dataset_name: str,
        phase_to_record=(SimulationPhase.END_OF_TICK,),
        use_fixed_testing_dataset: bool = True,
        store_top_accuracy_model_count: int = 0,
        accuracy_file_name: str = "accuracy.csv",
        loss_file_name: str = "loss.csv",
        output_var_file_name: str = "output_var.csv",
        test_whole_dataset: bool = False,
        test_val_split: Optional[float] = None,
        test_accuracy_file_name: str = "accuracy_test.csv",
        test_loss_file_name: str = "loss_test.csv",
        val_accuracy_file_name: str = "accuracy_val.csv",
        val_loss_file_name: str = "loss_val.csv",
    ):
        super().__init__()
        self.interval = interval
        self.test_batch_size = test_batch_size
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.phase_to_record = phase_to_record
        self.use_fixed_testing_dataset = use_fixed_testing_dataset
        self.store_top_accuracy_model_count = store_top_accuracy_model_count
        self.store_top_accuracy_model = store_top_accuracy_model_count != 0
        self.test_whole_dataset = test_whole_dataset

        self.accuracy_file_name = accuracy_file_name
        self.loss_file_name = loss_file_name
        self.output_var_file_name = output_var_file_name

        if test_whole_dataset:
            self.test_val_split = test_val_split
        else:
            if test_val_split is not None:
                raise ValueError("test_val_split cannot be used when test_whole_dataset=False")
            self.test_val_split = None

        self.test_accuracy_file_name = test_accuracy_file_name
        self.test_loss_file_name = test_loss_file_name
        self.val_accuracy_file_name = val_accuracy_file_name
        self.val_loss_file_name = val_loss_file_name

        self.accuracy_file = None
        self.loss_file = None
        self.output_var_file = None
        self.test_accuracy_file = None
        self.test_loss_file = None
        self.val_accuracy_file = None
        self.val_loss_file = None

        self.node_order: Optional[List] = None
        self.test_model: Optional[nn.Module] = None
        self._adapter: Optional[ModelAdapter] = None
        self.criterion: Optional[nn.Module] = None
        self.collate_fn = None
        self.test_dataset = None
        self.val_dataset = None
        self._device_obj: Optional[Device] = None
        self._device: Optional[torch.device] = None
        self.store_top_accuracy_model_path: Optional[str] = None
        self.store_top_accuracy_model_buffer: Optional[Dict] = None
        self.logger: Optional[logging.Logger] = None
        self.performance_logger: Optional[logging.Logger] = None
        self.test_idx = None
        self.val_idx = None
        self.enable_profiler = False
        self._cached_test_batches = None
        self._cached_val_batches = None

    def _synchronize_for_timing(self) -> None:
        if not self.enable_profiler:
            return
        if self._device is not None and self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)

    def _time_call(self, func, *args, **kwargs):
        if not self.enable_profiler:
            return func(*args, **kwargs), 0.0
        self._synchronize_for_timing()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        self._synchronize_for_timing()
        return result, time.perf_counter() - start_time

    @staticmethod
    def _format_timing_entries(entries: list[tuple[str, float]]) -> str:
        if not entries:
            return "no timings"
        return ", ".join(f"{name}={elapsed:.3f}s" for name, elapsed in entries)

    @staticmethod
    def _format_performance_row(
        tick: int,
        category: str,
        entries: list[tuple[str, float]],
        *,
        total: Optional[float] = None,
    ) -> str:
        parts = [f"tick={tick}", category]
        if total is not None:
            parts.append(f"total={total:.3f}s")
        for name, elapsed in entries:
            if name == "total":
                continue
            parts.append(f"{name}={elapsed:.3f}s")
        return " | ".join(parts)

    def _emit_profile_row(
        self,
        tick: int,
        category: str,
        entries: list[tuple[str, float]],
        *,
        total: Optional[float] = None,
    ) -> None:
        if not self.enable_profiler:
            return
        message = self._format_performance_row(tick, category, entries, total=total)
        if self.logger is not None:
            self.logger.info(message)
        if self.performance_logger is not None:
            self.performance_logger.info(message)

    def _move_batch_to_device(self, batch: Any) -> Any:
        if torch.is_tensor(batch):
            return batch.to(self._device, non_blocking=True)
        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(value) for value in batch)
        if isinstance(batch, list):
            return [self._move_batch_to_device(value) for value in batch]
        return batch

    def _cache_eval_max_samples(self) -> int:
        return max(self.test_batch_size * 16, 512)

    def _maybe_cache_eval_batches(self, dataloader, cache_attr_name: str):
        cached_batches = getattr(self, cache_attr_name)
        if cached_batches is not None:
            return cached_batches
        if dataloader is None or self._device is None or self._device.type != "cuda":
            return dataloader
        try:
            total_samples = len(dataloader.dataset)  # type: ignore[attr-defined]
        except Exception:
            return dataloader
        if total_samples > self._cache_eval_max_samples():
            return dataloader

        cached_batches = []
        for batch in dataloader:
            cached_batches.append(self._move_batch_to_device(batch))
        setattr(self, cache_attr_name, cached_batches)
        if self.enable_profiler and self.logger is not None:
            self.logger.info(
                "Cached %s eval batches on GPU (%s samples, %s batches)",
                cache_attr_name.replace("_cached_", "").replace("_", " "),
                total_samples,
                len(cached_batches),
            )
        return cached_batches

    @staticmethod
    def get_service_name() -> str:
        return "test_accuracy_loss_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        ml_setup = kwargs.get("ml_setup")
        device_obj = kwargs.get("device")
        assert ml_setup is not None

        node_names = list(parameters.node_container.keys())  # type: ignore
        evaluation_model = ml_setup.model
        for node in parameters.node_container.values():  # type: ignore
            if getattr(node, 'is_using_model_stat', False):
                evaluation_model = node.allocated_gpu.model
                break

        self.initialize_without_runtime_parameters(
            output_path, node_names, evaluation_model, ml_setup.criterion,
            ml_setup.testing_data, ml_setup, device=device_obj,
        )

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.phase in self.phase_to_record:
            stats = {n: parameters.node_container[n].get_model_stat() for n in self.node_order}  # type: ignore
            self.trigger_without_runtime_parameters(parameters.current_tick, stats, parameters.phase.name)

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        output_path: str,
        node_names: list,
        model: nn.Module,
        criterion: Optional[nn.Module],
        test_dataset,
        ml_setup,
        logger: Optional[logging.Logger] = None,
        device: Optional[Device] = None,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
    ):
        self.logger = logger
        self.node_order = node_names
        self.criterion = criterion
        base_test_dataset = test_dataset if test_dataset is not None else ml_setup.testing_data

        # Determine device
        if device is not None:
            self._device_obj = device
        else:
            self._device_obj = Device(self._infer_model_device(model))
        self._device = self._device_obj.device

        # Open CSV files
        header = ",".join(["tick", "phase", *[str(n) for n in node_names]])
        self.accuracy_file = open(os.path.join(output_path, self.accuracy_file_name), "w+")
        self.loss_file = open(os.path.join(output_path, self.loss_file_name), "w+")
        self.output_var_file = open(os.path.join(output_path, self.output_var_file_name), "w+")
        for f in (self.accuracy_file, self.loss_file, self.output_var_file):
            f.write(header + "\n")
        if self.test_val_split is not None:
            self.test_accuracy_file = open(os.path.join(output_path, self.test_accuracy_file_name), "w+")
            self.test_loss_file = open(os.path.join(output_path, self.test_loss_file_name), "w+")
            self.val_accuracy_file = open(os.path.join(output_path, self.val_accuracy_file_name), "w+")
            self.val_loss_file = open(os.path.join(output_path, self.val_loss_file_name), "w+")
            for f in (self.test_accuracy_file, self.test_loss_file, self.val_accuracy_file, self.val_loss_file):
                f.write(header + "\n")

        if self.test_whole_dataset:
            if self.test_val_split is not None:
                assert 0.0 < self.test_val_split < 1.0
                perm = torch.randperm(len(base_test_dataset)).tolist() # type: ignore[arg-type]
                n_test = int(round(len(base_test_dataset) * self.test_val_split)) # type: ignore[arg-type]
                self.test_idx = perm[:n_test]
                self.val_idx = perm[n_test:]
                self.test_dataset = self._build_val_dataloader(
                    ml_setup,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    dataset_override=Subset(base_test_dataset, self.test_idx), # type: ignore[arg-type]
                )
                self.test_dataset = self._maybe_cache_eval_batches(self.test_dataset, "_cached_test_batches")
                self.val_dataset = self._build_val_dataloader(
                    ml_setup,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    dataset_override=Subset(base_test_dataset, self.val_idx), # type: ignore[arg-type]
                )
                self.val_dataset = self._maybe_cache_eval_batches(self.val_dataset, "_cached_val_batches")
            else:
                self.test_dataset = self._build_val_dataloader(
                    ml_setup,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    dataset_override=base_test_dataset,
                )
                self.test_dataset = self._maybe_cache_eval_batches(self.test_dataset, "_cached_test_batches")
        else:
            if self.use_fixed_testing_dataset:
                labels = np.array([base_test_dataset[i][1] for i in range(len(base_test_dataset))]) # type: ignore[index,arg-type]
                unique_labels = sorted(set(labels.tolist()))
                n_labels = len(unique_labels)
                assert self.test_batch_size % n_labels == 0, \
                    f"test_batch_size({self.test_batch_size}) must be divisible by n_labels({n_labels})"
                per_label = self.test_batch_size // n_labels
                balanced_idx = []
                for lbl in unique_labels:
                    idxs = np.where(labels == lbl)[0]
                    balanced_idx.extend(np.random.choice(idxs, per_label, replace=False).tolist())
                self.test_dataset = self._build_val_dataloader(
                    ml_setup,
                    batch_size=min(100, self.test_batch_size),
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    dataset_override=Subset(base_test_dataset, balanced_idx), # type: ignore[arg-type]
                )
                self.test_dataset = self._maybe_cache_eval_batches(self.test_dataset, "_cached_test_batches")
            else:
                self.test_dataset = self._build_val_dataloader(
                    ml_setup,
                    batch_size=self.test_batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    num_samples=self.test_batch_size,
                    dataset_override=base_test_dataset,
                )
                self.test_dataset = self._maybe_cache_eval_batches(self.test_dataset, "_cached_test_batches")

        # Reuse the provided evaluation model directly.
        self.test_model = model
        self.test_model = self.test_model.to(self._device)
        self._adapter = clone_adapter_for_model(ml_setup.adapter, self.test_model, criterion=self.criterion)

        # Top-accuracy buffer
        self.store_top_accuracy_model_buffer = {}
        if self.store_top_accuracy_model:
            self.store_top_accuracy_model_path = os.path.join(output_path, "top_accuracy_models")
            os.makedirs(self.store_top_accuracy_model_path, exist_ok=True)
            for n in node_names:
                self.store_top_accuracy_model_buffer[n] = OrderedDict()

    def _build_val_dataloader(
        self,
        ml_setup,
        *,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        num_samples: Optional[int] = None,
        dataset_override=None,
    ):
        loader_setup = copy.copy(ml_setup)
        if dataset_override is not None:
            loader_setup.testing_data = dataset_override
        loader_setup.override_test_loader = None

        loader_config = DataloaderConfig(
            batch_size=batch_size or self.test_batch_size,
            num_workers=num_workers or 0,
            num_samples=num_samples,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
        return loader_setup.val_dataloader(loader_config, ignore_override=False)

    @staticmethod
    def _infer_model_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            try:
                return next(model.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    def trigger_without_runtime_parameters(
        self,
        tick: int,
        node_names_and_model_stats: Dict,
        phase_str: Optional[str] = None,
    ):
        if tick % self.interval != 0:
            return
        assert self._device_obj is not None
        assert self.test_model is not None
        assert self._adapter is not None
        assert self.test_dataset is not None
        assert self.node_order is not None
        assert self.accuracy_file is not None
        assert self.loss_file is not None
        assert self.output_var_file is not None
        logger = self.logger

        row_acc, row_loss, row_var = [], [], []
        row_test_acc, row_test_loss, row_val_acc, row_val_loss = [], [], [], []
        final_accuracy: Dict = {}
        final_model: Dict = {}
        node_component_timings: list[tuple[str, float]] = []
        service_timings: list[tuple[str, float]] = []

        service_start = 0.0
        if self.enable_profiler:
            self._synchronize_for_timing()
            service_start = time.perf_counter()

        for node_name in self.node_order:
            model_stat = node_names_and_model_stats[node_name]
            # Load state dict — tensors stay on device
            node_timings: list[tuple[str, float]] = []
            _, elapsed = self._time_call(self.test_model.load_state_dict, model_stat)
            node_timings.append(("load_state_dict", elapsed))
            _, elapsed = self._time_call(self.test_model.to, self._device)
            node_timings.append(("model_to_device", elapsed))

            loss_test = acc_test = loss_val = acc_val = var = 0.0

            if self.test_whole_dataset:
                assert self.test_dataset is not None
                result, elapsed = self._time_call(
                    engine_val, self._adapter, self.test_dataset, device=self._device_obj
                )
                node_timings.append(("test_eval", elapsed))
                acc_test = result.accuracy or 0.0
                loss_test = result.avg_loss
                var = result.extra.get("variance", 0.0)
                result_val = None

                if self.test_val_split is not None:
                    assert self.val_dataset is not None
                    result_val, elapsed = self._time_call(
                        engine_val, self._adapter, self.val_dataset, device=self._device_obj
                    ) # type: ignore
                    node_timings.append(("val_eval", elapsed))
                    acc_val = result_val.accuracy or 0.0
                    loss_val = result_val.avg_loss

                combined_count = result.total_count
                if result_val is not None:
                    combined_count += result_val.total_count
                if combined_count > 0:
                    total_loss = result.total_loss
                    correct = result.total_correct or 0
                    if result_val is not None:
                        total_loss += result_val.total_loss
                        correct += result_val.total_correct or 0
                    loss = total_loss / combined_count
                    accuracy = correct / combined_count
                else:
                    loss = 0.0; accuracy = 0.0
            else:
                result, elapsed = self._time_call(
                    engine_val, self._adapter, self.test_dataset, device=self._device_obj
                )
                node_timings.append(("test_eval", elapsed))
                accuracy = result.accuracy or 0.0
                loss = result.avg_loss
                var = result.extra.get("variance", 0.0)

            row_acc.append('%.4E' % accuracy)
            row_loss.append('%.4E' % loss)
            row_var.append('%.4E' % var)
            if self.test_val_split is not None:
                row_test_acc.append('%.4E' % acc_test)
                row_test_loss.append('%.4E' % loss_test)
                row_val_acc.append('%.4E' % acc_val)
                row_val_loss.append('%.4E' % loss_val)
            final_accuracy[node_name] = accuracy
            final_model[node_name] = model_stat
            node_component_timings.extend(
                (f"node{node_name}.{component_name}", component_elapsed)
                for component_name, component_elapsed in node_timings
            )

        prefix = [str(tick), str(phase_str)]
        write_start = time.perf_counter() if self.enable_profiler else 0.0
        self.accuracy_file.write(",".join(prefix + row_acc) + "\n"); self.accuracy_file.flush()
        self.loss_file.write(",".join(prefix + row_loss) + "\n"); self.loss_file.flush()
        self.output_var_file.write(",".join(prefix + row_var) + "\n"); self.output_var_file.flush()
        if self.test_val_split is not None:
            assert self.test_accuracy_file is not None
            assert self.test_loss_file is not None
            assert self.val_accuracy_file is not None
            assert self.val_loss_file is not None
            self.test_accuracy_file.write(",".join(prefix + row_test_acc) + "\n"); self.test_accuracy_file.flush()
            self.test_loss_file.write(",".join(prefix + row_test_loss) + "\n"); self.test_loss_file.flush()
            self.val_accuracy_file.write(",".join(prefix + row_val_acc) + "\n"); self.val_accuracy_file.flush()
            self.val_loss_file.write(",".join(prefix + row_val_loss) + "\n"); self.val_loss_file.flush()
        if self.enable_profiler:
            service_timings.append(("write_csv", time.perf_counter() - write_start))

        _, elapsed = self._time_call(self._check_store_top_accuracy_model, final_accuracy, final_model, tick)
        if self.enable_profiler:
            service_timings.append(("store_top_accuracy", elapsed))
            self._synchronize_for_timing()
            self._emit_profile_row(
                tick,
                "service(test_accuracy_loss)",
                node_component_timings + service_timings,
                total=time.perf_counter() - service_start,
            )

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        files = [
            (self.accuracy_file_name, self.accuracy_file),
            (self.loss_file_name, self.loss_file),
            (self.output_var_file_name, self.output_var_file),
        ]
        if self.test_val_split is not None:
            files += [
                (self.test_accuracy_file_name, self.test_accuracy_file),
                (self.test_loss_file_name, self.test_loss_file),
                (self.val_accuracy_file_name, self.val_accuracy_file),
                (self.val_loss_file_name, self.val_loss_file),
            ]
        for fname, fobj in files:
            if fobj is None:
                continue
            with open(os.path.join(checkpoint_folder_path, fname), 'r', newline='') as infile:
                next(infile)
                for line in infile:
                    if int(line.split(",", 1)[0]) < restore_until_tick:
                        fobj.write(line)
            fobj.flush()

    def _check_store_top_accuracy_model(self, final_accuracy: Dict, final_model: Dict, tick: int):
        if not self.store_top_accuracy_model:
            return
        assert self.node_order is not None
        assert self.store_top_accuracy_model_buffer is not None
        assert self.store_top_accuracy_model_path is not None
        for node_name in self.node_order:
            accuracy = final_accuracy[node_name]
            model_stat = final_model[node_name]
            buf = self.store_top_accuracy_model_buffer[node_name]
            save_name = f"name_{node_name}_tick_{tick}_acc_{accuracy:.6f}.model.pt"
            save_path = os.path.join(self.store_top_accuracy_model_path, save_name)
            changed = False
            if accuracy not in buf:
                if len(buf) < self.store_top_accuracy_model_count:
                    buf[accuracy] = save_path
                    # Save CPU copy of state dict
                    cpu_stat = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_stat.items()}
                    save_model_state(save_path, cpu_stat, self.model_name, self.dataset_name)
                    changed = True
                else:
                    smallest_acc = next(iter(buf))
                    if smallest_acc < accuracy:
                        old_path = buf.pop(smallest_acc)
                        if os.path.exists(old_path):
                            os.remove(old_path)
                        buf[accuracy] = save_path
                        cpu_stat = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_stat.items()}
                        save_model_state(save_path, cpu_stat, self.model_name, self.dataset_name)
                        changed = True
            if changed:
                self.store_top_accuracy_model_buffer[node_name] = OrderedDict(sorted(buf.items()))

    def __del__(self):
        for f in (self.accuracy_file, self.loss_file, self.output_var_file,
                   self.test_accuracy_file, self.test_loss_file,
                   self.val_accuracy_file, self.val_loss_file):
            if f is not None:
                try:
                    f.flush(); f.close()
                except Exception:
                    pass
