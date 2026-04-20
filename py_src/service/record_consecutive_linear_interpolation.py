"""Record loss/accuracy along linear interpolations between consecutive model states.

At START_OF_TICK the current model state is cached; at END_OF_TICK the path
between the cached start and the updated end is sampled at evenly-spaced
interpolation points and each is evaluated.

Ported from DFL_torch/py_src/service/record_consecutive_linear_interpolation.py.
Uses LLR2's engine.train(backpropagation=False) for evaluation.
GPU optimization: interpolated state dicts are computed on-device; only scalars
leave the GPU for CSV writing.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Subset

from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.adapters import ModelAdapter, clone_adapter_for_model
from py_src.engine import Device, train as engine_train
from py_src.ml_setup.dataloader_util import DataloaderConfig


class ServiceConsecutiveLinearInterpolationRecorder(Service):
    """Evaluate model loss/accuracy at interpolated points between consecutive
    model states (one step of the path-finding algorithm).
    """

    def __init__(
        self,
        interval: int,
        batch_size: int,
        dataset_size: int,
        points_size: int,
        recorded_node_name,
        training_mode: bool = False,
        loss_filename: str = "consec_linear_interpolation_loss.csv",
        accuracy_filename: str = "consec_linear_interpolation_accuracy.csv",
    ):
        super().__init__()
        self.interval = interval
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.points_size = points_size
        self.recorded_node_name = recorded_node_name
        self.training_mode = training_mode
        self.loss_file_name = loss_filename
        self.accuracy_file_name = accuracy_filename

        self.loss_file = None
        self.accuracy_file = None
        self.dataloader = None
        self.test_model: Optional[nn.Module] = None
        self._adapter: Optional[ModelAdapter] = None
        self.criterion: Optional[nn.Module] = None
        self._device_obj: Optional[Device] = None
        self._device: Optional[torch.device] = None
        self.cache_state_model_stat: Optional[dict] = None
        self._state_targets: Optional[dict[str, torch.Tensor]] = None
        self._float_state_names: list[str] = []
        self._other_state_names: list[str] = []
        self.logger: Optional[logging.Logger] = None
        self.enable_profiler = False

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
    def get_service_name() -> str:
        return "consecutive_linear_interpolation_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        ml_setup = kwargs.get("ml_setup")
        device_obj = kwargs.get("device")
        assert ml_setup is not None
        assert self.recorded_node_name in parameters.node_container
        evaluation_model = ml_setup.model
        for node in parameters.node_container.values():  # type: ignore
            if getattr(node, 'is_using_model_stat', False):
                evaluation_model = node.allocated_gpu.model
                break
        self.initialize_without_runtime_parameters(
            output_path, evaluation_model, ml_setup.criterion,
            ml_setup.training_data, ml_setup, device=device_obj,
        )

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.phase in (SimulationPhase.START_OF_TICK, SimulationPhase.END_OF_TICK):
            node = parameters.node_container[self.recorded_node_name] # type: ignore
            self.trigger_without_runtime_parameters(parameters.current_tick, parameters.phase, node.get_model_stat()) # type: ignore

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        output_path: str,
        model: nn.Module,
        criterion: Optional[nn.Module],
        train_dataset,
        ml_setup,
        logger: Optional[logging.Logger] = None,
        device: Optional[Device] = None,
        num_workers: Optional[int] = None,
    ):
        self.logger = logger
        self.criterion = criterion
        base_train_dataset = train_dataset if train_dataset is not None else ml_setup.training_data

        if device is not None:
            self._device_obj = device
        else:
            self._device_obj = Device(self._infer_model_device(model))
        self._device = self._device_obj.device

        self.test_model = model
        self.test_model = self.test_model.to(self._device)
        self._adapter = clone_adapter_for_model(ml_setup.adapter, self.test_model, criterion=self.criterion)
        self._state_targets = self._get_state_targets(self.test_model)
        self._float_state_names = [
            name for name, tensor in self._state_targets.items() if tensor.dtype.is_floating_point
        ]
        self._other_state_names = [
            name for name, tensor in self._state_targets.items() if not tensor.dtype.is_floating_point
        ]

        if self.points_size == 0:
            return  # nothing to do

        subset_override = base_train_dataset
        if (
            self.dataset_size is not None
            and self.dataset_size > 0
            and hasattr(base_train_dataset, "__len__")
            and self.dataset_size < len(base_train_dataset)  # type: ignore[arg-type]
        ):
            indices = torch.randperm(len(base_train_dataset))[:self.dataset_size].tolist()  # type: ignore[arg-type]
            subset_override = Subset(base_train_dataset, indices)
        self.dataloader = self._build_probe_dataloader(
            ml_setup,
            num_workers=num_workers,
            dataset_override=subset_override,
        )

        # CSV files
        self.accuracy_file = open(os.path.join(output_path, self.accuracy_file_name), "w+")
        self.loss_file = open(os.path.join(output_path, self.loss_file_name), "w+")
        point_cols = [str(i) for i in range(1, self.points_size)]
        header = ",".join(["tick", "phase", *point_cols])
        self.accuracy_file.write(header + "\n"); self.accuracy_file.flush()
        self.loss_file.write(header + "\n"); self.loss_file.flush()

    def trigger_without_runtime_parameters(self, tick: int, phase: SimulationPhase, model_state: dict):
        if tick % self.interval != 0:
            return
        assert self._device is not None
        assert self._device_obj is not None
        assert self.test_model is not None
        assert self._adapter is not None
        assert self.dataloader is not None
        assert self._state_targets is not None
        if self.accuracy_file is None or self.loss_file is None:
            return
        logger = self.logger

        if phase == SimulationPhase.START_OF_TICK:
            assert self.cache_state_model_stat is None
            # Cache on same device as model — avoids CPU round-trip
            start_time = 0.0
            if self.enable_profiler:
                self._synchronize_for_timing()
                start_time = time.perf_counter()
            self.cache_state_model_stat = {k: v.detach().clone() for k, v in model_state.items()}
            if self.enable_profiler:
                self._synchronize_for_timing()
            if self.enable_profiler and logger is not None:
                logger.info(
                    "tick %s consecutive_points start internals: cache_start_state=%.3fs",
                    tick,
                    time.perf_counter() - start_time,
                )

        elif phase == SimulationPhase.END_OF_TICK:
            assert self.cache_state_model_stat is not None
            service_start = 0.0
            if self.enable_profiler:
                self._synchronize_for_timing()
                service_start = time.perf_counter()
            timing_entries: list[tuple[str, float]] = []
            start_stat = self.cache_state_model_stat
            start_time = 0.0
            if self.enable_profiler:
                self._synchronize_for_timing()
                start_time = time.perf_counter()
            end_stat = {k: v.detach().clone() for k, v in model_state.items()}
            if self.enable_profiler:
                self._synchronize_for_timing()
                timing_entries.append(("clone_end_state", time.perf_counter() - start_time))
                self._synchronize_for_timing()
                start_time = time.perf_counter()
            delta_stat = {
                name: end_stat[name] - start_stat[name]
                for name in self._float_state_names
            }
            if self.enable_profiler:
                self._synchronize_for_timing()
                timing_entries.append(("build_delta", time.perf_counter() - start_time))

            loss_results, acc_results = [], []
            original_training_mode = self.test_model.training
            interpolation_load_total = 0.0
            probe_eval_total = 0.0
            point_total_times: list[float] = []
            try:
                for i in range(1, self.points_size):
                    alpha = i / self.points_size
                    point_start = 0.0
                    if self.enable_profiler:
                        self._synchronize_for_timing()
                        point_start = time.perf_counter()
                    _, elapsed = self._time_call(self._load_interpolated_state, start_stat, end_stat, delta_stat, alpha)
                    interpolation_load_total += elapsed
                    result, elapsed = self._time_call(
                        engine_train,
                        self._adapter, self.dataloader,
                        optimizer=None, lr_scheduler=None,
                        device=self._device_obj,
                        scaler=None,
                        backpropagation=False,
                        training_mode=self.training_mode,
                    )
                    probe_eval_total += elapsed
                    if self.enable_profiler:
                        self._synchronize_for_timing()
                        point_total_times.append(time.perf_counter() - point_start)
                    loss_results.append('%.4E' % result.avg_loss)
                    acc_results.append('%.4E' % (result.accuracy or 0.0))
            finally:
                # Service probes must not leave the model in a modified state.
                _, elapsed = self._time_call(self._restore_state, end_stat)
                if self.enable_profiler:
                    timing_entries.append(("restore_end_state", elapsed))
                self.test_model.train(original_training_mode)
                self.cache_state_model_stat = None

            if self.enable_profiler:
                timing_entries.append(("interpolate_state_total", interpolation_load_total))
                timing_entries.append(("probe_eval_total", probe_eval_total))
            if self.enable_profiler and point_total_times:
                timing_entries.append(("point_avg", sum(point_total_times) / len(point_total_times)))
                timing_entries.append(("point_max", max(point_total_times)))

            prefix = [str(tick), str(phase.name)]
            write_start = time.perf_counter() if self.enable_profiler else 0.0
            self.accuracy_file.write(",".join(prefix + acc_results) + "\n"); self.accuracy_file.flush()
            self.loss_file.write(",".join(prefix + loss_results) + "\n"); self.loss_file.flush()
            if self.enable_profiler:
                timing_entries.append(("write_csv", time.perf_counter() - write_start))
                self._synchronize_for_timing()
                timing_entries.append(("total", time.perf_counter() - service_start))
            if self.enable_profiler and logger is not None:
                logger.info(
                    "tick %s consecutive_points end internals: %s",
                    tick,
                    self._format_timing_entries(timing_entries),
                )

    def _build_probe_dataloader(
        self,
        ml_setup,
        *,
        num_workers: Optional[int] = None,
        dataset_override=None,
    ):
        loader_setup = copy.copy(ml_setup)
        loader_setup.testing_data = dataset_override if dataset_override is not None else ml_setup.training_data
        loader_setup.override_test_loader = None

        loader_config = DataloaderConfig(
            batch_size=min(100, self.batch_size),
            num_workers=num_workers or 0,
            shuffle=False,
            pin_memory=True,
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

    @staticmethod
    def _get_state_targets(model: nn.Module) -> dict[str, torch.Tensor]:
        # Only include entries that are part of state_dict(). Some models
        # register non-persistent buffers (for example Hugging Face
        # ``position_ids``) that appear in named_buffers() but are not saved in
        # checkpoints. The interpolation service operates on saved model states,
        # so its live tensor targets must match that serializable key set.
        state_dict_keys = set(model.state_dict().keys())
        state_targets: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if name in state_dict_keys:
                state_targets[name] = parameter
        for name, buffer in model.named_buffers():
            if name in state_dict_keys:
                state_targets[name] = buffer
        return state_targets

    def _load_interpolated_state(
        self,
        start_stat: dict[str, torch.Tensor],
        end_stat: dict[str, torch.Tensor],
        delta_stat: dict[str, torch.Tensor],
        alpha: float,
    ) -> None:
        assert self._state_targets is not None
        with torch.no_grad():
            for name in self._float_state_names:
                target = self._state_targets[name]
                target.copy_(start_stat[name])
                target.add_(delta_stat[name], alpha=alpha)

            for name in self._other_state_names:
                target = self._state_targets[name]
                start_value = start_stat[name]
                end_value = end_stat[name]
                if torch.equal(start_value, end_value):
                    target.copy_(start_value)
                    continue
                blended = torch.round(
                    (1 - alpha) * start_value.to(torch.float32)
                    + alpha * end_value.to(torch.float32)
                )
                target.copy_(blended.to(dtype=target.dtype))

    def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
        assert self._state_targets is not None
        with torch.no_grad():
            for name, target in self._state_targets.items():
                target.copy_(state[name])

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        if self.accuracy_file is None:
            return
        for fname, fobj in [(self.accuracy_file_name, self.accuracy_file),
                             (self.loss_file_name, self.loss_file)]:
            with open(os.path.join(checkpoint_folder_path, fname), 'r', newline='') as infile:
                next(infile)
                for line in infile:
                    if int(line.split(",", 1)[0]) < restore_until_tick:
                        fobj.write(line)
            fobj.flush()

    def __del__(self):
        for f in (self.accuracy_file, self.loss_file):
            if f is not None:
                try:
                    f.flush(); f.close()
                except Exception:
                    pass
