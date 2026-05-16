"""Move model weights toward a destination while retraining to keep accuracy.

This is the LLR2 port of ``DFL_torch/tool/find_high_accuracy_path_v2.py``.

The port keeps the original CLI workflow and exposes a reusable runner with a
small setup/step/run interface.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.util
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import types
from datetime import datetime
from typing import Any, Dict, Optional, Iterable

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from py_src.ml_setup import MLSetup
from tool.find_high_accuracy_path.runtime_parameters import Checkpoint, RuntimeParameters, WorkMode
from tool.find_high_accuracy_path.find_parameters import (
    ParameterGeneral,
    ParameterMove,
    ParameterRebuildNorm,
    ParameterTrain,
)
from tool.find_high_accuracy_path.functions import _optimizer_to, _try_get_criterion, rebuild_norm_layer_function

from py_src.engine import Device, train as engine_train
from py_src.ml_setup.dataloader_util import DataloaderConfig
from py_src.ml_setup.factory import get_ml_setup_from_config
from py_src.model_average import move_model_state_toward
from py_src.model_opti_save_load import (
    load_model_state_file,
    load_optimizer_state_file,
    save_model_state,
    save_optimizer_state,
)
from py_src.model_variance_correct import VarianceCorrectionType, VarianceCorrector
from py_src.service import (
    record_consecutive_linear_interpolation,
    record_cosine_similarity,
    record_model_stat,
    record_test_accuracy_loss,
    record_training_loss_accuracy,
    record_variance,
    record_weights_difference,
)
from py_src.simulation_runtime_parameters import SimulationPhase
from py_src.special_torch_layers import find_layers_according_to_name_and_keyword, find_normalization_layers
from py_src.util import (
    setup_logging,
    geodesic_distance,
    prompt_selection,
)

logger = logging.getLogger("find_high_accuracy_path")

REPORT_FINISH_TIME_PER_TICK = 100
ENABLE_REBUILD_NORM = False


def _require_model_type_name(model_type) -> str:
    assert model_type is not None
    return model_type.name


def _require_dataset_type_name(dataset_type) -> str:
    assert dataset_type is not None
    return dataset_type.name


def _register_config_import_aliases() -> None:
    """Expose compatibility aliases so original DFL_torch configs still import."""
    import py_src.ml_setup as ml_setup_pkg

    if not hasattr(ml_setup_pkg, "MlSetup"):
        ml_setup_pkg.MlSetup = ml_setup_pkg.MLSetup  # type: ignore[attr-defined]

    from tool.find_high_accuracy_path import find_parameters as _find_parameters_module
    from tool.find_high_accuracy_path import functions as _functions_module
    from tool.find_high_accuracy_path import runtime_parameters as _runtime_parameters_module

    for package_name in ("find_high_accuracy_path", "find_high_accuracy_path_v2"):
        package_module = sys.modules.get(package_name)
        if package_module is None:
            package_module = types.ModuleType(package_name)
            package_module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = package_module

        package_module.runtime_parameters = _runtime_parameters_module  # type: ignore[attr-defined]
        package_module.find_parameters = _find_parameters_module  # type: ignore[attr-defined]
        package_module.functions = _functions_module  # type: ignore[attr-defined]

        sys.modules[f"{package_name}.runtime_parameters"] = _runtime_parameters_module
        sys.modules[f"{package_name}.find_parameters"] = _find_parameters_module
        sys.modules[f"{package_name}.functions"] = _functions_module


def load_configuration(config_file_path: str):
    """Dynamically import a Python config file and return the module."""
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    _register_config_import_aliases()

    spec = importlib.util.spec_from_file_location("_find_high_accuracy_path_config", config_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import config file: {config_file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_find_high_accuracy_path_config"] = module
    spec.loader.exec_module(module)
    return module


def atomic_torch_save(obj, target_path: str) -> None:
    target_dir = os.path.dirname(target_path) or "."
    target_name = os.path.basename(target_path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target_dir,
            prefix=f"{target_name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
        torch.save(obj, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _parse_save_ticks(save_ticks: Optional[str]) -> Optional[set[int]]:
    if save_ticks is None:
        return None

    text = save_ticks.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    output: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if end < start:
                raise ValueError(f"Invalid save tick range: {part}")
            output.update(range(start, end + 1))
        else:
            output.add(int(part))
    return output


def _move_nested_tensors(obj: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().to(device)
    if isinstance(obj, dict):
        return {k: _move_nested_tensors(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_nested_tensors(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_nested_tensors(v, device) for v in obj)
    return copy.deepcopy(obj)


def _clone_state_dict(model_state: Dict[str, Any], device: Optional[torch.device] = None) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in model_state.items():
        if torch.is_tensor(value):
            tensor = value.detach()
            if device is not None:
                tensor = tensor.to(device)
            output[key] = tensor.clone()
        else:
            output[key] = copy.deepcopy(value)
    return output


def _state_dict_to_cpu(model_state: Dict[str, Any]) -> Dict[str, Any]:
    return _clone_state_dict(model_state, device=torch.device("cpu"))


def _ensure_state_dict_device(model_state: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in model_state.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _state_dicts_equal(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> bool:
    if state_a.keys() != state_b.keys():
        return False
    for key in state_a:
        value_a = state_a[key]
        value_b = state_b[key]
        if torch.is_tensor(value_a) and torch.is_tensor(value_b):
            if not torch.equal(value_a, value_b):
                return False
        elif value_a != value_b:
            return False
    return True


def _compute_file_sha256(path: Optional[str]) -> Optional[str]:
    if path is None or not os.path.exists(path):
        return None

    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _apply_ml_setup_compatibility(ml_setup) -> None:
    """Expose DFL_torch-era attribute names on the new MLSetup object."""
    if not hasattr(ml_setup, "model_name"):
        ml_setup.model_name = ml_setup.model_type.name
    if not hasattr(ml_setup, "dataset_name"):
        ml_setup.dataset_name = ml_setup.dataset_type.name
    if not hasattr(ml_setup, "training_batch_size"):
        ml_setup.training_batch_size = ml_setup.default_batch_size
    if not hasattr(ml_setup, "collate_fn"):
        ml_setup.collate_fn = ml_setup.default_collate_fn
    if not hasattr(ml_setup, "sampler_fn"):
        ml_setup.sampler_fn = ml_setup.default_sampler_fn
    if not hasattr(ml_setup, "override_training_dataset_loader"):
        ml_setup.override_training_dataset_loader = ml_setup.override_train_loader
    if not hasattr(ml_setup, "override_testing_dataset_loader"):
        ml_setup.override_testing_dataset_loader = ml_setup.override_test_loader
    if getattr(ml_setup, "criterion", None) is None:
        ml_setup.criterion = _try_get_criterion(ml_setup)


def get_files_to_process(start_folder: str, end_folder: str, mode: str) -> list[tuple[str, str]]:
    if not os.path.isdir(start_folder):
        logger.critical("%s does not exist", start_folder)
        return []
    if not os.path.isdir(end_folder):
        logger.critical("%s does not exist", end_folder)
        return []

    starts = sorted(name for name in os.listdir(start_folder) if name.endswith("model.pt"))
    ends = sorted(name for name in os.listdir(end_folder) if name.endswith("model.pt"))

    if mode == "auto":
        mode = "each_to_each" if len(starts) == len(ends) else "all_to_all"

    pairs: list[tuple[str, str]] = []
    if mode == "all_to_all":
        pairs = [
            (os.path.join(start_folder, start_name), os.path.join(end_folder, end_name))
            for start_name in starts
            for end_name in ends
        ]
    elif mode == "each_to_each":
        if len(starts) != len(ends):
            logger.critical("each_to_each requires matching file counts")
            return []
        if starts != ends:
            logger.critical("each_to_each requires matching file names")
            return []
        pairs = [
            (os.path.join(start_folder, file_name), os.path.join(end_folder, file_name))
            for file_name in starts
        ]
    elif mode == "one_to_all":
        if len(starts) != 1:
            logger.critical("one_to_all requires exactly one starting model")
            return []
        pairs = [
            (os.path.join(start_folder, starts[0]), os.path.join(end_folder, end_name))
            for end_name in ends
        ]
    elif mode == "all_to_one":
        if len(ends) != 1:
            logger.critical("all_to_one requires exactly one destination model")
            return []
        pairs = [
            (os.path.join(start_folder, start_name), os.path.join(end_folder, ends[0]))
            for start_name in starts
        ]
    else:
        logger.critical("Unknown mapping mode: %s", mode)
        return []

    return sorted(pairs)


def calculate_layer_wise_projection_to_variance_sphere(
    source_state: Dict[str, Any],
    variance_sphere_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Scale each floating-point layer to match the norm of the variance sphere."""

    def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(tensor.reshape(-1).to(torch.float32))

    output = _clone_state_dict(source_state)
    for key, source_value in source_state.items():
        if key not in variance_sphere_state:
            continue
        target_value = variance_sphere_state[key]
        if not (torch.is_tensor(source_value) and torch.is_tensor(target_value)):
            continue
        if not (source_value.dtype.is_floating_point and target_value.dtype.is_floating_point):
            continue

        source_norm = l2_norm(source_value)
        if source_norm.item() == 0.0:
            continue

        target_norm = l2_norm(target_value.to(source_value.device))
        scale = (target_norm / source_norm).to(source_value.dtype)
        output[key] = source_value * scale
    return output


def _build_compensate_destination(
    source_state: Dict[str, Any],
    end_state: Dict[str, Any],
    layer_names: list[str],
) -> tuple[Dict[str, Any], list[str]]:
    destination: Dict[str, Any] = {}
    skipped_layers: list[str] = []

    for layer_name in sorted(set(layer_names)):
        if layer_name not in source_state or layer_name not in end_state:
            skipped_layers.append(layer_name)
            continue

        source_value = source_state[layer_name]
        end_value = end_state[layer_name]
        if not (torch.is_tensor(source_value) and torch.is_tensor(end_value)):
            skipped_layers.append(layer_name)
            continue
        if not (source_value.dtype.is_floating_point and end_value.dtype.is_floating_point):
            skipped_layers.append(layer_name)
            continue

        destination[layer_name] = source_value.detach().clone() * 2 - end_value

    return destination, skipped_layers


def _capture_attention_state(
    model_state: Dict[str, Any],
    attention_layers: list[str],
) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}
    for layer_name in attention_layers:
        if layer_name in model_state:
            captured[layer_name] = model_state[layer_name].detach().clone()
        bias_name = layer_name.replace(".weight", ".bias")
        if bias_name in model_state:
            captured[bias_name] = model_state[bias_name].detach().clone()
    return captured


def _apply_attention_policy(
    old_attention_state: Dict[str, Any],
    target_state: Dict[str, Any],
    attention_layers: list[str],
    policy: Optional[str],
) -> None:
    if not attention_layers or policy == None:
        return
    if policy != "ignore_kv":
        raise NotImplementedError(f"Attention policy {policy!r} is not implemented")

    for layer_name in attention_layers:
        if layer_name not in old_attention_state or layer_name not in target_state:
            continue

        source_weight = old_attention_state[layer_name]
        target_weight = target_state[layer_name]
        assert source_weight.shape == target_weight.shape

        source_q, source_k, _ = source_weight.chunk(3, dim=0)
        _, _, target_v = target_weight.chunk(3, dim=0)
        target_state[layer_name] = torch.cat([source_q, source_k, target_v], dim=0)

        bias_name = layer_name.replace(".weight", ".bias")
        if bias_name in old_attention_state and bias_name in target_state:
            source_bias = old_attention_state[bias_name]
            target_bias = target_state[bias_name]
            assert source_bias.shape == target_bias.shape
            source_qb, source_kb, _ = source_bias.chunk(3, dim=0)
            _, _, target_vb = target_bias.chunk(3, dim=0)
            target_state[bias_name] = torch.cat([source_qb, source_kb, target_vb], dim=0)


def load_existing_optimizer_stat(
    optimizer: torch.optim.Optimizer,
    optimizer_stat_dict_path: str,
    device: torch.device,
    log: Optional[logging.Logger] = None,
) -> None:
    assert os.path.exists(optimizer_stat_dict_path), f"Optimizer state missing: {optimizer_stat_dict_path}"
    optimizer_state, _, _ = load_optimizer_state_file(optimizer_stat_dict_path)
    test_optimizer = copy.deepcopy(optimizer)
    try:
        test_optimizer.load_state_dict(optimizer_state)
        optimizer.load_state_dict(optimizer_state)
        _optimizer_to(optimizer, device)
        if log is not None:
            log.info("Successfully loaded optimizer state from %s", optimizer_stat_dict_path)
    except Exception as exc:
        if log is not None:
            log.warning("Failed to load optimizer state: %s", exc)
        else:
            raise


def pre_train(
    adapter,
    optimizer: torch.optim.Optimizer,
    dataloader: Iterable,
    device: Device,
    scaler,
    train_iteration: int = 0,
    train_model_weights: bool = False,
    train_optimizer: bool = False,
    log: Optional[logging.Logger] = None,
) -> None:
    """Warm up the optimizer and optionally the model before path stepping."""
    if not train_model_weights and not train_optimizer:
        if log is not None:
            log.info("Skipping pre-training")
        return

    if log is not None:
        log.info(
            "Pre-training: iterations=%s update_weights=%s update_optimizer=%s",
            train_iteration,
            train_model_weights,
            train_optimizer,
        )

    model = adapter.get_model()
    old_model_state = None if train_model_weights else _clone_state_dict(model.state_dict())
    old_optimizer_state = None if train_optimizer else copy.deepcopy(optimizer.state_dict())

    engine_train(
        adapter,
        dataloader,
        optimizer,
        None,
        device=device,
        scaler=scaler,
        max_rounds=train_iteration,
    )

    if old_model_state is not None:
        model.load_state_dict(old_model_state)
        if log is not None:
            log.info("Pre-training complete; restored original model weights")
    if old_optimizer_state is not None:
        optimizer.load_state_dict(old_optimizer_state)
        _optimizer_to(optimizer, device.device)
        if log is not None:
            log.info("Pre-training complete; restored original optimizer state")

    model.to(device.device)


class FindHighAccuracyPathRunner:
    """Reusable stateful runner for a single start/end path."""

    def __init__(self):
        self.runtime_parameter: Optional[RuntimeParameters] = None
        self.config_file = None
        self.checkpoint_content: Optional[Checkpoint] = None

        self.start_point: Optional[str] = None
        self.end_point: Optional[str] = None
        self.arg_output_folder_path: Optional[str] = None

        self.child_logger: Optional[logging.Logger] = None
        self.performance_logger: Optional[logging.Logger] = None
        self.device_obj: Optional[Device] = None
        self.device: Optional[torch.device] = None
        self.scaler = None

        self.current_ml_setup: Optional[MLSetup] = None
        self.model: Optional[torch.nn.Module] = None
        self.adapter: Optional[Any] = None
        self.dataloader: Optional[Iterable] = None
        self.criterion: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.initial_model_stat: Optional[Dict[str, Any]] = None
        self.starting_point_stat: Optional[Dict[str, Any]] = None
        self.end_model_stat_dict: Optional[Dict[str, Any]] = None
        self.current_phase_start_model_stat: Optional[Dict[str, Any]] = None
        self.initial_optimizer_state_dict: Optional[dict[str, Any]] = None
        self.base_optimizer_group_lrs: list[float] = []
        self.param_name_by_id: dict[int, str] = {}
        self.variance_sphere_model: Optional[Dict[str, Any]] = None
        self.target_variance = None

        self.general_parameter: Optional[ParameterGeneral] = None
        self.parameter_train: Optional[ParameterTrain] = None
        self.parameter_move: Optional[ParameterMove] = None
        self.parameter_rebuild_norm: Optional[ParameterRebuildNorm] = None

        self.weight_diff_service = None
        self.weight_change_service = None
        self.distance_to_origin_service = None
        self.record_variance_service = None
        self.record_model_service = None
        self.record_test_accuracy_loss_service = None
        self.record_training_loss_service = None
        self.record_consecutive_points_service = None
        self.record_cosine_similarity_service = None

        self.model_state_of_last_tick: Optional[Dict[str, Any]] = None
        self.latest_checkpoint_path: Optional[str] = None
        self.timer: Optional[float] = None

        self.re_init_norm_layer_list = False
        self.norm_layer_names: list[str] = []
        self.compensate_move_layer: list[str] = []
        self.compensate_movex2_layer: list[str] = []
        self.attention_layer: list[str] = []
        self.ignore_move_layers: list[str] = []
        self.ratio_step_size: Optional[dict[str, float]] = None

        self.initialized = False
        self.finalized = False
        self.finished = False

    def _synchronize_for_timing(self) -> None:
        if self.runtime_parameter is None or not self.runtime_parameter.enable_profiler:
            return
        if self.device is None:
            return
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _time_call(self, func, *args, **kwargs):
        if self.runtime_parameter is None or not self.runtime_parameter.enable_profiler:
            return func(*args, **kwargs), 0.0
        self._synchronize_for_timing()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        self._synchronize_for_timing()
        return result, time.perf_counter() - start_time

    @staticmethod
    def _format_timing_entries(entries: list[tuple[str, float]]) -> str:
        if not entries:
            return "no timed steps"
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

    def _emit_performance_row(
        self,
        tick: int,
        category: str,
        entries: list[tuple[str, float]],
        *,
        total: Optional[float] = None,
    ) -> None:
        if self.runtime_parameter is None or not self.runtime_parameter.enable_profiler:
            return
        message = self._format_performance_row(tick, category, entries, total=total)
        if self.child_logger is not None:
            self.child_logger.info(message)
        if self.performance_logger is not None:
            self.performance_logger.info(message)

    def _setup_performance_logger(self) -> None:
        assert self.runtime_parameter is not None
        assert self.child_logger is not None
        if not self.runtime_parameter.enable_profiler:
            self.performance_logger = None
            return

        os.makedirs(self.runtime_parameter.output_folder_path, exist_ok=True)
        performance_logger = logging.getLogger(f"{self.child_logger.name}.performance")
        performance_logger.setLevel(logging.INFO)
        performance_logger.propagate = False
        performance_log_file = os.path.abspath(
            os.path.join(self.runtime_parameter.output_folder_path, "performance.log")
        )
        if not any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(handler.baseFilename) == performance_log_file
            for handler in performance_logger.handlers
        ):
            file_handler = logging.FileHandler(performance_log_file)
            file_handler.setFormatter(
                logging.Formatter("[%(asctime)s %(name)s] %(levelname)s: %(message)s")
            )
            performance_logger.addHandler(file_handler)
        self.performance_logger = performance_logger

    def _warn_about_resume_config_mismatch(self, _checkpoint_runtime_parameter: RuntimeParameters) -> None:
        assert self.runtime_parameter is not None
        assert self.child_logger is not None
        assert self.checkpoint_content is not None
        checkpoint_content = self.checkpoint_content

        current_config_path = os.path.abspath(self.runtime_parameter.config_file_path)
        checkpoint_config_path = checkpoint_content.checkpoint_config_path
        checkpoint_config_hash = checkpoint_content.checkpoint_config_sha256
        assert checkpoint_config_path is not None
        assert checkpoint_config_hash is not None
        checkpoint_config_path = os.path.abspath(checkpoint_config_path)

        current_config_hash = _compute_file_sha256(self.runtime_parameter.config_file_path)
        assert current_config_hash is not None

        path_mismatch = checkpoint_config_path != current_config_path
        hash_mismatch = checkpoint_config_hash != current_config_hash
        if not path_mismatch and not hash_mismatch:
            return

        details: list[str] = []
        if path_mismatch:
            details.append(f"checkpoint config path: {checkpoint_config_path}")
            details.append(f"current config path: {current_config_path}")
        if hash_mismatch:
            details.append("config file contents differ from the checkpoint snapshot")

        warning_message = "Checkpoint was created with a different config.\n" + "\n".join(details)
        if self.runtime_parameter.silence_mode:
            self.child_logger.warning(warning_message)
            return

        selection = prompt_selection(
            ["Continue with current config", "Quit"],
            prompt_message=warning_message,
            allow_quit=True,
        )
        if selection != "Continue with current config":
            self.child_logger.critical("Aborted because checkpoint config does not match the current config")

    def _resolve_resume_parameter(
        self,
        parameter_name: str,
        config_parameter: Any,
        checkpoint_parameter: Any,
    ) -> Any:
        if config_parameter is not None:
            return config_parameter
        if checkpoint_parameter is None:
            raise RuntimeError(
                f"Config did not provide {parameter_name} at resume tick and checkpoint does not contain one"
            )
        assert self.child_logger is not None
        assert self.runtime_parameter is not None
        self.child_logger.info(
            "Using checkpointed %s at tick %s because the current config did not provide one",
            parameter_name,
            self.runtime_parameter.current_tick,
        )
        return copy.deepcopy(checkpoint_parameter)

    def _build_train_optimizer(self, runtime_parameter: RuntimeParameters, current_ml_setup: MLSetup) -> Optional[torch.optim.Optimizer]:
        assert self.config_file is not None
        assert self.model is not None
        optimizer = self.config_file.get_optimizer_train(
            runtime_parameter,
            current_ml_setup,
            self.model.parameters(),
        )
        if optimizer is not None or self.checkpoint_content is None:
            return optimizer

        bootstrap_runtime_parameter = copy.deepcopy(runtime_parameter)
        bootstrap_runtime_parameter.current_tick = 0
        optimizer = self.config_file.get_optimizer_train(
            bootstrap_runtime_parameter,
            current_ml_setup,
            self.model.parameters(),
        )
        if optimizer is not None:
            assert self.child_logger is not None
            self.child_logger.info(
                "get_optimizer_train(...) returned None at resume tick %s; "
                "falling back to tick 0 construction before loading checkpoint state",
                runtime_parameter.current_tick,
            )
        return optimizer

    def _resolve_dataloader_worker_count(self, explicit_workers: Optional[int]) -> int:
        if explicit_workers is not None:
            return explicit_workers
        assert self.runtime_parameter is not None
        total_cpu_count = self.runtime_parameter.total_cpu_count or os.cpu_count() or 1
        process_worker_count = max(1, self.runtime_parameter.worker_count or 1)
        cpu_budget_per_process = max(1, total_cpu_count // process_worker_count)
        # Conservative default: enough workers to overlap host preprocessing
        # and H2D copies, but capped to avoid oversubscribing many path workers.
        return min(8, max(1, cpu_budget_per_process // 8))

    def setup(
        self,
        index: int,
        runtime_parameter: RuntimeParameters,
        checkpoint_file_path: Optional[str] = None,
    ) -> None:
        self.runtime_parameter = runtime_parameter
        checkpoint_runtime_parameter: Optional[RuntimeParameters] = None
        checkpoint_general_parameter = None
        checkpoint_move_parameter = None
        checkpoint_train_parameter = None
        checkpoint_rebuild_norm_parameter = None
        runtime_parameter.save_ticks = (
            _parse_save_ticks(runtime_parameter.save_ticks)
            if isinstance(runtime_parameter.save_ticks, str)
            else runtime_parameter.save_ticks
        )

        self.config_file = load_configuration(runtime_parameter.config_file_path)
        self.checkpoint_content = None

        if checkpoint_file_path is None:
            self.start_point, self.end_point = runtime_parameter.start_and_end_point_for_paths[index]
            assert self.start_point is not None
            assert self.end_point is not None
            start_point = self.start_point
            end_point = self.end_point
            start_file_name = os.path.basename(start_point).replace(".model.pt", "")
            if end_point == "origin":
                end_file_name = "origin"
            elif end_point == "inf":
                end_file_name = "inf"
            elif end_point == "mean":
                end_file_name = "mean"
            elif end_point == "to_vs":
                end_file_name = "to_vs"
            else:
                end_file_name = os.path.basename(end_point).replace(".model.pt", "")
            runtime_parameter.task_name = f"{start_file_name}-{end_file_name}"
        else:
            logger.info("Loading checkpoint from %s", checkpoint_file_path)
            self.checkpoint_content = torch.load(checkpoint_file_path, map_location="cpu", weights_only=False)
            assert self.checkpoint_content is not None
            checkpoint_runtime_parameter = self.checkpoint_content.current_runtime_parameter
            assert checkpoint_runtime_parameter is not None
            assert checkpoint_runtime_parameter.task_name is not None
            runtime_parameter.task_name = checkpoint_runtime_parameter.task_name
            checkpoint_general_parameter = self.checkpoint_content.current_general_parameter
            checkpoint_move_parameter = self.checkpoint_content.current_move_parameter
            checkpoint_train_parameter = self.checkpoint_content.current_train_parameter
            checkpoint_rebuild_norm_parameter = self.checkpoint_content.current_rebuild_norm_parameter

        self.arg_output_folder_path = os.path.join(runtime_parameter.output_folder_path, runtime_parameter.task_name)
        output_path = self.arg_output_folder_path
        os.makedirs(output_path, exist_ok=True)

        self.child_logger = logging.getLogger(f"find_high_accuracy_path.{runtime_parameter.task_name}")
        assert self.child_logger is not None
        child_logger = self.child_logger
        setup_logging(
            child_logger,
            runtime_parameter.task_name,
            log_file_path=os.path.join(output_path, "info.log"),
            exit_on_critical=True,
        )
        child_logger.info("Logging setup complete")
        self._setup_performance_logger()
        if checkpoint_runtime_parameter is not None:
            self._warn_about_resume_config_mismatch(checkpoint_runtime_parameter)

        self.device_obj = Device.cpu() if runtime_parameter.use_cpu else Device.auto()
        assert self.device_obj is not None
        device_obj = self.device_obj
        self.device = device_obj.device
        assert self.device is not None
        device = self.device

        total_cpu_count = runtime_parameter.total_cpu_count or os.cpu_count() or 1
        worker_count = max(1, runtime_parameter.worker_count or 1)
        thread_per_process = max(1, total_cpu_count // worker_count)
        torch.set_num_threads(thread_per_process)

        self._load_model_and_setup()
        assert self.current_ml_setup is not None
        current_ml_setup = self.current_ml_setup
        assert self.model is not None
        model = self.model
        assert self.config_file is not None
        if checkpoint_runtime_parameter is not None:
            runtime_parameter.current_tick = checkpoint_runtime_parameter.current_tick
        else:
            runtime_parameter.current_tick = 0

        self.general_parameter = self._resolve_resume_parameter(
            "general parameters",
            self.config_file.get_parameter_general(runtime_parameter, current_ml_setup),
            checkpoint_general_parameter,
        )
        general_parameter = self.general_parameter
        assert general_parameter is not None
        general_parameter.fill_default()

        assert general_parameter.max_tick is not None, f"max_tick is not set"
        runtime_parameter.max_tick = general_parameter.max_tick
        runtime_parameter.test_dataset_use_whole = (
            general_parameter.test_dataset_use_whole
            if general_parameter.test_dataset_use_whole is not None
            else False
        )

        child_logger.info("test_dataset_use_whole = %s", runtime_parameter.test_dataset_use_whole)

        num_workers = self._resolve_dataloader_worker_count(general_parameter.dataloader_worker)
        self.dataloader = current_ml_setup.train_dataloader(
            DataloaderConfig(
                num_workers=num_workers,
                prefetch_factor=general_parameter.dataloader_prefetch_factor,
            )
        )
        self.criterion = _try_get_criterion(current_ml_setup)
        assert self.dataloader is not None

        model.to(device)
        self.optimizer = self._build_train_optimizer(runtime_parameter, current_ml_setup)
        if self.optimizer is None:
            raise RuntimeError("get_optimizer_train(...) returned None during initialization")
        optimizer = self.optimizer

        if checkpoint_file_path is not None:
            assert self.checkpoint_content is not None
            checkpoint_content: Checkpoint = self.checkpoint_content
            assert checkpoint_content.current_optimizer_stat is not None
            optimizer.load_state_dict(checkpoint_content.current_optimizer_stat)
            _optimizer_to(optimizer, device)

        self.initial_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        self.base_optimizer_group_lrs = [group["lr"] for group in optimizer.param_groups]
        self.param_name_by_id = {id(param): name for name, param in model.named_parameters()}

        self.parameter_train = self._resolve_resume_parameter(
            "train parameters",
            self.config_file.get_parameter_train(runtime_parameter, current_ml_setup),
            checkpoint_train_parameter,
        )
        parameter_train = self.parameter_train
        assert parameter_train is not None
        parameter_train.fill_default()

        self.parameter_move = self._resolve_resume_parameter(
            "move parameters",
            self.config_file.get_parameter_move(runtime_parameter, current_ml_setup),
            checkpoint_move_parameter,
        )
        parameter_move = self.parameter_move
        assert parameter_move is not None
        parameter_move.fill_default()

        self.parameter_rebuild_norm = self._resolve_resume_parameter(
            "rebuild-norm parameters",
            self.config_file.get_parameter_rebuild_norm(runtime_parameter, current_ml_setup),
            checkpoint_rebuild_norm_parameter,
        )
        assert self.parameter_rebuild_norm is not None

        self.scaler = device_obj.make_scaler() if runtime_parameter.use_amp else None
        assert self.end_model_stat_dict is not None
        self.end_model_stat_dict = _ensure_state_dict_device(self.end_model_stat_dict, device)
        if self.current_phase_start_model_stat is not None:
            self.current_phase_start_model_stat = _ensure_state_dict_device(
                self.current_phase_start_model_stat,
                device,
            )
        if self.variance_sphere_model is not None:
            self.variance_sphere_model = _ensure_state_dict_device(self.variance_sphere_model, device)
        self._update_ratio_step_size()

        self._initialize_services()

        variance_record = VarianceCorrector(VarianceCorrectionType.FollowOthers)
        assert self.starting_point_stat is not None
        variance_record.add_variance(self.starting_point_stat)
        self.target_variance = variance_record.get_variance()

        self.re_init_norm_layer_list = checkpoint_file_path is not None
        if checkpoint_file_path is not None:
            checkpoint_folder_path = os.path.dirname(checkpoint_file_path)
            assert self.checkpoint_content is not None

            for service_instance in (
                self.weight_diff_service,
                self.weight_change_service,
                self.distance_to_origin_service,
                self.record_variance_service,
                self.record_model_service,
                self.record_test_accuracy_loss_service,
                self.record_training_loss_service,
                self.record_consecutive_points_service,
                self.record_cosine_similarity_service,
            ):
                if service_instance is not None:
                    service_instance.continue_from_checkpoint(
                        checkpoint_folder_path,
                        runtime_parameter.current_tick,
                    )

        self.timer = time.time()
        self.initialized = True

    def _load_model_and_setup(self) -> None:
        assert self.runtime_parameter is not None
        assert self.child_logger is not None
        runtime_parameter = self.runtime_parameter
        child_logger = self.child_logger

        if self.checkpoint_content is None:
            assert self.start_point is not None
            start_point = self.start_point
            start_model_state, start_model_name, dataset_name_in_file = load_model_state_file(start_point)
            start_model_type = RuntimeParameters.coerce_model_type(start_model_name)
            assert(start_model_type is not None)
            dataset_type_in_file = RuntimeParameters.coerce_dataset_type(dataset_name_in_file)
            if dataset_type_in_file is None:
                dataset_type_in_file = runtime_parameter.dataset_type

            child_logger.info("Loading start model from %s", start_point)
            if runtime_parameter.dataset_type is not None:
                assert dataset_type_in_file == runtime_parameter.dataset_type
                self.current_ml_setup = get_ml_setup_from_config(
                    start_model_type.name,
                    dataset_type=runtime_parameter.dataset_type.name,
                    preset=runtime_parameter.pytorch_preset_version or 1,
                    use_dali=runtime_parameter.use_dali,
                    dali_device_id=runtime_parameter.dali_device_id,
                )
            else:
                dataset_type_name = dataset_type_in_file.name if dataset_type_in_file is not None else "default"
                self.current_ml_setup = get_ml_setup_from_config(
                    start_model_type.name,
                    dataset_type=dataset_type_name,
                    preset=runtime_parameter.pytorch_preset_version or 1,
                    use_dali=runtime_parameter.use_dali,
                    dali_device_id=runtime_parameter.dali_device_id,
                )

            assert self.current_ml_setup is not None
            current_ml_setup = self.current_ml_setup
            _apply_ml_setup_compatibility(current_ml_setup)

            runtime_parameter.model_type = current_ml_setup.model_type
            runtime_parameter.dataset_type = current_ml_setup.dataset_type
            child_logger.info("Model type: %s", runtime_parameter.model_type.name)
            child_logger.info("Dataset type: %s", runtime_parameter.dataset_type.name)

            self.initial_model_stat = _clone_state_dict(current_ml_setup.model.state_dict())
            self.starting_point_stat = _clone_state_dict(start_model_state)
            self.current_phase_start_model_stat = _clone_state_dict(start_model_state)

            self.model = current_ml_setup.model
            assert self.model is not None
            model = self.model
            model.load_state_dict(start_model_state)
            self.adapter = current_ml_setup.adapter

            if runtime_parameter.work_mode == WorkMode.to_origin:
                self.end_model_stat_dict = {
                    key: torch.zeros_like(value) if torch.is_tensor(value) else copy.deepcopy(value)
                    for key, value in start_model_state.items()
                }
                child_logger.info("Work mode: to_origin")
            elif runtime_parameter.work_mode == WorkMode.to_inf:
                self.end_model_stat_dict = {
                    key: value.detach().clone() * 2 if torch.is_tensor(value) else copy.deepcopy(value)
                    for key, value in start_model_state.items()
                }
                child_logger.info("Work mode: to_inf")
            elif runtime_parameter.work_mode == WorkMode.to_mean:
                self.end_model_stat_dict = {
                    key: torch.full_like(value, value.float().mean().item()) if torch.is_tensor(value) else copy.deepcopy(value)
                    for key, value in start_model_state.items()
                }
                child_logger.info("Work mode: to_mean")
            elif runtime_parameter.work_mode == WorkMode.to_certain_model:
                assert self.end_point is not None
                end_point = self.end_point
                self.end_model_stat_dict, end_model_name, _ = load_model_state_file(end_point)
                end_model_type = RuntimeParameters.coerce_model_type(end_model_name)
                assert end_model_type is not None
                assert end_model_type == start_model_type, f"start({start_model_type.name}) != end({end_model_type.name})"
                child_logger.info("Work mode: to_certain_model at %s", end_point)
            elif runtime_parameter.work_mode == WorkMode.to_vs:
                assert runtime_parameter.variance_sphere_file_path is not None
                variance_sphere_model, variance_sphere_name, _ = load_model_state_file(
                    runtime_parameter.variance_sphere_file_path
                )
                variance_sphere_type = RuntimeParameters.coerce_model_type(variance_sphere_name)
                assert variance_sphere_type is not None
                assert variance_sphere_type == start_model_type
                self.variance_sphere_model = variance_sphere_model
                self.end_model_stat_dict = calculate_layer_wise_projection_to_variance_sphere(
                    start_model_state,
                    self.variance_sphere_model,
                )
                child_logger.info(
                    "Work mode: to_variance_sphere at %s",
                    runtime_parameter.variance_sphere_file_path,
                )
            else:
                raise NotImplementedError(f"Unknown work mode: {runtime_parameter.work_mode}")

            assert self.end_model_stat_dict is not None
            assert not _state_dicts_equal(start_model_state, self.end_model_stat_dict), (
                "Starting model is identical to destination model"
            )

        else:
            assert self.checkpoint_content is not None
            checkpoint_content: Checkpoint = self.checkpoint_content
            checkpoint_runtime = checkpoint_content.current_runtime_parameter
            assert checkpoint_runtime is not None
            runtime_parameter.work_mode = checkpoint_runtime.work_mode
            runtime_parameter.dataset_type = checkpoint_runtime.dataset_type
            runtime_parameter.model_type = checkpoint_runtime.model_type
            runtime_parameter.save_format = checkpoint_runtime.save_format
            runtime_parameter.save_interval = checkpoint_runtime.save_interval
            runtime_parameter.save_ticks = checkpoint_runtime.save_ticks
            if isinstance(runtime_parameter.save_ticks, str):
                runtime_parameter.save_ticks = _parse_save_ticks(runtime_parameter.save_ticks)

            self.current_ml_setup = get_ml_setup_from_config(
                _require_model_type_name(runtime_parameter.model_type),
                dataset_type=_require_dataset_type_name(runtime_parameter.dataset_type),
                preset=runtime_parameter.pytorch_preset_version or 1,
            )
            assert self.current_ml_setup is not None
            current_ml_setup = self.current_ml_setup
            _apply_ml_setup_compatibility(current_ml_setup)

            self.initial_model_stat = checkpoint_content.init_model_stat
            self.starting_point_stat = checkpoint_content.start_model_stat
            self.end_model_stat_dict = checkpoint_content.end_model_stat
            self.current_phase_start_model_stat = checkpoint_content.current_phase_start_model_stat
            self.model = current_ml_setup.model
            assert self.model is not None
            model = self.model
            assert checkpoint_content.current_model_stat is not None
            model.load_state_dict(checkpoint_content.current_model_stat)
            assert self.current_phase_start_model_stat is not None
            self.adapter = current_ml_setup.adapter

            if runtime_parameter.work_mode == WorkMode.to_vs:
                if runtime_parameter.variance_sphere_file_path is None:
                    raise RuntimeError("variance_sphere_file_path is required to resume a to_vs run")
                self.variance_sphere_model, variance_sphere_name, _ = load_model_state_file(
                    runtime_parameter.variance_sphere_file_path
                )
                assert RuntimeParameters.coerce_model_type(variance_sphere_name) == runtime_parameter.model_type

    def _initialize_services(self) -> None:
        assert self.arg_output_folder_path is not None
        assert self.runtime_parameter is not None
        assert self.general_parameter is not None
        assert self.child_logger is not None
        assert self.model is not None
        assert self.current_ml_setup is not None
        assert self.device_obj is not None
        assert self.starting_point_stat is not None
        assert self.end_model_stat_dict is not None
        output_path = self.arg_output_folder_path
        runtime_parameter = self.runtime_parameter
        general_parameter = self.general_parameter
        child_logger = self.child_logger
        model = self.model
        current_ml_setup = self.current_ml_setup
        criterion = self.criterion
        device_obj = self.device_obj
        starting_point_stat = self.starting_point_stat
        end_model_stat_dict = self.end_model_stat_dict
        num_workers = self._resolve_dataloader_worker_count(general_parameter.dataloader_worker)

        child_logger.info("Initializing services")
        current_state = model.state_dict()
        all_model_states = [current_state, end_model_stat_dict]

        self.weight_diff_service = record_weights_difference.ServiceWeightsDifferenceRecorder(1)
        self.weight_diff_service.initialize_without_runtime_parameters(
            all_model_states,
            output_path,
            logger=child_logger,
        )

        self.weight_change_service = record_weights_difference.ServiceWeightsDifferenceRecorder(
            1,
            l1_save_file_name="weight_change_l1.csv",
            l2_save_file_name="weight_change_l2.csv",
        )
        self.model_state_of_last_tick = _clone_state_dict(current_state)
        self.weight_change_service.initialize_without_runtime_parameters(
            all_model_states,
            output_path,
            logger=child_logger,
        )

        self.distance_to_origin_service = record_weights_difference.ServiceDistanceToOriginRecorder(1, [0])
        self.distance_to_origin_service.initialize_without_runtime_parameters(
            {0: starting_point_stat},
            output_path,
            logger=child_logger,
        )

        self.record_variance_service = record_variance.ServiceVarianceRecorder(1)
        self.record_variance_service.initialize_without_runtime_parameters(
            [0],
            [starting_point_stat],
            output_path,
            logger=child_logger,
        )

        if not runtime_parameter.service_cosine_similarity_disable:
            self.record_cosine_similarity_service = record_cosine_similarity.ServiceCosineSimilarityRecorder(1)
            if runtime_parameter.service_cosine_similarity_ref_model is None:
                reference_model = starting_point_stat
            else:
                reference_model, reference_model_name, _ = load_model_state_file(
                    runtime_parameter.service_cosine_similarity_ref_model
                )
                assert RuntimeParameters.coerce_model_type(reference_model_name) == runtime_parameter.model_type
            self.record_cosine_similarity_service.initialize_without_runtime_parameters(
                {0: reference_model},
                output_path,
                logger=child_logger,
            )

        if runtime_parameter.save_format != "none":
            self.record_model_service = record_model_stat.ModelStatRecorder(
                sys.maxsize,
                _require_model_type_name(runtime_parameter.model_type),
                _require_dataset_type_name(runtime_parameter.dataset_type),
            )
            self.record_model_service.initialize_without_runtime_parameters(
                [0],
                output_path,
                logger=child_logger,
                save_format=runtime_parameter.save_format,
            )
            if runtime_parameter.save_ticks:
                child_logger.info("record_model_service is enabled at explicit ticks")
            else:
                child_logger.info(
                    "record_model_service is enabled every %s tick(s)",
                    runtime_parameter.save_interval,
                )
        else:
            child_logger.info("record_model_service is disabled")

        if not runtime_parameter.service_test_accuracy_loss_disable:
            test_dataset_interval = max(1, general_parameter.test_dataset_interval or 10)
            self.record_test_accuracy_loss_service = record_test_accuracy_loss.ServiceTestAccuracyLossRecorder(
                interval=test_dataset_interval,
                test_batch_size=runtime_parameter.service_test_accuracy_loss_batch_size,
                model_name=_require_model_type_name(runtime_parameter.model_type),
                dataset_name=_require_dataset_type_name(runtime_parameter.dataset_type),
                store_top_accuracy_model_count=runtime_parameter.store_top_accuracy_model_count,
                use_fixed_testing_dataset=True,
                test_whole_dataset=runtime_parameter.test_dataset_use_whole,
                test_val_split=general_parameter.split_test_val,
            )
            child_logger.info("record_test_accuracy_loss_service interval = %s tick(s)", test_dataset_interval)
            self.record_test_accuracy_loss_service.enable_profiler = runtime_parameter.enable_profiler
            self.record_test_accuracy_loss_service.performance_logger = self.performance_logger
            self.record_test_accuracy_loss_service.initialize_without_runtime_parameters(
                output_path=output_path,
                node_names=[0],
                model=model,
                criterion=criterion,
                test_dataset=current_ml_setup.testing_data,
                ml_setup=current_ml_setup,
                logger=child_logger,
                device=device_obj,
                num_workers=num_workers,
                prefetch_factor=general_parameter.dataloader_prefetch_factor,
            )

        self.record_training_loss_service = record_training_loss_accuracy.ServiceTrainingLossAccuracyRecorder(1)
        self.record_training_loss_service.initialize_without_runtime_parameters(
            output_path,
            [0],
            logger=child_logger,
        )

        if runtime_parameter.linear_interpolation_points_size > 0:
            self.record_consecutive_points_service = (
                record_consecutive_linear_interpolation.ServiceConsecutiveLinearInterpolationRecorder(
                    interval=1,
                    batch_size=runtime_parameter.service_test_accuracy_loss_batch_size,
                    dataset_size=runtime_parameter.linear_interpolation_dataset_size,
                    points_size=runtime_parameter.linear_interpolation_points_size,
                    recorded_node_name=0,
                )
            )
            self.record_consecutive_points_service.enable_profiler = runtime_parameter.enable_profiler
            self.record_consecutive_points_service.performance_logger = self.performance_logger
            self.record_consecutive_points_service.initialize_without_runtime_parameters(
                output_path=output_path,
                model=model,
                criterion=criterion,
                train_dataset=current_ml_setup.training_data,
                ml_setup=current_ml_setup,
                logger=child_logger,
                device=device_obj,
                num_workers=num_workers,
                prefetch_factor=general_parameter.dataloader_prefetch_factor,
            )

    def _refresh_layer_lists(self) -> None:
        assert self.child_logger is not None
        assert self.model is not None
        assert self.starting_point_stat is not None
        assert self.parameter_move is not None
        assert self.runtime_parameter is not None
        child_logger = self.child_logger
        model = self.model
        starting_point_stat = self.starting_point_stat
        parameter_move = self.parameter_move
        runtime_parameter = self.runtime_parameter

        self.re_init_norm_layer_list = False
        self.norm_layer_names = []
        self.compensate_move_layer = []
        self.compensate_movex2_layer = []
        self.attention_layer = []
        self.ignore_move_layers = []

        norm_results = find_normalization_layers(model)
        batch_norm_layer_names, _ = find_layers_according_to_name_and_keyword(
            starting_point_stat,
            [],
            [],
            norm_results.batch_normalization,
        )
        layer_norm_layer_names, _ = find_layers_according_to_name_and_keyword(
            starting_point_stat,
            [],
            [],
            norm_results.layer_normalization,
        )

        assert len(norm_results.group_normalization) == 0, "Group normalization is not supported yet"
        assert len(norm_results.instance_normalization) == 0, "Instance normalization is not supported yet"

        batch_norm_layer_names.sort()
        layer_norm_layer_names.sort()
        self.norm_layer_names.extend(batch_norm_layer_names)
        self.norm_layer_names.extend(layer_norm_layer_names)

        ignored_from_config, _ = find_layers_according_to_name_and_keyword(
            starting_point_stat,
            parameter_move.layer_skip_move,
            parameter_move.layer_skip_move_keyword,
        )
        self.ignore_move_layers.extend(ignored_from_config)

        layer_compensate_x2, _ = find_layers_according_to_name_and_keyword(
            starting_point_stat,
            parameter_move.layer_compensate_x2,
            parameter_move.layer_compensate_x2_keyword,
        )

        attention_layers, _ = find_layers_according_to_name_and_keyword(
            starting_point_stat,
            parameter_move.layer_attention,
            parameter_move.layer_attention_keyword,
        )
        self.attention_layer = sorted(attention_layers)
        if self.attention_layer:
            child_logger.info(
                "Found %s attention layers (policy=%s): %s",
                len(self.attention_layer),
                parameter_move.layer_attention_policy,
                self.attention_layer,
            )

        if runtime_parameter.work_mode in (WorkMode.to_inf, WorkMode.to_mean, WorkMode.to_origin):
            for layer_name in layer_compensate_x2:
                assert layer_name in layer_norm_layer_names, f"{layer_name} is not a layer norm layer"
            self.compensate_move_layer.extend(layer_norm_layer_names)
            self.compensate_movex2_layer.extend(layer_compensate_x2)
            self.ignore_move_layers.extend(layer_norm_layer_names)
            self.ignore_move_layers.extend(batch_norm_layer_names)

        self.compensate_move_layer = sorted(set(self.compensate_move_layer) - set(ignored_from_config))
        self.compensate_movex2_layer = sorted(set(self.compensate_movex2_layer) - set(ignored_from_config))
        self.ignore_move_layers = sorted(set(self.ignore_move_layers))

        moved_layers = sorted(
            set(starting_point_stat.keys())
            - set(self.ignore_move_layers)
            - set(self.compensate_move_layer)
            - set(self.compensate_movex2_layer)
        )
        child_logger.info("ignore_move (%s): %s", len(self.ignore_move_layers), self.ignore_move_layers)
        child_logger.info(
            "compensate_move (%s): %s",
            len(self.compensate_move_layer),
            self.compensate_move_layer,
        )
        child_logger.info("move (%s): %s", len(moved_layers), moved_layers)

        if not runtime_parameter.silence_mode:
            input("Check the planned layer movement and press Enter to continue, or Ctrl+C to quit")

    def _update_ratio_step_size(self) -> None:
        assert self.parameter_move is not None
        assert self.end_model_stat_dict is not None
        assert self.current_phase_start_model_stat is not None
        parameter_move = self.parameter_move
        end_model_stat_dict = self.end_model_stat_dict
        current_phase_start_model_stat = self.current_phase_start_model_stat

        if parameter_move.ratio_step_size is None:
            self.ratio_step_size = None
            return

        ratio_step_size: dict[str, float] = {}
        for layer_name, current_weights in current_phase_start_model_stat.items():
            if layer_name not in end_model_stat_dict:
                continue
            distance = geodesic_distance(current_weights, end_model_stat_dict[layer_name])
            if distance is not None:
                ratio_step_size[layer_name] = distance.item() * parameter_move.ratio_step_size
        self.ratio_step_size = ratio_step_size

    def _maybe_save_checkpoint(self) -> None:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if not runtime_parameter.checkpoint_interval:
            return
        if runtime_parameter.debug_check_config_mode:
            return
        if runtime_parameter.current_tick % runtime_parameter.checkpoint_interval != 0:
            return

        assert self.arg_output_folder_path is not None
        assert self.model is not None
        assert self.optimizer is not None
        assert self.child_logger is not None
        assert self.general_parameter is not None
        assert self.parameter_move is not None
        assert self.parameter_train is not None
        assert self.parameter_rebuild_norm is not None
        assert self.starting_point_stat is not None
        assert self.end_model_stat_dict is not None
        assert self.current_phase_start_model_stat is not None
        assert self.initial_model_stat is not None
        output_folder_path = self.arg_output_folder_path
        model = self.model
        optimizer = self.optimizer
        child_logger = self.child_logger
        general_parameter = self.general_parameter
        parameter_move = self.parameter_move
        parameter_train = self.parameter_train
        parameter_rebuild_norm = self.parameter_rebuild_norm
        starting_point_stat = self.starting_point_stat
        end_model_stat_dict = self.end_model_stat_dict
        current_phase_start_model_stat = self.current_phase_start_model_stat
        initial_model_stat = self.initial_model_stat

        checkpoint_path = os.path.join(
            output_folder_path,
            f"checkpoint_{runtime_parameter.current_tick}.checkpoint.pt",
        )
        previous_checkpoint_path = self.latest_checkpoint_path

        checkpoint = Checkpoint()
        checkpoint.current_model_stat = _state_dict_to_cpu(model.state_dict())
        checkpoint.current_optimizer_stat = _move_nested_tensors(optimizer.state_dict(), "cpu")
        checkpoint.current_runtime_parameter = copy.deepcopy(runtime_parameter)
        checkpoint.current_general_parameter = copy.deepcopy(general_parameter)
        checkpoint.current_move_parameter = copy.deepcopy(parameter_move)
        checkpoint.current_train_parameter = copy.deepcopy(parameter_train)
        checkpoint.current_rebuild_norm_parameter = copy.deepcopy(parameter_rebuild_norm)
        checkpoint.start_model_stat = _state_dict_to_cpu(starting_point_stat)
        checkpoint.end_model_stat = _state_dict_to_cpu(end_model_stat_dict)
        checkpoint.current_phase_start_model_stat = _state_dict_to_cpu(current_phase_start_model_stat)
        checkpoint.init_model_stat = _state_dict_to_cpu(initial_model_stat)
        checkpoint.checkpoint_config_path = os.path.abspath(runtime_parameter.config_file_path)
        checkpoint.checkpoint_config_sha256 = _compute_file_sha256(runtime_parameter.config_file_path)

        child_logger.info("Saving checkpoint to %s", checkpoint_path)
        atomic_torch_save(checkpoint, checkpoint_path)
        self.latest_checkpoint_path = checkpoint_path

        if previous_checkpoint_path and previous_checkpoint_path != checkpoint_path and os.path.exists(previous_checkpoint_path):
            os.remove(previous_checkpoint_path)

    def _update_dynamic_end_model(self) -> None:
        assert self.runtime_parameter is not None
        assert self.model is not None
        assert self.device is not None
        runtime_parameter = self.runtime_parameter
        model = self.model
        device = self.device

        if runtime_parameter.work_mode == WorkMode.to_inf:
            current_state = model.state_dict()
            self.end_model_stat_dict = {
                key: value.detach().clone() * 2 if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in current_state.items()
            }
        elif runtime_parameter.work_mode == WorkMode.to_vs:
            current_state = model.state_dict()
            assert self.variance_sphere_model is not None
            self.end_model_stat_dict = calculate_layer_wise_projection_to_variance_sphere(
                current_state,
                self.variance_sphere_model,
            )

        assert self.end_model_stat_dict is not None
        self.end_model_stat_dict = _ensure_state_dict_device(self.end_model_stat_dict, device)

    def _maybe_log_eta(self) -> None:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.current_tick == 0:
            return
        if runtime_parameter.current_tick % REPORT_FINISH_TIME_PER_TICK != 0:
            return

        assert self.child_logger is not None
        assert self.timer is not None
        child_logger = self.child_logger
        timer = self.timer
        elapsed = time.time() - timer
        self.timer = time.time()
        current_time = self.timer
        remaining_chunks = (
            runtime_parameter.max_tick - runtime_parameter.current_tick
        ) // REPORT_FINISH_TIME_PER_TICK
        eta = datetime.fromtimestamp(current_time + remaining_chunks * elapsed)
        child_logger.info(
            "%s ticks took %.1fs; ETA: %s",
            REPORT_FINISH_TIME_PER_TICK,
            elapsed,
            eta,
        )

    def _update_parameters_for_current_tick(self) -> bool:
        assert self.runtime_parameter is not None
        assert self.current_ml_setup is not None
        assert self.child_logger is not None
        assert self.config_file is not None
        assert self.model is not None
        runtime_parameter = self.runtime_parameter
        current_ml_setup = self.current_ml_setup
        child_logger = self.child_logger
        model = self.model
        parameter_updated = False

        new_parameter_train = self.config_file.get_parameter_train(runtime_parameter, current_ml_setup)
        if new_parameter_train is not None:
            parameter_updated = True
            new_parameter_train.fill_default()
            self.parameter_train = new_parameter_train
            child_logger.info("Updated train parameters at tick %s", runtime_parameter.current_tick)

        new_parameter_move = self.config_file.get_parameter_move(runtime_parameter, current_ml_setup)
        if new_parameter_move is not None:
            parameter_updated = True
            new_parameter_move.fill_default()
            self.parameter_move = new_parameter_move
            self.current_phase_start_model_stat = _clone_state_dict(model.state_dict())
            self.re_init_norm_layer_list = True
            self._update_ratio_step_size()

        if self.re_init_norm_layer_list:
            self._refresh_layer_lists()

        new_parameter_rebuild_norm = self.config_file.get_parameter_rebuild_norm(
            runtime_parameter,
            current_ml_setup,
        )
        if new_parameter_rebuild_norm is not None:
            parameter_updated = True
            self.parameter_rebuild_norm = new_parameter_rebuild_norm
            if ENABLE_REBUILD_NORM and new_parameter_rebuild_norm.rebuild_norm_for_max_rounds:
                assert self.starting_point_stat is not None
                extra_layers, _ = find_layers_according_to_name_and_keyword(
                    self.starting_point_stat,
                    new_parameter_rebuild_norm.rebuild_norm_layer,
                    new_parameter_rebuild_norm.rebuild_norm_layer_keyword,
                )
                self.norm_layer_names = sorted(set(self.norm_layer_names).union(extra_layers))

        return parameter_updated

    def _maybe_first_tick_setup(self) -> None:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.current_tick != 0:
            return
        if runtime_parameter.debug_check_config_mode:
            return

        assert self.parameter_train is not None
        assert self.optimizer is not None
        assert self.dataloader is not None
        assert self.device_obj is not None
        assert self.child_logger is not None
        assert self.adapter is not None
        parameter_train = self.parameter_train
        optimizer = self.optimizer
        dataloader = self.dataloader
        device_obj = self.device_obj
        child_logger = self.child_logger

        if parameter_train.pretrain_optimizer and parameter_train.load_existing_optimizer:
            raise RuntimeError("Cannot enable both pretrain_optimizer and load_existing_optimizer")

        if parameter_train.pretrain_optimizer or parameter_train.pretrain_model_weights:
            assert parameter_train.pretrain_iterations is not None
            pretrain_optimizer = parameter_train.pretrain_optimizer == True
            pretrain_model_weights = parameter_train.pretrain_model_weights == True
            pre_train(
                self.adapter,
                optimizer,
                dataloader,
                device_obj,
                self.scaler,
                train_iteration=parameter_train.pretrain_iterations,
                train_optimizer=pretrain_optimizer,
                train_model_weights=pretrain_model_weights,
                log=child_logger,
            )

        if parameter_train.load_existing_optimizer:
            assert self.start_point is not None
            assert self.device is not None
            optimizer_path = self.start_point.replace("model.pt", "optimizer.pt")
            load_existing_optimizer_stat(
                optimizer,
                optimizer_path,
                self.device,
                log=child_logger,
            )

    def _save_on_parameter_update(self) -> None:
        assert self.arg_output_folder_path is not None
        assert self.runtime_parameter is not None
        assert self.model is not None
        assert self.optimizer is not None
        output_path = self.arg_output_folder_path
        runtime_parameter = self.runtime_parameter
        model = self.model
        optimizer = self.optimizer
        save_model_state(
            os.path.join(output_path, f"{runtime_parameter.current_tick}.model.pt"),
            _state_dict_to_cpu(model.state_dict()),
            _require_model_type_name(runtime_parameter.model_type),
            _require_dataset_type_name(runtime_parameter.dataset_type),
        )
        save_optimizer_state(
            os.path.join(output_path, f"{runtime_parameter.current_tick}.optimizer.pt"),
            _move_nested_tensors(optimizer.state_dict(), "cpu"),
            _require_model_type_name(runtime_parameter.model_type),
            _require_dataset_type_name(runtime_parameter.dataset_type),
        )

    def _move_model(self) -> None:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.debug_check_config_mode:
            return

        assert self.model is not None
        assert self.parameter_move is not None
        assert self.end_model_stat_dict is not None
        assert self.child_logger is not None
        model = self.model
        parameter_move = self.parameter_move
        end_model_stat_dict = self.end_model_stat_dict
        child_logger = self.child_logger

        current_state = model.state_dict()
        old_attention_state = _capture_attention_state(current_state, self.attention_layer)

        step_size = 0.0 if parameter_move.step_size is None else parameter_move.step_size
        adoptive_step_size = 0.0 if parameter_move.adoptive_step_size is None else parameter_move.adoptive_step_size
        merge_bias_with_weights = parameter_move.merge_bias_with_weights == True
        target_state = move_model_state_toward(
            current_state,
            end_model_stat_dict,
            step_size,
            adoptive_step_size,
            enable_merge_bias_with_weight=merge_bias_with_weights,
            ignore_layers=self.ignore_move_layers,
            ratio_step_per_layer=self.ratio_step_size,
        )

        compensate_move_layer = list(self.compensate_move_layer)
        compensate_movex2_layer = list(self.compensate_movex2_layer)
        compensate_destination: Dict[str, Any] = {}
        requested_compensate_layers = compensate_move_layer + compensate_movex2_layer
        if requested_compensate_layers:
            compensate_destination, skipped_compensate_layers = _build_compensate_destination(
                current_state,
                end_model_stat_dict,
                requested_compensate_layers,
            )
            if skipped_compensate_layers:
                child_logger.info(
                    "skip_compensate (%s): %s",
                    len(skipped_compensate_layers),
                    skipped_compensate_layers,
                )
            compensate_move_layer = [
                layer_name for layer_name in compensate_move_layer if layer_name in compensate_destination
            ]
            compensate_movex2_layer = [
                layer_name for layer_name in compensate_movex2_layer if layer_name in compensate_destination
            ]

        if compensate_move_layer:
            if self.ratio_step_size is not None:
                child_logger.warning("ratio_step_size with compensate layers can be hard to interpret")
            target_state = move_model_state_toward(
                target_state,
                compensate_destination,
                step_size,
                adoptive_step_size,
                enable_merge_bias_with_weight=merge_bias_with_weights,
                move_layer=compensate_move_layer,
                ratio_step_per_layer=self.ratio_step_size,
            )
        if compensate_movex2_layer:
            target_state = move_model_state_toward(
                target_state,
                compensate_destination,
                step_size,
                adoptive_step_size,
                enable_merge_bias_with_weight=merge_bias_with_weights,
                move_layer=compensate_movex2_layer,
                ratio_step_per_layer=self.ratio_step_size,
            )

        _apply_attention_policy(
            old_attention_state,
            target_state,
            self.attention_layer,
            parameter_move.layer_attention_policy,
        )
        model.load_state_dict(target_state)

    def _apply_variance_correction(self, *, phase: str) -> None:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.debug_check_config_mode:
            return
        if runtime_parameter.work_mode != WorkMode.to_certain_model:
            return
        assert self.target_variance is not None
        assert self.child_logger is not None
        assert self.model is not None
        child_logger = self.child_logger
        model = self.model
        child_logger.info(
            "tick %s: rescale variance (%s)",
            runtime_parameter.current_tick,
            phase,
        )
        corrected_state = VarianceCorrector.scale_model_stat_to_variance(
            model.state_dict(),
            self.target_variance,
            ignore_layer_list=self.ignore_move_layers,
        )
        model.load_state_dict(corrected_state)

    def _adjust_learning_rates(self) -> None:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.debug_check_config_mode:
            return
        if runtime_parameter.work_mode not in (WorkMode.to_inf, WorkMode.to_origin, WorkMode.to_mean):
            return
        assert self.target_variance is not None
        assert self.optimizer is not None
        optimizer = self.optimizer
        for group_index, param_group in enumerate(optimizer.param_groups):
            base_lr = self.base_optimizer_group_lrs[group_index]
            updated_lrs = []
            for param in param_group["params"]:
                parameter_name = self.param_name_by_id.get(id(param))
                if parameter_name is None:
                    continue
                if parameter_name in self.norm_layer_names:
                    continue
                if "weight" not in parameter_name or not param.requires_grad:
                    continue

                target_variance = self.target_variance.get(parameter_name)
                if target_variance is None or target_variance <= 0:
                    continue

                current_variance = torch.var(param.data).item()
                if runtime_parameter.across_vs_lr_policy == "var":
                    scale = current_variance / target_variance
                elif runtime_parameter.across_vs_lr_policy == "std":
                    scale = math.sqrt(current_variance / target_variance)
                else:
                    raise NotImplementedError(runtime_parameter.across_vs_lr_policy)
                updated_lrs.append(base_lr * scale)

            if updated_lrs:
                param_group["lr"] = float(sum(updated_lrs) / len(updated_lrs))

    def _train_current_tick(self) -> tuple[float, float]:
        training_loss = 0.0
        training_accuracy = 0.0
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.debug_check_config_mode:
            return training_loss, training_accuracy

        assert self.dataloader is not None
        assert self.optimizer is not None
        assert self.device_obj is not None
        assert self.parameter_train is not None
        assert self.child_logger is not None
        assert self.adapter is not None
        dataloader = self.dataloader
        optimizer = self.optimizer
        device_obj = self.device_obj
        parameter_train = self.parameter_train
        child_logger = self.child_logger

        training_result = engine_train(
            self.adapter,
            dataloader,
            optimizer,
            None,
            device=device_obj,
            scaler=self.scaler,
            min_rounds=parameter_train.train_for_min_rounds or 0,
            max_rounds=parameter_train.train_for_max_rounds or 1,
            loss_threshold=parameter_train.train_until_loss,
        )
        training_loss = (
            training_result.moving_average_loss
            if training_result.moving_average_loss is not None
            else training_result.avg_loss
        )
        training_accuracy = 0.0 if training_result.accuracy is None else training_result.accuracy
        child_logger.info(
            "tick %s: trained %s step(s), loss=%.4f",
            runtime_parameter.current_tick,
            training_result.iterations,
            training_loss,
        )
        return training_loss, training_accuracy

    def _rebuild_norm_layers(self) -> None:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.debug_check_config_mode:
            return
        assert self.parameter_rebuild_norm is not None
        parameter_rebuild_norm = self.parameter_rebuild_norm
        if not ENABLE_REBUILD_NORM or not parameter_rebuild_norm.rebuild_norm_for_max_rounds:
            return

        assert self.current_ml_setup is not None
        assert self.model is not None
        assert self.config_file is not None
        current_ml_setup = self.current_ml_setup
        model = self.model
        rebuild_optimizer = self.config_file.get_optimizer_rebuild_norm(
            runtime_parameter,
            current_ml_setup,
            model.parameters(),
        )
        if rebuild_optimizer is None:
            return

        assert self.initial_model_stat is not None
        assert self.starting_point_stat is not None
        assert self.optimizer is not None
        assert self.dataloader is not None
        assert self.device is not None
        assert self.child_logger is not None
        rebuild_norm_layer_function(
            model,
            self.initial_model_stat,
            self.starting_point_stat,
            rebuild_optimizer,
            _move_nested_tensors(self.optimizer.state_dict(), "cpu"),
            self.norm_layer_names,
            current_ml_setup,
            self.dataloader,
            parameter_rebuild_norm,
            runtime_parameter,
            self.device,
            logger=self.child_logger,
        )

    def _should_run_services(self) -> bool:
        assert self.runtime_parameter is not None
        runtime_parameter = self.runtime_parameter
        if runtime_parameter.debug_check_config_mode:
            return runtime_parameter.current_tick % 1000 == 0
        return True

    def _run_services(self, training_loss: float, training_accuracy: float) -> None:
        run_service = self._should_run_services()
        assert self.model is not None
        assert self.runtime_parameter is not None
        assert self.child_logger is not None
        model = self.model
        runtime_parameter = self.runtime_parameter
        child_logger = self.child_logger
        current_state = model.state_dict()

        if not run_service:
            return

        assert self.weight_diff_service is not None
        assert self.weight_change_service is not None
        assert self.distance_to_origin_service is not None
        assert self.record_variance_service is not None
        assert self.record_training_loss_service is not None
        assert self.model_state_of_last_tick is not None
        assert self.end_model_stat_dict is not None

        service_timings: list[tuple[str, float]] = []
        services_start_time = 0.0
        if runtime_parameter.enable_profiler:
            self._synchronize_for_timing()
            services_start_time = time.perf_counter()

        _, elapsed = self._time_call(
            self.weight_diff_service.trigger_without_runtime_parameters,
            runtime_parameter.current_tick,
            [current_state, self.end_model_stat_dict],
        )
        service_timings.append(("weight_diff", elapsed))

        _, elapsed = self._time_call(
            self.weight_change_service.trigger_without_runtime_parameters,
            runtime_parameter.current_tick,
            [self.model_state_of_last_tick, current_state],
        )
        service_timings.append(("weight_change", elapsed))

        self.model_state_of_last_tick, elapsed = self._time_call(_clone_state_dict, current_state)
        service_timings.append(("clone_last_tick_state", elapsed))

        _, elapsed = self._time_call(
            self.distance_to_origin_service.trigger_without_runtime_parameters,
            runtime_parameter.current_tick,
            {0: current_state},
        )
        service_timings.append(("distance_to_origin", elapsed))

        _, elapsed = self._time_call(
            self.record_variance_service.trigger_without_runtime_parameters,
            runtime_parameter.current_tick,
            [0],
            [current_state],
        )
        service_timings.append(("record_variance", elapsed))

        if self.record_model_service is not None:
            should_save = False
            if runtime_parameter.save_ticks is None:
                should_save = runtime_parameter.current_tick % runtime_parameter.save_interval == 0
            else:
                should_save = runtime_parameter.current_tick in runtime_parameter.save_ticks
            if should_save:
                _, elapsed = self._time_call(
                    self.record_model_service.trigger_without_runtime_parameters,
                    runtime_parameter.current_tick,
                    [0],
                    [current_state],
                )
                service_timings.append(("record_model", elapsed))

        if self.record_test_accuracy_loss_service is not None:
            _, elapsed = self._time_call(
                self.record_test_accuracy_loss_service.trigger_without_runtime_parameters,
                runtime_parameter.current_tick,
                {0: current_state},
            )
            service_timings.append(("record_test_accuracy_loss", elapsed))

        _, elapsed = self._time_call(
            self.record_training_loss_service.trigger_without_runtime_parameters,
            runtime_parameter.current_tick,
            {0: training_loss},
            {0: training_accuracy},
        )
        service_timings.append(("record_training_loss", elapsed))

        if self.record_consecutive_points_service is not None:
            _, elapsed = self._time_call(
                self.record_consecutive_points_service.trigger_without_runtime_parameters,
                runtime_parameter.current_tick,
                SimulationPhase.END_OF_TICK,
                current_state,
            )
            service_timings.append(("record_consecutive_points", elapsed))

        if self.record_cosine_similarity_service is not None:
            _, elapsed = self._time_call(
                self.record_cosine_similarity_service.trigger_without_runtime_parameters,
                runtime_parameter.current_tick,
                {0: current_state},
            )
            service_timings.append(("record_cosine_similarity", elapsed))

        if runtime_parameter.enable_profiler:
            self._synchronize_for_timing()
            self._emit_performance_row(
                runtime_parameter.current_tick,
                "services(all)",
                service_timings,
                total=time.perf_counter() - services_start_time,
            )

    def finalize(self) -> None:
        if self.finalized:
            return

        assert self.arg_output_folder_path is not None
        assert self.runtime_parameter is not None
        assert self.model is not None
        assert self.optimizer is not None
        output_path = self.arg_output_folder_path
        runtime_parameter = self.runtime_parameter
        model = self.model
        optimizer = self.optimizer
        save_model_state(
            os.path.join(output_path, f"{runtime_parameter.current_tick}.model.pt"),
            _state_dict_to_cpu(model.state_dict()),
            _require_model_type_name(runtime_parameter.model_type),
            _require_dataset_type_name(runtime_parameter.dataset_type),
        )
        save_optimizer_state(
            os.path.join(output_path, f"{runtime_parameter.current_tick}.optimizer.pt"),
            _move_nested_tensors(optimizer.state_dict(), "cpu"),
            _require_model_type_name(runtime_parameter.model_type),
            _require_dataset_type_name(runtime_parameter.dataset_type),
        )

        self.finalized = True

    def step(self) -> bool:
        if not self.initialized:
            raise RuntimeError("Runner has not been set up")
        if self.finished:
            return False
        assert self.runtime_parameter is not None
        assert self.child_logger is not None
        assert self.model is not None
        runtime_parameter = self.runtime_parameter
        child_logger = self.child_logger
        model = self.model
        tick_index = runtime_parameter.current_tick

        if tick_index >= runtime_parameter.max_tick:
            self.finished = True
            self.finalize()
            return False

        child_logger.info("tick %s", tick_index)
        tick_timings: list[tuple[str, float]] = []
        tick_start_time = 0.0
        if runtime_parameter.enable_profiler:
            self._synchronize_for_timing()
            tick_start_time = time.perf_counter()

        _, elapsed = self._time_call(self._maybe_save_checkpoint)
        tick_timings.append(("save_checkpoint", elapsed))
        _, elapsed = self._time_call(self._update_dynamic_end_model)
        tick_timings.append(("update_dynamic_end_model", elapsed))
        _, elapsed = self._time_call(self._maybe_log_eta)
        tick_timings.append(("log_eta", elapsed))

        parameter_updated, elapsed = self._time_call(self._update_parameters_for_current_tick)
        tick_timings.append(("update_parameters", elapsed))
        _, elapsed = self._time_call(self._maybe_first_tick_setup)
        tick_timings.append(("first_tick_setup", elapsed))

        if parameter_updated:
            _, elapsed = self._time_call(self._save_on_parameter_update)
            tick_timings.append(("save_on_parameter_update", elapsed))

        if self._should_run_services() and self.record_consecutive_points_service is not None:
            _, elapsed = self._time_call(
                self.record_consecutive_points_service.trigger_without_runtime_parameters,
                runtime_parameter.current_tick,
                SimulationPhase.START_OF_TICK,
                model.state_dict(),
            )
            tick_timings.append(("record_consecutive_points_start", elapsed))

        _, elapsed = self._time_call(self._move_model)
        tick_timings.append(("move_model", elapsed))
        _, elapsed = self._time_call(self._apply_variance_correction, phase="pre-train")
        tick_timings.append(("variance_correction_pre_train", elapsed))
        _, elapsed = self._time_call(self._adjust_learning_rates)
        tick_timings.append(("adjust_learning_rates", elapsed))
        training_output, elapsed = self._time_call(self._train_current_tick)
        training_loss, training_accuracy = training_output
        tick_timings.append(("train_current_tick", elapsed))
        _, elapsed = self._time_call(self._rebuild_norm_layers)
        tick_timings.append(("rebuild_norm_layers", elapsed))
        _, elapsed = self._time_call(self._apply_variance_correction, phase="post-train")
        tick_timings.append(("variance_correction_post_train", elapsed))
        _, elapsed = self._time_call(self._run_services, training_loss, training_accuracy)
        tick_timings.append(("run_services", elapsed))

        runtime_parameter.current_tick += 1

        if (
            runtime_parameter.stop_when_training_loss_exceeds is not None
            and training_loss > runtime_parameter.stop_when_training_loss_exceeds
        ):
            child_logger.info(
                "Loss %.4f exceeded threshold %.4f; stopping",
                training_loss,
                runtime_parameter.stop_when_training_loss_exceeds,
            )
            self.finished = True

        if runtime_parameter.current_tick >= runtime_parameter.max_tick:
            self.finished = True

        if self.finished:
            _, elapsed = self._time_call(self.finalize)
            tick_timings.append(("finalize", elapsed))

        if runtime_parameter.enable_profiler:
            self._synchronize_for_timing()
            self._emit_performance_row(
                tick_index,
                "profile(total)",
                tick_timings,
                total=time.perf_counter() - tick_start_time,
            )

        return not self.finished

    def run(self) -> None:
        while self.step():
            pass


def process_file_func(
    index: int,
    runtime_parameter: RuntimeParameters,
    checkpoint_file_path: Optional[str] = None,
) -> None:
    local_runtime_parameter = copy.deepcopy(runtime_parameter)
    runner = FindHighAccuracyPathRunner()
    runner.setup(index, local_runtime_parameter, checkpoint_file_path)
    runner.run()


def _build_runtime_parameters_from_args(args) -> RuntimeParameters:
    runtime_parameter = RuntimeParameters()
    runtime_parameter.use_cpu = args.cpu
    runtime_parameter.use_amp = args.amp
    runtime_parameter.use_dali = args.dali
    runtime_parameter.dali_device_id = args.dali_device_id
    runtime_parameter.save_ticks = _parse_save_ticks(args.save_ticks)
    runtime_parameter.save_interval = args.save_interval
    runtime_parameter.save_format = args.save_format
    runtime_parameter.config_file_path = args.config
    runtime_parameter.dataset_type = RuntimeParameters.coerce_dataset_type(args.dataset)
    runtime_parameter.debug_check_config_mode = args.check_config
    runtime_parameter.verbose = args.verbose
    runtime_parameter.service_test_accuracy_loss_interval = args.test_interval
    runtime_parameter.service_test_accuracy_loss_batch_size = args.test_batch
    runtime_parameter.service_cosine_similarity_ref_model = args.cosine_similarity_ref
    runtime_parameter.store_top_accuracy_model_count = args.store_top_accuracy_model_count
    runtime_parameter.checkpoint_interval = args.checkpoint_interval
    runtime_parameter.pytorch_preset_version = args.torch_preset_version
    runtime_parameter.across_vs_lr_policy = args.across_vs_lr_policy
    runtime_parameter.silence_mode = args.silence
    runtime_parameter.linear_interpolation_points_size = args.linear_interpolation_points_size
    runtime_parameter.linear_interpolation_dataset_size = args.linear_interpolation_dataset_size
    runtime_parameter.stop_when_training_loss_exceeds = args.stop_when_loss_exceeds
    runtime_parameter.service_test_accuracy_loss_disable = args.disable_service_test
    runtime_parameter.service_cosine_similarity_disable = args.disable_service_cosine_similarity
    runtime_parameter.total_cpu_count = args.core
    runtime_parameter.worker_count = args.worker
    runtime_parameter.enable_profiler = args.profiler
    return runtime_parameter


def _configure_paths(args, runtime_parameter: RuntimeParameters) -> Optional[int]:
    if args.start_folder is None or args.end_folder is None:
        return None

    start_folder = args.start_folder
    end_folder = args.end_folder

    if runtime_parameter.across_vs_lr_policy == "std" and end_folder not in ("origin", "inf", "mean"):
        raise RuntimeError("across_vs_lr_policy='std' is only supported for origin/inf/mean runs")

    if end_folder == "origin":
        runtime_parameter.work_mode = WorkMode.to_origin
        assert args.mapping_mode == "auto", "mapping_mode must be 'auto' when moving to origin"
        model_files = sorted(name for name in os.listdir(start_folder) if name.endswith("model.pt"))
        paths_to_find: list[tuple[str, str]] = [(os.path.join(start_folder, name), "origin") for name in model_files]
    elif end_folder == "inf":
        runtime_parameter.work_mode = WorkMode.to_inf
        assert args.mapping_mode == "auto", "mapping_mode must be 'auto' when moving to inf"
        model_files = sorted(name for name in os.listdir(start_folder) if name.endswith("model.pt"))
        paths_to_find: list[tuple[str, str]] = [(os.path.join(start_folder, name), "inf") for name in model_files]
    elif end_folder == "mean":
        runtime_parameter.work_mode = WorkMode.to_mean
        assert args.mapping_mode == "auto", "mapping_mode must be 'auto' when moving to mean"
        model_files = sorted(name for name in os.listdir(start_folder) if name.endswith("model.pt"))
        paths_to_find: list[tuple[str, str]] = [(os.path.join(start_folder, name), "mean") for name in model_files]
    elif end_folder == "to_vs":
        runtime_parameter.work_mode = WorkMode.to_vs
        assert args.variance_sphere is not None, "--variance_sphere is required for 'to_vs'"
        runtime_parameter.variance_sphere_file_path = args.variance_sphere
        model_files = sorted(name for name in os.listdir(start_folder) if name.endswith("model.pt"))
        paths_to_find: list[tuple[str, str]] = [(os.path.join(start_folder, name), "to_vs") for name in model_files]
    else:
        runtime_parameter.work_mode = WorkMode.to_certain_model
        paths_to_find: list[tuple[str, str]] = get_files_to_process(start_folder, end_folder, args.mapping_mode)

    runtime_parameter.start_and_end_point_for_paths = paths_to_find
    logger.info("Paths to process (%s): %s", len(paths_to_find), paths_to_find)
    return len(paths_to_find)


def _create_output_folder(args, runtime_parameter: RuntimeParameters) -> str:
    if args.output_folder_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder = os.path.join(os.curdir, f"find_high_accuracy_path_{timestamp}")
    else:
        output_folder = os.path.join(os.curdir, args.output_folder_name)

    os.makedirs(output_folder)
    runtime_parameter.output_folder_path = output_folder

    with open(os.path.join(output_folder, "arguments.txt"), "x", encoding="utf-8") as output_file:
        output_file.write(str(args))

    shutil.copyfile(args.config, os.path.join(output_folder, os.path.basename(args.config)))
    shutil.copyfile(__file__, os.path.join(output_folder, os.path.basename(__file__)))
    return output_folder


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Move model weights toward a destination while maintaining accuracy",
    )
    parser.add_argument("start_folder", nargs="?", help="Folder with starting model(s)")
    parser.add_argument(
        "end_folder",
        nargs="?",
        help="Folder with destination model(s), or 'origin'/'inf'/'mean'/'to_vs'",
    )

    parser.add_argument("--variance_sphere", help="Variance-sphere model path for 'to_vs'")
    parser.add_argument("--config", default="find_high_accuracy_path_config.py", help="Config file")
    parser.add_argument(
        "--mapping_mode",
        default="auto",
        choices=["auto", "all_to_all", "each_to_each", "one_to_all", "all_to_one"],
    )

    parser.add_argument("-c", "--core", type=int, default=os.cpu_count(), help="CPU cores to use")
    parser.add_argument("-w", "--worker", type=int, default=1, help="Parallel workers")
    parser.add_argument("-d", "--dataset", default=None, help="Dataset override")

    parser.add_argument("--save_ticks", help="Ticks to save models, e.g. '1,2,3,5-10'")
    parser.add_argument("--save_interval", type=int, default=1, help="Model save interval")
    parser.add_argument("--save_format", default="none", choices=["none", "file", "lmdb"])
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--dali", action=argparse.BooleanOptionalAction, default=False, help="Use NVIDIA DALI for ImageNet dataloading")
    parser.add_argument("--dali_device_id", type=int, default=0, help="CUDA device id used by DALI pipelines")
    parser.add_argument("--check_config", action="store_true", help="Check config only")
    parser.add_argument("--cosine_similarity_ref", help="Reference model for cosine similarity")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output_folder_name", default=None)
    parser.add_argument("--store_top_accuracy_model_count", type=int, default=5)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--continue_from_checkpoint", help="Resume from checkpoint file")
    parser.add_argument("-A", "--across_vs_lr_policy", choices=["std", "var"], default="var")
    parser.add_argument("--stop_when_loss_exceeds", type=float)
    parser.add_argument("--test_interval", type=int, default=1)
    parser.add_argument("--test_batch", type=int, default=100)
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None)
    parser.add_argument("--silence", type=bool, default=True, help="disable the interactive check, default is True")
    parser.add_argument("--linear_interpolation_points_size", type=int, default=0)
    parser.add_argument("--linear_interpolation_dataset_size", type=int, default=1000)
    parser.add_argument("--disable_service_test", action="store_true")
    parser.add_argument("--disable_service_cosine_similarity", action="store_true")
    parser.add_argument("--profiler", action=argparse.BooleanOptionalAction, default=False, help="Enable per-tick and per-service timing logs")

    args = parser.parse_args(argv)
    if args.cpu and args.dali:
        parser.error("--dali requires CUDA; do not combine it with --cpu")

    setup_logging(logger, "main", exit_on_critical=True)
    logger.info("Logging initialized")

    runtime_parameter = _build_runtime_parameters_from_args(args)

    path_count = _configure_paths(args, runtime_parameter)
    if path_count is None and not args.continue_from_checkpoint:
        logger.critical("Provide start/end folders or --continue_from_checkpoint")
        sys.exit(1)

    _create_output_folder(args, runtime_parameter)
    logger.info("Runtime parameters:\n%s", runtime_parameter.print())

    if not runtime_parameter.use_cpu:
        assert torch.cuda.is_available(), "CUDA is not available; use --cpu to force CPU"
        for index in range(torch.cuda.device_count()):
            logger.info("GPU %s: %s", index, torch.cuda.get_device_name(index))

    if args.continue_from_checkpoint:
        process_file_func(0, runtime_parameter, args.continue_from_checkpoint)
        return

    assert path_count is not None
    worker_count = min(runtime_parameter.worker_count, path_count)
    runtime_parameter.worker_count = worker_count
    logger.info("Using %s worker(s)", worker_count)

    if worker_count == 1:
        for index in range(path_count):
            process_file_func(index, runtime_parameter, None)
    else:
        assert runtime_parameter.silence_mode, "silence_mode must be enabled for multi-worker runs"
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(process_file_func, index, runtime_parameter, None)
                for index in range(path_count)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
