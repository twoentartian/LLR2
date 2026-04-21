"""Runtime parameters and checkpoint dataclasses for find_high_accuracy_path."""

from enum import Enum, auto
from typing import Iterable, Optional, Set

from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType

from .find_parameters import ParameterGeneral, ParameterMove, ParameterRebuildNorm, ParameterTrain

class WorkMode(Enum):
    unknown = auto()
    to_origin = auto()
    to_inf = auto()
    to_certain_model = auto()
    to_mean = auto()
    to_vs = auto()


class RuntimeParameters:
    """Holds all mutable runtime state for one path-finding run."""

    # Static configuration (set once before run)
    start_and_end_point_for_paths: list[tuple[str, str]] = None # type: ignore
    use_cpu: bool = None # type: ignore
    use_amp: bool = None # type: ignore
    use_dali: bool = False
    dali_device_id: int = 0
    work_mode: WorkMode = WorkMode.unknown
    output_folder_path: str = None # type: ignore
    total_cpu_count: int = None # type: ignore
    worker_count: int = None # type: ignore

    save_ticks: Optional[Set[int]] = None # type: ignore
    save_interval: int = None # type: ignore
    save_format: str = None # type: ignore

    config_file_path: str = None # type: ignore
    dataset_type: Optional[DatasetType] = None # type: ignore
    model_type: Optional[ModelType] = None # type: ignore
    pytorch_preset_version: int = None # type: ignore
    store_top_accuracy_model_count: int = None # type: ignore
    checkpoint_interval: int = None # type: ignore
    task_name: str = None # type: ignore
    silence_mode: bool = None # type: ignore
    across_vs_lr_policy: str = None # type: ignore
    linear_interpolation_points_size: int = None # type: ignore
    linear_interpolation_dataset_size: int = None # type: ignore
    variance_sphere_file_path: str = None # type: ignore
    variance_sphere_model = None
    stop_when_training_loss_exceeds: float = None # type: ignore

    # Real-time values
    current_tick: int = None # type: ignore
    max_tick: int = None # type: ignore
    debug_check_config_mode: bool = None # type: ignore
    test_dataset_use_whole: bool = None # type: ignore
    verbose: bool = False

    # Service configuration
    service_test_accuracy_loss_disable: bool = None # type: ignore
    service_test_accuracy_loss_interval: int = None # type: ignore
    service_test_accuracy_loss_batch_size: int = None # type: ignore
    service_cosine_similarity_disable: bool = None # type: ignore
    service_cosine_similarity_ref_model: str = None # type: ignore
    enable_profiler: bool = False

    @staticmethod
    def coerce_model_type(value: Optional[ModelType | str]) -> Optional[ModelType]:
        if value is None:
            return None
        if isinstance(value, ModelType):
            return value
        try:
            return ModelType[value]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model type {value!r}. Valid options: {[e.name for e in ModelType]}"
            ) from exc

    @staticmethod
    def coerce_dataset_type(value: Optional[DatasetType | str]) -> Optional[DatasetType]:
        if value is None:
            return None
        if isinstance(value, DatasetType):
            return value
        try:
            return DatasetType[value]
        except KeyError as exc:
            raise ValueError(
                f"Unknown dataset type {value!r}. Valid options: {[e.name for e in DatasetType]}"
            ) from exc

    def print(self) -> str:
        lines = []
        for attr in sorted(dir(self)):
            if attr.startswith("__") or callable(getattr(self, attr)):
                continue
            value = getattr(self, attr)
            if isinstance(value, (ModelType, DatasetType)):
                value = value.name
            lines.append(f"runtime_parameters.{attr} = {value!r}")
        return "\n".join(lines)


class Checkpoint:
    current_model_stat: Optional[dict] = None
    current_optimizer_stat: Optional[dict] = None
    start_model_stat: Optional[dict] = None
    end_model_stat: Optional[dict] = None
    init_model_stat: Optional[dict] = None
    current_phase_start_model_stat: Optional[dict] = None
    checkpoint_config_path: Optional[str] = None
    checkpoint_config_sha256: Optional[str] = None

    current_runtime_parameter: Optional[RuntimeParameters] = None
    current_general_parameter: Optional[ParameterGeneral] = None
    current_move_parameter: Optional[ParameterMove] = None
    current_train_parameter: Optional[ParameterTrain] = None
    current_rebuild_norm_parameter: Optional[ParameterRebuildNorm] = None
