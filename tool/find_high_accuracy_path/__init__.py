"""Support modules and lazy exports for the find_high_accuracy_path tool."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

from .find_parameters import ParameterGeneral, ParameterMove, ParameterRebuildNorm, ParameterTrain
from .runtime_parameters import Checkpoint, RuntimeParameters, WorkMode

_IMPL_MODULE_NAME = "tool._find_high_accuracy_path_impl"
_IMPL_EXPORTS = {"FindHighAccuracyPathRunner", "process_file_func", "main"}


def _load_impl_module():
    module = sys.modules.get(_IMPL_MODULE_NAME)
    if module is not None:
        return module

    module_path = pathlib.Path(__file__).resolve().parent.parent / "find_high_accuracy_path.py"
    spec = importlib.util.spec_from_file_location(_IMPL_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load implementation module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_IMPL_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def __getattr__(name: str):
    if name in _IMPL_EXPORTS:
        return getattr(_load_impl_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Checkpoint",
    "RuntimeParameters",
    "WorkMode",
    "ParameterGeneral",
    "ParameterMove",
    "ParameterTrain",
    "ParameterRebuildNorm",
    "FindHighAccuracyPathRunner",
    "process_file_func",
    "main",
]
