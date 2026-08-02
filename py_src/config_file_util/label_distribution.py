"""Common label-distribution policies for simulator configurations.

The returned arrays are weights.  ``Node.set_label_distribution`` normalizes
them before constructing a training sampler.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from py_src.simulation_runtime_parameters import RuntimeParameters


def label_distribution_default(target_node: Any, parameters: RuntimeParameters):
    """Use the ML setup's default training data loader."""
    del target_node, parameters
    return None


def label_distribution_iid(target_node: Any, parameters: RuntimeParameters) -> np.ndarray:
    """Assign equal sampling weight to every dataset label."""
    del target_node
    return np.ones(len(parameters.dataset_label), dtype=np.float64)


def label_distribution_non_iid_dirichlet(
    target_node: Any,
    parameters: RuntimeParameters,
    alpha: float,
) -> np.ndarray:
    """Draw per-label weights from a symmetric Dirichlet distribution."""
    del target_node
    return np.random.dirichlet(np.full(len(parameters.dataset_label), alpha, dtype=np.float64))


def label_distribution_first_half(target_node: Any, parameters: RuntimeParameters) -> np.ndarray:
    """Sample only labels in the first half of the label list."""
    del target_node
    size = len(parameters.dataset_label) // 2
    return np.concatenate(
        [np.ones(size, dtype=np.float64), np.zeros(len(parameters.dataset_label) - size, dtype=np.float64)]
    )


def label_distribution_second_half(target_node: Any, parameters: RuntimeParameters) -> np.ndarray:
    """Sample only labels in the second half of the label list."""
    del target_node
    size = len(parameters.dataset_label) // 2
    return np.concatenate(
        [np.zeros(size, dtype=np.float64), np.ones(len(parameters.dataset_label) - size, dtype=np.float64)]
    )
