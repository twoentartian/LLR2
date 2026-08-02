from __future__ import annotations

import random

import networkx as nx
import torch

from py_src import ml_setup, model_average, node
from py_src.config_file_util import label_distribution
from py_src.ml_setup import MLSetup
from py_src.model_variance_correct import VarianceCorrectionType, VarianceCorrector
from py_src.service.record_test_accuracy_loss import ServiceTestAccuracyLossRecorder
from py_src.service.record_training_loss_accuracy import (
    ServiceTrainingLossAccuracyRecorder,
)
from py_src.service.record_variance import ServiceVarianceRecorder
from py_src.service.record_weights_difference import ServiceWeightsDifferenceRecorder
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


config_name = "default_config"

max_tick = 10000
# save_name = "LeNet4__GL_N50"
force_use_cpu = False

# Retained for compatibility with DFL_torch configs. LLR2 selects its execution
# strategy through the simulator's device option.
override_use_model_stat = None
override_allocate_all_models = None


# Network presets: "GL", "FL", or "single".
preset_network = "GL"
preset_variance_correction = None  # None or "VC"
preset_network_size = 50
preset_network_degree = 8  # Only valid for GL.

# One worker pool is shared by all node training batches.  This is a total
# process count, not a per-node count.
preset_shared_training_loader_workers = 4
preset_shared_training_loader_prefetch_factor = 2


_ml_setup: MLSetup | None = None


def get_ml_setup() -> MLSetup:
    global _ml_setup
    if _ml_setup is None:
        # _ml_setup = ml_setup.resnet18_cifar10()
        _ml_setup = ml_setup.lenet4_mnist()
        # _ml_setup = ml_setup.lenet5_mnist()
        # _ml_setup = ml_setup.cct_7_3x1_cifar10()
        # _ml_setup = ml_setup.simplenet_cifar10()
    return _ml_setup


def get_optimizer(
    target_node: node.Node,
    model: torch.nn.Module,
    parameters: RuntimeParameters,
    setup: MLSetup,
):
    """Return the optimizer and optional scheduler used by one node."""
    del target_node, setup
    assert model is not None
    if parameters.phase == SimulationPhase.INITIALIZING:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )
        return optimizer, None
    return None, None


def get_average_algorithm(
    target_node: node.Node,
    parameters: RuntimeParameters,
) -> model_average.ModelAverager:
    del target_node, parameters
    if preset_network == "FL":
        return model_average.StandardModelAverager()

    if preset_variance_correction == "VC":
        variance_corrector = VarianceCorrector(VarianceCorrectionType.FollowOthers)
        return model_average.ConservativeModelAverager(
            0.5,
            variance_corrector=variance_corrector,
        )

    return model_average.ConservativeModelAverager(0.5)


def get_average_buffer_size(
    target_node: node.Node,
    parameters: RuntimeParameters,
) -> int:
    return len(list(parameters.topology.neighbors(target_node.name)))


_current_topology: nx.Graph | None = None


def get_topology(parameters: RuntimeParameters) -> nx.Graph | None:
    """Create the selected topology during simulator initialization."""
    global _current_topology
    if parameters.phase != SimulationPhase.INITIALIZING:
        return None

    if preset_network == "FL":
        _current_topology = nx.star_graph(preset_network_size)
    elif preset_network == "GL":
        _current_topology = nx.random_regular_graph(
            preset_network_degree,
            preset_network_size,
        )
    elif preset_network == "single":
        _current_topology = nx.Graph()
        _current_topology.add_node(0)
    else:
        raise ValueError(f"unsupported network preset: {preset_network!r}")

    return _current_topology


def node_behavior_control(parameters: RuntimeParameters) -> None:
    """Apply initial per-node training and communication behavior."""
    if (
        parameters.phase != SimulationPhase.INITIALIZING
        or parameters.current_tick != 0
    ):
        return

    for node_name, target_node in parameters.node_container.items():
        target_node.send_model_after_P_training = 1
        if preset_network == "FL" and node_name == 0:
            target_node.enable_training = False


def get_next_training_time(
    target_node: node.Node,
    parameters: RuntimeParameters,
) -> int:
    if parameters.phase == SimulationPhase.INITIALIZING:
        if preset_network == "FL" and target_node.name != 0:
            return 5
        return 0

    if preset_network == "GL":
        return target_node.next_training_tick + random.randint(8, 12)
    if preset_network in {"FL", "single"}:
        return target_node.next_training_tick + 10
    raise ValueError(f"unsupported network preset: {preset_network!r}")


def get_label_distribution(
    target_node: node.Node,
    parameters: RuntimeParameters,
):
    """Use IID label sampling when each node is initialized."""
    if parameters.phase == SimulationPhase.INITIALIZING:
        return label_distribution.label_distribution_iid(target_node, parameters)
    return None


def get_service_list():
    setup = get_ml_setup()
    return [
        ServiceVarianceRecorder(100,phase=[SimulationPhase.AFTER_AVERAGING],),
        ServiceTrainingLossAccuracyRecorder(100),
        ServiceTestAccuracyLossRecorder(100,100,setup.model_type.name,setup.dataset_type.name,),
        # ServiceWeightsDifferenceRecorder(20),
    ]


performance_disable_training = False
performance_disable_communication = False
