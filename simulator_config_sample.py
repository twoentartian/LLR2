from __future__ import annotations

import networkx as nx
import random
import torch
import math

from py_src import ml_setup, node, model_average, nx_lib
from py_src.ml_setup import MLSetup
from py_src.service.record_weights_difference import ServiceWeightsDifferenceRecorder
from py_src.simulation_runtime_parameters import SimulationPhase, RuntimeParameters
from py_src.config_file_util import label_distribution
from py_src.service.record_variance import ServiceVarianceRecorder
from py_src.service.record_training_loss_accuracy import ServiceTrainingLossAccuracyRecorder
from py_src.service.record_test_accuracy_loss import ServiceTestAccuracyLossRecorder
from py_src.model_variance_correct import VarianceCorrector, VarianceCorrectionType

config_name = "default_config"

max_tick = 1000000        # total simulation ticks

force_use_cpu = False

"""do you want to put all models in GPU or only keep model stat in memory and share a model in gpu?"""
"""None | False: let simulator decide"""
override_use_model_stat = None
override_allocate_all_models = None

""""""""" Global Machine learning related parameters """""""""""
""" predefined: """
def get_ml_setup():
    if not hasattr(get_ml_setup, "__ml_setup"):
        # get_ml_setup.__ml_setup = ml_setup.resnet18_cifar10()
        get_ml_setup.__ml_setup = ml_setup.lenet4_mnist()
        # get_ml_setup.__ml_setup = ml_setup.cct_7_3x1_cifar10()
    return get_ml_setup.__ml_setup

""" or self defined: """
# ml_parameters = ml_setup.MLSetup()
# ml_parameters.model = None
# ml_parameters.training_data = None
# ml_parameters.testing_data = None


def get_average_algorithm(target_node: node.Node, parameters: RuntimeParameters):
    # variance_correction = VarianceCorrector(VarianceCorrectionType.FollowOthers)
    # return model_average.StandardModelAverager(variance_corrector=variance_correction)
    return model_average.StandardModelAverager()
    # return model_average.ConservativeModelAverager(0.5)

def get_average_buffer_size(target_node: node.Node, parameters: RuntimeParameters):
    neighbors = list(parameters.topology.neighbors(target_node.name))
    R = 1
    return R * len(neighbors)


""""""""""" Global Topology related parameters """""""""""
"""topology will be updated every tick if not None"""
def get_topology(parameters: RuntimeParameters) -> nx.Graph:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # get_topology.current_topology = nx.complete_graph(50)
        G: nx.Graph = nx.random_regular_graph(3, 10)
        # G = nx.Graph()
        # G.add_node(0)
        get_topology.current = G
        # get_topology.current = nx_lib.load_topology_from_edge_list_file(f"topology/ca-netscience.data")
        # G = nx.Graph()
        # # # G.add_edge(0,1)
        # G.add_node(0)
        # get_topology.current = G
        return get_topology.current

    return None

""""""""""" Node related parameters """""""""""
"""parameters here will only be initialized once at beginning"""
def node_behavior_control(parameters: RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:
        for node_name, node_target in parameters.node_container.items():
            node_target.send_model_after_P_training = 1


""""""""""" Training time related parameters """""""""""
"""this function will be called after training to get the next training tick"""
def get_next_training_time(target_node: node.Node, parameters: RuntimeParameters) -> int:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        return 0
    return target_node.next_training_tick + random.randint(8, 12)


""""""""""" Dataset related parameters """""""""""
"""label distribution will be updated every tick if not None"""
def get_label_distribution(target_node: node.Node, parameters: RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # return label_distribution.label_distribution_non_iid_dirichlet(target_node, parameters, 0.5)
        get_label_distribution.current = label_distribution.label_distribution_iid(target_node, parameters)
        return get_label_distribution.current

        # if target_node.name == 0:
        #     return label_distribution.label_distribution_first_half(target_node, parameters)
        # elif target_node.name == 1:
        #     return label_distribution.label_distribution_second_half(target_node, parameters)
        # else:
        #     raise NotImplementedError

    return None



""""""""""" Dataset related parameters """""""""""
def get_optimizer(target_node: node.Node, model: torch.nn.Module, parameters: RuntimeParameters, ml_setup: MLSetup):
    """warning: you are not allowed to change optimizer during simulation when use_model_stat == True"""
    assert model is not None
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0)
        lr_scheduler = None
        return optimizer, lr_scheduler
    return None, None


""""""""""" Service related parameters """""""""""
def get_service_list():
    setup = get_ml_setup()
    service_list = []

    service_list.append(ServiceVarianceRecorder(1000))
    service_list.append(ServiceTrainingLossAccuracyRecorder(1000))
    service_list.append(ServiceTestAccuracyLossRecorder(1000,100,setup.model_type.name,setup.dataset_type.name))
    service_list.append(ServiceWeightsDifferenceRecorder(1000))

    return service_list
