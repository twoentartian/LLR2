from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import torch

import simulator
import py_src.node as node_module
from py_src.config_file_util import label_distribution
from py_src.ml_setup import MLSetup
from py_src.model_average import (
    ConservativeModelAverager,
    ModelAverager,
    StandardModelAverager,
)
from py_src.ml_setup_dataset.dataset_intermediate_layer import (
    DatasetWithFastLabelSelection,
    LabelProbabilitySampler,
)
from py_src.node import Node
from py_src.node_behavior_control_lib import global_broadcast
from py_src.nx_lib import (
    get_inter_community_edges,
    load_topology_from_edge_list_file,
    split_to_equal_size_communities,
)
from py_src.simulation_runtime_parameters import RuntimeParameters


class _DummyNode:
    def __init__(self, model_stat):
        self.model_stat = model_stat

    def get_model_stat(self):
        return self.model_stat.copy()

    def set_model_stat(self, model_stat):
        self.model_stat = model_stat.copy()


class SimulatorConfigSupportTest(unittest.TestCase):
    def test_ml_setup_declares_dataset_labels(self):
        setup = MLSetup()

        self.assertEqual(setup.dataset_label, [])

    def test_runtime_parameters_declare_simulator_state(self):
        parameters = RuntimeParameters()

        self.assertEqual(parameters.max_tick, 0)
        self.assertEqual(parameters.current_tick, 0)
        self.assertEqual(parameters.node_container, {})
        self.assertEqual(parameters.dataset_label, [])
        self.assertEqual(parameters.service_container, {})
        self.assertFalse(parameters.mpi_enabled)
        self.assertEqual(parameters.output_path, "")
        self.assertTrue(parameters.average_on_cpu)
        self.assertFalse(parameters.performance_disable_training)
        self.assertFalse(parameters.performance_disable_communication)

    def test_simulator_classes_are_defined_in_py_src_modules(self):
        self.assertEqual(ModelAverager.__module__, "py_src.model_average")
        self.assertEqual(StandardModelAverager.__module__, "py_src.model_average")
        self.assertEqual(ConservativeModelAverager.__module__, "py_src.model_average")
        self.assertEqual(Node.__module__, "py_src.node")
        self.assertIs(simulator.Node, Node)

    def test_dataset_intermediate_classes_are_defined_in_dataset_layer(self):
        self.assertEqual(
            DatasetWithFastLabelSelection.__module__,
            "py_src.ml_setup_dataset.dataset_intermediate_layer",
        )
        self.assertEqual(
            LabelProbabilitySampler.__module__,
            "py_src.ml_setup_dataset.dataset_intermediate_layer",
        )
        self.assertIs(
            simulator.DatasetWithFastLabelSelection,
            DatasetWithFastLabelSelection,
        )
        self.assertIs(
            node_module.DatasetWithFastLabelSelection,
            DatasetWithFastLabelSelection,
        )

    def test_sample_config_loads_with_module_symbols_available(self):
        config_path = Path(__file__).resolve().parents[1] / "simulator_config_sample.py"
        config = simulator.load_configuration(str(config_path))
        parameters = RuntimeParameters()
        parameters.phase = simulator.SimulationPhase.INITIALIZING
        parameters.topology = nx.complete_graph(2)

        class Target:
            name = 0

        averager = config.get_average_algorithm(Target(), parameters)
        self.assertIsInstance(averager, ConservativeModelAverager)

    def test_model_averagers_preserve_behavior(self):
        self_model = {"weight": torch.tensor([2.0])}
        received_models = [
            {"weight": torch.tensor([4.0])},
            {"weight": torch.tensor([6.0])},
        ]

        standard = StandardModelAverager()
        conservative = ConservativeModelAverager(0.5)
        for model in received_models:
            standard.add_model(model)
            conservative.add_model(model)

        self.assertEqual(standard.get_model(self_model=self_model)["weight"].item(), 5.0)
        self.assertEqual(
            conservative.get_model(self_model=self_model)["weight"].item(),
            3.5,
        )

    def test_common_label_distributions(self):
        parameters = RuntimeParameters()
        parameters.dataset_label = list(range(5))

        self.assertIsNone(label_distribution.label_distribution_default(None, parameters))
        np.testing.assert_array_equal(
            label_distribution.label_distribution_iid(None, parameters),
            np.ones(5),
        )
        np.testing.assert_array_equal(
            label_distribution.label_distribution_first_half(None, parameters),
            [1, 1, 0, 0, 0],
        )
        np.testing.assert_array_equal(
            label_distribution.label_distribution_second_half(None, parameters),
            [0, 0, 1, 1, 1],
        )

        dirichlet = label_distribution.label_distribution_non_iid_dirichlet(None, parameters, 0.5)
        self.assertEqual(dirichlet.shape, (5,))
        self.assertTrue(np.isclose(dirichlet.sum(), 1.0))

    def test_global_broadcast_copies_source_model_to_all_nodes(self):
        parameters = RuntimeParameters()
        parameters.mpi_enabled = False
        parameters.node_container = {
            0: _DummyNode({"weight": 1}),
            1: _DummyNode({"weight": 2}),
            2: _DummyNode({"weight": 3}),
        }

        global_broadcast(parameters, 1)

        self.assertEqual(
            [node.model_stat for node in parameters.node_container.values()],
            [{"weight": 2}, {"weight": 2}, {"weight": 2}],
        )

    def test_networkx_config_helpers(self):
        topology = nx.cycle_graph(8)
        communities = split_to_equal_size_communities(topology, 3)

        self.assertEqual(
            sorted(node for community in communities for node in community),
            list(range(8)),
        )
        self.assertLessEqual(max(map(len, communities)) - min(map(len, communities)), 1)
        self.assertTrue(
            all(
                source in topology and destination in topology
                for source, destination in get_inter_community_edges(topology, communities)
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            edge_list = Path(temp_dir) / "topology.data"
            edge_list.write_text("0 1\n1 2\n", encoding="utf-8")
            loaded = load_topology_from_edge_list_file(str(edge_list))
        self.assertEqual(set(loaded.edges()), {(0, 1), (1, 2)})


if __name__ == "__main__":
    unittest.main()
