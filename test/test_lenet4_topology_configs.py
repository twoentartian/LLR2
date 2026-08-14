from __future__ import annotations

import unittest
from pathlib import Path

import networkx as nx

import simulator
from py_src.model_average import (
    StandardModelAverager,
    VarianceCorrectionType,
)
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


CONFIG_CASES = {
    "RING": (
        50,
        "simulator_config_lenet4_ring.py",
        "simulator_config_lenet4_ring_vc.py",
    ),
    "SW": (
        50,
        "simulator_config_lenet4_sw.py",
        "simulator_config_lenet4_sw_vc.py",
    ),
    "2COMM": (
        50,
        "simulator_config_lenet4_2comm.py",
        "simulator_config_lenet4_2comm_vc.py",
    ),
    "SF": (
        50,
        "simulator_config_lenet4_sf.py",
        "simulator_config_lenet4_sf_vc.py",
    ),
    "STAR": (
        50,
        "simulator_config_lenet4_star.py",
        "simulator_config_lenet4_star_vc.py",
    ),
    "DROP": (
        50,
        "simulator_config_lenet4_drop.py",
        "simulator_config_lenet4_drop_vc.py",
    ),
    "HIER_N50": (
        50,
        "simulator_config_lenet4_hier_n50.py",
        "simulator_config_lenet4_hier_n50_vc.py",
    ),
    "HIER_N100": (
        100,
        "simulator_config_lenet4_hier_n100.py",
        "simulator_config_lenet4_hier_n100_vc.py",
    ),
}


class LeNet4TopologyConfigTest(unittest.TestCase):
    @staticmethod
    def _load_config(config_filename):
        config_path = Path(__file__).resolve().parents[1] / "config" / config_filename
        return simulator.load_configuration(str(config_path))

    @staticmethod
    def _initial_topology(config_module):
        parameters = RuntimeParameters()
        parameters.phase = SimulationPhase.INITIALIZING
        parameters.current_tick = 0
        topology = config_module.get_topology(parameters)
        if topology is None:
            raise AssertionError("initial topology must not be None")
        return topology

    def test_all_sixteen_configs_have_paired_topologies_and_vc(self):
        for topology_name, case in CONFIG_CASES.items():
            node_count, no_vc_filename, vc_filename = case
            with self.subTest(topology=topology_name):
                no_vc_config = self._load_config(no_vc_filename)
                vc_config = self._load_config(vc_filename)
                no_vc_graph = self._initial_topology(no_vc_config)
                vc_graph = self._initial_topology(vc_config)

                self.assertEqual(no_vc_graph.number_of_nodes(), node_count)
                self.assertEqual(set(no_vc_graph.edges), set(vc_graph.edges))
                self.assertIsNone(no_vc_config.preset_variance_correction)
                self.assertEqual(vc_config.preset_variance_correction, "VC")

                parameters = RuntimeParameters()
                no_vc_averager = no_vc_config.get_average_algorithm(
                    None,
                    parameters,
                )
                vc_averager = vc_config.get_average_algorithm(None, parameters)
                self.assertIsInstance(no_vc_averager, StandardModelAverager)
                self.assertIsNone(no_vc_averager.variance_corrector)
                self.assertIsInstance(vc_averager, StandardModelAverager)
                self.assertIsNotNone(vc_averager.variance_corrector)
                self.assertEqual(
                    vc_averager.variance_corrector.variance_correction_type,
                    VarianceCorrectionType.FollowOthers,
                )

    def test_topology_constructions_match_the_experiment_design(self):
        graphs = {
            topology_name: self._initial_topology(
                self._load_config(case[1])
            )
            for topology_name, case in CONFIG_CASES.items()
        }

        self.assertTrue(all(degree == 2 for _, degree in graphs["RING"].degree))
        self.assertTrue(nx.is_connected(graphs["SW"]))
        self.assertTrue(
            all(degree == 8 for _, degree in graphs["2COMM"].degree)
        )
        cross_edges = [
            (source, target)
            for source, target in graphs["2COMM"].edges
            if graphs["2COMM"].nodes[source]["community"]
            != graphs["2COMM"].nodes[target]["community"]
        ]
        self.assertEqual(len(cross_edges), 20)
        self.assertEqual(graphs["SF"].number_of_edges(), 184)
        self.assertEqual(graphs["STAR"].number_of_edges(), 49)
        self.assertEqual(max(dict(graphs["STAR"].degree).values()), 49)
        self.assertLessEqual(graphs["DROP"].number_of_edges(), 200)

        hierarchy_expectations = {
            "HIER_N50": {
                "core": 2,
                "distribution": 6,
                "access": 6,
                "leaf": 36,
            },
            "HIER_N100": {
                "core": 4,
                "distribution": 12,
                "access": 12,
                "leaf": 72,
            },
        }
        for hierarchy_name, expected_roles in hierarchy_expectations.items():
            hierarchy_graph = graphs[hierarchy_name]
            self.assertTrue(nx.is_connected(hierarchy_graph))
            role_nodes = {}
            for node_name, node_data in hierarchy_graph.nodes(data=True):
                role = node_data["role"]
                role_nodes.setdefault(role, []).append(node_name)
            self.assertEqual(
                {role: len(nodes) for role, nodes in role_nodes.items()},
                expected_roles,
            )

            core_count = expected_roles["core"]
            distribution_count = expected_roles["distribution"]
            access_count = expected_roles["access"]
            leaf_count = expected_roles["leaf"]
            expected_edge_count = (
                core_count * distribution_count
                + distribution_count * access_count
                + leaf_count
            )
            self.assertEqual(
                hierarchy_graph.number_of_edges(),
                expected_edge_count,
            )
            for leaf_node in role_nodes["leaf"]:
                self.assertEqual(hierarchy_graph.degree[leaf_node], 1)
                parent_node = next(iter(hierarchy_graph.neighbors(leaf_node)))
                self.assertEqual(
                    hierarchy_graph.nodes[parent_node]["role"],
                    "access",
                )

    def test_hierarchy_configs_expose_layer_sizes(self):
        hierarchy_50 = self._load_config(CONFIG_CASES["HIER_N50"][1])
        hierarchy_100 = self._load_config(CONFIG_CASES["HIER_N100"][1])

        self.assertEqual(
            (
                hierarchy_50.preset_hierarchy_core_count,
                hierarchy_50.preset_hierarchy_distribution_count,
                hierarchy_50.preset_hierarchy_access_count,
                hierarchy_50.preset_hierarchy_leaf_count,
                hierarchy_50.preset_hierarchy_leaves_per_access,
            ),
            (2, 6, 6, 36, 6),
        )
        self.assertEqual(
            (
                hierarchy_100.preset_hierarchy_core_count,
                hierarchy_100.preset_hierarchy_distribution_count,
                hierarchy_100.preset_hierarchy_access_count,
                hierarchy_100.preset_hierarchy_leaf_count,
                hierarchy_100.preset_hierarchy_leaves_per_access,
            ),
            (4, 12, 12, 72, 6),
        )

    def test_drop_changes_reproducibly_with_tick(self):
        drop_config = self._load_config(CONFIG_CASES["DROP"][1])
        parameters = RuntimeParameters()
        parameters.phase = SimulationPhase.START_OF_TICK

        parameters.current_tick = 7
        graph_at_seven = drop_config.get_topology(parameters)
        repeated_graph_at_seven = drop_config.get_topology(parameters)
        parameters.current_tick = 8
        graph_at_eight = drop_config.get_topology(parameters)

        self.assertEqual(
            set(graph_at_seven.edges),
            set(repeated_graph_at_seven.edges),
        )
        self.assertNotEqual(
            set(graph_at_seven.edges),
            set(graph_at_eight.edges),
        )
        self.assertEqual(set(graph_at_seven.nodes), set(graph_at_eight.nodes))

    def test_drop_refreshes_each_node_average_buffer_size(self):
        drop_config = self._load_config(CONFIG_CASES["DROP"][1])
        parameters = RuntimeParameters()
        parameters.phase = SimulationPhase.START_OF_TICK
        parameters.current_tick = 12
        parameters.topology = drop_config.get_topology(parameters)

        class DummyNode:
            def __init__(self, node_name):
                self.name = node_name
                self.buffer_size = None

            def set_average_buffer_size(self, buffer_size):
                self.buffer_size = buffer_size

        parameters.node_container = {
            node_name: DummyNode(node_name)
            for node_name in parameters.topology.nodes
        }
        drop_config.node_behavior_control(parameters)

        for node_name, target_node in parameters.node_container.items():
            self.assertEqual(
                target_node.buffer_size,
                parameters.topology.degree[node_name],
            )


if __name__ == "__main__":
    unittest.main()
