"""Record per-layer weight variance for each node.

Ported from DFL_torch/py_src/service/record_variance.py.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch

from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


class ServiceVarianceRecorder(Service):
    """Record weight variance per layer for a set of nodes, writing one CSV
    per node under an output ``variance/`` subdirectory.
    """

    def __init__(
        self,
        interval: int,
        phase: Optional[list] = None,
        record_node=None,
        layer_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.interval = interval
        self.record_phase = phase or [SimulationPhase.END_OF_TICK]
        self.record_node = record_node
        self.layer_names = list(layer_names) if layer_names is not None else None
        self.save_path: Optional[str] = None
        self.save_files: Dict = {}
        self.header_order: Optional[List[str]] = None
        self.known_nodes_to_record: set = set()
        self.logger: Optional[logging.Logger] = None

    @staticmethod
    def get_service_name() -> str:
        return "variance_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        node_names, model_stats = [], []
        for name, node in parameters.node_container.items(): # type: ignore
            if self._should_record(name):
                self.known_nodes_to_record.add(name)
                node_names.append(name)
                model_stats.append(node.get_model_stat())
        self.initialize_without_runtime_parameters(node_names, model_stats, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return
        if parameters.phase in self.record_phase:
            node_names, model_stats = [], []
            for name in self.known_nodes_to_record:
                node_names.append(name)
                model_stats.append(parameters.node_container[name].get_model_stat()) # type: ignore
            self.trigger_without_runtime_parameters(parameters.current_tick, node_names, model_stats,
                                                     phase_str=parameters.phase.name)

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        node_names: list,
        model_stats: list,
        output_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger
        assert len(node_names) == len(model_stats)
        self.save_path = os.path.join(output_path, "variance")
        os.makedirs(self.save_path, exist_ok=True)
        for name, stat in zip(node_names, model_stats):
            self.known_nodes_to_record.add(name)
            f = open(os.path.join(self.save_path, f"{name}.csv"), "w+")
            self._write_header(stat, f)
            self.save_files[name] = f

    def trigger_without_runtime_parameters(
        self,
        tick: int,
        node_names: list,
        model_stats: list,
        phase_str: Optional[str] = None,
    ):
        assert len(node_names) == len(model_stats)
        for name, stat in zip(node_names, model_stats):
            self._write_row(tick, phase_str, stat, self.save_files[name])

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        for name in self.known_nodes_to_record:
            src = os.path.join(checkpoint_folder_path, "variance", f"{name}.csv")
            if not os.path.exists(src):
                continue
            f = self.save_files[name]
            with open(src, 'r', newline='') as infile:
                if next(infile, None) is None:
                    continue
                for line in infile:
                    if int(line.split(",", 1)[0]) < restore_until_tick:
                        f.write(line)
            f.flush()

    # ---- internals ----------------------------------------------------------

    def _should_record(self, node_name) -> bool:
        return self.record_node is None or node_name in self.record_node

    def _write_header(self, model_stat: dict, f):
        if self.header_order is None:
            if self.layer_names is None:
                self.header_order = [name for name in model_stat if 'weight' in name]
            else:
                allowed_names = set(self.layer_names)
                self.header_order = [name for name in model_stat if name in allowed_names]
        header = ",".join(["tick", "phase", *self.header_order])
        f.write(header + "\n")
        f.flush()

    def _write_row(self, tick: int, phase_str, model_stat: dict, f):
        assert self.header_order is not None
        row = [str(tick), str(phase_str)]
        for layer_name in self.header_order:
            var = torch.var(model_stat[layer_name]).item()
            row.append(f"{var:.4E}")
        f.write(",".join(row) + "\n")
        f.flush()

    def __del__(self):
        for f in self.save_files.values():
            try:
                f.flush(); f.close()
            except Exception:
                pass
