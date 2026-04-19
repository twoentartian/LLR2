"""Record per-layer weight differences and distances to zero.

Ported from DFL_torch/py_src/service/record_weights_difference.py.
GPU optimization: all tensor operations run on the tensors' own device;
only scalar results are moved to CPU via .item() for CSV writing.
"""

from __future__ import annotations

import logging
import os
from glob import glob
from typing import Dict, List, Optional

import torch

from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


# ---------------------------------------------------------------------------
# ServiceWeightsDifferenceRecorder
# ---------------------------------------------------------------------------

class ServiceWeightsDifferenceRecorder(Service):
    """Record per-layer L1 / L2 distance between a list of model state dicts
    (measuring the deviation of each layer from the mean across all models).
    """

    def __init__(
        self,
        interval: int,
        l1_save_file_name: str = "weight_difference_l1.csv",
        l2_save_file_name: str = "weight_difference_l2.csv",
    ):
        super().__init__()
        self.interval = interval
        self.l1_save_file_name = l1_save_file_name
        self.l2_save_file_name = l2_save_file_name
        self.l1_save_file = None
        self.l2_save_file = None
        self.layer_order: Optional[List[str]] = None
        self.last_l1_distance = None
        self.last_l2_distance = None
        self.logger: Optional[logging.Logger] = None

    @staticmethod
    def get_service_name() -> str:
        return "weight_difference_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        model_stats = []
        for _, node in parameters.node_container.items(): # type: ignore
            model_stats.append(node.get_model_stat())
            break
        self.initialize_without_runtime_parameters(model_stats, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.phase == SimulationPhase.END_OF_TICK and parameters.current_tick % self.interval == 0:
            model_stats = [node.get_model_stat() for node in parameters.node_container.values()] # type: ignore
            self.trigger_without_runtime_parameters(parameters.current_tick, model_stats)

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        model_stats: list,
        output_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger
        self.l1_save_file = open(os.path.join(output_path, self.l1_save_file_name), "w+")
        self.l2_save_file = open(os.path.join(output_path, self.l2_save_file_name), "w+")
        self.layer_order = list(model_stats[0].keys())
        header = ",".join(["tick", *self.layer_order])
        self.l1_save_file.write(header + "\n")
        self.l2_save_file.write(header + "\n")

    def trigger_without_runtime_parameters(self, tick: int, model_stats: list):
        assert self.layer_order is not None
        assert self.l1_save_file is not None
        assert self.l2_save_file is not None
        l1_vals, l2_vals = [], []
        for layer_name in self.layer_order:
            # Stack on the tensors' own device — avoids CPU round-trip
            weights = torch.stack([s[layer_name].float() for s in model_stats])
            mean_w = weights.mean(dim=0)
            diff = weights - mean_w
            l1_vals.append(f"{diff.abs().sum().item():.4e}")
            l2_vals.append(f"{diff.pow(2).sum().item():.4e}")
        self.l1_save_file.write(",".join([str(tick), *l1_vals]) + "\n")
        self.l1_save_file.flush()
        self.l2_save_file.write(",".join([str(tick), *l2_vals]) + "\n")
        self.l2_save_file.flush()
        self.last_l1_distance = l1_vals
        self.last_l2_distance = l2_vals

    def get_last_distance(self):
        return self.last_l1_distance, self.last_l2_distance

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        assert self.l1_save_file is not None
        assert self.l2_save_file is not None
        for fname, fobj in [(self.l1_save_file_name, self.l1_save_file),
                             (self.l2_save_file_name, self.l2_save_file)]:
            with open(os.path.join(checkpoint_folder_path, fname), 'r', newline='') as infile:
                next(infile)  # skip header
                for line in infile:
                    if int(line.split(",", 1)[0]) < restore_until_tick:
                        fobj.write(line)
            fobj.flush()

    def __del__(self):
        for f in (self.l1_save_file, self.l2_save_file):
            if f is not None:
                try:
                    f.flush(); f.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# ServiceDistanceToOriginRecorder
# ---------------------------------------------------------------------------

class ServiceDistanceToOriginRecorder(Service):
    """Record per-layer L1 / L2 norm (distance to zero) for each node."""

    def __init__(
        self,
        interval: int,
        nodes_to_record: List[int],
        l1_save_file_name: str = "distance_to_origin_l1.csv",
        l2_save_file_name: str = "distance_to_origin_l2.csv",
    ):
        super().__init__()
        self.interval = interval
        self.nodes_to_record = nodes_to_record
        self.l1_save_file_name = l1_save_file_name
        self.l2_save_file_name = l2_save_file_name
        self.l1_save_file: Dict = {}
        self.l2_save_file: Dict = {}
        self.layer_order: Optional[List[str]] = None
        self.logger: Optional[logging.Logger] = None

    @staticmethod
    def get_service_name() -> str:
        return "distance_to_origin_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        stats = {
            name: node.get_model_stat()
            for name, node in parameters.node_container.items() # type: ignore
            if name in self.nodes_to_record
        }
        self.initialize_without_runtime_parameters(stats, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.phase == SimulationPhase.END_OF_TICK and parameters.current_tick % self.interval == 0:
            stats = {
                name: node.get_model_stat()
                for name, node in parameters.node_container.items() # type: ignore
            }
            self.trigger_without_runtime_parameters(parameters.current_tick, stats)

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        node_name_and_model_stat: dict,
        output_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger
        first_stat = next(iter(node_name_and_model_stat.values()))
        self.layer_order = list(first_stat.keys())
        header = ",".join(["tick", *self.layer_order])
        for node_name in node_name_and_model_stat:
            f1 = open(os.path.join(output_path, f"{node_name}__{self.l1_save_file_name}"), "w+")
            f2 = open(os.path.join(output_path, f"{node_name}__{self.l2_save_file_name}"), "w+")
            f1.write(header + "\n")
            f2.write(header + "\n")
            self.l1_save_file[node_name] = f1
            self.l2_save_file[node_name] = f2

    def trigger_without_runtime_parameters(self, tick: int, node_name_and_model_stat: dict):
        assert self.layer_order is not None
        assert self.l1_save_file is not None
        assert self.l2_save_file is not None
        for node_name, model_stat in node_name_and_model_stat.items():
            if node_name not in self.l1_save_file:
                continue
            l1_vals, l2_vals = [], []
            for layer_name in self.layer_order:
                t = model_stat[layer_name]
                l1_vals.append(f"{t.abs().sum().item():.4e}")
                l2_vals.append(f"{t.float().pow(2).sum().sqrt().item():.4e}")
            self.l1_save_file[node_name].write(",".join([str(tick), *l1_vals]) + "\n")
            self.l1_save_file[node_name].flush()
            self.l2_save_file[node_name].write(",".join([str(tick), *l2_vals]) + "\n")
            self.l2_save_file[node_name].flush()

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        assert self.l1_save_file is not None
        assert self.l2_save_file is not None
        for suffix, files_dict in [(self.l1_save_file_name, self.l1_save_file),
                                    (self.l2_save_file_name, self.l2_save_file)]:
            for path in glob(os.path.join(checkpoint_folder_path, f"*{suffix}")):
                node_name = int(os.path.basename(path).split("__")[0])
                if node_name not in files_dict:
                    continue
                with open(path, 'r', newline='') as infile:
                    next(infile)
                    for line in infile:
                        if int(line.split(",", 1)[0]) < restore_until_tick:
                            files_dict[node_name].write(line)
                files_dict[node_name].flush()

    def __del__(self):
        for d in (self.l1_save_file, self.l2_save_file):
            for f in d.values():
                try:
                    f.flush(); f.close()
                except Exception:
                    pass
