"""Record per-layer cosine similarity against a fixed reference model.

Ported from DFL_torch/py_src/service/record_cosine_similarity.py.
GPU optimization: similarity is computed on the tensors' own device;
only scalars are moved to CPU via .item() for CSV writing.
The reference model is stored on CPU to save GPU memory (loaded once).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


class ServiceCosineSimilarityRecorder(Service):
    """Record layer-wise cosine similarity between the current model state and a
    fixed reference state dict.
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
        self.reference_model_state: Dict = {}   # stored on CPU
        self.known_nodes_to_record: set = set()
        self.logger: Optional[logging.Logger] = None

    @staticmethod
    def get_service_name() -> str:
        return "cosine_similarity_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        node_stats: Dict = {}
        for name, node in parameters.node_container.items(): # type: ignore
            if self._should_record(name):
                self.known_nodes_to_record.add(name)
                node_stats[name] = node.get_model_stat()
        self.initialize_without_runtime_parameters(node_stats, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return
        if parameters.phase in self.record_phase:
            stats = {name: parameters.node_container[name].get_model_stat() # type: ignore
                     for name in self.known_nodes_to_record}
            self.trigger_without_runtime_parameters(parameters.current_tick, stats,
                                                     phase_str=parameters.phase.name)

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        node_names_and_model_stats: Dict,
        output_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger
        self.save_path = os.path.join(output_path, "cosine_similarity")
        os.makedirs(self.save_path, exist_ok=True)
        for node_name, model_stat in node_names_and_model_stats.items():
            self.known_nodes_to_record.add(node_name)
            f = open(os.path.join(self.save_path, f"{node_name}.csv"), "w+")
            self._write_header(model_stat, f)
            self.save_files[node_name] = f
            self.set_reference_model_state(node_name, model_stat)

    def trigger_without_runtime_parameters(
        self,
        tick: int,
        node_names_and_model_stats: Dict,
        phase_str: Optional[str] = None,
    ):
        for node_name, model_stat in node_names_and_model_stats.items():
            if node_name not in self.save_files:
                continue
            self._write_row(tick, phase_str, node_name, model_stat, self.save_files[node_name])

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        for name in self.known_nodes_to_record:
            src = os.path.join(checkpoint_folder_path, "cosine_similarity", f"{name}.csv")
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

    def set_reference_model_state(self, node_name, model_stat: dict):
        """Store a CPU copy of the reference model state."""
        if self.layer_names is None:
            keys_to_store = list(model_stat.keys())
        else:
            allowed_names = set(self.layer_names)
            keys_to_store = [name for name in model_stat if name in allowed_names]
        self.reference_model_state[node_name] = {
            k: model_stat[k].detach().cpu().float() for k in keys_to_store
        }

    # ---- internals ----------------------------------------------------------

    def _should_record(self, node_name) -> bool:
        return self.record_node is None or node_name in self.record_node

    @staticmethod
    def _layerwise_cosine_similarity(current: dict, reference: dict) -> Dict[str, float]:
        """Compute cosine similarity per layer (all ops on current tensor's device)."""
        sims: Dict[str, float] = {}
        for key in current:
            if key not in reference:
                continue
            v1 = current[key].detach().float().view(-1)
            v2 = reference[key].detach().float().to(v1.device).view(-1)  # move reference shard to same device
            if v1.numel() != v2.numel():
                sims[key] = float("nan")
                continue
            if v1.numel() > 1:
                sim = F.cosine_similarity(v1, v2, dim=0).item()
            else:
                sim = 1.0 if torch.equal(v1, v2) else 0.0
            sims[key] = sim
        return sims

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

    def _write_row(self, tick: int, phase_str, node_name, model_stat: dict, f):
        ref = self.reference_model_state[node_name]
        sims = self._layerwise_cosine_similarity(model_stat, ref)
        row = [str(tick), str(phase_str)]
        for layer_name in self.header_order:  # type: ignore
            row.append(f"{sims.get(layer_name, 0.0):.7E}")
        f.write(",".join(row) + "\n")
        f.flush()

    def __del__(self):
        for f in self.save_files.values():
            try:
                f.flush(); f.close()
            except Exception:
                pass
