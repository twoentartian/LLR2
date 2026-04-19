"""Save model state dicts to disk (file or LMDB) at configurable ticks.

Ported from DFL_torch/py_src/service/record_model_stat.py.
State dicts are moved to CPU before serialisation (writing to disk is
inherently a CPU operation and is done infrequently).
"""

from __future__ import annotations

import io
import logging
import os
from typing import Dict, List, Optional

import torch

from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.model_opti_save_load import save_model_state
from py_src import lmdb_pack


class ModelStatRecorder(Service):
    """Persist model state dicts to disk in either ``file`` or ``lmdb`` format."""

    def __init__(
        self,
        interval: int,
        model_name: str,
        dataset_name: str,
        phase: SimulationPhase = SimulationPhase.END_OF_TICK,
        record_node=None,
        record_at_tick: Optional[List[int]] = None,
    ):
        super().__init__()
        self.interval = interval
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.record_phase = phase
        self.record_node = record_node
        self.record_at_tick: List[int] = record_at_tick or []
        self.known_nodes_to_record: set = set()
        self.save_path: Optional[str] = None
        self.save_path_for_each_node: Optional[Dict] = None
        self.save_format: Optional[str] = None
        self.save_lmdb = None
        self.write_count: int = 0
        self.logger: Optional[logging.Logger] = None

    @staticmethod
    def get_service_name() -> str:
        return "model_stat_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, save_format: str = "lmdb", *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        node_names = [n for n in parameters.node_container if self._should_record(n)]  # type: ignore
        for n in node_names:
            self.known_nodes_to_record.add(n)
        self.initialize_without_runtime_parameters(node_names, output_path, save_format=save_format)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if (parameters.current_tick % self.interval != 0 and
                parameters.current_tick not in self.record_at_tick):
            return
        if parameters.phase == self.record_phase:
            node_names = list(self.known_nodes_to_record)
            model_stats = [parameters.node_container[n].get_model_stat() for n in node_names]  # type: ignore
            self.trigger_without_runtime_parameters(parameters.current_tick, node_names, model_stats)

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        node_names: list,
        output_path: str,
        logger: Optional[logging.Logger] = None,
        save_format: str = "lmdb",
        lmdb_db_name: Optional[str] = None,
    ):
        self.logger = logger
        assert save_format in ("lmdb", "file"), "save_format must be 'lmdb' or 'file'"
        self.save_format = save_format
        for n in node_names:
            self.known_nodes_to_record.add(n)

        if save_format == "lmdb":
            db_name = lmdb_db_name or "model_stat"
            self.save_path = os.path.join(output_path, f"{db_name}.lmdb")
            os.makedirs(self.save_path, exist_ok=True)
            try:
                import lmdb
                self.save_lmdb = lmdb.open(self.save_path, map_size=4 * 1024 ** 4)
            except ImportError:
                raise ImportError("lmdb is required for save_format='lmdb'. "
                                  "Install it with: pip install lmdb")
        else:
            self.save_path = os.path.join(output_path, "model_stat")
            os.makedirs(self.save_path, exist_ok=True)
            self.save_path_for_each_node = {}
            for n in node_names:
                node_dir = os.path.join(self.save_path, str(n))
                os.makedirs(node_dir, exist_ok=True)
                self.save_path_for_each_node[n] = node_dir

    def trigger_without_runtime_parameters(self, tick: int, node_names: list, model_stats: list):
        assert len(node_names) == len(model_stats)
        self.write_count += 1

        if self.save_format == "lmdb":
            with self.save_lmdb.begin(write=True) as txn:  # type: ignore
                for node_name, model_stat in zip(node_names, model_stats):
                    key = lmdb_pack.generate_lmdb_index_from_node_name_and_tick(node_name, tick)
                    buf = io.BytesIO()
                    # Move to CPU before serialising
                    cpu_stat = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_stat.items()}
                    torch.save(cpu_stat, buf)
                    txn.put(key.encode(), buf.getvalue())
            if self.write_count >= 1000:
                self._maybe_resize_lmdb()
                self.write_count = 0
        else:
            for node_name, model_stat in zip(node_names, model_stats):
                path = os.path.join(self.save_path_for_each_node[node_name], f"{tick}.model.pt")  # type: ignore
                cpu_stat = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_stat.items()}
                save_model_state(path, cpu_stat, self.model_name, self.dataset_name)

    def continue_from_checkpoint(
        self, checkpoint_folder_path: str, restore_until_tick: int,
        lmdb_db_name: Optional[str] = None, *args, **kwargs,
    ):
        if self.save_format is None:
            return
        if self.save_format == "lmdb":
            import lmdb
            db_name = lmdb_db_name or "model_stat"
            src_path = os.path.join(checkpoint_folder_path, f"{db_name}.lmdb")
            src_db = lmdb.open(src_path, readonly=True, lock=False)
            with self.save_lmdb.begin(write=True) as w: # type: ignore
                with src_db.begin() as r:
                    for key, val in r.cursor():
                        _, t = lmdb_pack.get_node_name_and_tick_from_lmdb_index(key)
                        if t < restore_until_tick:
                            w.put(key, val)
        else:
            raise NotImplementedError("File-format checkpoint restore not yet implemented")

    # ---- helpers ------------------------------------------------------------

    def _should_record(self, node_name) -> bool:
        return self.record_node is None or node_name in self.record_node

    def _maybe_resize_lmdb(self, threshold: float = 0.8):
        if self.save_lmdb is None:
            return
        data_file = os.path.join(self.save_path, 'data.mdb')  # type: ignore
        current_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
        current_mapsize = self.save_lmdb.info()['map_size']
        if current_size > current_mapsize * threshold:
            self.save_lmdb.set_mapsize(current_mapsize * 2)

    def __del__(self):
        if self.save_lmdb is not None:
            try:
                self.save_lmdb.close()
            except Exception:
                pass
