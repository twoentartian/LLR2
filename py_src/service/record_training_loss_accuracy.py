"""Record per-tick training loss and accuracy to CSV.

Ported from DFL_torch/py_src/service/record_training_loss_accuracy.py.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


class ServiceTrainingLossAccuracyRecorder(Service):
    """Write one training-loss and one training-accuracy row per tick."""

    def __init__(
        self,
        interval: int,
        loss_file_name: str = "training_loss.csv",
        accuracy_file_name: str = "training_accuracy.csv",
    ):
        super().__init__()
        self.interval = interval
        self.loss_file_name = loss_file_name
        self.accuracy_file_name = accuracy_file_name
        self.loss_file = None
        self.accuracy_file = None
        self.node_order: Optional[List] = None
        self.logger: Optional[logging.Logger] = None

    @staticmethod
    def get_service_name() -> str:
        return "training_loss_recorder"

    # ---- simulator interface ------------------------------------------------

    def initialize(self, parameters: RuntimeParameters, output_path: str, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        node_order = list(parameters.node_container.keys()) # type: ignore
        self.initialize_without_runtime_parameters(output_path, node_order)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.phase == SimulationPhase.AFTER_TRAINING and parameters.current_tick % self.interval == 0:
            losses = {n: parameters.node_container[n].most_recent_loss.detach().cpu().item() for n in self.node_order} # type: ignore
            accuracies = {n: parameters.node_container[n].most_recent_accuracy for n in self.node_order} # type: ignore
            self.trigger_without_runtime_parameters(parameters.current_tick, losses, accuracies)

    # ---- standalone interface -----------------------------------------------

    def initialize_without_runtime_parameters(
        self,
        output_path: str,
        node_order: list,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger
        self.node_order = node_order
        self.loss_file = open(os.path.join(output_path, self.loss_file_name), "w+")
        self.accuracy_file = open(os.path.join(output_path, self.accuracy_file_name), "w+")
        header = ",".join(["tick", *[str(n) for n in node_order]])
        self.loss_file.write(header + "\n")
        self.accuracy_file.write(header + "\n")

    def trigger_without_runtime_parameters(
        self,
        tick: int,
        node_name_and_loss: Dict,
        node_name_and_accuracy: Dict,
    ):
        assert self.node_order is not None
        assert self.loss_file is not None
        assert self.accuracy_file is not None
        loss_row = [str(tick)] + ['%.4e' % node_name_and_loss[n] for n in self.node_order]
        self.loss_file.write(",".join(loss_row) + "\n")
        self.loss_file.flush()

        acc_row = [str(tick)] + ['%.4e' % node_name_and_accuracy[n] for n in self.node_order]
        self.accuracy_file.write(",".join(acc_row) + "\n")
        self.accuracy_file.flush()

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        assert self.loss_file is not None
        assert self.accuracy_file is not None
        for fname, fobj in [(self.loss_file_name, self.loss_file),
                             (self.accuracy_file_name, self.accuracy_file)]:
            src_path = os.path.join(checkpoint_folder_path, fname)
            if not os.path.exists(src_path):
                continue
            with open(src_path, 'r', newline='') as infile:
                if next(infile, None) is None:
                    continue
                for line in infile:
                    if int(line.split(",", 1)[0]) < restore_until_tick:
                        fobj.write(line)
            fobj.flush()

    def __del__(self):
        for f in (self.loss_file, self.accuracy_file):
            if f is not None:
                try:
                    f.flush(); f.close()
                except Exception:
                    pass
