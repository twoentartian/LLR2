"""Disk-backed top-k checkpoints for train/validation metric gaps."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from torch import nn

from py_src.model_opti_save_load import save_model_state


logger = logging.getLogger("misalignment_measurement_2")


class GapCheckpointTracker:
    """Keep the union of two rankings without retaining models in GPU RAM.

    Scores are absolute train/val differences at the end of each epoch.
    Equal scores prefer the earlier epoch. Only finite metric points qualify.
    A checkpoint is deleted only when it leaves *both* top-k rankings.
    Pass the original model rather than its torch.compile wrapper so saved
    state dictionaries remain compatible with the project's model loader.
    """

    def __init__(
        self,
        model: nn.Module,
        output_folder: Path,
        direction_name: str,
        model_type_name: str,
        dataset_type_name: str,
        *,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.model = model
        self.output_folder = Path(output_folder)
        self.checkpoint_folder = self.output_folder / f"gap_checkpoints_{direction_name}"
        self.checkpoint_folder.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.checkpoint_folder / "selection.json"
        self.model_type_name = model_type_name
        self.dataset_type_name = dataset_type_name
        self.top_k = top_k
        self._points: dict[int, dict[str, Any]] = {}
        self._accuracy_epochs: list[int] = []
        self._loss_epochs: list[int] = []
        self._last_epoch = -1
        self._write_manifest()

    def observe(self, row: dict[str, float]) -> None:
        """Consider a completed epoch, saving weights only if it enters top-k."""
        epoch = int(row["epoch"])
        if epoch <= self._last_epoch:
            raise ValueError("checkpoint observations must have increasing epochs")
        self._last_epoch = epoch
        metrics = {
            "train_loss": float(row["loss_a"]),
            "val_loss": float(row["loss_b"]),
            "train_accuracy": float(row["train_accuracy"]),
            "val_accuracy": float(row["val_accuracy"]),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            logger.warning("skipping non-finite gap checkpoint at epoch %d", epoch)
            return
        point = {
            "epoch": epoch,
            **metrics,
            "abs_accuracy_gap": abs(metrics["train_accuracy"] - metrics["val_accuracy"]),
            "abs_loss_gap": abs(metrics["train_loss"] - metrics["val_loss"]),
            "checkpoint": f"{self.checkpoint_folder.name}/epoch_{epoch:06d}.model.pt",
        }
        if "learning_rate" in row and math.isfinite(float(row["learning_rate"])):
            point["learning_rate"] = float(row["learning_rate"])
        candidates = {**self._points, epoch: point}
        accuracy_epochs = sorted(
            candidates, key=lambda item: (-candidates[item]["abs_accuracy_gap"], item)
        )[: self.top_k]
        loss_epochs = sorted(
            candidates, key=lambda item: (-candidates[item]["abs_loss_gap"], item)
        )[: self.top_k]
        retained = set(accuracy_epochs) | set(loss_epochs)
        if epoch not in retained:
            return

        # Clone parameters AND buffers (e.g. BatchNorm running statistics).
        # Serialization is synchronous; later optimizer steps cannot alter it.
        state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
        save_model_state(
            str(self.output_folder / point["checkpoint"]),
            state,
            self.model_type_name,
            self.dataset_type_name,
        )
        retired = set(self._points) - retained
        retired_paths = [self.output_folder / self._points[item]["checkpoint"] for item in retired]
        self._points = {item: candidates[item] for item in retained}
        self._accuracy_epochs = accuracy_epochs
        self._loss_epochs = loss_epochs
        self._write_manifest()
        for path in retired_paths:
            # These exact files were created by this tracker; no broad cleanup.
            path.unlink()

    def selection(self) -> dict[str, Any]:
        """Return JSON-ready rankings and a de-duplicated union of points."""
        points = {}
        for epoch, point in self._points.items():
            accuracy_rank = self._accuracy_epochs.index(epoch) + 1 if epoch in self._accuracy_epochs else None
            loss_rank = self._loss_epochs.index(epoch) + 1 if epoch in self._loss_epochs else None
            points[epoch] = {
                **point,
                "accuracy_gap_rank": accuracy_rank,
                "loss_gap_rank": loss_rank,
                "selected_by": [
                    name for name, rank in (("accuracy_gap", accuracy_rank), ("loss_gap", loss_rank))
                    if rank is not None
                ],
            }
        return {
            "top_k": self.top_k,
            "gap_definition": "absolute_train_minus_val",
            "epoch_definition": "zero_based_epoch_after_updates",
            "tie_break": "earlier_epoch_first",
            "top_accuracy_gap": [points[epoch] for epoch in self._accuracy_epochs],
            "top_loss_gap": [points[epoch] for epoch in self._loss_epochs],
            "union": [points[epoch] for epoch in sorted(points)],
        }

    def _write_manifest(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as outfile:
            json.dump(self.selection(), outfile, indent=2, sort_keys=True, allow_nan=False)
