"""Helper functions for the find_high_accuracy_path tool.

Ported from DFL_torch/tool/find_high_accuracy_path_v2/functions.py.
Uses LLR2's engine for training.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from py_src.adapters import StandardAdapter, clone_adapter_for_model
from py_src.util import MovingAverage


def rebuild_norm_layer_function(
    model: torch.nn.Module,
    initial_model_state: dict,
    start_model_state: dict,
    rebuild_norm_optimizer: torch.optim.Optimizer,
    training_optimizer_state: dict,
    norm_layers: list,
    ml_setup,
    dataloader,
    parameter_rebuild_norm,
    runtime_parameter,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
):
    """Rebuild only normalization layers while keeping other layers frozen.

    After training the norm layers, their weights converge to match the batch
    statistics of the non-norm weights.
    """
    model_stat = model.state_dict()

    # Optionally reset norm-layer weights
    reset_printed = False
    for layer_name in norm_layers:
        if layer_name not in model_stat:
            continue
        if parameter_rebuild_norm.rebuild_norm_use_initial_norm_weights:
            if not reset_printed and logger:
                logger.info("reset norm weights to initial model weights")
                reset_printed = True
            model_stat[layer_name] = initial_model_state[layer_name].detach().clone()
        elif parameter_rebuild_norm.rebuild_norm_use_start_model_norm_weights:
            if not reset_printed and logger:
                logger.info("reset norm weights to starting model weights")
                reset_printed = True
            model_stat[layer_name] = start_model_state[layer_name].detach().clone()
    model.load_state_dict(model_stat)

    model.train()
    model.to(device)
    criterion = _try_get_criterion(ml_setup)
    adapter = clone_adapter_for_model(ml_setup.adapter, model, criterion=criterion)
    rebuild_norm_optimizer.load_state_dict(training_optimizer_state)
    _optimizer_to(rebuild_norm_optimizer, device)

    scaler = torch.cuda.amp.GradScaler() if runtime_parameter.use_amp and device.type == "cuda" else None
    moving_average = MovingAverage(parameter_rebuild_norm.rebuild_norm_for_min_rounds)

    step = 0
    while True:
        done = False
        for batch_idx, batch in enumerate(dataloader):
            step += 1
            result = adapter.train_step(
                batch=batch,
                batch_idx=batch_idx,
                optimizer=rebuild_norm_optimizer,
                lr_scheduler=None,
                device=device,
                scaler=scaler,
                backpropagation=True,
                zero_grad=True,
                step_optimizer=True,
            )
            if result.optimizer_was_run:
                adapter.post_train_step()

            loss_val = result.loss
            moving_average.add(loss_val)

            if runtime_parameter.verbose and step % 10 == 0 and logger:
                logger.info(f"tick {runtime_parameter.current_tick}: rebuild norm step {step}, "
                            f"loss={moving_average.get_average():.3f}")

            if step >= parameter_rebuild_norm.rebuild_norm_for_max_rounds:
                done = True
                break
            if (moving_average.get_average() <= parameter_rebuild_norm.rebuild_norm_until_loss and
                    step >= parameter_rebuild_norm.rebuild_norm_for_min_rounds):
                done = True
                break

        if done:
            if logger:
                logger.info(f"tick {runtime_parameter.current_tick}: rebuild norm done in {step} steps, "
                            f"loss={moving_average.get_average():.3f}")
            break


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_get_criterion(ml_setup) -> Optional[torch.nn.Module]:
    """Extract criterion from an MLSetup when the adapter exposes one."""
    criterion = getattr(ml_setup, "criterion", None)
    if criterion is not None:
        return criterion
    if isinstance(ml_setup.adapter, StandardAdapter):
        return ml_setup.adapter._criterion
    return None


def _get_criterion(ml_setup) -> torch.nn.Module:
    """Extract criterion from an MLSetup."""
    criterion = _try_get_criterion(ml_setup)
    if criterion is not None:
        return criterion
    raise AttributeError("Cannot extract criterion from adapter type: "
                         f"{type(ml_setup.adapter).__name__}")


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device):
    """Move all optimizer state tensors to *device* in-place."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
