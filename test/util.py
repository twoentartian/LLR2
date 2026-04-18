import os

import sys

import lightning as L

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from py_src.ml_setup import ApplicationType, MLSetup
from py_src.complete_ml_setup import FastTrainingSetup
from py_src.engine import Device, TrainResult, ValResult, train, val
from py_src.ml_setup.dataloader_util import DataloaderConfig

# ---------------------------------------------------------------------------
# Single-batch train and interface (for unit tests and smoke-checks)
# ---------------------------------------------------------------------------

def run_single_batch(
    ml_setup: MLSetup,
    *,
    use_cpu: bool = True,
    amp: bool = False,
    preset: int = 0,
    run_val: bool = True,
    batch_size: int = 8,
) -> "tuple[TrainResult, ValResult | None]":
    """Train for one batch and (optionally) evaluate for one batch.

    Intended for unit testing and quick smoke-checks.  The model and adapter
    stored in *ml_setup* are used directly (no deepcopy).

    Parameters
    ----------
    ml_setup:
        Fully configured :class:`MLSetup`.
    use_cpu:
        Force CPU execution (default ``True`` so tests run anywhere).
    amp:
        Enable automatic mixed precision.
    preset:
        Hyperparameter preset index forwarded to
        :func:`~py_src.complete_ml_setup.FastTrainingSetup.get_optimizer_lr_scheduler_epoch`.
    run_val:
        Whether to run the validation pass.  Always skipped for diffusion
        models regardless of this flag.

    Returns
    -------
    tuple[TrainResult, ValResult | None]
        ``(train_result, val_result)`` where *val_result* is ``None`` when
        validation was skipped.
    """
    device = Device.cpu() if use_cpu else Device.auto()

    one_batch_cfg = DataloaderConfig(num_samples=batch_size, num_workers=0, pin_memory=False)

    train_loader = ml_setup.train_dataloader(one_batch_cfg)

    model = ml_setup.model
    adapter = ml_setup.adapter
    model.to(device.device)

    # Build optimizer / scheduler — mirrors the logic in training_model()
    if isinstance(model, L.LightningModule):
        optimizer_lit, lr_scheduler_lit = model.configure_optimizers()  # type: ignore[call-arg]
        optimizer_cfg, lr_scheduler_cfg, _ = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(ml_setup, model, preset)
        optimizer = optimizer_cfg if optimizer_cfg is not None else optimizer_lit
        lr_scheduler = lr_scheduler_cfg if lr_scheduler_cfg is not None else lr_scheduler_lit
    else:
        optimizer, lr_scheduler, _ = FastTrainingSetup.get_optimizer_lr_scheduler_epoch(ml_setup, model, preset)

    scaler = device.make_scaler() if amp else None

    train_result = train(
        adapter, train_loader, optimizer, lr_scheduler, # type: ignore
        device=device, scaler=scaler, max_steps=1,
    )

    val_result = None
    if run_val and ml_setup.application_type == ApplicationType.classifier:
        val_loader = ml_setup.val_dataloader(one_batch_cfg)
        val_result = val(adapter, val_loader, device=device)

    return train_result, val_result
