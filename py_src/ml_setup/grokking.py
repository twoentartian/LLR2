from __future__ import annotations

import contextlib
import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from py_src.adapters import CustomStepAdapter
from py_src.ml_setup.ml_setup import ApplicationType, MLSetup
from py_src.ml_setup_dataset import ArithmeticDataset, DatasetSetup, DatasetType
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_model.transformer_for_grokking import TransformerForGrokking
from py_src.types import StepOutput

DEFAULT_GROKKING_VOCAB_LEN = 2000


class GrokkingArithmeticDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style wrapper for arithmetic sequences used by the grokking model."""

    def __init__(self, dataset: ArithmeticDataset) -> None:
        if dataset.data.ndim != 2 or dataset.data.size(1) < 2:
            raise ValueError("ArithmeticDataset must contain rank-2 token data with sequence length >= 2")
        self._dataset = dataset
        self.tokenizer = dataset.tokenizer
        self.sequence_length = int(dataset.data.size(1) - 1)
        eq_token_index = self.tokenizer.stoi["="]
        eq_positions = (dataset.data[0, 1:] == eq_token_index).nonzero(as_tuple=False)
        if eq_positions.numel() != 1:
            raise ValueError("Expected exactly one '=' token in grokking arithmetic sequences")
        self.eq_position = int(eq_positions.item())

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._dataset.data[index]
        return {
            "text": row[:-1],
            "target": row[1:],
        }


def _wrap_grokking_dataset(dataset: Dataset) -> GrokkingArithmeticDataset:
    if isinstance(dataset, GrokkingArithmeticDataset):
        return dataset
    if isinstance(dataset, ArithmeticDataset):
        return GrokkingArithmeticDataset(dataset)
    raise TypeError(
        "Grokking setups require ArithmeticDataset or GrokkingArithmeticDataset overrides, "
        f"got {type(dataset).__name__}"
    )


def _build_arithmetic_dataset(
    dataset_type: DatasetType,
    *,
    operator: str,
    train_percentage: float,
    modulus: int,
    operand_length: Optional[int],
    train_split_type: Literal[
        "random",
        "chessboard",
        "updown",
        "leftright",
        "tl_to_br",
        "tr_to_bl",
        "interlace_row",
        "interlace_col",
        "chessboard_random",
    ],
) -> DatasetSetup:
    train_data, val_data = ArithmeticDataset.splits(
        train_pct=train_percentage,
        operator=operator,
        modulus=modulus,
        operand_length=operand_length,
        train_split_type=train_split_type,
    )
    return DatasetSetup(
        dataset_type,
        GrokkingArithmeticDataset(train_data),
        GrokkingArithmeticDataset(val_data),
    )


def _resolve_dataset_setup(
    *,
    dataset_type: DatasetType,
    override_dataset: Optional[DatasetSetup],
    operator: str,
    train_percentage: float,
    modulus: int,
    operand_length: Optional[int],
    train_split_type: Literal[
        "random",
        "chessboard",
        "updown",
        "leftright",
        "tl_to_br",
        "tr_to_bl",
        "interlace_row",
        "interlace_col",
        "chessboard_random",
    ],
) -> DatasetSetup:
    if override_dataset is None:
        return _build_arithmetic_dataset(
            dataset_type,
            operator=operator,
            train_percentage=train_percentage,
            modulus=modulus,
            operand_length=operand_length,
            train_split_type=train_split_type,
        )

    return DatasetSetup(
        override_dataset.dataset_type,
        _wrap_grokking_dataset(override_dataset.train_data),
        _wrap_grokking_dataset(override_dataset.valdation_data),
    )


def _step(
    batch: dict[str, torch.Tensor],
    model: TransformerForGrokking,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler,
    *,
    eq_position: int,
    train: bool,
) -> StepOutput:
    should_update = train and optimizer is not None
    if should_update:
        optimizer.zero_grad(set_to_none=True)

    x = batch["text"]
    y = batch["target"]

    autocast_context = contextlib.nullcontext()
    if should_update and x.device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast_context = torch.autocast(device_type=x.device.type, dtype=amp_dtype)

    grad_context = contextlib.nullcontext() if should_update else torch.inference_mode()
    with grad_context:
        with autocast_context:
            logits, _, _ = model(x=x)
            logits = logits.transpose(-2, -1)
            target_rhs = y[..., eq_position + 1:]
            logits_rhs = logits[..., eq_position + 1:]
            loss = F.cross_entropy(logits_rhs, target_rhs, reduction="mean")

    predictions = logits_rhs.argmax(dim=-2)
    correct_rows = (predictions == target_rhs).all(dim=-1)

    if should_update:
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    extra = {}
    if not train:
        extra["variance"] = float(logits_rhs.var(unbiased=False).item())

    return StepOutput(
        loss=float(loss.item()),
        sample_count=int(y.shape[0]),
        correct_count=int(correct_rows.sum().item()),
        extra=extra,
    )


def _train_step(batch_index, batch, model, optimizer, lr_scheduler, extra_ctx) -> StepOutput:
    del batch_index
    assert extra_ctx is not None
    return _step(
        batch,
        model,
        optimizer,
        lr_scheduler,
        eq_position=extra_ctx["eq_position"],
        train=True,
    )


def _evaluation_step(batch_index, batch, model, extra_ctx) -> StepOutput:
    del batch_index
    assert extra_ctx is not None
    return _step(
        batch,
        model,
        None,
        None,
        eq_position=extra_ctx["eq_position"],
        train=False,
    )


def build_grokking_model(
    dataset,
    *,
    n_layers=None,
    n_heads=None,
    d_model=None,
    context_len=None,
    vocab_len=None,
    position_encoding=None,
):
    if hasattr(dataset, "sequence_length"):
        min_context_len = int(dataset.sequence_length)
    else:
        min_context_len = int(dataset.data.shape[1] - 1)
    actual_context_len = max(50, min_context_len) if context_len is None else max(context_len, min_context_len)
    actual_vocab_len = max(
        DEFAULT_GROKKING_VOCAB_LEN if vocab_len is None else vocab_len,
        len(dataset.tokenizer),
    )
    return TransformerForGrokking(
        n_layers=2 if n_layers is None else n_layers,
        n_heads=4 if n_heads is None else n_heads,
        d_model=128 if d_model is None else d_model,
        max_context_len=actual_context_len,
        vocab_len=actual_vocab_len,
        trainable_position_encoding=position_encoding == "trainable",
    )


def _make_grokking_setup(
    *,
    dataset_type: DatasetType,
    operator: str,
    override_dataset: Optional[DatasetSetup],
    train_percentage: float,
    operand_length: Optional[int],
    modulus: int,
    train_split_type: Literal[
        "random",
        "chessboard",
        "updown",
        "leftright",
        "tl_to_br",
        "tr_to_bl",
        "interlace_row",
        "interlace_col",
        "chessboard_random",
    ] = "random",
) -> MLSetup:
    dataset_setup = _resolve_dataset_setup(
        dataset_type=dataset_type,
        override_dataset=override_dataset,
        operator=operator,
        train_percentage=train_percentage,
        modulus=modulus,
        operand_length=operand_length,
        train_split_type=train_split_type,
    )
    train_dataset = _wrap_grokking_dataset(dataset_setup.train_data)
    val_dataset = _wrap_grokking_dataset(dataset_setup.valdation_data)

    model = build_grokking_model(train_dataset)
    adapter = CustomStepAdapter(
        model,
        _train_step,
        _evaluation_step,
        extra_ctx={"eq_position": train_dataset.eq_position},
    )

    return MLSetup(
        model=model,
        adapter=adapter,
        model_type=ModelType.transformer_for_grokking,
        application_type=ApplicationType.classifier,
        training_data=train_dataset,
        testing_data=val_dataset,
        dataset_type=dataset_setup.dataset_type,
        default_batch_size=max(1, min(512, math.ceil(len(train_dataset) / 2.0))),
        criterion=nn.CrossEntropyLoss(),
        has_normalization_layer=True,
    )


def arithmetic_addition_grokking(
    override_dataset: Optional[DatasetSetup] = None,
    *,
    train_percentage: float = 50,
    operand_length: Optional[int] = None,
    modulus: int = 97,
    train_split_type: Literal[
        "random",
        "chessboard",
        "updown",
        "leftright",
        "tl_to_br",
        "tr_to_bl",
        "interlace_row",
        "interlace_col",
        "chessboard_random",
    ] = "random",
) -> MLSetup:
    return _make_grokking_setup(
        dataset_type=DatasetType.arithmetic_addition,
        operator="+",
        override_dataset=override_dataset,
        train_percentage=train_percentage,
        operand_length=operand_length,
        modulus=modulus,
        train_split_type=train_split_type,
    )


def arithmetic_cubepoly_grokking(
    override_dataset: Optional[DatasetSetup] = None,
    *,
    train_percentage: float = 50,
    operand_length: Optional[int] = None,
    modulus: int = 97,
    train_split_type: Literal[
        "random",
        "chessboard",
        "updown",
        "leftright",
        "tl_to_br",
        "tr_to_bl",
        "interlace_row",
        "interlace_col",
        "chessboard_random",
    ] = "random",
) -> MLSetup:
    return _make_grokking_setup(
        dataset_type=DatasetType.arithmetic_cubepoly,
        operator="**3+",
        override_dataset=override_dataset,
        train_percentage=train_percentage,
        operand_length=operand_length,
        modulus=modulus,
        train_split_type=train_split_type,
    )


def arithmetic_cube2_grokking(
    override_dataset: Optional[DatasetSetup] = None,
    *,
    train_percentage: float = 50,
    operand_length: Optional[int] = None,
    modulus: int = 97,
    train_split_type: Literal[
        "random",
        "chessboard",
        "updown",
        "leftright",
        "tl_to_br",
        "tr_to_bl",
        "interlace_row",
        "interlace_col",
        "chessboard_random",
    ] = "random",
) -> MLSetup:
    return _make_grokking_setup(
        dataset_type=DatasetType.arithmetic_cube2,
        operator=f"x**3+x*y**2+y_mod_{modulus}",
        override_dataset=override_dataset,
        train_percentage=train_percentage,
        operand_length=operand_length,
        modulus=modulus,
        train_split_type=train_split_type,
    )


def arithmetic_unknown_exp_grokking(
    override_dataset: Optional[DatasetSetup] = None,
    *,
    train_percentage: float = 50,
    operand_length: Optional[int] = None,
    modulus: int = 97,
    train_split_type: Literal[
        "random",
        "chessboard",
        "updown",
        "leftright",
        "tl_to_br",
        "tr_to_bl",
        "interlace_row",
        "interlace_col",
        "chessboard_random",
    ] = "random",
) -> MLSetup:
    return _make_grokking_setup(
        dataset_type=DatasetType.arithmetic_exp_unknown,
        operator="unknown",
        override_dataset=override_dataset,
        train_percentage=train_percentage,
        operand_length=operand_length,
        modulus=modulus,
        train_split_type=train_split_type,
    )


__all__ = [
    "GrokkingArithmeticDataset",
    "build_grokking_model",
    "arithmetic_addition_grokking",
    "arithmetic_cubepoly_grokking",
    "arithmetic_cube2_grokking",
    "arithmetic_unknown_exp_grokking",
]
