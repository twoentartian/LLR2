from __future__ import annotations

import unittest

import torch

from py_src.adapters import ModelAdapter, StandardAdapter
from py_src.engine import Device, train
from py_src.types import StepOutput


class _FixedLossAdapter(ModelAdapter):
    def __init__(self, losses: list[float]):
        self._model = torch.nn.Linear(1, 1)
        self._losses = losses
        self._index = 0

    def get_model(self) -> torch.nn.Module:
        return self._model

    def train_step(
        self,
        batch,
        batch_idx,
        optimizer,
        lr_scheduler,
        device,
        scaler,
        backpropagation=True,
    ) -> StepOutput:
        loss = self._losses[self._index % len(self._losses)]
        self._index += 1
        return StepOutput(loss=loss, sample_count=1, correct_count=1)

    def val_step(self, batch, batch_idx, device) -> StepOutput:
        raise NotImplementedError


class _ModeTrackingAdapter(ModelAdapter):
    def __init__(self):
        self._model = torch.nn.Linear(1, 1)
        self.saw_training_mode: bool | None = None
        self.saw_grad_enabled: bool | None = None
        self.post_step_calls = 0

    def get_model(self) -> torch.nn.Module:
        return self._model

    def train_step(
        self,
        batch,
        batch_idx,
        optimizer,
        lr_scheduler,
        device,
        scaler,
        backpropagation=True,
    ) -> StepOutput:
        self.saw_training_mode = self._model.training
        self.saw_grad_enabled = torch.is_grad_enabled()
        return StepOutput(loss=0.5, sample_count=1, correct_count=1)

    def val_step(self, batch, batch_idx, device) -> StepOutput:
        raise NotImplementedError

    def post_train_step(self) -> None:
        self.post_step_calls += 1


class _DummyScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


class _ScaledLossWrapper:
    def __init__(self, loss: torch.Tensor) -> None:
        self._loss = loss

    def backward(self) -> None:
        self._loss.backward()


class _OverflowSkippingScaler:
    def __init__(self) -> None:
        self._scale = 8.0

    def scale(self, loss: torch.Tensor) -> _ScaledLossWrapper:
        return _ScaledLossWrapper(loss)

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        # Simulate GradScaler skipping optimizer.step() due to overflow.
        return None

    def update(self) -> None:
        self._scale = self._scale / 2.0

    def get_scale(self) -> float:
        return self._scale


class TestEngineTrain(unittest.TestCase):
    def test_adaptive_train_can_span_multiple_dataloader_passes(self):
        adapter = _FixedLossAdapter([1.0, 0.8])
        dataloader = [object(), object()]

        result = train(
            adapter,
            dataloader,
            optimizer=None,
            lr_scheduler=None,
            device=Device.cpu(),
            backpropagation=False,
            min_rounds=2,
            max_rounds=5,
            loss_threshold=0.0,
        )

        self.assertEqual(result.iterations, 5)
        self.assertEqual(result.stop_reason, "max_rounds")
        self.assertGreater(result.iterations, len(dataloader))

    def test_adaptive_train_stops_when_moving_average_reaches_threshold(self):
        adapter = _FixedLossAdapter([0.6, 0.4, 0.1, 0.1])
        dataloader = [object(), object()]

        result = train(
            adapter,
            dataloader,
            optimizer=None,
            lr_scheduler=None,
            device=Device.cpu(),
            backpropagation=False,
            min_rounds=2,
            max_rounds=10,
            loss_threshold=0.25,
        )

        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.stop_reason, "loss_threshold")
        self.assertIsNotNone(result.moving_average_loss)
        assert result.moving_average_loss is not None
        self.assertLessEqual(result.moving_average_loss, 0.25)

    def test_forward_only_can_run_in_eval_mode_without_post_step_side_effects(self):
        adapter = _ModeTrackingAdapter()

        result = train(
            adapter,
            [object()],
            optimizer=None,
            lr_scheduler=None,
            device=Device.cpu(),
            backpropagation=False,
            training_mode=False,
            max_rounds=1,
        )

        self.assertEqual(result.iterations, 1)
        self.assertFalse(adapter.saw_training_mode)
        self.assertFalse(adapter.saw_grad_enabled)
        self.assertEqual(adapter.post_step_calls, 0)

    def test_scheduler_is_not_advanced_when_scaler_skips_optimizer_step(self):
        model = torch.nn.Linear(4, 2)
        adapter = StandardAdapter(model, torch.nn.CrossEntropyLoss())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = _DummyScheduler()
        scaler = _OverflowSkippingScaler()

        batch = (torch.randn(2, 4), torch.tensor([0, 1], dtype=torch.int64))

        result = train(
            adapter,
            [batch],
            optimizer=optimizer,
            lr_scheduler=scheduler,
            device=Device.cpu(),
            scaler=scaler, # type: ignore[arg-type]
            max_rounds=1,
        )

        self.assertEqual(result.iterations, 1)
        self.assertEqual(scheduler.step_calls, 0)
