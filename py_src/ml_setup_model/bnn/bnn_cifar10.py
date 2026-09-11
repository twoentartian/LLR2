"""CIFAR-10 BinaryNet adapted from RWTH-IDS/efficient-binary-neural-networks.

Copyright (c) IDS, RWTH Aachen. MIT license; see
py_src/third_party/efficient_binary_neural_networks/LICENSE.

Implements config/binary_cifar10.yml (VGGNet7_binary, CPBA, width_scale=1).
Parameters remain full precision: only the forward operands are binarized.
This preserves the upstream STE/update rule while making state_dict, device
transfers, deepcopy and training checkpoint resume safe.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        # Match upstream rounding, including zero -> -1 (not torch.sign).
        return value.add(1).div(2).clamp(0, 1).round().mul(2).sub(1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binarize(value):
    return _BinarySTE.apply(value)


class BinaryConv2d(nn.Conv2d):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, value):
        if self.padding != (0, 0):
            value = F.pad(value, self._reversed_padding_repeated_twice, mode="replicate")
        return F.conv2d(value, binarize(self.weight), self.bias, self.stride,
                        0, self.dilation, self.groups)


class BinaryLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, value):
        return F.linear(value, binarize(self.weight), self.bias)


class FloatingConv2d(nn.Conv2d):
    """Floating-point convolution with the BNN layer's edge padding."""

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, value):
        if self.padding != (0, 0):
            value = F.pad(value, self._reversed_padding_repeated_twice, mode="replicate")
        return F.conv2d(value, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class FloatingLinear(nn.Linear):
    """Floating-point counterpart of :class:`BinaryLinear`."""

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class VGGNet7Binary(nn.Module):
    """Six binary convolutions and three binary linear layers, no biases."""

    def __init__(self):
        super().__init__()
        channels = (3, 128, 128, 256, 256, 512, 512)
        for i in range(1, 7):
            setattr(self, f"conv{i}", BinaryConv2d(
                channels[i - 1], channels[i], 3, padding=0 if i == 1 else 1, bias=False,
            ))
            setattr(self, f"bn{i}", nn.BatchNorm2d(channels[i]))
        self.fc1 = BinaryLinear(512 * 3 * 3, 1024, bias=False)
        self.fc2 = BinaryLinear(1024, 1024, bias=False)
        self.fc3 = BinaryLinear(1024, 10, bias=False)
        self.bn7 = nn.BatchNorm1d(1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.bn9 = nn.BatchNorm1d(10, affine=False)

    def forward(self, value):
        for i in range(1, 7):
            value = getattr(self, f"conv{i}")(value)
            if i in (2, 4, 6):
                value = F.max_pool2d(value, 2)
            value = binarize(F.hardtanh(getattr(self, f"bn{i}")(value)))
        value = value.flatten(1)
        value = binarize(F.hardtanh(self.bn7(self.fc1(value))))
        value = binarize(F.hardtanh(self.bn8(self.fc2(value))))
        return self.bn9(self.fc3(value))


class VGGNet7Floating(nn.Module):
    """VGGNet7 with floating-point weights and BNN binary activations."""

    def __init__(self):
        super().__init__()
        channels = (3, 128, 128, 256, 256, 512, 512)
        for i in range(1, 7):
            setattr(self, f"conv{i}", FloatingConv2d(
                channels[i - 1], channels[i], 3, padding=0 if i == 1 else 1, bias=False,
            ))
            setattr(self, f"bn{i}", nn.BatchNorm2d(channels[i]))
        self.fc1 = FloatingLinear(512 * 3 * 3, 1024, bias=False)
        self.fc2 = FloatingLinear(1024, 1024, bias=False)
        self.fc3 = FloatingLinear(1024, 10, bias=False)
        self.bn7 = nn.BatchNorm1d(1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.bn9 = nn.BatchNorm1d(10, affine=False)

    def forward(self, value):
        for i in range(1, 7):
            value = getattr(self, f"conv{i}")(value)
            if i in (2, 4, 6):
                value = F.max_pool2d(value, 2)
            value = binarize(F.hardtanh(getattr(self, f"bn{i}")(value)))
        value = value.flatten(1)
        value = binarize(F.hardtanh(self.bn7(self.fc1(value))))
        value = binarize(F.hardtanh(self.bn8(self.fc2(value))))
        return self.bn9(self.fc3(value))


class BinaryAdam(torch.optim.Adam):
    """Adam followed by clipping only binary-layer latent weights to [-1, 1]."""

    def __init__(self, model, **kwargs):
        binary_ids = {
            id(module.weight) for module in model.modules()
            if isinstance(module, (BinaryConv2d, BinaryLinear))
        }
        super().__init__([
            {"params": [p for p in model.parameters() if id(p) in binary_ids], "binary": True},
            {"params": [p for p in model.parameters() if id(p) not in binary_ids], "binary": False},
        ], **kwargs)

    def step(self, closure=None):
        loss = super().step(closure)
        with torch.no_grad():
            for group in self.param_groups:
                if group["binary"]:
                    for param in group["params"]:
                        param.clamp_(-1, 1)
        return loss
