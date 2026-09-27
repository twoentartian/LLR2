"""Sanity checks for the SGLD-based local learning coefficient estimator."""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from tool.calculate_rlct import estimate_llc


def _gaussian_nll(model, batch):
    data, label = batch
    return 0.5 * ((model(data) - label) ** 2).mean()


def test_regular_linear_regression_llc_is_half_parameter_count():
    torch.manual_seed(0)
    n, d = 10000, 20
    data = torch.randn(n, d)
    label = data @ torch.randn(d, 1) + torch.randn(n, 1)
    model = torch.nn.Linear(d, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.linalg.lstsq(data, label).solution.T)
    original_weight = model.weight.detach().clone()

    dataset = TensorDataset(data, label)
    result = estimate_llc(
        model,
        _gaussian_nll,
        DataLoader(dataset, batch_size=500, shuffle=True),
        DataLoader(dataset, batch_size=5000),
        n,
        epsilon=3e-5,
        gamma=1.0,
        nbeta=None,
        num_chains=2,
        num_draws=1000,
        num_burnin=200,
        device=torch.device("cpu"),
        log_interval=0,
    )

    assert result["num_parameters"] == d
    assert math.isclose(result["nbeta"], n / math.log(n))
    assert abs(result["llc_mean"] - d / 2) < 2.0
    assert len(result["traces"]) == 2 and len(result["traces"][0]) == 1200
    assert torch.equal(model.weight, original_weight)
