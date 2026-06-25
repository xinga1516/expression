from __future__ import annotations

import pytest
import torch

from scripts.train import ZINBLoss, pearson_loss, weighted_mse_loss
from src.model import MODEL_REGISTRY, build_model


pytestmark = pytest.mark.unit


def test_weighted_mse_loss_weights_nonzero_targets() -> None:
    pred = torch.tensor([0.0, 1.0])
    target_with_nonzero = torch.tensor([0.0, 2.0])
    unweighted_loss = weighted_mse_loss(pred, target_with_nonzero, nonzero_weight=1.0)
    weighted_loss = weighted_mse_loss(pred, target_with_nonzero, nonzero_weight=10.0)

    assert weighted_loss > unweighted_loss


def test_pearson_loss_perfect_and_constant_vectors_are_finite() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 4.0, 6.0])
    assert pearson_loss(pred, target).item() == pytest.approx(0.0, abs=1e-6)

    constant_loss = pearson_loss(torch.ones(3), torch.ones(3))
    assert torch.isfinite(constant_loss)


def test_zinb_loss_is_finite_for_zero_heavy_counts() -> None:
    loss_fn = ZINBLoss()
    y_true = torch.tensor([0.0, 0.0, 3.0, 10.0])
    mu = torch.tensor([0.1, 1.0, 3.0, 9.0])
    theta = torch.ones(4)
    pi = torch.tensor([0.8, 0.2, 0.1, 0.05])

    loss = loss_fn(y_true, mu, theta, pi)

    assert torch.isfinite(loss)


def test_registered_models_scalar_output_shapes() -> None:
    promoter = torch.zeros(2, 400, 5)
    promoter[:, :, 4] = 1.0
    expr = torch.rand(2, 5)

    for model_name in MODEL_REGISTRY:
        model = build_model(model_name, expr_dim=5, hidden_size=4, output_mode="scalar")
        output = model(promoter, expr)
        assert output.shape == (2, 1), model_name


def test_zinb_capable_models_output_shapes() -> None:
    promoter = torch.zeros(2, 400, 5)
    promoter[:, :, 4] = 1.0
    expr = torch.rand(2, 5)

    for model_name in ["LSTMmodel", "ConvAttentionModel", "CNNFlattenPromoterModel"]:
        model = build_model(model_name, expr_dim=5, hidden_size=4, output_mode="zinb")
        mu_ratio, theta, pi = model(promoter, expr)
        assert mu_ratio.shape == (2, 1)
        assert theta.shape == (2, 1)
        assert pi.shape == (2, 1)
        assert torch.all(mu_ratio > 0)
        assert torch.all(theta > 0)
        assert torch.all((pi >= 0) & (pi <= 1))
