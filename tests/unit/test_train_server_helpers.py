from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train import (
    apply_vae_fine_tune_schedule,
    count_vae_parameters,
    count_model_parameters,
    dataloader_worker_kwargs,
    estimate_batch_input_mib,
    set_vae_trainable,
)
from src.utils import (
    get_vae_fine_tune_start_epoch_from_log,
    plot_loss_curves_from_logfile,
    plot_val_metrics,
    plot_zero_nonzero_loss_curves,
)
from src.model import build_model


pytestmark = pytest.mark.unit


def test_dataloader_worker_kwargs_for_zero_and_worker_modes() -> None:
    dataset = TensorDataset(torch.arange(8))

    no_worker_kwargs = dataloader_worker_kwargs(num_workers=0, prefetch_factor=2)
    assert no_worker_kwargs == {}
    no_worker_loader = DataLoader(dataset, batch_size=2, num_workers=0, **no_worker_kwargs)
    assert len(next(iter(no_worker_loader))) == 1

    worker_kwargs = dataloader_worker_kwargs(num_workers=1, prefetch_factor=2)
    assert worker_kwargs == {"persistent_workers": True, "prefetch_factor": 2}
    worker_loader = DataLoader(dataset, batch_size=2, num_workers=1, **worker_kwargs)
    assert len(next(iter(worker_loader))) == 1


def test_resource_summary_helpers_return_expected_values() -> None:
    model = build_model("SimpleGeneModel", expr_dim=5, hidden_size=4)

    total, trainable = count_model_parameters(model)
    input_mib = estimate_batch_input_mib(batch_size=2, expr_dim=5)

    assert total > 0
    assert trainable == total
    assert input_mib == pytest.approx(2 * (400 * 5 + 5) * 4 / (1024 ** 2))


class DummyVAEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vae_encoder = nn.Linear(3, 2)
        self.vae_fine_tune = False


def test_vae_fine_tune_schedule_freezes_and_unfreezes() -> None:
    model = DummyVAEModel()
    original_weight = model.vae_encoder.weight

    assert not set_vae_trainable(model, False)
    assert not any(param.requires_grad for param in model.vae_encoder.parameters())
    assert not model.vae_fine_tune
    assert not model.vae_encoder.training

    before_start = apply_vae_fine_tune_schedule(
        model,
        epoch=2,
        vae_fine_tune_start_epoch=3,
        force_initial_fine_tune=False,
    )
    assert not before_start
    assert not any(param.requires_grad for param in model.vae_encoder.parameters())

    at_start = apply_vae_fine_tune_schedule(
        model,
        epoch=3,
        vae_fine_tune_start_epoch=3,
        force_initial_fine_tune=False,
    )
    assert at_start
    assert all(param.requires_grad for param in model.vae_encoder.parameters())
    assert model.vae_encoder.weight is original_weight
    assert original_weight.requires_grad
    assert model.vae_fine_tune
    assert model.vae_encoder.training
    vae_total, vae_trainable = count_vae_parameters(model)
    assert vae_total == vae_trainable


def test_vae_fine_tune_force_overrides_start_epoch() -> None:
    model = DummyVAEModel()

    trainable = apply_vae_fine_tune_schedule(
        model,
        epoch=0,
        vae_fine_tune_start_epoch=10,
        force_initial_fine_tune=True,
    )

    assert trainable
    assert all(param.requires_grad for param in model.vae_encoder.parameters())


def test_get_vae_fine_tune_start_epoch_from_log_prefers_active_epoch() -> None:
    df = pytest.importorskip("pandas").DataFrame(
        {
            "epoch": [0, 1, 2, 3],
            "vae_fine_tune_start_epoch": [2, 2, 2, 2],
            "vae_fine_tune_active": [0, 0, 1, 1],
        }
    )

    assert get_vae_fine_tune_start_epoch_from_log(df) == 2


def test_get_vae_fine_tune_start_epoch_from_log_uses_config_fallback() -> None:
    df = pytest.importorskip("pandas").DataFrame(
        {
            "epoch": [0, 1],
            "vae_fine_tune_start_epoch": [5, 5],
        }
    )

    assert get_vae_fine_tune_start_epoch_from_log(df) == 5


def test_loss_plots_accept_vae_fine_tune_marker_columns(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    log_file = tmp_path / "train_log.csv"
    step_log_file = tmp_path / "step_train_loss.csv"
    pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.2, 1.0, 0.9],
            "lr": [1e-4, 1e-4, 1e-4],
            "val_loss_ema": [1.2, 1.1, 1.0],
            "train_loss_zero": [0.2, 0.15, 0.1],
            "train_loss_nonzero": [1.3, 1.1, 0.9],
            "val_loss_zero": [0.25, 0.2, 0.15],
            "val_loss_nonzero": [1.4, 1.2, 1.0],
            "val_pearson_nonzero": [0.0, 0.1, 0.2],
            "val_zero_accuracy": [0.3, 0.4, 0.5],
            "vae_fine_tune_start_epoch": [1, 1, 1],
            "vae_fine_tune_active": [0, 1, 1],
        }
    ).to_csv(log_file, index=False)
    pd.DataFrame(
        {
            "global_step": [0, 1, 2, 3, 4, 5],
            "epoch": [0, 0, 1, 1, 2, 2],
            "train_loss": [1.0, 0.9, 0.8, 0.7, 0.65, 0.6],
        }
    ).to_csv(step_log_file, index=False)

    loss_png = tmp_path / "loss.png"
    zero_png = tmp_path / "zero.png"
    metrics_png = tmp_path / "metrics.png"
    plot_loss_curves_from_logfile(log_file, save_path=loss_png, step_log_file=step_log_file)
    plot_zero_nonzero_loss_curves(log_file, save_path=zero_png)
    plot_val_metrics(log_file, save_path=metrics_png)

    assert loss_png.exists()
    assert zero_png.exists()
    assert metrics_png.exists()
