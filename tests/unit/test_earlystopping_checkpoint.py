from __future__ import annotations

import pytest
import torch

from scripts.train import compute_val_loss_ema, should_save_best_model
from src.earlystopping import EarlyStopping
import src.utils as utils


pytestmark = pytest.mark.unit


def test_early_stopping_respects_min_delta_and_patience() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.1)

    stopper(1.0)
    stopper(0.95)
    assert stopper.best_score == pytest.approx(1.0)
    assert stopper.counter == 1

    stopper(0.8)
    assert stopper.best_score == pytest.approx(0.8)
    assert stopper.counter == 0

    stopper(0.79)
    stopper(0.78)
    assert stopper.early_stop


def test_ema_helper_and_best_model_decision_lock_current_behavior() -> None:
    ema0 = compute_val_loss_ema(None, monitor_loss=1.0, ema_alpha=0.9)
    ema1 = compute_val_loss_ema(ema0, monitor_loss=0.8, ema_alpha=0.9)
    ema2 = compute_val_loss_ema(ema1, monitor_loss=1.2, ema_alpha=0.9)

    assert ema0 == pytest.approx(1.0)
    assert ema1 == pytest.approx(0.98)
    assert ema2 == pytest.approx(1.002)

    stopper = EarlyStopping(patience=3, min_delta=0.01)
    stopper(ema0)
    stopper(ema1)
    assert should_save_best_model(ema1, stopper.best_score)

    stopper(ema2)
    assert not should_save_best_model(ema2, stopper.best_score)


def test_save_and_resume_checkpoint_round_trip(tmp_path) -> None:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    stopper = EarlyStopping(patience=2)
    stopper(1.2)
    checkpoint = tmp_path / "last.ckpt"

    utils.save_checkpoint(checkpoint, 3, model, optimizer, scheduler, stopper, [2.0], [1.2])

    new_model = torch.nn.Linear(3, 1)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
    new_stopper = EarlyStopping(patience=2)
    start_epoch, loaded_stopper, train_losses, val_losses = utils.resume_from_checkpoint(
        checkpoint,
        new_model,
        new_optimizer,
        new_scheduler,
        new_stopper,
        torch.device("cpu"),
    )

    assert start_epoch == 4
    assert loaded_stopper.best_score == pytest.approx(1.2)
    assert train_losses == [2.0]
    assert val_losses == [1.2]


def test_robust_save_model_writes_safetensors(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    output = tmp_path / "best_model.safetensors"

    utils.robust_save_model(model, output)

    assert output.exists()
    assert not output.with_suffix(".tmp").exists()


def test_robust_save_model_removes_temp_file_on_failure(tmp_path, monkeypatch) -> None:
    model = torch.nn.Linear(2, 1)
    output = tmp_path / "best_model.safetensors"

    def fail_save_file(_state, temp_path) -> None:
        temp_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("forced failure")

    monkeypatch.setattr(utils, "save_file", fail_save_file)

    with pytest.raises(RuntimeError):
        utils.robust_save_model(model, output)

    assert not output.exists()
    assert not output.with_suffix(".tmp").exists()
