from __future__ import annotations

import pytest

from scripts.train import compute_val_loss_ema, should_save_best_model
from src.earlystopping import EarlyStopping


pytestmark = pytest.mark.regression


def test_best_model_selection_uses_ema_not_raw_validation_loss() -> None:
    stopper = EarlyStopping(patience=5, min_delta=0.01)
    ema = compute_val_loss_ema(None, 1.0, 0.9)
    stopper(ema)
    assert should_save_best_model(ema, stopper.best_score)

    raw_val_improves = 0.99
    ema = compute_val_loss_ema(ema, raw_val_improves, 0.9)
    stopper(ema)
    assert not should_save_best_model(ema, stopper.best_score)

    raw_val_worsens_but_ema_improves_from_previous_best = 0.5
    ema = compute_val_loss_ema(ema, raw_val_worsens_but_ema_improves_from_previous_best, 0.9)
    stopper(ema)
    assert should_save_best_model(ema, stopper.best_score)
