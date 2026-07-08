from __future__ import annotations

import pytest
import torch

from scripts.train import build_warmup_constant_scheduler


def test_warmup_scheduler_reaches_base_lr_and_stays_constant() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4)
    scheduler = build_warmup_constant_scheduler(optimizer, warmup_epochs=5)

    observed_lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        observed_lrs.append(optimizer.param_groups[0]["lr"])

    assert observed_lrs[0] == pytest.approx(5e-5)
    assert observed_lrs[5] == pytest.approx(5e-4)
    assert observed_lrs[5:] == pytest.approx([5e-4] * 4)


def test_warmup_scheduler_rejects_nonpositive_duration() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4)

    with pytest.raises(ValueError, match="warmup_epochs must be at least 1"):
        build_warmup_constant_scheduler(optimizer, warmup_epochs=0)
