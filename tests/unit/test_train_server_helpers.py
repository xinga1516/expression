from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train import count_model_parameters, dataloader_worker_kwargs, estimate_batch_input_mib
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
