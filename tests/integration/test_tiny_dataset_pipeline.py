from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import MyDataset
from src.model import build_model


pytestmark = pytest.mark.integration


def test_tiny_dataset_dataloader_model_forward(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    promoters, exprs, ys = next(iter(loader))
    model = build_model("LSTMmodel", expr_dim=exprs.shape[1], hidden_size=4, output_mode="scalar")

    out = model(promoters, exprs).squeeze(1)

    assert promoters.shape == (4, 400, 5)
    assert exprs.shape == (4, 5)
    assert ys.shape == (4,)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
