from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.dataset import MyDataset, PromoterOneHotEncoder


pytestmark = pytest.mark.unit


def test_promoter_one_hot_encoder_encodes_pads_and_truncates() -> None:
    encoder = PromoterOneHotEncoder(length=6)
    encoded = encoder("ACGTNXZ")

    assert encoded.shape == (6, 5)
    assert torch.equal(encoded[0], torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32))
    assert torch.equal(encoded[1], torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32))
    assert torch.equal(encoded[2], torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32))
    assert torch.equal(encoded[3], torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32))
    assert torch.equal(encoded[4], torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32))
    assert torch.equal(encoded[5], torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32))


def test_dataset_masks_target_and_returns_target(tiny_data_dir) -> None:
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=False,
    )

    promoter, expr, y = dataset.in_getitem(pro_i=0, cell_i=0)
    target_idx = int(dataset.promoter2expr_idx[0])

    assert promoter.shape == (400, 5)
    assert y.item() == pytest.approx(10.0)
    assert expr[target_idx].item() == 0.0
    assert expr.sum().item() == pytest.approx(6.0)


def test_dataset_log1p_cpm_target_uses_masked_library_size(tiny_data_dir) -> None:
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        log1p_cpm_target=True,
    )

    _promoter, _expr, y = dataset.in_getitem(pro_i=0, cell_i=0)
    expected = math.log1p(10.0 / 6.0 * 1e6)
    assert y.item() == pytest.approx(expected)


def test_dynamic_and_preencoded_promoter_tensors_match(tiny_data_dir) -> None:
    dynamic = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=False,
    )
    preencoded = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=True,
    )

    assert dynamic.promoter_tensor is None
    assert preencoded.promoter_tensor is not None
    assert torch.equal(dynamic.get_promoter_tensor(1), preencoded.get_promoter_tensor(1))
    assert torch.equal(dynamic[0][0], preencoded[0][0])


def test_resample_cells_updates_subset_size(tiny_data_dir) -> None:
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        mode="train",
        cell_ratio=0.5,
        seed=1,
    )
    before = dataset.cells.copy()

    dataset.resample_cells(seed=2)

    assert dataset.C == max(1, int(len(dataset._original_cells) * 0.5))
    assert len(dataset.cells) == dataset.C
    assert not np.array_equal(before, dataset.cells)
