from __future__ import annotations

import numpy as np
import pytest

from src.dataset import MyDataset
from src.utils import BalancedEpochSubsetSampler, ZeroNonZeroSampler


pytestmark = pytest.mark.unit


def _is_nonzero_sample(dataset: MyDataset, flat_idx: int) -> bool:
    pro_i = flat_idx // dataset.C
    cell_i = flat_idx % dataset.C
    cell_row = int(dataset.cells[cell_i])
    gene_idx = int(dataset.promoter2expr_idx[pro_i])
    return float(dataset.X[cell_row, gene_idx]) != 0.0


def test_balanced_sampler_length_bounds_and_promoter_balance(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    sampler = BalancedEpochSubsetSampler(dataset, samples_per_epoch=8, seed=7)

    indices = list(iter(sampler))
    promoter_counts = np.bincount([idx // dataset.C for idx in indices], minlength=dataset.P)

    assert len(sampler) == 8
    assert len(indices) == 8
    assert min(indices) >= 0
    assert max(indices) < len(dataset)
    assert promoter_counts.max() - promoter_counts.min() <= 1


def test_balanced_sampler_auto_samples_uses_dataset_size_for_small_dataset(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    sampler = BalancedEpochSubsetSampler(dataset, samples_per_epoch=0, seed=7)

    assert len(sampler) == len(dataset)
    assert len(list(iter(sampler))) == len(dataset)


def test_zero_nonzero_sampler_respects_requested_ratio(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    sampler = ZeroNonZeroSampler(dataset, nonzero_ratio=0.5, samples_per_epoch=8, seed=3)

    indices = list(iter(sampler))
    n_nonzero = sum(_is_nonzero_sample(dataset, idx) for idx in indices)

    assert len(indices) == 8
    assert n_nonzero == 4


def test_zero_nonzero_sampler_includes_duplicate_gene_windows(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    sampler = ZeroNonZeroSampler(dataset, nonzero_ratio=1.0, samples_per_epoch=4, seed=5)
    nz = set(sampler.nz_indices.tolist())

    assert 0 * dataset.C + 0 in nz
    assert 1 * dataset.C + 0 in nz


def test_zero_nonzero_sampler_rebuild_after_cell_resample(tiny_data_dir) -> None:
    dataset = MyDataset(
        tiny_data_dir / "promoter_train.csv",
        tiny_data_dir / "integrated_data.h5ad",
        mode="train",
        cell_ratio=0.5,
        seed=1,
    )
    sampler = ZeroNonZeroSampler(dataset, nonzero_ratio=0.5, samples_per_epoch=4, seed=5)
    before_c = sampler.C

    dataset.resample_cells(seed=9)
    sampler.rebuild(dataset)

    assert sampler.C == dataset.C == before_c
    assert sampler.total_len == dataset.P * dataset.C


def test_zero_nonzero_sampler_replace_false_raises_when_pool_too_small(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    sampler = ZeroNonZeroSampler(
        dataset,
        nonzero_ratio=1.0,
        samples_per_epoch=len(dataset),
        seed=5,
        replace=False,
    )

    with pytest.raises(ValueError):
        list(iter(sampler))
