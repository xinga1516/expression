from __future__ import annotations

import pytest

from src.dataset import MyDataset
from src.utils import ZeroNonZeroSampler


pytestmark = pytest.mark.regression


def test_augmented_windows_for_same_gene_all_enter_nonzero_pool(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    sampler = ZeroNonZeroSampler(dataset, nonzero_ratio=1.0, samples_per_epoch=4, seed=1)
    nz = set(sampler.nz_indices.tolist())

    for cell_pos in [0, 1, 3, 5]:
        assert 0 * dataset.C + cell_pos in nz
        assert 1 * dataset.C + cell_pos in nz
