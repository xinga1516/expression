from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from src.gpu_cache import GpuCachedPairLoader

pytestmark = pytest.mark.unit


class TinyCachedDataset:
    def __init__(self) -> None:
        self.P = 2
        self.C = 3
        self.expr_dim = 3
        self.sequence_length = 4
        self.promoter_shift_max = 1
        self.sequence_column = "sequence"
        self.mode = "train"
        self.return_indices = True
        self.log1p_cpm_target = False
        self.promoters = pd.DataFrame(
            {
                "sequence": ["AACCGG", "TTGGCC"],
                "positive_sequence": ["CCGGAA", "GGCCAA"],
                "control_sequence": ["GGTTAA", "CCAATT"],
            }
        )
        self.cells = np.arange(self.C, dtype=np.int64)
        self.expression_X = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        self.X = self.expression_X.copy()
        self.target_value_X = self.expression_X.copy()
        self.promoter2expr_idx = np.asarray([0, 1], dtype=np.int64)
        self.input_expr_indices = np.asarray([0, 1, 2], dtype=np.int64)
        self.input_gene_idx_to_position = {0: 0, 1: 1, 2: 2}

    def uses_runtime_sequence_shift(self) -> bool:
        return True

    def _apply_expression_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


def test_gpu_cached_pair_loader_returns_contrastive_cached_sequences() -> None:
    dataset = TinyCachedDataset()
    loader = GpuCachedPairLoader(
        dataset,
        batch_size=2,
        device="cpu",
        samples_per_epoch=4,
        seed=3,
        sampler_mode="random",
        contrastive_positive_column="positive_sequence",
        contrastive_negative_column="control_sequence",
        contrastive_negative_shift_max=1,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)

    batch = loader._make_batch_from_pair_indices(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 2], dtype=torch.long),
        generator=generator,
    )

    assert len(batch) == 7
    promoters, exprs, ys, pro_indices, cell_indices, positive, negative = batch
    assert promoters.shape == (2, 4, 5)
    assert positive.shape == (2, 4, 5)
    assert negative.shape == (2, 4, 5)
    assert pro_indices.tolist() == [0, 1]
    assert cell_indices.tolist() == [0, 2]
    assert ys.tolist() == [1.0, 8.0]
    assert exprs[0, 0].item() == 0.0
    assert exprs[1, 1].item() == 0.0


def test_cached_intergenic_negative_uses_requested_random_crop() -> None:
    dataset = TinyCachedDataset()
    loader = GpuCachedPairLoader(
        dataset,
        batch_size=2,
        device="cpu",
        samples_per_epoch=4,
        seed=3,
        sampler_mode="random",
        contrastive_negative_column="control_sequence",
        contrastive_negative_shift_max=1,
    )
    assert loader.negative_promoters_device is not None
    full_negative = loader.negative_promoters_device.index_select(
        0, torch.tensor([0, 1], dtype=torch.long)
    )

    with patch(
        "src.gpu_cache.torch.randint",
        return_value=torch.tensor([0, 2], dtype=torch.long),
    ) as mocked_randint:
        cropped_negative = loader._crop_sequences_for_model(
            full_negative,
            shift_max=1,
        )

    mocked_randint.assert_called_once()
    assert torch.equal(cropped_negative[0], full_negative[0, :4])
    assert torch.equal(cropped_negative[1], full_negative[1, 2:])
