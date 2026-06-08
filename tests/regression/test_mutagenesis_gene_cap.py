from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.model_test import select_top_expressed_pairs


pytestmark = pytest.mark.regression


class _DominantGeneDataset:
    def __init__(self) -> None:
        n_cells = 200
        self.P = 8
        self.C = n_cells
        self.cells = np.arange(n_cells, dtype=np.int64)
        self.promoter2expr_idx = np.array([0, 0, 0, 0, 1, 2, 3, 4], dtype=np.int32)
        self.promoters = pd.DataFrame(
            {
                "gene_id": ["dominant"] * 4 + ["g1", "g2", "g3", "g4"],
                "chrom": ["2R"] * 8,
                "start": list(range(8)),
                "end": [400 + i for i in range(8)],
                "strand": ["+"] * 8,
                "sequence": ["A" * 400] * 8,
            }
        )
        x = np.zeros((n_cells, 5), dtype=np.float32)
        x[:, 0] = np.linspace(1000, 1, n_cells)
        x[:, 1] = np.linspace(900, 1, n_cells)
        x[:, 2] = np.linspace(800, 1, n_cells)
        x[:, 3] = np.linspace(700, 1, n_cells)
        x[:, 4] = np.linspace(600, 1, n_cells)
        self.X = sparse.csr_matrix(x)
        self.log1p_cpm_target = False
        self.scrna = type(
            "Scrna",
            (),
            {
                "obs": pd.DataFrame({"sample_id": ["s"] * n_cells}),
                "obs_names": [f"cell{i}" for i in range(n_cells)],
            },
        )()


def test_mutagenesis_top1000_gene_cap_limits_dominant_gene() -> None:
    dataset = _DominantGeneDataset()

    top_pairs = select_top_expressed_pairs(dataset, top_n=1000, max_pairs_per_gene_ratio=0.1)

    assert len(top_pairs) <= 1000
    assert top_pairs.groupby("gene_id").size().max() <= 100
