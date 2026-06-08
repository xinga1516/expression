from __future__ import annotations

import pytest

from scripts.data_sanity import summarize_integrated_data
from scripts.project_test import (
    check_gene_id_alignment,
    check_h5ad,
    check_promoter_csv,
    check_target_values,
    load_h5ad,
)


pytestmark = pytest.mark.sanity


def test_project_test_core_checks_on_tiny_data(tiny_data_dir) -> None:
    adata = load_h5ad(tiny_data_dir / "integrated_data.h5ad")
    h5ad_info = check_h5ad(adata)
    promoter_info = check_promoter_csv(tiny_data_dir / "promoter_train.csv")
    alignment = check_gene_id_alignment(h5ad_info["gene_ids"], promoter_info["gene_ids"], "tiny")

    check_target_values(adata, promoter_info["gene_ids"], n_cells=3, seed=1)

    assert h5ad_info["missing_count"] == 0
    assert h5ad_info["duplicated_count"] == 0
    assert promoter_info["missing_count"] == 0
    assert alignment["missing_from_h5ad"] == []
    assert alignment["ambiguous_in_h5ad"] == []


def test_data_sanity_summary_on_tiny_h5ad(tiny_data_dir) -> None:
    summary = summarize_integrated_data(tiny_data_dir / "integrated_data.h5ad")

    assert summary["matrix"]["n_cells"] == 6
    assert summary["matrix"]["n_genes"] == 5
    assert summary["matrix"]["nnz"] > 0
    assert 0 < summary["matrix"]["density"] < 1
