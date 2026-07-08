from __future__ import annotations

import pandas as pd
import pytest

from scripts.stage1_training_ablation import load_paired_deltas


pytestmark = pytest.mark.unit


def test_load_paired_deltas_matches_ids(tmp_path) -> None:
    run_names = (
        "stage1_shift420_promoter_seed7",
        "stage1_shift420_combined_seed7",
        "stage1_shift420_combined_fixedlr_seed7",
    )
    values = ([0.1, 0.2], [0.2, 0.4], [0.3, 0.5])
    for run_name, pearson in zip(run_names, values):
        test_dir = tmp_path / run_name / "test"
        test_dir.mkdir(parents=True)
        pd.DataFrame({"gene_id": ["g1", "g2"], "pearson_r": pearson}).to_csv(
            test_dir / "per_gene_metrics.csv", index=False
        )
        pd.DataFrame({"cell_id": ["c1", "c2"], "pearson_r": pearson}).to_csv(
            test_dir / "per_cell_metrics.csv", index=False
        )

    paired = load_paired_deltas(tmp_path)

    assert len(paired) == 12
    selected = paired[
        (paired["comparison"] == "combined_vs_mse")
        & (paired["level"] == "per_gene")
    ]
    assert selected["pearson_delta"].tolist() == pytest.approx([0.1, 0.2])
