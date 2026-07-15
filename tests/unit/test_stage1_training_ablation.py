from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.stage1_training_ablation import load_paired_deltas, write_two_run_ablation_outputs
from scripts.stage2_contrastive_ablation import write_violin_outputs


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


def test_write_two_run_ablation_outputs_reuses_paired_contract(tmp_path) -> None:
    for run_name, values in (("baseline", [0.1, 0.2]), ("treatment", [0.3, 0.4])):
        run_dir = tmp_path / run_name
        test_dir = run_dir / "test"
        test_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps({"seed": 7, "contrastive_weight": 0.4 if run_name == "treatment" else 0.0}),
            encoding="utf-8",
        )
        (test_dir / "test_metrics.json").write_text(
            json.dumps(
                {
                    "mse": 1.0,
                    "rmse": 1.0,
                    "pearson_r": values[0],
                    "spearman_r": values[1],
                    "nonzero_rmse": 1.1,
                    "zero_rmse": 0.9,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame({"gene_id": ["g1", "g2"], "pearson_r": values}).to_csv(
            test_dir / "per_gene_metrics.csv", index=False
        )
        pd.DataFrame({"cell_id": ["c1", "c2"], "pearson_r": values}).to_csv(
            test_dir / "per_cell_metrics.csv", index=False
        )

    output_dir = tmp_path / "summary"
    result = write_two_run_ablation_outputs(
        runs_root=tmp_path,
        output_dir=output_dir,
        baseline_run="baseline",
        treatment_run="treatment",
        baseline_label="cw=0",
        treatment_label="cw=0.4",
        comparison="contrastive",
        title="test",
        repeats=20,
        confidence=0.95,
        random_seed=42,
    )

    paired = pd.read_csv(output_dir / "paired_deltas.csv")
    assert result == {"global_rows": 2, "paired_rows": 4, "bootstrap_rows": 2}
    assert paired["pearson_delta"].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert (output_dir / "ablation_summary.png").exists()
    assert (output_dir / "README.md").exists()


def test_write_violin_outputs_for_two_stage2_runs(tmp_path) -> None:
    for run_name, values in (("baseline", [0.1, 0.2]), ("treatment", [0.3, 0.4])):
        test_dir = tmp_path / run_name / "test"
        test_dir.mkdir(parents=True)
        pd.DataFrame({"gene_id": ["g1", "g2"], "pearson_r": values}).to_csv(
            test_dir / "per_gene_metrics.csv", index=False
        )
        pd.DataFrame({"cell_id": ["c1", "c2"], "pearson_r": values}).to_csv(
            test_dir / "per_cell_metrics.csv", index=False
        )

    output_dir = tmp_path / "summary"
    result = write_violin_outputs(
        runs_root=tmp_path,
        output_dir=output_dir,
        baseline_run="baseline",
        treatment_run="treatment",
        baseline_label="cw=0",
        treatment_label="cw=0.4",
        seed=7,
        random_seed=42,
        extreme_count=1,
    )

    assert result == {"violin_rows": 8, "extreme_rows": 8}
    assert (output_dir / "stage2_per_gene_pearson_violin_compare.png").exists()
    assert (output_dir / "stage2_per_cell_pearson_violin_compare.svg").exists()
