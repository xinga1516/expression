from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from scripts.model_test import compute_input_ablation_metrics, compute_test_metrics, run_sequence_mutagenesis
from src.dataset import MyDataset
from src.model import build_model


pytestmark = pytest.mark.integration


def test_model_test_metrics_and_tiny_mutagenesis_outputs(tiny_data_dir, tmp_path) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_test.csv", tiny_data_dir / "integrated_data.h5ad")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    model = build_model("LSTMmodel", expr_dim=dataset.X.shape[1], hidden_size=4, output_mode="scalar")

    metrics = compute_test_metrics(
        model=model,
        loader=loader,
        device=next(model.parameters()).device,
        max_samples=8,
        spearman_samples=8,
        seed=42,
    )

    assert metrics["num_samples"] == 8
    assert "mse" in metrics
    assert "pearson_r" in metrics
    assert "spearman_r" in metrics

    ablation_metrics = compute_input_ablation_metrics(
        model=model,
        loader=loader,
        device=next(model.parameters()).device,
        max_samples=8,
        spearman_samples=8,
        seed=42,
        repeats=2,
    )

    assert set(ablation_metrics["condition"]) == {
        "original",
        "shuffle_promoter",
        "shuffle_expression",
        "shuffle_both",
    }
    for column in ["mse", "pearson_r", "spearman_r", "mean_abs_pred_delta", "repeat"]:
        assert column in ablation_metrics.columns

    run_sequence_mutagenesis(
        model=model,
        dataset=dataset,
        output_dir=tmp_path,
        top_n=4,
        mutation_batch_size=128,
        max_pairs_per_gene_ratio=0.5,
        motif_window_size=9,
        motif_top_windows=4,
        motif_top_k=10,
        motif_min_support=1,
        known_motif_file=None,
        known_motif_min_score_ratio=0.8,
        known_motif_max_hits_per_motif=10,
        device=next(model.parameters()).device,
    )

    mut_dir = tmp_path / "sequence_mutagenesis"
    assert (mut_dir / "top4_pairs.csv").exists()
    assert (mut_dir / "mutation_effects_long.csv").exists()
    assert (mut_dir / "position_importance.csv").exists()
    assert (mut_dir / "motif_windows.csv").exists()
    assert (mut_dir / "de_novo_motifs.csv").exists()
