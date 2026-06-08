from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.model_test import (
    parse_meme_motifs,
    select_top_expressed_pairs,
    shuffle_expression_batch,
    shuffle_promoter_batch,
    update_spearman_reservoir,
    write_de_novo_motif_outputs,
    write_known_motif_outputs,
)
from src.dataset import MyDataset


pytestmark = pytest.mark.unit


def test_select_top_expressed_pairs_respects_gene_cap(tiny_data_dir) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")

    top_pairs = select_top_expressed_pairs(dataset, top_n=5, max_pairs_per_gene_ratio=0.4)

    assert len(top_pairs) <= 5
    assert top_pairs.groupby("gene_id").size().max() <= 2
    assert int(top_pairs["max_pairs_per_gene"].iloc[0]) == 2


def test_update_spearman_reservoir_all_samples_and_limited_samples() -> None:
    rng = np.random.default_rng(42)
    y_true = np.arange(10, dtype=np.float64)
    y_pred = y_true[::-1].copy()
    reservoir_true: list[np.ndarray] = []
    reservoir_pred: list[np.ndarray] = []

    seen = update_spearman_reservoir(y_true, y_pred, reservoir_true, reservoir_pred, 0, 0, rng)
    assert seen == 10
    assert np.array_equal(np.concatenate(reservoir_true), y_true.astype(np.float32))

    reservoir_true = []
    reservoir_pred = []
    seen = update_spearman_reservoir(y_true, y_pred, reservoir_true, reservoir_pred, 0, 4, rng)
    assert seen == 10
    assert len(np.concatenate(reservoir_true)) == 4
    assert len(np.concatenate(reservoir_pred)) == 4


def test_shuffle_promoter_batch_preserves_one_hot_and_base_counts() -> None:
    promoters = torch.zeros(2, 8, 5)
    promoters[0, torch.arange(8), torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])] = 1.0
    promoters[1, torch.arange(8), torch.tensor([4, 3, 2, 1, 0, 4, 3, 2])] = 1.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)

    shuffled = shuffle_promoter_batch(promoters, generator)

    assert shuffled.shape == promoters.shape
    assert torch.allclose(shuffled.sum(dim=2), torch.ones(2, 8))
    assert torch.allclose(shuffled.sum(dim=1), promoters.sum(dim=1))


def test_shuffle_expression_batch_preserves_rows() -> None:
    exprs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(11)

    shuffled = shuffle_expression_batch(exprs, generator)

    assert shuffled.shape == exprs.shape
    assert sorted(map(tuple, shuffled.tolist())) == sorted(map(tuple, exprs.tolist()))


def test_parse_meme_motifs_reads_pwm(tiny_meme_file) -> None:
    motifs = parse_meme_motifs(tiny_meme_file)

    assert len(motifs) == 1
    assert motifs[0]["name"] == "toy_acgt"
    assert motifs[0]["pwm"].shape == (4, 4)
    assert motifs[0]["pwm"][0, 0] == pytest.approx(0.9)


def test_write_de_novo_motif_outputs(tiny_data_dir, tiny_mutation_effects, tmp_path) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    effects, top_pairs = tiny_mutation_effects

    write_de_novo_motif_outputs(dataset, effects, top_pairs, tmp_path, 9, 3, 20, 1)

    assert (tmp_path / "motif_windows.csv").exists()
    assert (tmp_path / "de_novo_motifs.csv").exists()
    assert (tmp_path / "important_motif_pwm.csv").exists()
    assert (tmp_path / "important_motif_pwm.png").exists()
    assert len(pd.read_csv(tmp_path / "motif_windows.csv")) == 3


def test_write_known_motif_outputs_with_and_without_file(tiny_data_dir, tiny_mutation_effects, tiny_meme_file, tmp_path) -> None:
    dataset = MyDataset(tiny_data_dir / "promoter_train.csv", tiny_data_dir / "integrated_data.h5ad")
    effects, top_pairs = tiny_mutation_effects

    write_known_motif_outputs(dataset, effects, top_pairs, tmp_path, None, 0.8, 10)
    assert not (tmp_path / "known_motif_hits.csv").exists()

    write_known_motif_outputs(dataset, effects, top_pairs, tmp_path, str(tiny_meme_file), 0.7, 10)

    assert (tmp_path / "known_motif_hits.csv").exists()
    assert (tmp_path / "known_motif_summary.csv").exists()
