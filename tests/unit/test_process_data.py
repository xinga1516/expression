from __future__ import annotations

import pandas as pd
import pytest
from Bio.Seq import Seq

from scripts.process_data import augment_promoter_windows, split_train_val


pytestmark = pytest.mark.unit


def test_augment_promoter_windows_creates_shifted_400bp_windows(tiny_genome_fasta) -> None:
    df = pd.DataFrame(
        {
            "gene_id": ["g0"],
            "chrom": ["2R"],
            "start": [10],
            "end": [410],
            "strand": ["+"],
            "sequence": ["N" * 400],
        }
    )

    augmented = augment_promoter_windows(df, tiny_genome_fasta, shift_bp=2)

    assert len(augmented) == 5
    assert set(augmented["augment_offset"]) == {-2, -1, 0, 1, 2}
    assert set(augmented["length"]) == {400}
    assert augmented["sequence"].str.len().eq(400).all()


def test_augment_promoter_windows_reverse_complements_negative_strand(tiny_genome_fasta) -> None:
    df = pd.DataFrame(
        {
            "gene_id": ["g1"],
            "chrom": ["3R"],
            "start": [20],
            "end": [420],
            "strand": ["-"],
            "sequence": ["N" * 400],
        }
    )

    augmented = augment_promoter_windows(df, tiny_genome_fasta, shift_bp=0)
    genome_seq = ("TGCA" * 126)[:500]
    expected = str(Seq(genome_seq[20:420]).reverse_complement()).upper()

    assert augmented.iloc[0]["sequence"] == expected


def test_augment_promoter_windows_skips_out_of_bounds(tiny_genome_fasta) -> None:
    df = pd.DataFrame(
        {
            "gene_id": ["g0"],
            "chrom": ["2R"],
            "start": [1],
            "end": [401],
            "strand": ["+"],
            "sequence": ["N" * 400],
        }
    )

    augmented = augment_promoter_windows(df, tiny_genome_fasta, shift_bp=2)

    assert set(augmented["augment_offset"]) == {-1, 0, 1, 2}


def test_split_train_val_by_gene_keeps_gene_windows_together(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "gene_id": ["g0", "g0", "g1", "g2", "g3", "g4"],
            "chrom": ["2R", "2R", "2R", "2R", "3R", "X"],
            "start": [0, 1, 100, 200, 300, 400],
            "end": [400, 401, 500, 600, 700, 800],
            "strand": ["+"] * 6,
            "sequence": ["A" * 400] * 6,
            "length": [400] * 6,
        }
    )

    train_genes, val_genes, test_genes = split_train_val(
        df,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        output_dir=tmp_path,
        by_gene=True,
    )

    assert set(train_genes).isdisjoint(val_genes)
    assert set(train_genes).isdisjoint(test_genes)
    assert set(val_genes).isdisjoint(test_genes)
    for split_file in ["promoter_train.csv", "promoter_val.csv", "promoter_test.csv"]:
        assert (tmp_path / split_file).exists()
    combined = {
        "train": set(pd.read_csv(tmp_path / "promoter_train.csv")["gene_id"]),
        "val": set(pd.read_csv(tmp_path / "promoter_val.csv")["gene_id"]),
        "test": set(pd.read_csv(tmp_path / "promoter_test.csv")["gene_id"]),
    }
    assert "g0" in combined["train"] or "g0" in combined["val"] or "g0" in combined["test"]
    assert sum("g0" in genes for genes in combined.values()) == 1
