from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_reused_split_sequence_assets as reuse_assets

pytestmark = pytest.mark.unit


def write_source_promoters(source_dir: Path) -> None:
    rows = {
        "train": [
            {
                "gene_id": "g_plus",
                "chrom": "2R",
                "start": 100,
                "end": 140,
                "strand": "+",
                "sequence": "A" * 40,
                "control_chrom": "2R",
                "control_start": 300,
                "control_end": 340,
                "control_strand": "+",
                "control_sequence": "C" * 40,
            }
        ],
        "val": [
            {
                "gene_id": "g_minus",
                "chrom": "2R",
                "start": 400,
                "end": 440,
                "strand": "-",
                "sequence": "G" * 40,
                "control_chrom": "2R",
                "control_start": 500,
                "control_end": 540,
                "control_strand": "-",
                "control_sequence": "T" * 40,
            }
        ],
        "test": [
            {
                "gene_id": "g_test",
                "chrom": "2R",
                "start": 600,
                "end": 640,
                "strand": "+",
                "sequence": "A" * 40,
                "control_chrom": "2R",
                "control_start": 700,
                "control_end": 740,
                "control_strand": "+",
                "control_sequence": "C" * 40,
            }
        ],
    }
    for split, split_rows in rows.items():
        pd.DataFrame(split_rows).to_csv(source_dir / f"promoter_{split}.csv", index=False)


def test_build_reused_split_assets_keeps_stage1_panels_and_reextracts_sequences(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(reuse_assets, "PROJECT_ROOT", tmp_path)
    source_dir = tmp_path / "data" / "promoter_stage1_v1"
    source_dir.mkdir(parents=True)
    write_source_promoters(source_dir)
    (source_dir / "gene_splits.tsv").write_text("gene_id\tsplit\ng_plus\ttrain\n", encoding="utf-8")
    (source_dir / "cell_train.txt").write_text("cell0\n", encoding="utf-8")
    (source_dir / "cell_val.txt").write_text("cell1\n", encoding="utf-8")
    (source_dir / "cell_test.txt").write_text("cell2\n", encoding="utf-8")
    (source_dir / "input_gene_panel_train.txt").write_text("g_plus\n", encoding="utf-8")
    (source_dir / "cells.tsv").write_text("cell_id\tpanel\ncell0\ttrain\n", encoding="utf-8")
    (source_dir / "frozen_eval_cells.tsv").write_text("cell_id\ncell2\n", encoding="utf-8")
    fasta = tmp_path / "genome.fa"
    genome_sequence = "ACGT" * 300
    fasta.write_text(">2R\n" + genome_sequence + "\n", encoding="utf-8")

    reuse_assets.build_assets(
        argparse.Namespace(
            source_data="promoter_stage1_v1",
            output_data="promoter_stage2_v1",
            genome_fasta=str(fasta),
            sequence_window_length=44,
            positive_shift_bp=4,
        )
    )

    output_dir = tmp_path / "data" / "promoter_stage2_v1"
    train = pd.read_csv(output_dir / "promoter_train.csv")
    val = pd.read_csv(output_dir / "promoter_val.csv")
    report = json.loads((output_dir / "audit_report.json").read_text(encoding="utf-8"))

    assert (output_dir / "input_gene_panel_train.txt").read_text(encoding="utf-8") == "g_plus\n"
    assert (output_dir / "gene_splits.tsv").read_text(encoding="utf-8") == "gene_id\tsplit\ng_plus\ttrain\n"
    assert len(train.loc[0, "sequence"]) == 44
    assert train.loc[0, "start"] == 98
    assert train.loc[0, "end"] == 142
    assert len(train.loc[0, "control_sequence"]) == 44
    assert train.loc[0, "control_start"] == 298
    assert train.loc[0, "control_end"] == 342
    assert len(train.loc[0, "positive_sequence"]) == 44
    assert train.loc[0, "positive_shift_bp"] == 4
    assert len(val.loc[0, "sequence"]) == 44
    assert report["reused_gene_splits"]
    assert report["reused_cell_splits"]
    assert report["reused_input_gene_panel"]
    assert report["sequence_length_min"] == 44
    assert report["control_length_min"] == 44
