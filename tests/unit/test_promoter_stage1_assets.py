from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.build_promoter_stage1_assets as stage1_assets
from scripts.build_promoter_stage1_assets import (
    add_intergenic_controls,
    build_assets,
    choose_gene_splits,
    read_gtf_gene_table,
    select_input_gene_panel,
)


pytestmark = pytest.mark.unit


def test_read_gtf_gene_table_marks_only_protein_coding_stage1_eligible(tmp_path) -> None:
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(
        "\n".join(
            [
                '2R\tFlyBase\tgene\t1\t100\t.\t+\t.\tgene_id "g_pc"; gene_symbol "Pc";',
                '2R\tFlyBase\tmRNA\t1\t100\t.\t+\t.\tgene_id "g_pc"; gene_symbol "Pc"; transcript_id "tx_pc"; transcript_symbol "Pc-RA";',
                '2R\tFlyBase\tCDS\t20\t80\t.\t+\t0\tgene_id "g_pc"; gene_symbol "Pc"; transcript_id "tx_pc"; transcript_symbol "Pc-RA";',
                '2R\tFlyBase\tgene\t201\t300\t.\t+\t.\tgene_id "g_lnc"; gene_symbol "lncRNA:Foo";',
                '2R\tFlyBase\tncRNA\t201\t300\t.\t+\t.\tgene_id "g_lnc"; gene_symbol "lncRNA:Foo"; transcript_id "tx_lnc"; transcript_symbol "lncRNA:Foo-RA";',
                '2R\tFlyBase\tgene\t401\t450\t.\t+\t.\tgene_id "g_trna"; gene_symbol "tRNA:Foo";',
                '2R\tFlyBase\ttRNA\t401\t450\t.\t+\t.\tgene_id "g_trna"; gene_symbol "tRNA:Foo"; transcript_id "tx_t"; transcript_symbol "tRNA:Foo-RA";',
                '2R\tFlyBase\tgene\t501\t550\t.\t+\t.\tgene_id "g_mir"; gene_symbol "mir";',
                '2R\tFlyBase\tmiRNA\t501\t550\t.\t+\t.\tgene_id "g_mir"; gene_symbol "mir"; transcript_id "tx_m"; transcript_symbol "mir-RA";',
                '2R\tFlyBase\tgene\t601\t650\t.\t+\t.\tgene_id "g_pseudo"; gene_symbol "Pseudo";',
                '2R\tFlyBase\tpseudogene\t601\t650\t.\t+\t.\tgene_id "g_pseudo"; gene_symbol "Pseudo"; transcript_id "tx_p"; transcript_symbol "Pseudo-RA";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    genes = read_gtf_gene_table(gtf).set_index("gene_id")

    assert genes.loc["g_pc", "gene_class"] == "protein_coding"
    assert bool(genes.loc["g_pc", "is_stage1_promoter_eligible"])
    assert genes.loc["g_lnc", "gene_class"] == "lncRNA_candidate"
    assert not bool(genes.loc["g_lnc", "is_stage1_promoter_eligible"])
    assert genes.loc["g_lnc", "excluded_reason"] == "lncRNA_candidate_not_used_in_stage1"
    for gene_id in ["g_trna", "g_mir", "g_pseudo"]:
        assert not bool(genes.loc[gene_id, "is_stage1_promoter_eligible"])
        assert genes.loc[gene_id, "excluded_reason"]


def test_choose_gene_splits_are_mutually_exclusive() -> None:
    genes = pd.DataFrame(
        {
            "gene_id": [f"g{i}" for i in range(20)],
            "contig": ["2R"] * 10 + ["3R"] * 10,
            "strand": ["+"] * 20,
            "tss": list(range(20)),
            "gene_class": ["protein_coding"] * 20,
        }
    )

    split = choose_gene_splits(genes, val_contig="2L", test_contig="3L", min_contig_genes=5, seed=1)

    groups = {name: set(df["gene_id"]) for name, df in split.groupby("split")}
    assert groups.get("train", set()).isdisjoint(groups.get("val", set()))
    assert groups.get("train", set()).isdisjoint(groups.get("test", set()))
    assert groups.get("val", set()).isdisjoint(groups.get("test", set()))


def test_add_intergenic_controls_writes_control_sequence(tmp_path) -> None:
    genome = tmp_path / "tiny.fa"
    genome.write_text(">2R\n" + ("ACGT" * 300) + "\n", encoding="utf-8")
    gtf_genes = pd.DataFrame(
        {
            "gene_id": ["g0"],
            "contig": ["2R"],
            "start0": [10],
            "end0": [30],
        }
    )
    promoters = pd.DataFrame(
        {
            "gene_id": ["g0"],
            "chrom": ["2R"],
            "start": [100],
            "end": [140],
            "strand": ["+"],
            "sequence": ["ACGT" * 10],
            "length": [40],
        }
    )

    with_controls = add_intergenic_controls(promoters, genome, gtf_genes, seed=3, attempts=200)

    assert with_controls.loc[0, "match_status"] == "matched"
    assert len(with_controls.loc[0, "control_sequence"]) == 40
    assert not (10 < int(with_controls.loc[0, "control_end"]) and int(with_controls.loc[0, "control_start"]) < 30)


def test_select_input_gene_panel_uses_train_cells_and_scanpy_hvg(monkeypatch, tiny_adata) -> None:
    called: dict[str, object] = {}

    def fake_highly_variable_genes(adata: Any, n_top_genes: int, flavor: str, inplace: bool) -> None:
        called["shape"] = adata.shape
        called["n_top_genes"] = n_top_genes
        called["flavor"] = flavor
        adata.var["highly_variable"] = [True, False, True, False]
        adata.var["highly_variable_rank"] = [1.0, np.nan, 0.0, np.nan]

    monkeypatch.setattr(
        stage1_assets.sc.pp,
        "highly_variable_genes",
        fake_highly_variable_genes,
    )

    panel = select_input_gene_panel(
        tiny_adata,
        train_gene_ids=["g0", "g1", "g2", "g3"],
        max_genes=2,
        train_cell_ids=["cell0", "cell2"],
        method="hvg",
        hvg_flavor="cell_ranger",
    )

    assert called["shape"] == (2, 4)
    assert called["n_top_genes"] == 2
    assert called["flavor"] == "cell_ranger"
    assert panel == ["g2", "g0"]


def test_build_assets_matches_controls_before_split(monkeypatch, tmp_path, tiny_adata) -> None:
    monkeypatch.setattr(stage1_assets, "PROJECT_ROOT", tmp_path)
    source_dir = tmp_path / "data" / "source"
    source_dir.mkdir(parents=True)
    output_name = "stage1"
    tiny_adata.write_h5ad(source_dir / "integrated_data.h5ad")
    promoters = pd.DataFrame(
        {
            "gene_id": ["g0", "g1", "g2"],
            "chrom": ["2R", "2R", "2R"],
            "start": [100, 200, 300],
            "end": [140, 240, 340],
            "strand": ["+", "+", "+"],
            "sequence": ["ACGT" * 10, "CGTA" * 10, "GTAC" * 10],
        }
    )
    promoters.to_csv(source_dir / "promoter_train.csv", index=False)
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(
        "\n".join(
            [
                '2R\tFlyBase\tgene\t1\t80\t.\t+\t.\tgene_id "g0"; gene_symbol "G0";',
                '2R\tFlyBase\tmRNA\t1\t80\t.\t+\t.\tgene_id "g0"; gene_symbol "G0"; transcript_id "tx0";',
                '2R\tFlyBase\tCDS\t10\t60\t.\t+\t0\tgene_id "g0"; gene_symbol "G0"; transcript_id "tx0";',
                '2R\tFlyBase\tgene\t81\t160\t.\t+\t.\tgene_id "g1"; gene_symbol "G1";',
                '2R\tFlyBase\tmRNA\t81\t160\t.\t+\t.\tgene_id "g1"; gene_symbol "G1"; transcript_id "tx1";',
                '2R\tFlyBase\tCDS\t90\t140\t.\t+\t0\tgene_id "g1"; gene_symbol "G1"; transcript_id "tx1";',
                '2R\tFlyBase\tgene\t161\t240\t.\t+\t.\tgene_id "g2"; gene_symbol "G2";',
                '2R\tFlyBase\tmRNA\t161\t240\t.\t+\t.\tgene_id "g2"; gene_symbol "G2"; transcript_id "tx2";',
                '2R\tFlyBase\tCDS\t170\t220\t.\t+\t0\tgene_id "g2"; gene_symbol "G2"; transcript_id "tx2";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    genome = tmp_path / "tiny.fa"
    genome.write_text(">2R\n" + ("ACGT" * 200) + "\n", encoding="utf-8")

    def fake_add_intergenic_controls(
        promoters_df: pd.DataFrame,
        genome_fasta: Path,
        gtf_genes: pd.DataFrame,
        seed: int,
        attempts: int,
    ) -> pd.DataFrame:
        controls = pd.DataFrame(
            {
                "control_id": ["c0", "c1", ""],
                "control_kind": ["matched_intergenic", "matched_intergenic", ""],
                "control_chrom": ["2R", "2R", ""],
                "control_start": [400, 500, np.nan],
                "control_end": [440, 540, np.nan],
                "control_strand": ["+", "+", ""],
                "control_sequence": ["T" * 40, "A" * 40, ""],
                "promoter_gc": [0.5, 0.5, 0.5],
                "control_gc": [0.0, 0.0, np.nan],
                "gc_diff": [0.5, 0.5, np.nan],
                "match_status": ["matched", "matched", "no_nonoverlap_candidate"],
            }
        )
        return pd.concat([promoters_df.reset_index(drop=True), controls], axis=1)

    def fake_choose_gene_splits(
        eligible_genes: pd.DataFrame,
        val_contig: str,
        test_contig: str,
        min_contig_genes: int,
        seed: int,
    ) -> pd.DataFrame:
        assert set(eligible_genes["gene_id"]) == {"g0", "g1"}
        split = eligible_genes.copy()
        split["split"] = ["train", "test"]
        split["split_strategy"] = "test_stub"
        return split

    monkeypatch.setattr(stage1_assets, "add_intergenic_controls", fake_add_intergenic_controls)
    monkeypatch.setattr(stage1_assets, "choose_gene_splits", fake_choose_gene_splits)

    build_assets(
        argparse.Namespace(
            source_data="source",
            output_data=output_name,
            gtf=str(gtf),
            genome_fasta=str(genome),
            val_contig="2L",
            test_contig="3L",
            min_contig_genes=200,
            input_gene_panel_size=2,
            input_gene_panel_method="variance",
            hvg_flavor="cell_ranger",
            max_eval_cells=10,
            control_attempts=10,
            seed=42,
        )
    )

    output_dir = tmp_path / "data" / output_name
    splits = pd.read_csv(output_dir / "gene_splits.tsv", sep="\t")
    genes = pd.read_csv(output_dir / "genes.tsv", sep="\t")

    assert set(splits["gene_id"]) == {"g0", "g1"}
    assert genes.set_index("gene_id").loc["g2", "excluded_reason"] == "missing_matched_intergenic_control"
