from __future__ import annotations

import pandas as pd
import pytest
from Bio.Seq import Seq
from Bio import SeqIO

from scripts.build_utr_stage_assets import extract_utr_windows, read_stop_landmarks


pytestmark = pytest.mark.unit


def test_read_stop_landmarks_and_extract_downstream_windows(tmp_path) -> None:
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(
        "\n".join(
            [
                '2R\tFlyBase\tgene\t1\t120\t.\t+\t.\tgene_id "g_plus"; gene_symbol "Gp";',
                '2R\tFlyBase\tmRNA\t1\t120\t.\t+\t.\tgene_id "g_plus"; transcript_id "txp";',
                '2R\tFlyBase\tCDS\t10\t89\t.\t+\t0\tgene_id "g_plus"; transcript_id "txp";',
                '2R\tFlyBase\tstop_codon\t90\t92\t.\t+\t0\tgene_id "g_plus"; transcript_id "txp";',
                '2R\tFlyBase\tgene\t401\t520\t.\t-\t.\tgene_id "g_minus"; gene_symbol "Gm";',
                '2R\tFlyBase\tmRNA\t401\t520\t.\t-\t.\tgene_id "g_minus"; transcript_id "txm";',
                '2R\tFlyBase\tCDS\t432\t510\t.\t-\t0\tgene_id "g_minus"; transcript_id "txm";',
                '2R\tFlyBase\tstop_codon\t429\t431\t.\t-\t0\tgene_id "g_minus"; transcript_id "txm";',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fasta = tmp_path / "tiny.fa"
    genome_seq = ("ACGT" * 200)[:800]
    fasta.write_text(">2R\n" + genome_seq + "\n", encoding="utf-8")
    genome = SeqIO.to_dict(SeqIO.parse(str(fasta), "fasta"))
    genes = pd.DataFrame(
        {
            "gene_id": ["g_plus", "g_minus"],
            "split": ["train", "test"],
            "split_strategy": ["stub", "stub"],
            "gene_class": ["protein_coding", "protein_coding"],
        }
    )

    stops = read_stop_landmarks(gtf)
    windows = extract_utr_windows(genes, stops, genome, sequence_length=20).set_index("gene_id")

    assert windows.loc["g_plus", "utr_status"] == "extracted"
    assert windows.loc["g_plus", "utr_start"] == 92
    assert windows.loc["g_plus", "utr_sequence"] == genome_seq[92:112]
    assert windows.loc["g_minus", "utr_status"] == "extracted"
    assert windows.loc["g_minus", "utr_end"] == 428
    expected_minus = str(Seq(genome_seq[408:428]).reverse_complement()).upper()
    assert windows.loc["g_minus", "utr_sequence"] == expected_minus
