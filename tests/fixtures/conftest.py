from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def dna_sequence(pattern: str = "ACGT", length: int = 400) -> str:
    return (pattern * ((length // len(pattern)) + 1))[:length]


@pytest.fixture
def tiny_adata() -> ad.AnnData:
    x = sparse.csr_matrix(
        np.array(
            [
                [10.0, 0.0, 5.0, 0.0, 1.0],
                [9.0, 8.0, 0.0, 0.0, 2.0],
                [0.0, 7.0, 3.0, 1.0, 0.0],
                [6.0, 0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 2.0, 5.0, 0.0],
                [3.0, 1.0, 0.0, 0.0, 9.0],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["s0", "s0", "s1", "s1", "s2", "s2"],
            "total_counts": np.asarray(x.sum(axis=1)).ravel(),
        },
        index=[f"cell{i}" for i in range(x.shape[0])],
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"g{i}" for i in range(x.shape[1])],
            "gene_symbol": [f"G{i}" for i in range(x.shape[1])],
        },
        index=[f"g{i}" for i in range(x.shape[1])],
    )
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers["counts"] = x.copy()
    dense = x.toarray().astype(np.float32)
    totals = np.maximum(dense.sum(axis=1, keepdims=True), 1.0)
    log_cpm = np.log1p(dense / totals * 1e6).astype(np.float32)
    adata.layers["cpm"] = sparse.csr_matrix(log_cpm)
    adata.layers["logcpm"] = sparse.csr_matrix(log_cpm)
    return adata


@pytest.fixture
def tiny_promoters_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_id": ["g0", "g0", "g1", "g2"],
            "chrom": ["2R", "2R", "2R", "3R"],
            "start": [10, 12, 20, 30],
            "end": [410, 412, 420, 430],
            "strand": ["+", "+", "-", "+"],
            "sequence": [
                dna_sequence("ACGT"),
                dna_sequence("CGTA"),
                dna_sequence("TATA"),
                dna_sequence("GGCA"),
            ],
            "length": [400, 400, 400, 400],
            "augment_offset": [0, 1, 0, 0],
        }
    )


@pytest.fixture
def tiny_data_dir(tmp_path: Path, tiny_adata: ad.AnnData, tiny_promoters_df: pd.DataFrame) -> Path:
    data_dir = tmp_path / "tiny_data"
    data_dir.mkdir()
    tiny_adata.write_h5ad(data_dir / "integrated_data.h5ad")
    tiny_promoters_df.to_csv(data_dir / "promoter_train.csv", index=False)
    tiny_promoters_df.iloc[:3].to_csv(data_dir / "promoter_val.csv", index=False)
    tiny_promoters_df.iloc[1:].to_csv(data_dir / "promoter_test.csv", index=False)
    (data_dir / "sample_tissue.json").write_text('{"s0": "tissue0", "s1": "tissue1", "s2": "tissue2"}')
    return data_dir


@pytest.fixture
def tiny_genome_fasta(tmp_path: Path) -> Path:
    fasta = tmp_path / "tiny_genome.fa"
    fasta.write_text(
        f">2R\n{dna_sequence('ACGT', 500)}\n"
        f">3R\n{dna_sequence('TGCA', 500)}\n",
        encoding="utf-8",
    )
    return fasta


@pytest.fixture
def tiny_meme_file(tmp_path: Path) -> Path:
    meme = tmp_path / "tiny_motifs.meme"
    meme.write_text(
        "\n".join(
            [
                "MEME version 4",
                "",
                "ALPHABET= ACGT",
                "",
                "MOTIF toy_acgt",
                "letter-probability matrix: alength= 4 w= 4 nsites= 10 E= 0",
                "0.90 0.03 0.03 0.04",
                "0.03 0.90 0.03 0.04",
                "0.03 0.03 0.90 0.04",
                "0.03 0.03 0.04 0.90",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return meme


@pytest.fixture
def tiny_mutation_effects() -> tuple[pd.DataFrame, pd.DataFrame]:
    effects: list[dict[str, Any]] = []
    top_pairs = pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "pro_i": [0, 1, 2],
            "gene_id": ["g0", "g0", "g1"],
            "cell_id": ["cell0", "cell1", "cell2"],
        }
    )
    for rank, pro_i, gene_id, cell_id in top_pairs.itertuples(index=False):
        for pos in range(400):
            for alt in ["A", "C", "G"]:
                delta = 10.0 if pos == rank + 3 else 0.1
                effects.append(
                    {
                        "rank": int(rank),
                        "pro_i": int(pro_i),
                        "cell_row": int(rank) - 1,
                        "cell_id": str(cell_id),
                        "gene_id": str(gene_id),
                        "target_score": 1.0,
                        "raw_target": 1.0,
                        "wt_pred": 1.0,
                        "position_0based": pos,
                        "position_1based": pos + 1,
                        "ref_base": "T",
                        "alt_base": alt,
                        "mut_pred": 1.0 + delta,
                        "delta": delta,
                        "abs_delta": abs(delta),
                    }
                )
    return pd.DataFrame(effects), top_pairs
