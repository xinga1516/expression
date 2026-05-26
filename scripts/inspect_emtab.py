# -*- coding: utf-8 -*-
"""Inspect the E-MTAB-10519 normalised scRNA-seq dataset.

Data source: ArrayExpress E-MTAB-10519
Format: Matrix Market (.mtx + .mtx_rows + .mtx_cols)
"""
import pathlib
import numpy as np
from scipy.io import mmread

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
#DATA_DIR = PROJECT_ROOT / "data" / "E-MTAB-10519-normalised-files"
DATA_DIR = PROJECT_ROOT / "data" / "E-MTAB-10519-raw"


def inspect() -> None:
    # mtx_path = DATA_DIR / "E-MTAB-10519.aggregated_filtered_normalised_counts.mtx"
    # rows_path = DATA_DIR / "E-MTAB-10519.aggregated_filtered_normalised_counts.mtx_rows"
    # cols_path = DATA_DIR / "E-MTAB-10519.aggregated_filtered_normalised_counts.mtx_cols"
    mtx_path = DATA_DIR / "E-MTAB-10519.aggregated_filtered_counts.mtx"
    rows_path = DATA_DIR / "E-MTAB-10519.aggregated_filtered_counts.mtx_rows"
    cols_path = DATA_DIR / "E-MTAB-10519.aggregated_filtered_counts.mtx_cols"

    # ── Matrix ──
    print("Reading matrix ...")
    X = mmread(str(mtx_path))
    n_genes, n_cells = X.shape
    nz = X.nnz
    density = nz / (n_genes * n_cells) * 100
    values = X.data

    print(f"\n{'='*60}")
    print(f"  E-MTAB-10519 — Aggregated filtered counts")
    print(f"{'='*60}")
    print(f"  Genes:  {n_genes:,}")
    print(f"  Cells:  {n_cells:,}")
    print(f"  Non-zeros: {nz:,}  ({density:.1f}% dense)")

    # ── Value distribution ──
    print(f"\n  ── Value distribution ──")
    for p in [0.1, 1, 10, 25, 50, 75, 90, 99, 99.9]:
        print(f"    P{p:5.1f}: {np.percentile(values, p):12.4f}")
    print(f"    {'Max':>5s}: {values.max():12.2f}")

    # ── Per-cell ──
    col_sums = np.asarray(X.sum(axis=0)).ravel()
    cell_nz = np.diff(X.tocsr().indptr)

    print(f"\n  ── Per-cell stats ──")
    print(f"    Total counts (sum):")
    print(f"      min={col_sums.min():.1f}  median={np.median(col_sums):.1f}  max={col_sums.max():.1f}")
    print(f"    Genes detected:")
    print(f"      min={cell_nz.min()}  median={np.median(cell_nz):.0f}  max={cell_nz.max()}")

    # ── Per-gene ──
    gene_nz = np.diff(X.tocsc().indptr)
    gene_means = np.asarray(X.mean(axis=1)).ravel()

    print(f"\n  ── Per-gene stats ──")
    print(f"    Cells detected in:")
    print(f"      genes with 0 cells: {(gene_nz == 0).sum()}")
    print(f"      median={np.median(gene_nz):.0f} / {n_cells} cells ({np.median(gene_nz)/n_cells*100:.1f}%)")
    print(f"    Mean expression: median={np.median(gene_means):.4f}  mean={gene_means.mean():.4f}")

    # ── Gene IDs ──
    with open(rows_path) as f:
        gene_ids = [line.strip().split('\t')[0] for line in f]
    print(f"\n  ── Gene IDs ──")
    print(f"    Total: {len(gene_ids)}")
    print(f"    Format: FlyBase IDs (FBgn...)")
    print(f"    Sample: {gene_ids[:5]}")

    # ── Cell metadata from IDs ──
    with open(cols_path) as f:
        cell_ids = [line.strip() for line in f]
    # Parse SAMEA prefix (sample-level) and barcode suffix
    samples = sorted(set(c.split('-')[0] for c in cell_ids))
    print(f"\n  ── Cell metadata ──")
    print(f"    Unique SAMEA (sample) prefixes: {len(samples)}")
    for s in samples:
        count = sum(1 for c in cell_ids if c.startswith(s))
        print(f"      {s}: {count:,} cells")


if __name__ == "__main__":
    inspect()
