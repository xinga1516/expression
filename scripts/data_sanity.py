import json
import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


project_root = pathlib.Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "log_processed"#"highquality"
out_dir = data_dir#project_root / "outputs"


def _summary_stats(arr: np.ndarray) -> dict[str, Optional[float]]:
    if arr.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def summarize_integrated_data(h5ad_path: pathlib.Path) -> dict:
    adata = sc.read_h5ad(h5ad_path)
    X = adata.X

    if sparse.issparse(X):
        X_csr = X.tocsr()
    else:
        X_csr = sparse.csr_matrix(X)

    n_cells, n_genes = X_csr.shape
    total_entries = n_cells * n_genes
    nnz = int(X_csr.nnz)
    density = float(nnz / total_entries) if total_entries > 0 else 0.0
    sparsity_ratio = 1.0 - density

    # Non-zero values distribution
    nz_values = np.asarray(X_csr.data)

    # Per-cell distributions
    cell_total_counts = np.asarray(X_csr.sum(axis=1)).ravel()
    cell_nnz = np.diff(X_csr.indptr)

    # Per-gene distributions
    gene_total_counts = np.asarray(X_csr.sum(axis=0)).ravel()
    X_csc = X_csr.tocsc()
    gene_nnz = np.diff(X_csc.indptr)
    gene_detection_rate = gene_nnz / max(n_cells, 1)

    summary = {
        "matrix": {
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "nnz": nnz,
            "density": density,
            "sparsity": sparsity_ratio,
        },
        "nonzero_values": _summary_stats(nz_values),
        "cell_total_counts": _summary_stats(cell_total_counts),
        "cell_nonzero_genes": _summary_stats(cell_nnz.astype(np.float64)),
        "gene_total_counts": _summary_stats(gene_total_counts),
        "gene_nonzero_cells": _summary_stats(gene_nnz.astype(np.float64)),
        "gene_detection_rate": _summary_stats(gene_detection_rate.astype(np.float64)),
    }
    return summary


def plot_gene_variance_vs_median(
    h5ad_path: pathlib.Path,
    save_path: pathlib.Path | None = None,
    n_label: int = 100,
) -> None:
    """Scatter plot: per-gene variance (x) vs median (y) across cells.

    Uses the processed dataset. Annotates the n_label genes with smallest
    and largest variance with their gene_symbol.
    """
    adata = sc.read_h5ad(h5ad_path)
    X = adata.X
    if sparse.issparse(X):
        X_csc = X.tocsc()
    else:
        X_csc = sparse.csc_matrix(X)

    n_cells, n_genes = X_csc.shape
    gene_names = adata.var["gene_symbol"].values

    variances = np.empty(n_genes, dtype=np.float64)
    medians = np.empty(n_genes, dtype=np.float64)
    #means = np.empty(n_genes, dtype=np.float64)

    print(f"Computing per-gene variance & median for {n_genes} genes...")
    for i in range(n_genes):
        col = X_csc[:, i].toarray().ravel()
        variances[i] = float(np.var(col))
        medians[i] = float(np.mean(col))

    # Sort by variance to find extremes
    order = np.argsort(variances)
    n_label = min(n_label, n_genes // 2)
    low_idx = order[:1]   # lowest variance
    high_idx = order[-n_label:]  # highest variance

    # sort by median to find extremes
    order_median = np.argsort(medians)
    low_idx_median = order_median[:1]   # lowest median
    high_idx_median = order_median[-n_label:]  # highest median

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(variances, medians, s=1, alpha=0.3, color="steelblue", rasterized=True)

    # Highlight known Drosophila housekeeping genes (low variance, high median)
    hk_genes = [
        "Act5C",                                # actins
        "RpL32", "RpL13A", "RpS18", "RpL11",    # ribosomal proteins
        "Gapdh1", "Gapdh2",                      # GAPDH
        "alphaTub84B", "betaTub56D",             # tubulins
        "eEF1alpha1",                            # elongation factors
        "14-3-3epsilon",                          # signaling
        "8SrRNA",
    ]
    # 合并hk_genes和模糊匹配得到的基因列表（名字中包含 "18SrRNA" 的基因，比如 18SrRNA, 28SrRNA 等）
    srrna = np.array(["18SrRNA" in str(g) for g in gene_names], dtype=bool)
    print(f"Found {srrna.sum()} genes matching '18SrRNA' pattern for housekeeping annotation.")
    hk_mask = np.isin(gene_names, hk_genes) | srrna
    if hk_mask.any():
        hk_idx = np.where(hk_mask)[0]
        ax.scatter(variances[hk_idx], medians[hk_idx], s=60, color="red",
                   edgecolors="darkred", linewidths=1.0, zorder=5,
                   label=f"Housekeeping ({len(hk_idx)} genes)")
        for idx in hk_idx:
            ax.annotate(str(gene_names[idx]), (variances[idx], medians[idx]),
                        fontsize=7, color="darkred", fontweight="bold", rotation=25,
                        ha="left", va="bottom")
    

    # Annotate extremes
    for idx in np.concatenate([low_idx, high_idx, low_idx_median, high_idx_median]):
        ax.annotate(gene_names[idx], (variances[idx], medians[idx]),
                    fontsize=5, alpha=0.8, rotation=25,
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, linewidth=0.5))

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Variance across cells")
    ax.set_ylabel("Mean expression across cells")
    ax.set_title(f"Gene variability ({n_genes} genes)\n"
                 f"Annotated: {n_label} lowest + {n_label} highest variance genes")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Gene variance plot saved to: {save_path}")
    plt.close()


def plot_fca_gene_variance_vs_median(
    csv_path: pathlib.Path,
    save_path: pathlib.Path | None = None,
    n_label: int = 100,
) -> None:
    """Plot per-gene variance vs median using Fly Cell Atlas logCPM data.

    Reads the FCA_expression_logCPM.csv file. Each point is a gene, x-axis is
    variance across 259 cell types, y-axis is median expression. Annotates
    variance extremes and Drosophila housekeeping genes in red.
    """
    df = pd.read_csv(csv_path)
    meta_cols = ["flybase_id", "entrez_id", "gene_symbol", "gene_name"]
    expr_cols = [c for c in df.columns if c not in meta_cols]
    expr_mat = df[expr_cols].values

    n_genes, n_types = expr_mat.shape
    gene_names = df["gene_symbol"].values

    variances = expr_mat.var(axis=1)
    medians = np.median(expr_mat, axis=1)

    # Variance extremes
    order = np.argsort(variances)
    n_label = min(n_label, n_genes // 2)
    low_idx = order[:n_label]
    high_idx = order[-n_label:]

    # Median extremes
    order_median = np.argsort(medians)
    low_idx_median = order_median[:1]
    high_idx_median = order_median[-n_label:]

    # Drosophila housekeeping genes
    hk_genes = [
        "Act5C", 
        "RpL32", "RpL13A", "RpS18", "RpL11",
        "Gapdh1", "Gapdh2",
        "alphaTub84B", "betaTub56D",
        "eEF1alpha1", 
        "14-3-3epsilon",
    ]
    # 精确匹配 hk_genes，或模糊匹配含有 "8SrRNA" 字符串的基因
    hk_mask = np.isin(gene_names, hk_genes) | np.array(["8SrRNA" in str(g) for g in gene_names], dtype=bool)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(variances, medians, s=1, alpha=0.25, color="steelblue", rasterized=True)

    # HK genes in red
    if hk_mask.any():
        hk_idx = np.where(hk_mask)[0]
        ax.scatter(variances[hk_idx], medians[hk_idx], s=60, color="red",
                   edgecolors="darkred", linewidths=1.0, zorder=5,
                   label=f"Housekeeping ({len(hk_idx)} genes)")
        for idx in hk_idx:
            ax.annotate(str(gene_names[idx]), (variances[idx], medians[idx]),
                        fontsize=7, color="darkred", fontweight="bold", rotation=25,
                        ha="left", va="bottom")

    # Annotate variance extremes
    for idx in np.concatenate([low_idx, high_idx, low_idx_median, high_idx_median]):
        ax.annotate(str(gene_names[idx]), (variances[idx], medians[idx]),
                    fontsize=4, alpha=0.7, rotation=25,
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, linewidth=0.3))

    ax.set_xlabel("Variance across cell types (log2 CPM)")
    ax.set_ylabel("Median expression across cell types (log2 CPM)")
    ax.set_title(f"FCA Gene variability — {n_genes} genes x {n_types} cell types\n"
                 f"Red: Drosophila housekeeping genes  |  "
                 f"Annotated: {n_label} lowest + {n_label} highest variance")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"FCA gene variance plot saved to: {save_path}")
    plt.close()


def plot_loom_gene_variance_vs_median(
    loom_path: pathlib.Path,
    save_path: pathlib.Path | None = None,
    n_label: int = 100,
    chunk_size: int = 10000,
) -> None:
    """Scatter plot of per-gene variance vs mean from raw loom UMI counts.

    Reads the dense (genes × cells) matrix in chunks in a single pass
    (Welford algorithm) to stay memory-friendly.  Each point is one gene;
    x-axis is variance across cells, y-axis is mean expression (both on
    log-log axes).

    Annotates variance extremes and Drosophila housekeeping genes.
    """
    import h5py

    f = h5py.File(str(loom_path), "r")
    X = f["matrix"]           # shape (n_genes, n_cells)
    gene_names = f["row_attrs"]["Gene"][:].astype(str)
    n_genes, n_cells = X.shape
    print(f"Loom matrix: {n_genes} genes × {n_cells} cells")

    # One-pass mean + variance via Welford algorithm:
    # accumulates sum_x and sum_x2 per gene across chunks of cells.
    sum_x = np.zeros(n_genes, dtype=np.float64)
    sum_x2 = np.zeros(n_genes, dtype=np.float64)

    print(f"Scanning cells in chunks of {chunk_size} ...")
    n_chunks = 0
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        block = X[:, start:end].astype(np.float64)  # (genes, chunk)
        sum_x += block.sum(axis=1)
        sum_x2 += (block * block).sum(axis=1)
        n_chunks += 1
        if n_chunks % 20 == 0:
            print(f"  processed {end}/{n_cells} cells ({100*end/n_cells:.0f}%)")

    f.close()

    means = sum_x / n_cells
    variances = (sum_x2 / n_cells) - (means * means)
    variances = np.maximum(variances, 0.0)  # clip numeric noise

    gene_names_arr = np.asarray([str(g) for g in gene_names])

    # --- plot ---
    n_label = min(n_label, n_genes // 2)
    order = np.argsort(variances)
    low_idx = order[:n_label]
    high_idx = order[-n_label:]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(variances, means, s=1, alpha=0.25, color="steelblue", rasterized=True)

    # Housekeeping genes
    hk_genes = [
        "Act5C",
        "RpL32", "RpL13A", "RpS18", "RpL11",
        "Gapdh1", "Gapdh2",
        "alphaTub84B", "betaTub56D",
        "eEF1alpha1",
        "14-3-3epsilon",
    ]
    hk_mask = np.isin(gene_names_arr, hk_genes) | np.array(
        ["8SrRNA" in g for g in gene_names_arr], dtype=bool
    )
    if hk_mask.any():
        hk_idx = np.where(hk_mask)[0]
        ax.scatter(variances[hk_idx], means[hk_idx], s=70, color="red",
                   edgecolors="darkred", linewidths=1.2, zorder=5,
                   label=f"Housekeeping ({len(hk_idx)} genes)")
        for idx in hk_idx:
            ax.annotate(str(gene_names_arr[idx]), (variances[idx], means[idx]),
                        fontsize=7, color="darkred", fontweight="bold", rotation=25,
                        ha="left", va="bottom")

    # Annotate variance extremes
    for idx in np.concatenate([low_idx, high_idx]):
        ax.annotate(str(gene_names_arr[idx]), (variances[idx], means[idx]),
                    fontsize=4.5, alpha=0.7, rotation=25,
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, linewidth=0.3))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Variance across cells (raw UMI counts)")
    ax.set_ylabel("Mean expression across cells (raw UMI counts)")
    ax.set_title(f"Loom raw UMI — gene variability ({n_genes} genes × {n_cells} cells)\n"
                 f"Red: Drosophila housekeeping genes  |  "
                 f"Annotated: {n_label} lowest + {n_label} highest variance genes")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Loom gene variance vs mean plot saved to: {save_path}")
    plt.close()


def plot_emtab_gene_variance_vs_median(
    mtx_dir: pathlib.Path,
    save_path: pathlib.Path | None = None,
    n_label: int = 50,
) -> None:
    """Per-gene variance vs median for E-MTAB-10519 sparse MTX data.

    Reads the aggregated filtered normalised counts (CPM) from Matrix Market
    format, log2(CPM+1) transforms, then computes per-gene variance and median
    across 527k cells.  All operations are sparse-safe.
    """
    from scipy.io import mmread
    from scipy import sparse

    mtx_path = mtx_dir / "E-MTAB-10519.aggregated_filtered_normalised_counts.mtx"
    rows_path = mtx_dir / "E-MTAB-10519.aggregated_filtered_normalised_counts.mtx_rows"

    print("Reading MTX ...")
    X = mmread(str(mtx_path))
    n_genes, n_cells = X.shape
    print(f"  {n_genes} genes x {n_cells} cells  |  {X.nnz:,} non-zeros  |  "
          f"{X.nnz/(n_genes*n_cells)*100:.1f}% dense")

    # ── log2(CPM + 1) transform ──
    print("log2(CPM + 1) transform ...")
    X.data = np.log2(X.data + 1.0)

    # ── Per-gene mean & variance (sparse CSR, one pass) ──
    print("Computing per-gene stats ...")
    X_csr = X.tocsr()
    ones_cells = np.ones(n_cells, dtype=np.float64)

    row_sums = X_csr.dot(ones_cells)                           # sum of log2(cpm+1)
    row_sq_sums = X_csr.power(2).dot(ones_cells)               # sum of squares
    means = row_sums / n_cells
    variances = (row_sq_sums / n_cells) - (means * means)
    variances = np.maximum(variances, 0.0)

    # ── Per-gene median ──
    # For genes detected in <50% cells, median = 0 (the log2(0+1) = 0).
    # For the few highly detected genes, compute actual median of non-zeros.
    gene_nz = np.diff(X_csr.indptr)
    medians = np.zeros(n_genes, dtype=np.float64)
    dense_mask = gene_nz > (n_cells // 2)
    n_dense = int(dense_mask.sum())
    print(f"  {n_dense} genes detected in >50% cells — computing exact median ...")
    for i in np.where(dense_mask)[0]:
        nz_vals = X_csr.data[X_csr.indptr[i]:X_csr.indptr[i + 1]]
        n_zeros = n_cells - len(nz_vals)
        medians[i] = float(np.median(np.concatenate([np.zeros(n_zeros, dtype=nz_vals.dtype), nz_vals])))

    # ── Gene IDs → symbols ──
    with open(rows_path) as f:
        fb_ids = [line.strip().split('\t')[0] for line in f]
    # Try to map FlyBase → symbol via FCA CSV, fall back to FBgn ID
    fca_csv = mtx_dir.parent / "FCA_expression_logCPM.csv"
    id2sym: dict[str, str] = {}
    if fca_csv.exists():
        fca = pd.read_csv(fca_csv, usecols=["flybase_id", "gene_symbol"])
        id2sym = dict(zip(fca["flybase_id"], fca["gene_symbol"]))
    gene_names = np.array([id2sym.get(fb, fb) for fb in fb_ids])

    # ── Plot ──
    n_label = min(n_label, n_genes // 2)
    order = np.argsort(variances)
    low_idx = order[:n_label]
    high_idx = order[-n_label:]

    hk_genes = [
        "Act5C", "Act42A",
        "RpL32", "RpL13A", "RpS18", "RpL11",
        "Gapdh1", "Gapdh2",
        "alphaTub84B", "betaTub56D",
        "eEF1alpha1", "eEF1alpha2",
        "14-3-3epsilon",
    ]
    hk_mask = np.isin(gene_names, hk_genes)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(variances, medians, s=0.5, alpha=0.2, color="steelblue", rasterized=True)

    if hk_mask.any():
        hk_idx = np.where(hk_mask)[0]
        ax.scatter(variances[hk_idx], medians[hk_idx], s=60, color="red",
                   edgecolors="darkred", linewidths=1.0, zorder=5,
                   label=f"Housekeeping ({len(hk_idx)} genes)")
        for idx in hk_idx:
            ax.annotate(str(gene_names[idx]), (variances[idx], medians[idx]),
                        fontsize=7, color="darkred", fontweight="bold", rotation=25,
                        ha="left", va="bottom")

    for idx in np.concatenate([low_idx, high_idx]):
        ax.annotate(str(gene_names[idx]), (variances[idx], medians[idx]),
                    fontsize=4, alpha=0.7, rotation=25,
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, linewidth=0.3))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Variance across cells (log2 CPM+1)")
    ax.set_ylabel("Median expression across cells (log2 CPM+1)")
    ax.set_title(f"E-MTAB-10519 — gene variability ({n_genes} genes × {n_cells:,} cells)\n"
                 f"Red: Drosophila housekeeping genes  |  "
                 f"Annotated: {n_label} lowest + {n_label} highest variance")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"E-MTAB-10519 gene variance plot saved to: {save_path}")
    plt.close()


def main() -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # # Gene variance vs median plot from raw loom UMI counts
    # loom_path = project_root / "data" / "raw" / "s_fca_biohub_all_wo_blood_10x.loom"
    # if loom_path.exists():
    #     plot_loom_gene_variance_vs_median(
    #         loom_path,
    #         save_path=out_dir / "loom_gene_variance_vs_median.png",
    #     )

    # h5ad_path = data_dir / "integrated_data.h5ad"
    # summary = summarize_integrated_data(h5ad_path)

    # print("=== 训练前数据分布摘要（integrated_data.h5ad）===")
    # print(json.dumps(summary, indent=2, ensure_ascii=False))

    # out_path = out_dir / "data_sanity_summary.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(summary, f, indent=2, ensure_ascii=False)
    # print(f"\n摘要已保存到: {out_path}")

    # Gene variance vs median plot (scRNA-seq h5ad)
    data_h5ad = data_dir / "integrated_data.h5ad"
    if data_h5ad.exists():
        plot_gene_variance_vs_median(
            data_h5ad,
            save_path=out_dir / "gene_variance_vs_mean.png",
        )

    # # FCA gene variance vs median plot (logCPM CSV from Fly Cell Atlas)
    # fca_csv = project_root / "data" / "FCA_expression_logCPM.csv"
    # if fca_csv.exists():
    #     plot_fca_gene_variance_vs_median(
    #         fca_csv,
    #         save_path=project_root / "data" / "FCA_gene_variance_vs_median.png",
    #     )

    # # E-MTAB-10519 gene variance vs median (sparse MTX, 527k cells)
    # emtab_dir = project_root / "data" / "E-MTAB-10519-normalised-files"
    # emtab_mtx = emtab_dir / "E-MTAB-10519.aggregated_filtered_normalised_counts.mtx"
    # if emtab_mtx.exists():
    #     plot_emtab_gene_variance_vs_median(
    #         emtab_dir,
    #         save_path=out_dir / "emtab_gene_variance_vs_median.png",
    #     )


if __name__ == "__main__":
    main()