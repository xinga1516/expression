import json
import pathlib

import numpy as np
import scanpy as sc
from scipy import sparse


project_root = pathlib.Path(__file__).resolve().parent.parent
data_dir = project_root / "data" / "log_processed"#"highquality"
out_dir = data_dir#project_root / "outputs"


def _summary_stats(arr: np.ndarray):
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


def summarize_integrated_data(h5ad_path: pathlib.Path):
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


def main():
    h5ad_path = data_dir / "integrated_data.h5ad"
    summary = summarize_integrated_data(h5ad_path)

    print("=== 训练前数据分布摘要（integrated_data.h5ad）===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_sanity_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n摘要已保存到: {out_path}")


if __name__ == "__main__":
    main()