"""Convert TPM-normalized h5ad back to raw UMI counts using stored total_counts.

TPM = raw_counts / total_counts * 1e6
=> raw_counts = round(TPM * total_counts / 1e6)
"""

import numpy as np
import scanpy as sc
from scipy import sparse
from pathlib import Path


def un_tpm(adata: sc.AnnData) -> sc.AnnData:
    """Reverse TPM normalization to recover raw UMI counts."""
    total_counts = adata.obs["total_counts"].values.astype(np.float64)

    X = adata.X
    if not sparse.issparse(X):
        raise ValueError("Expected sparse matrix")

    # Convert to COO for row-wise scaling
    X_coo = X.tocoo()
    row = X_coo.row
    col = X_coo.col
    data = X_coo.data.astype(np.float64)

    # Reverse TPM: raw = TPM * total_counts / 1e6
    scales = total_counts[row] / 1e6
    raw_data = np.round(data * scales).astype(np.int64)

    # Clip negative values (float32 rounding can produce tiny negatives)
    raw_data = np.maximum(raw_data, 0)

    n_rows, n_cols = X.shape
    raw_sparse = sparse.coo_matrix((raw_data, (row, col)), shape=(n_rows, n_cols), dtype=np.int64)
    raw_csr = raw_sparse.tocsr()

    adata_raw = sc.AnnData(
        X=raw_csr,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    return adata_raw


def main() -> None:
    base = Path("/mnt/d/code/pro_model/cluster test")
    src = base / "data" / "processed" / "integrated_data.h5ad"
    dst = base / "data" / "processed" / "integrated_data_raw.h5ad"

    print(f"Loading TPM data: {src}")
    adata = sc.read(src)
    print(f"  shape={adata.shape}, cell0_total_counts={adata.obs['total_counts'].iloc[0]:.0f}")

    print("Reversing TPM to raw counts...")
    adata_raw = un_tpm(adata)

    # Verify
    X = adata_raw.X
    row0 = np.asarray(X[0].toarray()).ravel()
    print(f"  cell0_raw_sum={row0.sum():.0f}, nz={(row0>0).sum()}, dtype={X.dtype}")
    print(f"  all_integers={np.allclose(X.data, X.data.astype(np.int64))}")

    print(f"Saving: {dst}")
    adata_raw.write_h5ad(dst)
    print("Done.")


if __name__ == "__main__":
    main()
