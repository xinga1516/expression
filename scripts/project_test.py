from __future__ import annotations

from typing import Any
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _clean_gene_ids(values: pd.Series) -> pd.Series:
    return pd.Series(values, dtype="string").str.strip()


def load_h5ad(h5ad_file: Path) -> Any:
    if not h5ad_file.exists():
        raise FileNotFoundError(f"h5ad file not found: {h5ad_file}")
    return sc.read_h5ad(h5ad_file)


def check_h5ad(adata: Any, gene_col: str = "gene_id") -> dict:
    print(f"[h5ad] shape={adata.shape}")

    X = adata.X
    if sparse.issparse(X):
        print(f"[h5ad] X sparse format={X.getformat()} nnz={X.nnz}")
    else:
        print(f"[h5ad] X dense dtype={getattr(X, 'dtype', 'unknown')}")

    if gene_col not in adata.var.columns:
        raise ValueError(f"[h5ad] missing adata.var['{gene_col}']")

    gene_ids = _clean_gene_ids(adata.var[gene_col])
    missing = gene_ids.isna() | (gene_ids == "")
    duplicated = gene_ids[~missing].duplicated(keep=False)

    print(
        "[h5ad] gene_id "
        f"total={len(gene_ids)} unique={gene_ids[~missing].nunique()} "
        f"missing={int(missing.sum())} duplicated_rows={int(duplicated.sum())}"
    )

    if duplicated.any():
        examples = gene_ids[duplicated].drop_duplicates().head(10).tolist()
        print(f"[h5ad][WARN] duplicated gene_id examples: {examples}")

    return {
        "gene_ids": gene_ids,
        "missing_count": int(missing.sum()),
        "duplicated_count": int(duplicated.sum()),
    }


def check_promoter_csv(promoter_file: Path, gene_col: str = "gene_id") -> dict:
    if not promoter_file.exists():
        raise FileNotFoundError(f"promoter CSV not found: {promoter_file}")

    df = pd.read_csv(promoter_file)
    print(f"[csv] {promoter_file.name} rows={len(df)} cols={list(df.columns)}")

    if gene_col not in df.columns:
        raise ValueError(f"[csv] {promoter_file} missing column '{gene_col}'")

    gene_ids = _clean_gene_ids(df[gene_col])
    missing = gene_ids.isna() | (gene_ids == "")
    duplicated_rows = gene_ids[~missing].duplicated(keep=False)

    print(
        f"[csv] {promoter_file.name} gene_id "
        f"unique={gene_ids[~missing].nunique()} missing={int(missing.sum())} "
        f"duplicated_rows={int(duplicated_rows.sum())}"
    )

    required_cols = {"gene_id", "sequence"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        print(f"[csv][WARN] {promoter_file.name} missing expected columns: {missing_cols}")

    if "sequence" in df.columns:
        seq_len = df["sequence"].astype("string").str.len()
        print(
            f"[csv] {promoter_file.name} sequence length "
            f"min={int(seq_len.min())} max={int(seq_len.max())} "
            f"bad_length_rows={int((seq_len != 400).sum())}"
        )

    return {
        "df": df,
        "gene_ids": gene_ids,
        "missing_count": int(missing.sum()),
        "duplicated_rows": int(duplicated_rows.sum()),
    }


def check_gene_id_alignment(h5ad_gene_ids: pd.Series, promoter_gene_ids: pd.Series, label: str) -> dict:
    h5ad_valid = h5ad_gene_ids.dropna()
    h5ad_valid = h5ad_valid[h5ad_valid != ""]
    promoter_valid = promoter_gene_ids.dropna()
    promoter_valid = promoter_valid[promoter_valid != ""]

    h5ad_set = set(h5ad_valid.tolist())
    promoter_set = set(promoter_valid.tolist())
    missing_from_h5ad = sorted(promoter_set - h5ad_set)
    unused_in_promoter = sorted(h5ad_set - promoter_set)

    h5ad_counts = h5ad_valid.value_counts()
    ambiguous = sorted(g for g in promoter_set if h5ad_counts.get(g, 0) > 1)

    print(
        f"[align] {label} promoter_unique={len(promoter_set)} "
        f"in_h5ad={len(promoter_set) - len(missing_from_h5ad)} "
        f"missing_from_h5ad={len(missing_from_h5ad)} "
        f"ambiguous_in_h5ad={len(ambiguous)}"
    )

    if missing_from_h5ad:
        print(f"[align][FAIL] {label} missing examples: {missing_from_h5ad[:10]}")
    if ambiguous:
        print(f"[align][FAIL] {label} ambiguous examples: {ambiguous[:10]}")
    if unused_in_promoter:
        print(f"[align] {label} h5ad genes not used by promoter CSV: {len(unused_in_promoter)}")

    return {
        "missing_from_h5ad": missing_from_h5ad,
        "ambiguous_in_h5ad": ambiguous,
        "unused_in_promoter_count": len(unused_in_promoter),
    }


def check_target_values(adata: Any, promoter_gene_ids: pd.Series, n_cells: int = 200, seed: int = 42) -> None:
    valid_promoter_gene_ids = promoter_gene_ids.dropna()
    valid_promoter_gene_ids = valid_promoter_gene_ids[valid_promoter_gene_ids != ""].drop_duplicates()
    gene_to_idx = {gene_id: i for i, gene_id in enumerate(_clean_gene_ids(adata.var["gene_id"]))}
    gene_indices = [gene_to_idx[g] for g in valid_promoter_gene_ids if g in gene_to_idx]

    if not gene_indices:
        print("[target][SKIP] no promoter genes found in h5ad")
        return

    rng = np.random.default_rng(seed)
    cell_count = min(n_cells, adata.n_obs)
    cell_indices = rng.choice(adata.n_obs, size=cell_count, replace=False)
    X_subset = adata.X[cell_indices][:, gene_indices]

    if sparse.issparse(X_subset):
        nnz = X_subset.nnz
        total = X_subset.shape[0] * X_subset.shape[1]
        data = X_subset.data
    else:
        nnz = int(np.count_nonzero(X_subset))
        total = X_subset.size
        data = np.asarray(X_subset)[np.asarray(X_subset) != 0]

    frac_nonzero = nnz / max(total, 1)
    if len(data) > 0:
        print(
            f"[target] sampled cells={cell_count} genes={len(gene_indices)} "
            f"nonzero_frac={frac_nonzero:.6f} min_nz={data.min():.6f} max={data.max():.6f}"
        )
    else:
        print(
            f"[target][WARN] sampled cells={cell_count} genes={len(gene_indices)} "
            "all sampled targets are zero"
        )


def resolve_promoter_files(data_dir: Path, promoter_files: list[str] | None) -> list[Path]:
    if promoter_files:
        return [Path(p) if Path(p).is_absolute() else data_dir / p for p in promoter_files]
    return [
        data_dir / "promoter_train.csv",
        data_dir / "promoter_val.csv",
        data_dir / "promoter_test.csv",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic sanity checks for h5ad and promoter CSV files.")
    parser.add_argument("--data", default="processed", help="Data folder under data/, e.g. processed or highquality")
    parser.add_argument("--h5ad", default=None, help="Path to integrated_data.h5ad")
    parser.add_argument("--promoter", nargs="*", default=None, help="Promoter CSV filenames or paths")
    parser.add_argument("--gene-col", default="gene_id", help="Gene id column name")
    parser.add_argument("--sample-cells", type=int, default=200, help="Cells sampled for target non-zero check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / args.data
    h5ad_file = Path(args.h5ad) if args.h5ad else data_dir / "integrated_data.h5ad"
    promoter_files = resolve_promoter_files(data_dir, args.promoter)

    print(f"[path] data_dir={data_dir}")
    print(f"[path] h5ad={h5ad_file}")

    adata = load_h5ad(h5ad_file)
    h5ad_info = check_h5ad(adata, gene_col=args.gene_col)

    has_failure = h5ad_info["missing_count"] > 0 or h5ad_info["duplicated_count"] > 0
    all_promoter_gene_ids = []

    for promoter_file in promoter_files:
        promoter_info = check_promoter_csv(promoter_file, gene_col=args.gene_col)
        alignment = check_gene_id_alignment(
            h5ad_info["gene_ids"],
            promoter_info["gene_ids"],
            label=promoter_file.name,
        )
        all_promoter_gene_ids.append(promoter_info["gene_ids"])
        has_failure = has_failure or promoter_info["missing_count"] > 0
        has_failure = has_failure or bool(alignment["missing_from_h5ad"])
        has_failure = has_failure or bool(alignment["ambiguous_in_h5ad"])

    if all_promoter_gene_ids:
        merged_gene_ids = pd.concat(all_promoter_gene_ids, ignore_index=True).drop_duplicates()
        check_target_values(adata, merged_gene_ids, n_cells=args.sample_cells, seed=args.seed)

    if has_failure:
        raise SystemExit("[FAIL] data sanity checks found alignment problems")

    print("[OK] data sanity checks passed")


if __name__ == "__main__":
    main()