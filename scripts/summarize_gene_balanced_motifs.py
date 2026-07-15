"""Build a gene-balanced diagnostic from completed mutation effects.

The primary mutation output keeps all selected pairs and remains unchanged.
This diagnostic lets each gene contribute at most one important window before
motif aggregation, so pair multiplicity from one gene cannot create a false
cross-gene motif signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def centered_window_bounds(position: int, sequence_length: int, window_size: int) -> tuple[int, int]:
    window_size = min(max(1, int(window_size)), sequence_length)
    start = position - window_size // 2
    start = max(0, min(start, sequence_length - window_size))
    return start, start + window_size


def summarize_gene_balanced_run(
    run_dir: Path,
    promoter_file: Path,
    window_size: int,
    top_windows: int,
    top_k: int,
    min_support_genes: int,
) -> dict[str, Any]:
    mut_dir = run_dir / "test" / "sequence_mutagenesis"
    effects_file = mut_dir / "mutation_effects_long.csv"
    if not effects_file.exists():
        return {"run": run_dir.name, "status": "missing_effects"}

    effects = pd.read_csv(effects_file)
    promoters = pd.read_csv(promoter_file, usecols=["sequence"])
    if effects.empty:
        empty = pd.DataFrame()
        empty.to_csv(mut_dir / "gene_balanced_motif_windows.csv", index=False)
        empty.to_csv(mut_dir / "gene_balanced_de_novo_motifs.csv", index=False)
        return {"run": run_dir.name, "status": "empty", "max_support_genes": 0}

    pair_position = (
        effects
        .groupby(
            ["rank", "pro_i", "cell_id", "gene_id", "position_0based", "position_1based"],
            as_index=False,
        )
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            mean_signed_delta=("delta", "mean"),
        )
    )
    important_indices = pair_position.groupby("rank")["mean_abs_delta"].idxmax()
    important = pair_position.loc[important_indices].sort_values("mean_abs_delta", ascending=False)
    # One window per gene is the diagnostic's defining invariant.
    balanced = important.drop_duplicates("gene_id", keep="first").head(max(0, top_windows)).copy()

    rows: list[dict[str, Any]] = []
    for row in balanced.itertuples(index=False):
        pro_i = int(row.pro_i)
        sequence = str(promoters.iloc[pro_i]["sequence"]).upper()
        start, end = centered_window_bounds(int(row.position_0based), len(sequence), window_size)
        rows.append(
            {
                "rank": int(row.rank),
                "pro_i": pro_i,
                "gene_id": str(row.gene_id),
                "cell_id": str(row.cell_id),
                "important_position_0based": int(row.position_0based),
                "important_position_1based": int(row.position_1based),
                "window_start_0based": start,
                "window_end_0based": end,
                "motif_sequence": sequence[start:end],
                "mean_abs_delta": float(row.mean_abs_delta),
                "mean_signed_delta": float(row.mean_signed_delta),
            }
        )

    windows = pd.DataFrame(rows)
    windows.to_csv(mut_dir / "gene_balanced_motif_windows.csv", index=False)
    if windows.empty:
        motifs = pd.DataFrame()
    else:
        motifs = (
            windows.groupby("motif_sequence", as_index=False)
            .agg(
                support_pairs=("rank", "nunique"),
                support_genes=("gene_id", "nunique"),
                mean_abs_delta=("mean_abs_delta", "mean"),
                sum_abs_delta=("mean_abs_delta", "sum"),
                mean_signed_delta=("mean_signed_delta", "mean"),
            )
            .sort_values(["support_genes", "mean_abs_delta", "sum_abs_delta"], ascending=False)
            .head(max(0, top_k))
        )
    motifs.to_csv(mut_dir / "gene_balanced_de_novo_motifs.csv", index=False)

    max_support = int(motifs["support_genes"].max()) if not motifs.empty else 0
    return {
        "run": run_dir.name,
        "status": "ok",
        "gene_balanced_windows": int(len(windows)),
        "gene_balanced_motif_rows": int(len(motifs)),
        "max_support_genes": max_support,
        "motifs_passing_support_gate": int((motifs["support_genes"] >= min_support_genes).sum()) if not motifs.empty else 0,
        "top_motif": str(motifs.iloc[0]["motif_sequence"]) if not motifs.empty else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize mutation motifs with one window per gene.")
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--promoter-file", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--top-windows", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-support-genes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir in sorted(args.outputs_root.iterdir()):
        if (run_dir / "config.json").exists():
            rows.append(
                summarize_gene_balanced_run(
                    run_dir=run_dir,
                    promoter_file=args.promoter_file,
                    window_size=args.window_size,
                    top_windows=args.top_windows,
                    top_k=args.top_k,
                    min_support_genes=args.min_support_genes,
                )
            )
    summary = pd.DataFrame(rows)
    output_file = args.outputs_root / "stage2_gene_balanced_motif_summary.csv"
    summary.to_csv(output_file, index=False)
    max_support = int(summary["max_support_genes"].max()) if not summary.empty else 0
    print(f"[GeneBalancedMotifs] runs={len(summary)} max_support_genes={max_support} output={output_file}")


if __name__ == "__main__":
    main()
