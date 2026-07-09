from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_blank(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def int_or_blank(value: Any) -> int | str:
    if value is None or value == "":
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return ""


def sequence_length_summary(data_dir: Path, sequence_column: str, control_column: str) -> dict[str, Any]:
    train_csv = data_dir / "promoter_train.csv"
    rows = read_csv_rows(train_csv)
    if not rows:
        return {
            "asset_rows": 0,
            "sequence_min_len": "",
            "sequence_max_len": "",
            "control_min_len": "",
            "control_max_len": "",
        }
    seq_lengths = [len(str(row.get(sequence_column, ""))) for row in rows]
    control_lengths = [len(str(row.get(control_column, ""))) for row in rows]
    return {
        "asset_rows": len(rows),
        "sequence_min_len": min(seq_lengths),
        "sequence_max_len": max(seq_lengths),
        "control_min_len": min(control_lengths),
        "control_max_len": max(control_lengths),
    }


def motif_summary(run_dir: Path, support_gate: int) -> dict[str, Any]:
    motif_file = run_dir / "test" / "sequence_mutagenesis" / "de_novo_motifs.csv"
    motifs = read_csv_rows(motif_file)
    if not motifs:
        return {
            "motif_file": str(motif_file),
            "motif_rows": 0,
            "max_support_genes": "",
            "motifs_passing_support_gate": 0,
            "top_gate_motif": "",
        }
    support_values = [int_or_blank(row.get("support_genes")) for row in motifs]
    support_ints = [value for value in support_values if isinstance(value, int)]
    max_support = max(support_ints) if support_ints else ""
    passing = [row for row in motifs if isinstance(int_or_blank(row.get("support_genes")), int) and int_or_blank(row.get("support_genes")) >= support_gate]
    top_gate_motif = passing[0].get("motif_sequence", "") if passing else ""
    return {
        "motif_file": str(motif_file),
        "motif_rows": len(motifs),
        "max_support_genes": max_support,
        "motifs_passing_support_gate": len(passing),
        "top_gate_motif": top_gate_motif,
    }


def summarize_run(run_dir: Path, data_root: Path, support_gate: int) -> dict[str, Any]:
    cfg = load_json(run_dir / "config.json")
    metrics = load_json(run_dir / "test" / "test_metrics.json")
    sequence_length = int(cfg.get("sequence_length", metrics.get("sequence_length", 400)) or 400)
    sequence_column = str(cfg.get("sequence_column", metrics.get("sequence_column", "sequence")))
    control_column = str(cfg.get("contrastive_negative_column", "control_sequence"))
    data_name = cfg.get("data") or Path(str(metrics.get("data_dir", ""))).name
    data_dir = data_root / str(data_name) if data_name else Path(str(metrics.get("data_dir", "")))
    if not data_dir.exists() and metrics.get("data_dir"):
        data_dir = Path(str(metrics["data_dir"]))
    asset = sequence_length_summary(data_dir, sequence_column, control_column)
    control_max_len = int_or_blank(asset.get("control_max_len"))
    negative_crop_ready = isinstance(control_max_len, int) and control_max_len > sequence_length
    motif = motif_summary(run_dir, support_gate)
    return {
        "run": run_dir.name,
        "exp_name": cfg.get("exp_name", metrics.get("exp_name", run_dir.name)),
        "data": data_name,
        "seed": cfg.get("seed", ""),
        "contrastive_weight": cfg.get("contrastive_weight", ""),
        "contrastive_negative_shift_max": cfg.get("contrastive_negative_shift_max", ""),
        "contrastive_projection_dim": cfg.get("contrastive_projection_dim", 0),
        "sequence_length": sequence_length,
        "negative_crop_ready": negative_crop_ready,
        "mse": float_or_blank(metrics.get("mse")),
        "rmse": float_or_blank(metrics.get("rmse")),
        "pearson_r": float_or_blank(metrics.get("pearson_r")),
        "spearman_r": float_or_blank(metrics.get("spearman_r")),
        "nonzero_rmse": float_or_blank(metrics.get("nonzero_rmse")),
        "zero_rmse": float_or_blank(metrics.get("zero_rmse")),
        "per_gene_median_rmse": float_or_blank(metrics.get("per_gene_median_rmse")),
        "per_cell_median_rmse": float_or_blank(metrics.get("per_cell_median_rmse")),
        **asset,
        **motif,
    }


def write_summary(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "exp_name",
        "data",
        "seed",
        "contrastive_weight",
        "contrastive_negative_shift_max",
        "contrastive_projection_dim",
        "sequence_length",
        "negative_crop_ready",
        "mse",
        "rmse",
        "pearson_r",
        "spearman_r",
        "nonzero_rmse",
        "zero_rmse",
        "per_gene_median_rmse",
        "per_cell_median_rmse",
        "asset_rows",
        "sequence_min_len",
        "sequence_max_len",
        "control_min_len",
        "control_max_len",
        "motif_rows",
        "max_support_genes",
        "motifs_passing_support_gate",
        "top_gate_motif",
        "motif_file",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 2 contrastive runs and motif support gate.")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs" / "stage2")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--support-gate", type=int, default=5)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root if args.outputs_root.is_absolute() else PROJECT_ROOT / args.outputs_root
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    run_dirs = sorted(path for path in outputs_root.iterdir() if (path / "config.json").exists()) if outputs_root.exists() else []
    rows = [summarize_run(run_dir, data_root, int(args.support_gate)) for run_dir in run_dirs]
    output_csv = args.output_csv or outputs_root / "stage2_summary.csv"
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv
    write_summary(rows, output_csv)
    print(f"[Stage2Summary] runs={len(rows)} output={output_csv}")
    if rows:
        best_support = max((int_or_blank(row.get("max_support_genes")) for row in rows), default="")
        best_support_ints = [value for value in (int_or_blank(row.get("max_support_genes")) for row in rows) if isinstance(value, int)]
        if best_support_ints:
            best_support = max(best_support_ints)
            print(f"[Stage2Summary] best max_support_genes={best_support} gate={args.support_gate}")
        missing_crop = [row["run"] for row in rows if not row.get("negative_crop_ready")]
        if missing_crop:
            print(f"[Stage2Summary] WARNING negative crop not active for: {', '.join(missing_crop)}")


if __name__ == "__main__":
    main()
