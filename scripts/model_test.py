from pathlib import Path
import argparse
from collections import Counter, defaultdict
import heapq
import json
import math
import re
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MyDataset
from src.model import build_model
import src.utils as utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name under outputs/")
    parser.add_argument("--data", type=str, default=None, help="Fallback data directory name under data/")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Default: outputs/<exp_name>/checkpoints/best_model.safetensors")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all test samples.")
    parser.add_argument("--spearman-samples", type=int, default=1_000_000, help="Reservoir sample size for Spearman. 0 means store all samples.")
    parser.add_argument("--preencode-promoters", action="store_true", default=False, help="Pre-encode test promoters in memory.")
    parser.add_argument("--skip-standard-test", action="store_true", default=False, help="Skip metrics/scatter and only run requested extra analyses.")
    parser.add_argument("--run-input-ablation", action="store_true", default=False, help="Run promoter/expression input shuffling tests.")
    parser.add_argument("--ablation-repeats", type=int, default=3, help="Random repeats for input ablation.")
    parser.add_argument("--ablation-max-samples", type=int, default=None, help="Max samples for input ablation. Default: reuse --max-samples.")
    parser.add_argument("--run-mutagenesis", action="store_true", default=False, help="Run in silico point-mutagenesis on top expressed test promoter-cell pairs.")
    parser.add_argument("--top-n", type=int, default=100, help="Top expressed promoter-cell pairs for mutagenesis.")
    parser.add_argument("--mutation-batch-size", type=int, default=512, help="Batch size for mutated promoter forward passes.")
    parser.add_argument("--max-pairs-per-gene-ratio", type=float, default=0.1, help="Maximum fraction of mutagenesis top_n pairs allowed for one gene_id.")
    parser.add_argument("--motif-window-size", type=int, default=9, help="Window size for de novo motif extraction around important positions.")
    parser.add_argument("--motif-top-windows", type=int, default=200, help="Number of top important windows used for motif outputs.")
    parser.add_argument("--motif-top-k", type=int, default=20, help="Number of de novo motif sequences to report.")
    parser.add_argument("--motif-min-support", type=int, default=3, help="Minimum pair support for de novo motif sequence reporting.")
    parser.add_argument("--known-motif-file", type=str, default=None, help="Optional MEME DNA motif file for known motif matching.")
    parser.add_argument("--known-motif-min-score-ratio", type=float, default=0.8, help="Keep known motif hits with PWM score >= ratio * motif max score.")
    parser.add_argument("--known-motif-max-hits-per-motif", type=int, default=1000, help="Maximum saved known motif hits per motif.")
    return parser.parse_args()


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_dir(base_dir: Path, cfg: dict[str, Any], data_arg: str | None) -> Path:
    scrna_file = cfg.get("scrna_file", "")
    if scrna_file:
        return Path(scrna_file).parent
    if data_arg is None:
        raise ValueError("--data is required when config.json has no scrna_file")
    return base_dir / "data" / data_arg


def build_test_model(cfg: dict[str, Any], expr_dim: int, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    loss_type = cfg.get("loss_type", "mse")
    output_mode = "zinb" if loss_type == "zinb" else "scalar"
    model = build_model(
        cfg.get("model", "LSTMmodel"),
        expr_dim=expr_dim,
        hidden_size=cfg.get("hidden_size", 128),
        use_vae=cfg.get("use_vae", False),
        vae_encoder_path=cfg.get("vae_encoder_path", None),
        vae_fine_tune=cfg.get("vae_fine_tune", False),
        output_mode=output_mode,
        fusion=cfg.get("fusion", "gate"),
    )
    model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    model.to(device)
    model.eval()
    return model


def predict_model_values(model: torch.nn.Module, promoters: torch.Tensor, exprs: torch.Tensor) -> torch.Tensor:
    if getattr(model, "output_mode", "scalar") == "zinb":
        mu_ratio, _theta, _pi = model(promoters, exprs)
        return torch.log(mu_ratio.squeeze(1) * 1e6 + 1)
    return model(promoters, exprs).squeeze(1)


def compute_target_values(dataset: MyDataset, cell_rows: np.ndarray, gene_idx: int, values: np.ndarray, totals: np.ndarray) -> np.ndarray:
    if dataset.log1p_cpm_target:
        denom = np.maximum(totals[cell_rows] - values, 1.0)
        return np.log1p(values / denom * 1e6)
    return values.astype(np.float64, copy=False)


def select_top_expressed_pairs(dataset: MyDataset, top_n: int, max_pairs_per_gene_ratio: float) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    if max_pairs_per_gene_ratio <= 0:
        raise ValueError("max_pairs_per_gene_ratio must be > 0")

    max_pairs_per_gene = max(1, int(math.floor(top_n * max_pairs_per_gene_ratio)))

    x_csc = dataset.X.tocsc()
    totals = np.asarray(dataset.X.sum(axis=1)).ravel().astype(np.float64)
    cell_row_to_pos = {int(cell_row): cell_pos for cell_pos, cell_row in enumerate(dataset.cells)}
    gene_heaps: dict[str, list[tuple[float, int, int, int, int, float, str]]] = defaultdict(list)

    for pro_i in range(dataset.P):
        gene_idx = int(dataset.promoter2expr_idx[pro_i])
        gene_id = str(dataset.promoters["gene_id"].iloc[pro_i])
        col = x_csc[:, gene_idx]
        cell_rows = col.indices.astype(np.int64, copy=False)
        values = col.data.astype(np.float64, copy=False)
        if len(values) == 0:
            continue

        scores = compute_target_values(dataset, cell_rows, gene_idx, values, totals)
        for cell_row, raw_value, score in zip(cell_rows, values, scores):
            cell_pos = cell_row_to_pos.get(int(cell_row))
            if cell_pos is None:
                continue
            item = (float(score), int(pro_i), int(cell_pos), int(cell_row), gene_idx, float(raw_value), gene_id)
            gene_heap = gene_heaps[gene_id]
            if len(gene_heap) < max_pairs_per_gene:
                heapq.heappush(gene_heap, item)
            elif item[0] > gene_heap[0][0]:
                heapq.heapreplace(gene_heap, item)

    candidates = [item for gene_heap in gene_heaps.values() for item in gene_heap]
    selected = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_n]
    if len(selected) < top_n:
        print(
            f"[Mutagenesis] WARNING: selected {len(selected)} non-zero pairs after gene cap, "
            f"fewer than top_n={top_n}."
        )
    gene_counts = Counter(item[6] for item in selected)

    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        score, pro_i, cell_pos, cell_row, gene_idx, raw_value, gene_id = item
        promoter_row = dataset.promoters.iloc[pro_i]
        obs = dataset.scrna.obs.iloc[cell_row]
        rows.append({
            "rank": rank,
            "target_score": score,
            "raw_target": raw_value,
            "pro_i": pro_i,
            "cell_pos": cell_pos,
            "cell_row": cell_row,
            "cell_id": dataset.scrna.obs_names[cell_row],
            "sample_id": obs.get("sample_id", ""),
            "gene_idx": gene_idx,
            "gene_id": gene_id,
            "gene_pair_count": int(gene_counts[gene_id]),
            "max_pairs_per_gene": int(max_pairs_per_gene),
            "max_pairs_per_gene_ratio": float(max_pairs_per_gene_ratio),
            "chrom": promoter_row.get("chrom", ""),
            "start": promoter_row.get("start", ""),
            "end": promoter_row.get("end", ""),
            "strand": promoter_row.get("strand", ""),
        })
    return pd.DataFrame(rows)


def build_masked_expression(dataset: MyDataset, cell_row: int, target_idx: int) -> tuple[torch.Tensor, float]:
    expr = dataset.X[cell_row]
    if hasattr(expr, "toarray"):
        expr_arr = expr.toarray().astype("float32").squeeze()
    else:
        expr_arr = np.asarray(expr, dtype=np.float32).squeeze()
    y_raw = float(expr_arr[target_idx])
    expr_arr[target_idx] = 0.0
    return torch.from_numpy(expr_arr).float(), y_raw


def centered_window_bounds(position: int, sequence_length: int, window_size: int) -> tuple[int, int]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    window_size = min(window_size, sequence_length)
    left = window_size // 2
    start = position - left
    end = start + window_size
    if start < 0:
        start = 0
        end = window_size
    if end > sequence_length:
        end = sequence_length
        start = max(0, end - window_size)
    return start, end


def plot_pwm_heatmap(pwm: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, pwm.shape[0] * 0.6), 3))
    im = ax.imshow(pwm[["A", "C", "G", "T"]].to_numpy().T, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(pwm.shape[0]))
    ax.set_xticklabels(pwm["motif_position"].astype(str).tolist())
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["A", "C", "G", "T"])
    ax.set_xlabel("Motif position")
    ax.set_ylabel("Base")
    ax.set_title("Important motif PWM")
    fig.colorbar(im, ax=ax, label="Weighted frequency")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_de_novo_motif_outputs(
    dataset: MyDataset,
    effects: pd.DataFrame,
    top_pairs: pd.DataFrame,
    mut_dir: Path,
    motif_window_size: int,
    motif_top_windows: int,
    motif_top_k: int,
    motif_min_support: int,
) -> None:
    motif_window_columns = [
        "rank",
        "pro_i",
        "gene_id",
        "cell_id",
        "important_position_0based",
        "important_position_1based",
        "window_start_0based",
        "window_end_0based",
        "window_start_1based",
        "window_end_1based",
        "motif_sequence",
        "mean_abs_delta",
        "mean_signed_delta",
    ]
    motif_columns = [
        "motif_sequence",
        "support_pairs",
        "support_genes",
        "mean_abs_delta",
        "sum_abs_delta",
        "mean_signed_delta",
    ]
    pwm_columns = ["motif_position", "A", "C", "G", "T"]

    if effects.empty or top_pairs.empty:
        pd.DataFrame(columns=motif_window_columns).to_csv(mut_dir / "motif_windows.csv", index=False)
        pd.DataFrame(columns=motif_columns).to_csv(mut_dir / "de_novo_motifs.csv", index=False)
        pd.DataFrame(columns=pwm_columns).to_csv(mut_dir / "important_motif_pwm.csv", index=False)
        return

    pair_position = (
        effects
        .groupby(["rank", "pro_i", "cell_id", "gene_id", "position_0based", "position_1based"], as_index=False)
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            mean_signed_delta=("delta", "mean"),
        )
    )
    idx = pair_position.groupby("rank")["mean_abs_delta"].idxmax()
    important_positions = (
        pair_position
        .loc[idx]
        .sort_values("mean_abs_delta", ascending=False)
        .head(max(0, motif_top_windows))
    )

    window_rows: list[dict[str, Any]] = []
    for row in important_positions.itertuples(index=False):
        pro_i = int(row.pro_i)
        seq = str(dataset.promoters["sequence"].iloc[pro_i]).upper()
        pos = int(row.position_0based)
        start, end = centered_window_bounds(pos, len(seq), motif_window_size)
        window_rows.append({
            "rank": int(row.rank),
            "pro_i": pro_i,
            "gene_id": str(row.gene_id),
            "cell_id": str(row.cell_id),
            "important_position_0based": pos,
            "important_position_1based": int(row.position_1based),
            "window_start_0based": start,
            "window_end_0based": end,
            "window_start_1based": start + 1,
            "window_end_1based": end,
            "motif_sequence": seq[start:end],
            "mean_abs_delta": float(row.mean_abs_delta),
            "mean_signed_delta": float(row.mean_signed_delta),
        })

    motif_windows = pd.DataFrame(window_rows, columns=motif_window_columns)
    motif_windows.to_csv(mut_dir / "motif_windows.csv", index=False)

    if motif_windows.empty:
        pd.DataFrame(columns=motif_columns).to_csv(mut_dir / "de_novo_motifs.csv", index=False)
        pd.DataFrame(columns=pwm_columns).to_csv(mut_dir / "important_motif_pwm.csv", index=False)
        return

    motif_summary = (
        motif_windows
        .groupby("motif_sequence", as_index=False)
        .agg(
            support_pairs=("rank", "nunique"),
            support_genes=("gene_id", "nunique"),
            mean_abs_delta=("mean_abs_delta", "mean"),
            sum_abs_delta=("mean_abs_delta", "sum"),
            mean_signed_delta=("mean_signed_delta", "mean"),
        )
    )
    motif_summary = (
        motif_summary[motif_summary["support_pairs"] >= motif_min_support]
        .sort_values(["support_genes", "sum_abs_delta", "mean_abs_delta"], ascending=False)
        .head(max(0, motif_top_k))
    )
    motif_summary.to_csv(mut_dir / "de_novo_motifs.csv", index=False)

    pwm_counts = np.full((motif_window_size, 4), 1e-6, dtype=np.float64)
    base_to_col = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row in motif_windows.itertuples(index=False):
        seq = str(row.motif_sequence).upper()
        weight = max(float(row.mean_abs_delta), 0.0)
        for pos, base in enumerate(seq[:motif_window_size]):
            col = base_to_col.get(base)
            if col is not None:
                pwm_counts[pos, col] += weight
    row_sums = np.maximum(pwm_counts.sum(axis=1, keepdims=True), 1e-12)
    pwm_values = pwm_counts / row_sums
    pwm = pd.DataFrame(pwm_values, columns=["A", "C", "G", "T"])
    pwm.insert(0, "motif_position", np.arange(1, motif_window_size + 1))
    pwm.to_csv(mut_dir / "important_motif_pwm.csv", index=False)
    plot_pwm_heatmap(pwm, mut_dir / "important_motif_pwm.png")


def reverse_complement(seq: str) -> str:
    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(complement)[::-1].upper()


def parse_meme_motifs(motif_file: Path) -> list[dict[str, Any]]:
    motifs: list[dict[str, Any]] = []
    current_name: str | None = None
    lines = motif_file.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF"):
            parts = line.split()
            current_name = parts[1] if len(parts) > 1 else f"motif_{len(motifs) + 1}"
            i += 1
            continue
        if current_name is not None and line.startswith("letter-probability matrix"):
            width_match = re.search(r"\bw=\s*(\d+)", line)
            width = int(width_match.group(1)) if width_match else 0
            matrix: list[list[float]] = []
            i += 1
            while i < len(lines):
                values = lines[i].strip().split()
                if len(values) < 4:
                    break
                try:
                    row = [float(values[j]) for j in range(4)]
                except ValueError:
                    break
                matrix.append(row)
                i += 1
                if width > 0 and len(matrix) >= width:
                    break
            if matrix:
                pwm = np.asarray(matrix, dtype=np.float64)
                row_sums = np.maximum(pwm.sum(axis=1, keepdims=True), 1e-12)
                motifs.append({"name": current_name, "pwm": pwm / row_sums})
            current_name = None
            continue
        i += 1
    return motifs


def pwm_log_odds(pwm: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(pwm, 1e-9) / 0.25)


def score_pwm_window(seq: str, log_odds: np.ndarray) -> float | None:
    base_to_col = {"A": 0, "C": 1, "G": 2, "T": 3}
    score = 0.0
    for pos, base in enumerate(seq.upper()):
        col = base_to_col.get(base)
        if col is None:
            return None
        score += float(log_odds[pos, col])
    return score


def write_known_motif_outputs(
    dataset: MyDataset,
    effects: pd.DataFrame,
    top_pairs: pd.DataFrame,
    mut_dir: Path,
    known_motif_file: str | None,
    known_motif_min_score_ratio: float,
    known_motif_max_hits_per_motif: int,
) -> None:
    hit_columns = [
        "motif_name",
        "pro_i",
        "gene_id",
        "chrom",
        "promoter_start",
        "promoter_end",
        "hit_start_0based",
        "hit_end_0based",
        "hit_start_1based",
        "hit_end_1based",
        "strand",
        "matched_sequence",
        "pwm_score",
        "importance_score",
    ]
    summary_columns = [
        "motif_name",
        "hit_count",
        "support_genes",
        "mean_pwm_score",
        "max_pwm_score",
        "mean_importance_score",
        "max_importance_score",
    ]

    if known_motif_file is None:
        return

    motif_path = Path(known_motif_file)
    if not motif_path.exists():
        raise FileNotFoundError(f"Known motif file not found: {motif_path}")

    motifs = parse_meme_motifs(motif_path)
    if not motifs:
        print(f"[Mutagenesis] WARNING: no MEME motifs parsed from {motif_path}.")
        pd.DataFrame(columns=hit_columns).to_csv(mut_dir / "known_motif_hits.csv", index=False)
        pd.DataFrame(columns=summary_columns).to_csv(mut_dir / "known_motif_summary.csv", index=False)
        return

    if effects.empty or top_pairs.empty:
        pd.DataFrame(columns=hit_columns).to_csv(mut_dir / "known_motif_hits.csv", index=False)
        pd.DataFrame(columns=summary_columns).to_csv(mut_dir / "known_motif_summary.csv", index=False)
        return

    position_importance = (
        effects
        .groupby(["pro_i", "position_0based"], as_index=False)
        .agg(importance_score=("abs_delta", "mean"))
    )
    importance_by_promoter: dict[int, np.ndarray] = {}
    for pro_i, rows in position_importance.groupby("pro_i"):
        promoter_len = len(str(dataset.promoters["sequence"].iloc[int(pro_i)]))
        arr = np.zeros(promoter_len, dtype=np.float64)
        pos = rows["position_0based"].to_numpy(dtype=np.int64)
        vals = rows["importance_score"].to_numpy(dtype=np.float64)
        keep = (pos >= 0) & (pos < promoter_len)
        arr[pos[keep]] = vals[keep]
        importance_by_promoter[int(pro_i)] = arr

    unique_promoters = sorted(set(int(pro_i) for pro_i in top_pairs["pro_i"].tolist()))
    all_hits: list[dict[str, Any]] = []
    for motif in motifs:
        motif_name = str(motif["name"])
        pwm = np.asarray(motif["pwm"], dtype=np.float64)
        width = int(pwm.shape[0])
        if width <= 0:
            continue
        log_odds = pwm_log_odds(pwm)
        max_score = float(np.max(log_odds, axis=1).sum())
        min_score = known_motif_min_score_ratio * max_score
        motif_hits: list[tuple[float, float, int, dict[str, Any]]] = []
        hit_counter = 0

        for pro_i in unique_promoters:
            promoter_row = dataset.promoters.iloc[pro_i]
            seq = str(promoter_row["sequence"]).upper()
            if len(seq) < width:
                continue
            importance = importance_by_promoter.get(pro_i, np.zeros(len(seq), dtype=np.float64))
            for start in range(0, len(seq) - width + 1):
                forward_seq = seq[start:start + width]
                for strand, matched_seq in (("+", forward_seq), ("-", reverse_complement(forward_seq))):
                    score = score_pwm_window(matched_seq, log_odds)
                    if score is None or score < min_score:
                        continue
                    end = start + width
                    importance_score = float(np.mean(importance[start:end])) if end <= len(importance) else 0.0
                    hit = {
                        "motif_name": motif_name,
                        "pro_i": pro_i,
                        "gene_id": str(promoter_row["gene_id"]),
                        "chrom": promoter_row.get("chrom", ""),
                        "promoter_start": promoter_row.get("start", ""),
                        "promoter_end": promoter_row.get("end", ""),
                        "hit_start_0based": start,
                        "hit_end_0based": end,
                        "hit_start_1based": start + 1,
                        "hit_end_1based": end,
                        "strand": strand,
                        "matched_sequence": matched_seq,
                        "pwm_score": float(score),
                        "importance_score": importance_score,
                    }
                    hit_counter += 1
                    if known_motif_max_hits_per_motif <= 0:
                        motif_hits.append((importance_score, float(score), hit_counter, hit))
                    elif len(motif_hits) < known_motif_max_hits_per_motif:
                        heapq.heappush(motif_hits, (importance_score, float(score), hit_counter, hit))
                    else:
                        sort_key = (importance_score, float(score))
                        worst_key = (motif_hits[0][0], motif_hits[0][1])
                        if sort_key > worst_key:
                            heapq.heapreplace(motif_hits, (importance_score, float(score), hit_counter, hit))
        all_hits.extend(hit for _importance, _score, _counter, hit in motif_hits)

    hits = pd.DataFrame(all_hits, columns=hit_columns)
    if hits.empty:
        hits.to_csv(mut_dir / "known_motif_hits.csv", index=False)
        pd.DataFrame(columns=summary_columns).to_csv(mut_dir / "known_motif_summary.csv", index=False)
        return

    hits = hits.sort_values(["importance_score", "pwm_score"], ascending=False)
    hits.to_csv(mut_dir / "known_motif_hits.csv", index=False)
    known_summary = (
        hits
        .groupby("motif_name", as_index=False)
        .agg(
            hit_count=("motif_name", "count"),
            support_genes=("gene_id", "nunique"),
            mean_pwm_score=("pwm_score", "mean"),
            max_pwm_score=("pwm_score", "max"),
            mean_importance_score=("importance_score", "mean"),
            max_importance_score=("importance_score", "max"),
        )
        .sort_values(["max_importance_score", "mean_importance_score", "hit_count"], ascending=False)
    )
    known_summary.to_csv(mut_dir / "known_motif_summary.csv", index=False)


def run_sequence_mutagenesis(
    model: torch.nn.Module,
    dataset: MyDataset,
    output_dir: Path,
    top_n: int,
    mutation_batch_size: int,
    max_pairs_per_gene_ratio: float,
    motif_window_size: int,
    motif_top_windows: int,
    motif_top_k: int,
    motif_min_support: int,
    known_motif_file: str | None,
    known_motif_min_score_ratio: float,
    known_motif_max_hits_per_motif: int,
    device: torch.device,
) -> None:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    canonical_bases = ["A", "C", "G", "T"]

    mut_dir = output_dir / "sequence_mutagenesis"
    mut_dir.mkdir(parents=True, exist_ok=True)

    top_pairs = select_top_expressed_pairs(
        dataset,
        top_n=top_n,
        max_pairs_per_gene_ratio=max_pairs_per_gene_ratio,
    )
    top_pairs.to_csv(mut_dir / f"top{top_n}_pairs.csv", index=False)
    max_pairs = int(top_pairs["max_pairs_per_gene"].iloc[0]) if len(top_pairs) > 0 else 0
    print(f"[Mutagenesis] Selected {len(top_pairs)} top expressed test pairs with <= {max_pairs} pairs per gene.")

    long_rows: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for pair in top_pairs.itertuples(index=False):
            pro_i = int(pair.pro_i)
            cell_row = int(pair.cell_row)
            target_idx = int(pair.gene_idx)
            seq = str(dataset.promoters["sequence"].iloc[pro_i]).upper()

            wt_promoter = dataset.get_promoter_tensor(pro_i).to(device)
            expr_vec, y_raw = build_masked_expression(dataset, cell_row, target_idx)
            expr_vec = expr_vec.to(device)

            wt_pred = float(predict_model_values(
                model,
                wt_promoter.unsqueeze(0),
                expr_vec.unsqueeze(0),
            ).detach().cpu().item())

            mutated_tensors: list[torch.Tensor] = []
            mutation_meta: list[tuple[int, str, str]] = []
            for pos in range(400):
                ref_base = seq[pos] if pos < len(seq) else "N"
                if ref_base not in base_to_idx:
                    ref_base = "N"
                alt_bases = canonical_bases if ref_base == "N" else [b for b in canonical_bases if b != ref_base]
                for alt_base in alt_bases:
                    mutated = wt_promoter.detach().cpu().clone()
                    mutated[pos, :] = 0.0
                    mutated[pos, base_to_idx[alt_base]] = 1.0
                    mutated_tensors.append(mutated)
                    mutation_meta.append((pos, ref_base, alt_base))

            expr_batch_base = expr_vec.unsqueeze(0)
            mut_preds: list[np.ndarray] = []
            for start in range(0, len(mutated_tensors), mutation_batch_size):
                end = min(start + mutation_batch_size, len(mutated_tensors))
                promoter_batch = torch.stack(mutated_tensors[start:end], dim=0).to(device)
                expr_batch = expr_batch_base.expand(len(promoter_batch), -1)
                pred_batch = predict_model_values(model, promoter_batch, expr_batch)
                mut_preds.append(pred_batch.detach().cpu().numpy())
            pred_values = np.concatenate(mut_preds)

            for (pos, ref_base, alt_base), mut_pred in zip(mutation_meta, pred_values):
                delta = float(mut_pred - wt_pred)
                long_rows.append({
                    "rank": int(pair.rank),
                    "pro_i": pro_i,
                    "cell_row": cell_row,
                    "cell_id": pair.cell_id,
                    "gene_id": pair.gene_id,
                    "target_score": float(pair.target_score),
                    "raw_target": y_raw,
                    "wt_pred": wt_pred,
                    "position_0based": pos,
                    "position_1based": pos + 1,
                    "ref_base": ref_base,
                    "alt_base": alt_base,
                    "mut_pred": float(mut_pred),
                    "delta": delta,
                    "abs_delta": abs(delta),
                })

    effects = pd.DataFrame(long_rows)
    effects.to_csv(mut_dir / "mutation_effects_long.csv", index=False)

    summary = (
        effects
        .groupby(["position_0based", "position_1based"], as_index=False)
        .agg(
            median_signed_delta=("delta", "median"),
            median_abs_delta=("abs_delta", "median"),
            mean_signed_delta=("delta", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
            n_mutations=("delta", "count"),
        )
    )
    summary.to_csv(mut_dir / "position_importance.csv", index=False)

    write_de_novo_motif_outputs(
        dataset=dataset,
        effects=effects,
        top_pairs=top_pairs,
        mut_dir=mut_dir,
        motif_window_size=motif_window_size,
        motif_top_windows=motif_top_windows,
        motif_top_k=motif_top_k,
        motif_min_support=motif_min_support,
    )
    write_known_motif_outputs(
        dataset=dataset,
        effects=effects,
        top_pairs=top_pairs,
        mut_dir=mut_dir,
        known_motif_file=known_motif_file,
        known_motif_min_score_ratio=known_motif_min_score_ratio,
        known_motif_max_hits_per_motif=known_motif_max_hits_per_motif,
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(summary["position_1based"], summary["median_signed_delta"], linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Promoter position (1-based)")
    ax.set_ylabel("Median signed delta")
    ax.set_title("Sequence mutagenesis: median signed effect")
    plt.tight_layout()
    fig.savefig(mut_dir / "position_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(summary["position_1based"], summary["median_abs_delta"], linewidth=1)
    ax.set_xlabel("Promoter position (1-based)")
    ax.set_ylabel("Median |delta|")
    ax.set_title("Sequence mutagenesis: median absolute effect")
    plt.tight_layout()
    fig.savefig(mut_dir / "position_importance_abs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(summary["position_1based"], summary["mean_abs_delta"], linewidth=1)
    ax.set_xlabel("Promoter position (1-based)")
    ax.set_ylabel("Mean |delta|")
    ax.set_title("Sequence mutagenesis: mean absolute effect")
    plt.tight_layout()
    fig.savefig(mut_dir / "position_importance_mean_abs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Mutagenesis] Outputs saved to: {mut_dir}")


def update_spearman_reservoir(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reservoir_true: list[np.ndarray],
    reservoir_pred: list[np.ndarray],
    seen: int,
    limit: int,
    rng: np.random.Generator,
) -> int:
    if limit == 0:
        reservoir_true.append(y_true.astype(np.float32, copy=False))
        reservoir_pred.append(y_pred.astype(np.float32, copy=False))
        return seen + len(y_true)

    remaining = max(limit - seen, 0)
    if remaining > 0:
        take = min(remaining, len(y_true))
        reservoir_true.append(y_true[:take].astype(np.float32, copy=False))
        reservoir_pred.append(y_pred[:take].astype(np.float32, copy=False))
        y_true = y_true[take:]
        y_pred = y_pred[take:]
        seen += take

    if len(y_true) == 0:
        return seen

    if len(reservoir_true) == 1 and len(reservoir_true[0]) == limit:
        true_arr = reservoir_true[0]
        pred_arr = reservoir_pred[0]
    else:
        true_arr = np.concatenate(reservoir_true)
        pred_arr = np.concatenate(reservoir_pred)
    for yt, yp in zip(y_true, y_pred):
        seen += 1
        j = int(rng.integers(0, seen))
        if j < limit:
            true_arr[j] = yt
            pred_arr[j] = yp
    reservoir_true[:] = [true_arr]
    reservoir_pred[:] = [pred_arr]
    return seen


def compute_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    spearman_samples: int,
    seed: int,
) -> dict[str, float | int]:
    ''' compute
    '''
    is_zinb = getattr(model, "output_mode", "scalar") == "zinb"
    rng = np.random.default_rng(seed)

    count = 0
    sum_y = 0.0
    sum_p = 0.0
    sum_yy = 0.0
    sum_pp = 0.0
    sum_yp = 0.0
    sse = 0.0
    reservoir_true: list[np.ndarray] = []
    reservoir_pred: list[np.ndarray] = []
    spearman_seen = 0

    with torch.no_grad():
        for promoters, exprs, ys in loader:
            promoters = promoters.to(device, non_blocking=True)
            exprs = exprs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True).float()

            if is_zinb:
                mu_ratio, _theta, _pi = model(promoters, exprs)
                mu_ratio = mu_ratio.squeeze(1)
                lib_size = exprs.sum(dim=1) + ys
                pred = torch.log(mu_ratio * 1e6 + 1)
                true = torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6)
            else:
                pred = model(promoters, exprs).squeeze(1)
                true = ys

            if max_samples > 0 and count + true.numel() > max_samples:
                keep = max_samples - count
                true = true[:keep]
                pred = pred[:keep]

            y = true.detach().cpu().numpy().astype(np.float64, copy=False)
            p = pred.detach().cpu().numpy().astype(np.float64, copy=False)

            count += int(y.size)
            diff = p - y
            sse += float(np.sum(diff * diff))
            sum_y += float(np.sum(y))
            sum_p += float(np.sum(p))
            sum_yy += float(np.sum(y * y))
            sum_pp += float(np.sum(p * p))
            sum_yp += float(np.sum(y * p))

            spearman_seen = update_spearman_reservoir(
                y, p, reservoir_true, reservoir_pred, spearman_seen, spearman_samples, rng
            )

            if max_samples > 0 and count >= max_samples:
                break

    mse = sse / max(count, 1)
    denom_y = sum_yy - sum_y * sum_y / max(count, 1)
    denom_p = sum_pp - sum_p * sum_p / max(count, 1)
    denom = float(np.sqrt(max(denom_y, 0.0) * max(denom_p, 0.0)))
    pearson = (sum_yp - sum_y * sum_p / max(count, 1)) / denom if denom > 0 else float("nan")

    spearman_true = np.concatenate(reservoir_true) if reservoir_true else np.array([], dtype=np.float32)
    spearman_pred = np.concatenate(reservoir_pred) if reservoir_pred else np.array([], dtype=np.float32)
    if len(spearman_true) > 1:
        try:
            from scipy.stats import spearmanr
            spearman = float(spearmanr(spearman_true, spearman_pred)[0])
        except Exception:
            spearman = float("nan")
    else:
        spearman = float("nan")

    return {
        "num_samples": int(count),
        "mse": float(mse),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
        "spearman_num_samples": int(len(spearman_true)),
    }


def shuffle_promoter_batch(promoters: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if promoters.ndim != 3:
        raise ValueError("promoters must have shape (batch, length, channels)")
    length = promoters.shape[1]
    perms = [torch.randperm(length, generator=generator, device="cpu") for _ in range(promoters.shape[0])]
    perm_tensor = torch.stack(perms, dim=0).to(promoters.device)
    batch_idx = torch.arange(promoters.shape[0], device=promoters.device).unsqueeze(1)
    return promoters[batch_idx, perm_tensor, :]


def shuffle_expression_batch(exprs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if exprs.ndim != 2:
        raise ValueError("exprs must have shape (batch, expr_dim)")
    batch_size = exprs.shape[0]
    if batch_size <= 1:
        return exprs.clone()
    perm = torch.randperm(batch_size, generator=generator, device="cpu").to(exprs.device)
    if torch.equal(perm, torch.arange(batch_size, device=exprs.device)):
        perm = torch.roll(perm, shifts=1)
    return exprs[perm]


def predict_batch_values(
    model: torch.nn.Module,
    promoters: torch.Tensor,
    exprs: torch.Tensor,
) -> torch.Tensor:
    if getattr(model, "output_mode", "scalar") == "zinb":
        mu_ratio, _theta, _pi = model(promoters, exprs)
        return torch.log(mu_ratio.squeeze(1) * 1e6 + 1)
    return model(promoters, exprs).squeeze(1)


def compute_batch_true_values(
    model: torch.nn.Module,
    exprs: torch.Tensor,
    ys: torch.Tensor,
) -> torch.Tensor:
    if getattr(model, "output_mode", "scalar") == "zinb":
        lib_size = exprs.sum(dim=1) + ys
        return torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6)
    return ys


def _empty_ablation_state() -> dict[str, Any]:
    return {
        "count": 0,
        "sum_y": 0.0,
        "sum_p": 0.0,
        "sum_yy": 0.0,
        "sum_pp": 0.0,
        "sum_yp": 0.0,
        "sse": 0.0,
        "pred_delta_abs_sum": 0.0,
        "pred_delta_sum": 0.0,
        "reservoir_true": [],
        "reservoir_pred": [],
        "spearman_seen": 0,
    }


def _update_ablation_state(
    state: dict[str, Any],
    y: np.ndarray,
    p: np.ndarray,
    original_pred: np.ndarray,
    spearman_samples: int,
    rng: np.random.Generator,
) -> None:
    state["count"] += int(y.size)
    diff = p - y
    pred_delta = p - original_pred
    state["sse"] += float(np.sum(diff * diff))
    state["sum_y"] += float(np.sum(y))
    state["sum_p"] += float(np.sum(p))
    state["sum_yy"] += float(np.sum(y * y))
    state["sum_pp"] += float(np.sum(p * p))
    state["sum_yp"] += float(np.sum(y * p))
    state["pred_delta_abs_sum"] += float(np.sum(np.abs(pred_delta)))
    state["pred_delta_sum"] += float(np.sum(pred_delta))
    state["spearman_seen"] = update_spearman_reservoir(
        y,
        p,
        state["reservoir_true"],
        state["reservoir_pred"],
        int(state["spearman_seen"]),
        spearman_samples,
        rng,
    )


def _finalize_ablation_state(condition: str, repeat: int, state: dict[str, Any]) -> dict[str, float | int | str]:
    count = int(state["count"])
    mse = float(state["sse"]) / max(count, 1)
    denom_y = float(state["sum_yy"]) - float(state["sum_y"]) * float(state["sum_y"]) / max(count, 1)
    denom_p = float(state["sum_pp"]) - float(state["sum_p"]) * float(state["sum_p"]) / max(count, 1)
    denom = float(np.sqrt(max(denom_y, 0.0) * max(denom_p, 0.0)))
    pearson = (float(state["sum_yp"]) - float(state["sum_y"]) * float(state["sum_p"]) / max(count, 1)) / denom if denom > 0 else float("nan")

    reservoir_true = state["reservoir_true"]
    reservoir_pred = state["reservoir_pred"]
    spearman_true = np.concatenate(reservoir_true) if reservoir_true else np.array([], dtype=np.float32)
    spearman_pred = np.concatenate(reservoir_pred) if reservoir_pred else np.array([], dtype=np.float32)
    if len(spearman_true) > 1:
        try:
            from scipy.stats import spearmanr
            spearman = float(spearmanr(spearman_true, spearman_pred)[0])
        except Exception:
            spearman = float("nan")
    else:
        spearman = float("nan")

    return {
        "repeat": int(repeat),
        "condition": condition,
        "num_samples": count,
        "mse": float(mse),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
        "spearman_num_samples": int(len(spearman_true)),
        "mean_abs_pred_delta": float(state["pred_delta_abs_sum"]) / max(count, 1),
        "mean_pred_delta": float(state["pred_delta_sum"]) / max(count, 1),
    }


def compute_input_ablation_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    spearman_samples: int,
    seed: int,
    repeats: int,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    conditions = ("original", "shuffle_promoter", "shuffle_expression", "shuffle_both")
    rows: list[dict[str, float | int | str]] = []
    model.eval()

    with torch.no_grad():
        for repeat in range(repeats):
            torch_rng = torch.Generator(device="cpu")
            torch_rng.manual_seed(seed + repeat)
            spearman_rngs = {
                condition: np.random.default_rng(seed + repeat + idx)
                for idx, condition in enumerate(conditions)
            }
            states = {condition: _empty_ablation_state() for condition in conditions}
            count = 0

            for promoters, exprs, ys in loader:
                promoters = promoters.to(device, non_blocking=True)
                exprs = exprs.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True).float()

                true = compute_batch_true_values(model, exprs, ys)
                original_pred = predict_batch_values(model, promoters, exprs)

                if max_samples > 0 and count + true.numel() > max_samples:
                    keep = max_samples - count
                    true = true[:keep]
                    original_pred = original_pred[:keep]
                    promoters = promoters[:keep]
                    exprs = exprs[:keep]
                else:
                    keep = int(true.numel())

                perturbed_inputs = {
                    "original": (promoters, exprs),
                    "shuffle_promoter": (shuffle_promoter_batch(promoters, torch_rng), exprs),
                    "shuffle_expression": (promoters, shuffle_expression_batch(exprs, torch_rng)),
                    "shuffle_both": (
                        shuffle_promoter_batch(promoters, torch_rng),
                        shuffle_expression_batch(exprs, torch_rng),
                    ),
                }

                y = true.detach().cpu().numpy().astype(np.float64, copy=False)
                original_np = original_pred.detach().cpu().numpy().astype(np.float64, copy=False)

                for condition, (condition_promoters, condition_exprs) in perturbed_inputs.items():
                    if condition == "original":
                        pred = original_pred
                    else:
                        pred = predict_batch_values(model, condition_promoters, condition_exprs)
                    p = pred[:keep].detach().cpu().numpy().astype(np.float64, copy=False)
                    _update_ablation_state(states[condition], y, p, original_np, spearman_samples, spearman_rngs[condition])

                count += keep
                if max_samples > 0 and count >= max_samples:
                    break

            for condition in conditions:
                rows.append(_finalize_ablation_state(condition, repeat, states[condition]))

    df = pd.DataFrame(rows)
    baseline = df[df["condition"] == "original"][["repeat", "mse", "pearson_r", "spearman_r"]].rename(
        columns={
            "mse": "original_mse",
            "pearson_r": "original_pearson_r",
            "spearman_r": "original_spearman_r",
        }
    )
    df = df.merge(baseline, on="repeat", how="left")
    df["delta_mse"] = df["mse"] - df["original_mse"]
    df["delta_pearson_r"] = df["pearson_r"] - df["original_pearson_r"]
    df["delta_spearman_r"] = df["spearman_r"] - df["original_spearman_r"]
    return df


def write_input_ablation_outputs(metrics: pd.DataFrame, output_dir: Path) -> None:
    ablation_dir = output_dir / "input_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ablation_dir / "input_ablation_metrics.csv"
    json_path = ablation_dir / "input_ablation_metrics.json"
    metrics.to_csv(csv_path, index=False)

    summary = (
        metrics
        .groupby("condition", as_index=False)
        .agg(
            num_samples=("num_samples", "max"),
            mse=("mse", "mean"),
            pearson_r=("pearson_r", "mean"),
            spearman_r=("spearman_r", "mean"),
            mean_abs_pred_delta=("mean_abs_pred_delta", "mean"),
            mean_pred_delta=("mean_pred_delta", "mean"),
            delta_mse=("delta_mse", "mean"),
            delta_pearson_r=("delta_pearson_r", "mean"),
            delta_spearman_r=("delta_spearman_r", "mean"),
        )
    )
    payload = {
        "per_repeat": metrics.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    condition_order = ["original", "shuffle_promoter", "shuffle_expression", "shuffle_both"]
    summary = summary.set_index("condition").reindex(condition_order).reset_index()
    for metric_name, file_name, ylabel in (
        ("pearson_r", "input_ablation_pearson.png", "Pearson r"),
        ("mse", "input_ablation_mse.png", "MSE"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summary["condition"], summary[metric_name], color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Condition")
        ax.set_title(f"Input ablation: {ylabel}")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        fig.savefig(ablation_dir / file_name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("========== Input Ablation ==========")
    for row in summary.itertuples(index=False):
        print(
            f"{row.condition}: pearson={row.pearson_r:.6f} mse={row.mse:.6f} "
            f"delta_pearson={row.delta_pearson_r:.6f} delta_mse={row.delta_mse:.6f} "
            f"mean_abs_pred_delta={row.mean_abs_pred_delta:.6f}"
        )
    print(f"outputs: {ablation_dir}")


def main() -> None:
    args = parse_args()
    base_dir = PROJECT_ROOT
    run_dir = base_dir / "outputs" / args.exp_name
    test_dir = run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(run_dir)
    data_dir = resolve_data_dir(base_dir, cfg, args.data)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best_model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    loss_type = cfg.get("loss_type", "mse")
    is_umi = data_dir.name.startswith("umi_") and loss_type != "zinb"
    dataset = MyDataset(
        promoter_file=data_dir / "promoter_test.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        mode="test",
        seed=args.seed,
        log1p_cpm_target=is_umi,
        preencode_promoters=args.preencode_promoters,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expr_dim = int(cfg.get("expr_dim", dataset.X.shape[1]))
    model = build_test_model(cfg, expr_dim=expr_dim, checkpoint=checkpoint, device=device)

    if not args.skip_standard_test:
        metrics = compute_test_metrics(
            model=model,
            loader=loader,
            device=device,
            max_samples=args.max_samples,
            spearman_samples=args.spearman_samples,
            seed=args.seed,
        )
        metrics.update({
            "exp_name": args.exp_name,
            "data_dir": str(data_dir),
            "checkpoint": str(checkpoint),
            "loss_type": loss_type,
            "max_samples": int(args.max_samples),
        })

        metrics_path = test_dir / "test_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame([metrics]).to_csv(test_dir / "test_metrics.csv", index=False)

        utils.plot_per_promoter_scatter(
            model,
            dataset,
            is_umi=is_umi,
            n_promoters=3,
            save_path=test_dir / "per_promoter_scatter.png",
        )
        utils.plot_per_cell_scatter(
            model,
            dataset,
            is_umi=is_umi,
            n_cells=3,
            save_path=test_dir / "per_cell_scatter.png",
        )

        print("========== Test ==========")
        print(f"data: {data_dir}")
        print(f"checkpoint: {checkpoint}")
        print(f"samples: {metrics['num_samples']}")
        print(f"mse: {metrics['mse']:.6f}")
        print(f"pearson_r: {metrics['pearson_r']:.6f}")
        print(f"spearman_r: {metrics['spearman_r']:.6f} (n={metrics['spearman_num_samples']})")
        print(f"outputs: {test_dir}")

    if args.run_input_ablation:
        ablation_max_samples = args.max_samples if args.ablation_max_samples is None else args.ablation_max_samples
        ablation_metrics = compute_input_ablation_metrics(
            model=model,
            loader=loader,
            device=device,
            max_samples=ablation_max_samples,
            spearman_samples=args.spearman_samples,
            seed=args.seed,
            repeats=args.ablation_repeats,
        )
        write_input_ablation_outputs(ablation_metrics, test_dir)

    if args.run_mutagenesis:
        run_sequence_mutagenesis(
            model=model,
            dataset=dataset,
            output_dir=test_dir,
            top_n=args.top_n,
            mutation_batch_size=args.mutation_batch_size,
            max_pairs_per_gene_ratio=args.max_pairs_per_gene_ratio,
            motif_window_size=args.motif_window_size,
            motif_top_windows=args.motif_top_windows,
            motif_top_k=args.motif_top_k,
            motif_min_support=args.motif_min_support,
            known_motif_file=args.known_motif_file,
            known_motif_min_score_ratio=args.known_motif_min_score_ratio,
            known_motif_max_hits_per_motif=args.known_motif_max_hits_per_motif,
            device=device,
        )


if __name__ == "__main__":
    main()
