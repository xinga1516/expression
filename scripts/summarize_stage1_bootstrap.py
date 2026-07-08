from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (1, 7, 42)
COMPARISONS = {
    "promoter_vs_intergenic": "stage1_shift420_intergenic_seed{seed}",
    "promoter_vs_exprmatched": "stage1_shift420_exprmatched_seed{seed}",
}
LEVELS = {
    "per_gene": ("gene_id", "per_gene_metrics.csv"),
    "per_cell": ("cell_id", "per_cell_metrics.csv"),
}

MODEL_RUNS = {
    "exprmatched": ("Expression baseline", "stage1_shift420_exprmatched_seed{seed}"),
    "intergenic": ("Matched intergenic", "stage1_shift420_intergenic_seed{seed}"),
    "promoter": ("Real promoter", "stage1_shift420_promoter_seed{seed}"),
}
MODEL_COLORS = {
    "exprmatched": "#4C78A8",
    "intergenic": "#F58518",
    "promoter": "#54A24B",
}


def percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def bootstrap_paired_delta(
    deltas: np.ndarray,
    repeats: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError("deltas must be a non-empty one-dimensional array")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    mean_samples = np.empty(repeats, dtype=np.float64)
    median_samples = np.empty(repeats, dtype=np.float64)
    chunk_size = 250
    for start in range(0, repeats, chunk_size):
        end = min(start + chunk_size, repeats)
        indices = rng.integers(0, deltas.size, size=(end - start, deltas.size))
        sampled = deltas[indices]
        mean_samples[start:end] = sampled.mean(axis=1)
        median_samples[start:end] = np.median(sampled, axis=1)

    mean_low, mean_high = percentile_interval(mean_samples, confidence)
    median_low, median_high = percentile_interval(median_samples, confidence)
    return {
        "mean_delta": float(deltas.mean()),
        "mean_ci_low": mean_low,
        "mean_ci_high": mean_high,
        "median_delta": float(np.median(deltas)),
        "median_ci_low": median_low,
        "median_ci_high": median_high,
        "win_fraction": float(np.mean(deltas > 0)),
    }


def hierarchical_seed_bootstrap(
    deltas_by_seed: dict[int, np.ndarray],
    repeats: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    seeds = np.asarray(sorted(deltas_by_seed), dtype=np.int64)
    if seeds.size == 0:
        raise ValueError("deltas_by_seed must not be empty")

    bootstrap_means = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_seeds = rng.choice(seeds, size=seeds.size, replace=True)
        seed_means = []
        for sampled_seed in sampled_seeds:
            deltas = deltas_by_seed[int(sampled_seed)]
            sampled_pairs = rng.choice(deltas, size=deltas.size, replace=True)
            seed_means.append(float(sampled_pairs.mean()))
        bootstrap_means[repeat] = float(np.mean(seed_means))

    ci_low, ci_high = percentile_interval(bootstrap_means, confidence)
    observed_seed_means = np.asarray(
        [deltas_by_seed[int(seed)].mean() for seed in seeds], dtype=np.float64
    )
    return {
        "seed_count": int(seeds.size),
        "equal_seed_mean_delta": float(observed_seed_means.mean()),
        "mean_ci_low": ci_low,
        "mean_ci_high": ci_high,
        "min_seed_mean_delta": float(observed_seed_means.min()),
        "max_seed_mean_delta": float(observed_seed_means.max()),
        "all_seed_means_positive": bool(np.all(observed_seed_means > 0)),
    }


def load_paired_deltas(
    stage1_dir: Path,
    comparison: str,
    baseline_template: str,
    level: str,
    id_column: str,
    metrics_file: str,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        promoter_file = (
            stage1_dir
            / f"stage1_shift420_promoter_seed{seed}"
            / "test"
            / metrics_file
        )
        baseline_file = stage1_dir / baseline_template.format(seed=seed) / "test" / metrics_file
        promoter = pd.read_csv(promoter_file, usecols=[id_column, "pearson_r"])
        baseline = pd.read_csv(baseline_file, usecols=[id_column, "pearson_r"])
        paired = promoter.merge(
            baseline,
            on=id_column,
            how="inner",
            validate="one_to_one",
            suffixes=("_promoter", "_baseline"),
        ).dropna(subset=["pearson_r_promoter", "pearson_r_baseline"])
        paired.insert(0, "comparison", comparison)
        paired.insert(1, "level", level)
        paired.insert(2, "seed", seed)
        paired["pearson_delta"] = (
            paired["pearson_r_promoter"] - paired["pearson_r_baseline"]
        )
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def write_readme(
    output_dir: Path,
    by_seed: pd.DataFrame,
    across_seeds: pd.DataFrame,
    repeats: int,
    confidence: float,
    random_seed: int,
) -> None:
    lines = [
        "# Stage 1 配对 Bootstrap 记录",
        "",
        "该目录比较 real promoter 与 matched intergenic / expression baseline。",
        "所有比较先按同一 `cell_id` 或 `gene_id` 配对，再计算 `Pearson delta = promoter - baseline`。",
        "",
        f"- Bootstrap 次数：{repeats:,}",
        f"- 置信水平：{confidence:.1%}",
        f"- 随机种子：{random_seed}",
        "- 逐 seed CI：在该 seed 的配对 cell/gene 内有放回抽样。",
        "- 三 seed CI：分层 bootstrap，先有放回抽 seed，再在对应 seed 内有放回抽配对单位。",
        "",
        "## 三 Seed 汇总",
        "",
        "```text",
        across_seeds.to_string(index=False),
        "```",
        "",
        "## 逐 Seed 汇总",
        "",
        "```text",
        by_seed.to_string(index=False),
        "```",
        "",
        "正的 delta 表示 promoter 优于对应 baseline。CI 不跨 0 表示该配对均值提升在当前 bootstrap 设定下稳定。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def load_violin_data(stage1_dir: Path, seeds: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for level, (id_column, metrics_file) in LEVELS.items():
        for model_group, (model_label, run_template) in MODEL_RUNS.items():
            for seed in seeds:
                metrics_path = (
                    stage1_dir
                    / run_template.format(seed=seed)
                    / "test"
                    / metrics_file
                )
                frame = pd.read_csv(metrics_path, usecols=[id_column, "pearson_r"])
                frame = frame.rename(columns={id_column: "sample_id"})
                frame.insert(0, "level", level)
                frame.insert(1, "model_group", model_group)
                frame.insert(2, "model_label", model_label)
                frame.insert(3, "seed", int(seed))
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_violin_data(
    violin_data: pd.DataFrame,
    extreme_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_rows: list[dict[str, Any]] = []
    extreme_frames: list[pd.DataFrame] = []
    for (level, model_group, model_label), group in violin_data.groupby(
        ["level", "model_group", "model_label"], sort=False
    ):
        valid = group.dropna(subset=["pearson_r"]).copy()
        values = valid["pearson_r"].to_numpy(dtype=np.float64)
        stats_rows.append(
            {
                "source": level.replace("_", "-"),
                "model_group": model_group,
                "model_label": model_label,
                "n_total": int(len(group)),
                "n_valid": int(len(valid)),
                "n_extreme_points": int(min(2 * extreme_count, len(valid))),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                "min": float(np.min(values)) if len(values) else np.nan,
                "max": float(np.max(values)) if len(values) else np.nan,
            }
        )
        if valid.empty:
            continue
        low = valid.nsmallest(extreme_count, "pearson_r").copy()
        low["extreme_side"] = "low"
        high = valid.nlargest(extreme_count, "pearson_r").copy()
        high["extreme_side"] = "high"
        extremes = pd.concat([low, high], ignore_index=True).drop_duplicates(
            subset=["sample_id", "seed", "pearson_r"]
        )
        extremes.insert(0, "source", level.replace("_", "-"))
        extreme_frames.append(extremes)

    stats = pd.DataFrame(stats_rows)
    if extreme_frames:
        extreme_points = pd.concat(extreme_frames, ignore_index=True)
    else:
        extreme_points = pd.DataFrame(
            columns=[
                "source",
                "level",
                "model_group",
                "model_label",
                "seed",
                "sample_id",
                "pearson_r",
                "extreme_side",
            ]
        )
    return stats, extreme_points


def plot_pearson_violin(
    violin_data: pd.DataFrame,
    extreme_points: pd.DataFrame,
    level: str,
    output_dir: Path,
    random_seed: int,
) -> None:
    model_order = list(MODEL_RUNS)
    labels = [MODEL_RUNS[group][0] for group in model_order]
    values_by_model: list[np.ndarray] = []
    for model_group in model_order:
        values = violin_data.loc[
            (violin_data["level"] == level)
            & (violin_data["model_group"] == model_group),
            "pearson_r",
        ].dropna().to_numpy(dtype=np.float64)
        values_by_model.append(values)

    nonempty_positions = [
        index + 1 for index, values in enumerate(values_by_model) if len(values)
    ]
    nonempty_values = [values for values in values_by_model if len(values)]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    if nonempty_values:
        violins = ax.violinplot(
            nonempty_values,
            positions=nonempty_positions,
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.25,
        )
        for body, position in zip(violins["bodies"], nonempty_positions):
            model_group = model_order[position - 1]
            body.set_facecolor(MODEL_COLORS[model_group])
            body.set_edgecolor(MODEL_COLORS[model_group])
            body.set_alpha(0.52)
            body.set_linewidth(1.4)

    rng = np.random.default_rng(random_seed + (0 if level == "per_gene" else 10_000))
    for position, (model_group, values) in enumerate(
        zip(model_order, values_by_model), start=1
    ):
        if len(values) == 0:
            ax.text(position, 0, "no valid r", ha="center", va="bottom", fontsize=9)
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        mean = float(np.mean(values))
        ax.vlines(position, q25, q75, color="#263238", linewidth=5, alpha=0.75)
        ax.hlines(median, position - 0.22, position + 0.22, color="#263238", linewidth=2.2)
        ax.scatter(
            position,
            mean,
            s=38,
            facecolor="white",
            edgecolor="#263238",
            linewidth=1.3,
            zorder=5,
        )
        selected = extreme_points[
            (extreme_points["level"] == level)
            & (extreme_points["model_group"] == model_group)
        ]
        if not selected.empty:
            jitter = rng.uniform(-0.18, 0.18, size=len(selected))
            ax.scatter(
                position + jitter,
                selected["pearson_r"],
                s=15,
                color="#111111",
                alpha=0.58,
                edgecolor="white",
                linewidth=0.25,
                zorder=4,
            )

    source_label = "gene" if level == "per_gene" else "cell"
    valid_counts = [len(values) for values in values_by_model]
    tick_labels = [
        f"{label}\n(n={count:,})" for label, count in zip(labels, valid_counts)
    ]
    ax.axhline(0, color="#607D8B", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, len(model_order) + 1), tick_labels)
    ax.set_ylabel("Pearson r")
    ax.set_title(
        f"Stage 1 per-{source_label} Pearson across seeds 1, 7, and 42"
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        0.01,
        "White dot: mean; thick bar: IQR; line: median; black dots: lowest/highest 25",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#455A64",
    )
    plt.tight_layout()
    stem = f"stage1_{level}_pearson_violin_compare"
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def write_violin_outputs(
    stage1_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    random_seed: int,
    extreme_count: int = 25,
) -> dict[str, int]:
    violin_data = load_violin_data(stage1_dir, seeds)
    stats, extreme_points = summarize_violin_data(violin_data, extreme_count)
    stats.to_csv(output_dir / "stage1_pearson_compare_violin_stats.csv", index=False)
    extreme_points.to_csv(
        output_dir / "stage1_pearson_extreme_points.csv", index=False
    )
    for level in LEVELS:
        plot_pearson_violin(
            violin_data,
            extreme_points,
            level,
            output_dir,
            random_seed,
        )
    return {
        "violin_rows": int(len(violin_data)),
        "extreme_rows": int(len(extreme_points)),
    }

def run_summary(
    stage1_dir: Path,
    repeats: int,
    confidence: float,
    random_seed: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    output_dir = stage1_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    violin_result = write_violin_outputs(
        stage1_dir,
        output_dir,
        seeds,
        random_seed,
    )
    try:
        from scripts.stage1_training_ablation import write_training_ablation_outputs
    except ModuleNotFoundError:
        from stage1_training_ablation import write_training_ablation_outputs

    ablation_result = write_training_ablation_outputs(
        stage1_dir,
        output_dir,
        repeats,
        confidence,
        rng,
        bootstrap_paired_delta,
    )

    paired_frames = []
    for comparison, baseline_template in COMPARISONS.items():
        for level, (id_column, metrics_file) in LEVELS.items():
            paired_frames.append(
                load_paired_deltas(
                    stage1_dir,
                    comparison,
                    baseline_template,
                    level,
                    id_column,
                    metrics_file,
                    seeds,
                )
            )
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(output_dir / "paired_pearson_deltas.csv", index=False)

    by_seed_rows = []
    across_seed_rows = []
    for (comparison, level), group in paired.groupby(["comparison", "level"], sort=True):
        deltas_by_seed: dict[int, np.ndarray] = {}
        for seed, seed_group in group.groupby("seed", sort=True):
            deltas = seed_group["pearson_delta"].to_numpy(dtype=np.float64)
            deltas_by_seed[int(seed)] = deltas
            stats = bootstrap_paired_delta(deltas, repeats, confidence, rng)
            by_seed_rows.append(
                {
                    "comparison": comparison,
                    "level": level,
                    "seed": int(seed),
                    "n_pairs": int(deltas.size),
                    **stats,
                }
            )
        hierarchical = hierarchical_seed_bootstrap(
            deltas_by_seed, repeats, confidence, rng
        )
        across_seed_rows.append(
            {
                "comparison": comparison,
                "level": level,
                "total_paired_rows": int(len(group)),
                **hierarchical,
            }
        )

    by_seed = pd.DataFrame(by_seed_rows)
    across_seeds = pd.DataFrame(across_seed_rows)
    by_seed.to_csv(output_dir / "paired_bootstrap_by_seed.csv", index=False)
    across_seeds.to_csv(output_dir / "paired_bootstrap_across_seeds.csv", index=False)

    config = {
        "stage1_dir": str(stage1_dir),
        "seeds": list(seeds),
        "bootstrap_repeats": repeats,
        "confidence": confidence,
        "random_seed": random_seed,
        "delta_definition": "promoter pearson_r - baseline pearson_r",
        "comparisons": COMPARISONS,
        "levels": LEVELS,
    }
    (output_dir / "paired_bootstrap_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_readme(output_dir, by_seed, across_seeds, repeats, confidence, random_seed)
    return {
        "output_dir": output_dir,
        "paired_rows": int(len(paired)),
        "by_seed_rows": int(len(by_seed)),
        "across_seed_rows": int(len(across_seeds)),
        **violin_result,
        **ablation_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 1 paired Pearson deltas with bootstrap confidence intervals.")
    parser.add_argument("--stage1-dir", type=Path, default=Path("outputs/stage1"))
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    result = run_summary(
        stage1_dir=args.stage1_dir,
        repeats=args.bootstrap_repeats,
        confidence=args.confidence,
        random_seed=args.seed,
    )
    print(
        f"Wrote Stage 1 paired bootstrap summary to {result['output_dir']} "
        f"({result['paired_rows']} paired rows)."
    )


if __name__ == "__main__":
    main()
