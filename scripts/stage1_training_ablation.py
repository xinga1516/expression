from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = {
    "mse": ("MSE", "stage1_shift420_promoter_seed7"),
    "combined": ("Combined", "stage1_shift420_combined_seed7"),
    "combined_fixedlr": ("Combined + fixed LR", "stage1_shift420_combined_fixedlr_seed7"),
}
COMPARISONS = (
    ("combined_vs_mse", "combined", "mse"),
    ("combined_fixedlr_vs_mse", "combined_fixedlr", "mse"),
    ("combined_fixedlr_vs_combined", "combined_fixedlr", "combined"),
)
LEVELS = {"per_gene": ("gene_id", "per_gene_metrics.csv"), "per_cell": ("cell_id", "per_cell_metrics.csv")}


def load_global_summary(stage1_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, (label, run_name) in RUNS.items():
        run_dir = stage1_dir / run_name
        metrics = json.loads((run_dir / "test/test_metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        log = pd.read_csv(run_dir / "log/train_log.csv")
        best_rmse = int(log["val_rmse"].idxmin())
        best_pearson = int(log["val_pearson_all"].idxmax())
        rows.append({
            "strategy": strategy, "label": label, "run_name": run_name,
            "seed": int(config["seed"]), "loss_type": config["loss_type"],
            "pearson_lambda": float(config["pearson_lambda"]),
            "requested_epochs": int(config["epochs"]), "logged_epochs": int(len(log)),
            "initial_lr": float(config["learning_rate"]), "final_logged_lr": float(log.iloc[-1]["lr"]),
            "best_val_rmse_epoch": int(log.loc[best_rmse, "epoch"]),
            "best_val_rmse": float(log.loc[best_rmse, "val_rmse"]),
            "best_val_pearson_epoch": int(log.loc[best_pearson, "epoch"]),
            "best_val_pearson": float(log.loc[best_pearson, "val_pearson_all"]),
            "test_mse": float(metrics["mse"]), "test_rmse": float(metrics["rmse"]),
            "test_pearson_r": float(metrics["pearson_r"]), "test_spearman_r": float(metrics["spearman_r"]),
            "test_nonzero_rmse": float(metrics["nonzero_rmse"]), "test_zero_rmse": float(metrics["zero_rmse"]),
        })
    return pd.DataFrame(rows)


def load_paired_deltas(stage1_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for comparison, treatment, baseline in COMPARISONS:
        for level, (id_column, filename) in LEVELS.items():
            treatment_data = pd.read_csv(stage1_dir / RUNS[treatment][1] / "test" / filename, usecols=[id_column, "pearson_r"])
            baseline_data = pd.read_csv(stage1_dir / RUNS[baseline][1] / "test" / filename, usecols=[id_column, "pearson_r"])
            paired = treatment_data.merge(
                baseline_data, on=id_column, validate="one_to_one",
                suffixes=("_treatment", "_baseline"),
            ).dropna(subset=["pearson_r_treatment", "pearson_r_baseline"])
            paired.insert(0, "comparison", comparison)
            paired.insert(1, "level", level)
            paired = paired.rename(columns={id_column: "sample_id"})
            paired["pearson_delta"] = paired["pearson_r_treatment"] - paired["pearson_r_baseline"]
            frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def plot_summary(global_data: pd.DataFrame, paired_data: pd.DataFrame, output_dir: Path) -> None:
    labels = global_data["label"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    width = 0.34
    axes[0, 0].bar(x - width / 2, global_data["test_pearson_r"], width, label="Pearson", color="#4C78A8")
    axes[0, 0].bar(x + width / 2, global_data["test_spearman_r"], width, label="Spearman", color="#F58518")
    axes[0, 0].set_ylabel("Correlation")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].bar(x, global_data["test_rmse"], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0, 1].set_ylabel("Test RMSE (lower is better)")
    for axis in axes[0]:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    order = [item[0] for item in COMPARISONS]
    names = ["Combined - MSE", "Fixed LR - MSE", "Fixed LR - Combined"]
    for axis, level, title in zip(axes[1], ("per_cell", "per_gene"), ("Per-cell Pearson delta", "Per-gene Pearson delta")):
        subset = paired_data[paired_data["level"] == level].set_index("comparison").loc[order]
        y = np.arange(len(subset))
        axis.errorbar(
            subset["mean_delta"], y,
            xerr=np.vstack((subset["mean_delta"] - subset["mean_ci_low"], subset["mean_ci_high"] - subset["mean_delta"])),
            fmt="o", color="#263238", ecolor="#607D8B", capsize=4,
        )
        axis.axvline(0, color="#B0BEC5", linestyle="--", linewidth=1)
        axis.set_yticks(y, names)
        axis.set_xlabel("Paired Pearson r delta (95% bootstrap CI)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stage 1 seed 7 training-strategy ablation", fontsize=14)
    fig.text(0.5, 0.01, "Single-seed comparison; MSE and combined runs also differ in training budget.", ha="center", color="#455A64")
    plt.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(output_dir / "stage1_training_ablation_seed7.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "stage1_training_ablation_seed7.svg", bbox_inches="tight")
    plt.close(fig)


def write_ablation_readme(
    global_data: pd.DataFrame,
    paired_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    indexed = global_data.set_index("strategy")
    paired = paired_data.set_index(["comparison", "level"])
    lines = [
        "# Stage 1 seed 7 训练策略消融",
        "",
        "本比较使用相同的 held-out test genes、2,048 个冻结 test cells、模型结构和 promoter shift 设置。",
        "它是单 seed 消融，不替代三 seed Stage 1 正式结论。MSE run 训练 30 epochs，而两个 combined run 最多训练 80 epochs，因此 combined 与 MSE 同时包含 loss 和训练预算差异。",
        "",
        "## 全局测试指标",
        "",
        "| 策略 | RMSE | Pearson | Spearman | nonzero RMSE | zero RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in RUNS:
        row = indexed.loc[strategy]
        lines.append(
            f"| {row['label']} | {row['test_rmse']:.6f} | {row['test_pearson_r']:.6f} | "
            f"{row['test_spearman_r']:.6f} | {row['test_nonzero_rmse']:.6f} | {row['test_zero_rmse']:.6f} |"
        )
    lines.extend([
        "",
        "## 配对 Pearson 结果",
        "",
        "| 比较 | 层级 | mean delta | 95% CI | 胜率 |",
        "|---|---|---:|---:|---:|",
    ])
    labels = {
        "combined_vs_mse": "Combined - MSE",
        "combined_fixedlr_vs_mse": "Fixed LR - MSE",
        "combined_fixedlr_vs_combined": "Fixed LR - Combined",
    }
    for comparison, _, _ in COMPARISONS:
        for level in ("per_cell", "per_gene"):
            row = paired.loc[(comparison, level)]
            lines.append(
                f"| {labels[comparison]} | {level} | {row['mean_delta']:.6f} | "
                f"[{row['mean_ci_low']:.6f}, {row['mean_ci_high']:.6f}] | {row['win_fraction']:.2%} |"
            )
    lines.extend([
        "",
        "## 解释",
        "",
        "- Combined 相对 MSE 提高全局 Pearson/Spearman，且 per-cell 和 per-gene 配对提升稳定为正；但全局 RMSE 略差。",
        "- Fixed LR 是三者中全局 Pearson、Spearman 和 RMSE 最好的策略，并显著提高 per-cell Pearson。",
        "- Fixed LR 相对普通 combined 的 per-gene Pearson mean delta 为负，说明收益主要来自细胞层面和整体校准，不能据此宣称它对每个基因都更好。",
        "- 下一步应以相同 epochs/early-stopping 预算补跑 seeds 1 和 42，再决定是否把 fixed LR 设为 Stage 1 默认。",
    ])
    (output_dir / "stage1_training_ablation_seed7_README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_training_ablation_outputs(
    stage1_dir: Path,
    output_dir: Path,
    repeats: int,
    confidence: float,
    rng: np.random.Generator,
    bootstrap_fn: Callable[..., dict[str, float]],
) -> dict[str, int]:
    global_data = load_global_summary(stage1_dir)
    global_data.to_csv(output_dir / "stage1_training_ablation_seed7.csv", index=False)
    paired = load_paired_deltas(stage1_dir)
    paired.to_csv(output_dir / "stage1_training_ablation_paired_deltas.csv", index=False)
    rows: list[dict[str, Any]] = []
    for (comparison, level), group in paired.groupby(["comparison", "level"], sort=False):
        deltas = group["pearson_delta"].to_numpy(dtype=np.float64)
        rows.append({"comparison": comparison, "level": level, "n_pairs": len(deltas), **bootstrap_fn(deltas, repeats, confidence, rng)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "stage1_training_ablation_paired_bootstrap.csv", index=False)
    plot_summary(global_data, summary, output_dir)
    write_ablation_readme(global_data, summary, output_dir)
    return {"ablation_global_rows": len(global_data), "ablation_paired_rows": len(paired)}


def load_two_run_paired_deltas(
    runs_root: Path,
    baseline_run: str,
    treatment_run: str,
    comparison: str,
) -> pd.DataFrame:
    """Build per-cell and per-gene Pearson deltas for one matched run pair."""
    frames: list[pd.DataFrame] = []
    for level, (id_column, filename) in LEVELS.items():
        baseline = pd.read_csv(
            runs_root / baseline_run / "test" / filename,
            usecols=[id_column, "pearson_r"],
        )
        treatment = pd.read_csv(
            runs_root / treatment_run / "test" / filename,
            usecols=[id_column, "pearson_r"],
        )
        paired = treatment.merge(
            baseline,
            on=id_column,
            validate="one_to_one",
            suffixes=("_treatment", "_baseline"),
        ).dropna(subset=["pearson_r_treatment", "pearson_r_baseline"])
        paired.insert(0, "comparison", comparison)
        paired.insert(1, "level", level)
        paired = paired.rename(columns={id_column: "sample_id"})
        paired["pearson_delta"] = (
            paired["pearson_r_treatment"] - paired["pearson_r_baseline"]
        )
        frames.append(paired)
    return pd.concat(frames, ignore_index=True)


def load_two_run_global_summary(
    runs_root: Path,
    baseline_run: str,
    treatment_run: str,
    baseline_label: str,
    treatment_label: str,
) -> pd.DataFrame:
    """Load directly comparable global test metrics for a two-run ablation."""
    rows: list[dict[str, Any]] = []
    for strategy, run_name, label in (
        ("baseline", baseline_run, baseline_label),
        ("treatment", treatment_run, treatment_label),
    ):
        metrics = json.loads(
            (runs_root / run_name / "test" / "test_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (runs_root / run_name / "config.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "strategy": strategy,
                "label": label,
                "run_name": run_name,
                "seed": int(config.get("seed", -1)),
                "contrastive_weight": float(config.get("contrastive_weight", 0.0)),
                "contrastive_projection_dim": int(
                    config.get("contrastive_projection_dim", 0)
                ),
                "test_mse": float(metrics["mse"]),
                "test_rmse": float(metrics["rmse"]),
                "test_pearson_r": float(metrics["pearson_r"]),
                "test_spearman_r": float(metrics["spearman_r"]),
                "test_nonzero_rmse": float(metrics["nonzero_rmse"]),
                "test_zero_rmse": float(metrics["zero_rmse"]),
            }
        )
    return pd.DataFrame(rows)


def write_two_run_ablation_outputs(
    runs_root: Path,
    output_dir: Path,
    baseline_run: str,
    treatment_run: str,
    baseline_label: str,
    treatment_label: str,
    comparison: str,
    title: str,
    repeats: int,
    confidence: float,
    random_seed: int,
) -> dict[str, int]:
    """Write the Stage 1 ablation contract for another matched two-run comparison."""
    from scripts.summarize_stage1_bootstrap import bootstrap_paired_delta

    output_dir.mkdir(parents=True, exist_ok=True)
    global_data = load_two_run_global_summary(
        runs_root,
        baseline_run,
        treatment_run,
        baseline_label,
        treatment_label,
    )
    global_data.to_csv(output_dir / "global_metrics.csv", index=False)
    paired = load_two_run_paired_deltas(
        runs_root,
        baseline_run,
        treatment_run,
        comparison,
    )
    paired.to_csv(output_dir / "paired_deltas.csv", index=False)

    rng = np.random.default_rng(random_seed)
    summary_rows: list[dict[str, Any]] = []
    for level, group in paired.groupby("level", sort=False):
        deltas = group["pearson_delta"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "comparison": comparison,
                "level": level,
                "n_pairs": len(deltas),
                **bootstrap_paired_delta(deltas, repeats, confidence, rng),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "paired_bootstrap.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    labels = global_data["label"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, global_data["test_pearson_r"], width, label="Pearson", color="#4C78A8")
    axes[0].bar(x + width / 2, global_data["test_spearman_r"], width, label="Spearman", color="#F58518")
    axes[0].set_ylabel("Global test correlation")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    y = np.arange(len(summary))
    axes[1].errorbar(
        summary["mean_delta"],
        y,
        xerr=np.vstack(
            (
                summary["mean_delta"] - summary["mean_ci_low"],
                summary["mean_ci_high"] - summary["mean_delta"],
            )
        ),
        fmt="o",
        color="#263238",
        ecolor="#607D8B",
        capsize=4,
    )
    axes[1].axvline(0, color="#B0BEC5", linestyle="--", linewidth=1)
    axes[1].set_yticks(y, summary["level"].str.replace("_", "-"))
    axes[1].set_xlabel("Paired Pearson delta: treatment - baseline")
    axes[1].grid(axis="x", alpha=0.2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "ablation_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "ablation_summary.svg", bbox_inches="tight")
    plt.close(fig)

    summary_text = summary.to_string(index=False)
    readme = "\n".join(
        (
            f"# {title}",
            "",
            f"Baseline: `{baseline_run}` ({baseline_label})",
            f"Treatment: `{treatment_run}` ({treatment_label})",
            f"Bootstrap repeats: {repeats:,}; confidence: {confidence:.1%}; random seed: {random_seed}",
            "",
            "Positive paired Pearson deltas favor the treatment.",
            "",
            "```text",
            summary_text,
            "```",
            "",
        )
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return {
        "global_rows": len(global_data),
        "paired_rows": len(paired),
        "bootstrap_rows": len(summary),
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Write the standalone Stage 1 seed-7 training-strategy ablation."
    )
    parser.add_argument("--stage1-dir", type=Path, default=project_root / "outputs" / "stage1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from scripts.summarize_stage1_bootstrap import bootstrap_paired_delta

    args = parse_args()
    stage1_dir = args.stage1_dir
    if not stage1_dir.is_absolute():
        stage1_dir = Path(__file__).resolve().parent.parent / stage1_dir
    output_dir = args.output_dir or stage1_dir / "summary"
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    result = write_training_ablation_outputs(
        stage1_dir=stage1_dir,
        output_dir=output_dir,
        repeats=int(args.repeats),
        confidence=float(args.confidence),
        rng=np.random.default_rng(int(args.seed)),
        bootstrap_fn=bootstrap_paired_delta,
    )
    print(f"[Stage1Ablation] output={output_dir} {result}")


if __name__ == "__main__":
    main()
