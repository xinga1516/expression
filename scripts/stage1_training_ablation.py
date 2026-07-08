from __future__ import annotations

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
