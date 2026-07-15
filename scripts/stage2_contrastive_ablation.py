from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.stage1_training_ablation import write_two_run_ablation_outputs


LEVELS = {
    "per_gene": ("gene_id", "per_gene_metrics.csv"),
    "per_cell": ("cell_id", "per_cell_metrics.csv"),
}
VIOLIN_COLORS = {"baseline": "#4C78A8", "treatment": "#F58518"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 paired-ablation analysis contract on one Stage 2 contrastive comparison."
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/stage2"))
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--treatment-run", required=True)
    parser.add_argument("--baseline-label", default="cw=0")
    parser.add_argument("--treatment-label", default="cw=0.40")
    parser.add_argument("--comparison", default="cw040_vs_cw000")
    parser.add_argument("--title", default="Stage 2 seed 7 contrastive ablation")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap and display-jitter random seed.")
    parser.add_argument("--run-seed", type=int, default=7, help="Training seed represented by the compared run pair.")
    parser.add_argument("--violin-extreme-count", type=int, default=25)
    return parser.parse_args()


def write_violin_outputs(
    runs_root: Path,
    output_dir: Path,
    baseline_run: str,
    treatment_run: str,
    baseline_label: str,
    treatment_label: str,
    seed: int,
    random_seed: int,
    extreme_count: int,
) -> dict[str, int]:
    """Write Stage 1-style per-gene and per-cell Pearson violin outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_specs = (
        ("baseline", baseline_run, baseline_label),
        ("treatment", treatment_run, treatment_label),
    )
    frames: list[pd.DataFrame] = []
    stats_rows: list[dict[str, object]] = []
    extreme_frames: list[pd.DataFrame] = []
    for level, (id_column, filename) in LEVELS.items():
        for model_group, run_name, model_label in model_specs:
            frame = pd.read_csv(
                runs_root / run_name / "test" / filename,
                usecols=[id_column, "pearson_r"],
            ).rename(columns={id_column: "sample_id"})
            frame.insert(0, "level", level)
            frame.insert(1, "model_group", model_group)
            frame.insert(2, "model_label", model_label)
            frame.insert(3, "seed", seed)
            frames.append(frame)

            valid = frame.dropna(subset=["pearson_r"]).copy()
            values = valid["pearson_r"].to_numpy(dtype=np.float64)
            stats_rows.append(
                {
                    "level": level,
                    "model_group": model_group,
                    "model_label": model_label,
                    "n_total": int(len(frame)),
                    "n_valid": int(len(valid)),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                    "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                }
            )
            if not valid.empty:
                low = valid.nsmallest(extreme_count, "pearson_r").copy()
                low["extreme_side"] = "low"
                high = valid.nlargest(extreme_count, "pearson_r").copy()
                high["extreme_side"] = "high"
                extreme_frames.append(
                    pd.concat([low, high], ignore_index=True).drop_duplicates(
                        subset=["sample_id", "pearson_r"]
                    )
                )

    violin_data = pd.concat(frames, ignore_index=True)
    stats = pd.DataFrame(stats_rows)
    extreme_points = pd.concat(extreme_frames, ignore_index=True)
    violin_data.to_csv(output_dir / "stage2_pearson_violin_data.csv", index=False)
    stats.to_csv(output_dir / "stage2_pearson_violin_stats.csv", index=False)
    extreme_points.to_csv(output_dir / "stage2_pearson_violin_extreme_points.csv", index=False)

    for level in LEVELS:
        values_by_model = [
            violin_data.loc[
                (violin_data["level"] == level)
                & (violin_data["model_group"] == model_group),
                "pearson_r",
            ].dropna().to_numpy(dtype=np.float64)
            for model_group, _run_name, _label in model_specs
        ]
        fig, ax = plt.subplots(figsize=(8.8, 6.0))
        violins = ax.violinplot(
            values_by_model,
            positions=[1, 2],
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.25,
        )
        for body, (model_group, _run_name, _label) in zip(violins["bodies"], model_specs):
            body.set_facecolor(VIOLIN_COLORS[model_group])
            body.set_edgecolor(VIOLIN_COLORS[model_group])
            body.set_alpha(0.52)
            body.set_linewidth(1.4)

        rng = np.random.default_rng(random_seed + (0 if level == "per_gene" else 10_000))
        for position, ((model_group, _run_name, _label), values) in enumerate(
            zip(model_specs, values_by_model), start=1
        ):
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            ax.vlines(position, q25, q75, color="#263238", linewidth=5, alpha=0.75)
            ax.hlines(median, position - 0.22, position + 0.22, color="#263238", linewidth=2.2)
            ax.scatter(position, float(np.mean(values)), s=38, facecolor="white", edgecolor="#263238", linewidth=1.3, zorder=5)
            selected = extreme_points[
                (extreme_points["level"] == level)
                & (extreme_points["model_group"] == model_group)
            ]
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
        ax.axhline(0, color="#607D8B", linestyle="--", linewidth=1)
        ax.set_xticks(
            [1, 2],
            [f"{label}\n(n={len(values):,})" for (_group, _run, label), values in zip(model_specs, values_by_model)],
        )
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Stage 2 seed {seed} per-{source_label} Pearson")
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
        stem = f"stage2_{level}_pearson_violin_compare"
        fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    return {
        "violin_rows": int(len(violin_data)),
        "extreme_rows": int(len(extreme_points)),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.outputs_root / "summary" / f"stage2_{args.comparison}_seed7_ablation"
    )
    result = write_two_run_ablation_outputs(
        runs_root=args.outputs_root,
        output_dir=output_dir,
        baseline_run=args.baseline_run,
        treatment_run=args.treatment_run,
        baseline_label=args.baseline_label,
        treatment_label=args.treatment_label,
        comparison=args.comparison,
        title=args.title,
        repeats=args.repeats,
        confidence=args.confidence,
        random_seed=args.seed,
    )
    violin_result = write_violin_outputs(
        runs_root=args.outputs_root,
        output_dir=output_dir,
        baseline_run=args.baseline_run,
        treatment_run=args.treatment_run,
        baseline_label=args.baseline_label,
        treatment_label=args.treatment_label,
        seed=args.run_seed,
        random_seed=args.seed,
        extreme_count=args.violin_extreme_count,
    )
    print(f"[Stage2Ablation] output={output_dir} {result} {violin_result}")


if __name__ == "__main__":
    main()
