from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_test import (  # noqa: E402
    build_test_model,
    load_config,
    predict_model_values,
    read_cell_split,
    resolve_cell_split_dir,
    resolve_data_dir,
    resolve_eval_expression_data_config,
)
from src.dataset import MyDataset  # noqa: E402
from src.gpu_cache import GpuCachedPairLoader  # noqa: E402


PROTOCOL_FIELDS = (
    "seed",
    "git_hash",
    "train_promoter_file",
    "val_promoter_file",
    "scrna_file",
    "expr_dim",
    "use_cell_split",
    "cell_split_dir",
    "input_gene_panel_file",
    "expression_layer",
    "expression_transform",
    "target_count_layer",
    "target_value_layer",
    "target_transform",
    "sequence_length",
    "checkpoint_metric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure cell-varying promoter effects by applying one promoter model to "
            "real and matched-control sequences, with an expression-only residual."
        )
    )
    parser.add_argument("--promoter-exp", default="stage1_shift420_promoter_seed7")
    parser.add_argument("--expression-exp", default="stage1_shift420_exprmatched_seed7")
    parser.add_argument("--promoter-checkpoint", type=str, default=None)
    parser.add_argument("--expression-checkpoint", type=str, default=None)
    parser.add_argument("--control-sequence-column", default="control_sequence")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stage1_shift420_sequence_interaction_seed7",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates every test gene-cell pair.")
    parser.add_argument("--seed", type=int, default=7, help="DataLoader seed; does not change frozen test cells.")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument(
        "--no-gpu-cache-dataset",
        action="store_true",
        help="Use paired CPU DataLoaders even when CUDA is available.",
    )
    return parser.parse_args()


def resolve_run_dir(exp_name: str) -> Path:
    run_dir = Path(exp_name)
    if not run_dir.is_absolute():
        if run_dir.parts and run_dir.parts[0] == "outputs":
            run_dir = PROJECT_ROOT / run_dir
        else:
            run_dir = PROJECT_ROOT / "outputs" / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {run_dir}")
    return run_dir.resolve()


def resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_checkpoint(run_dir: Path, value: str | None) -> Path:
    checkpoint = resolve_optional_path(value)
    if checkpoint is None:
        checkpoint = run_dir / "checkpoints" / "best_model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def normalized_protocol_value(field: str, value: Any) -> Any:
    if field.endswith("_file") or field.endswith("_dir"):
        if value in {None, ""}:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())
    return value


def validate_matched_protocol(configs: dict[str, dict[str, Any]]) -> None:
    reference_name = "promoter"
    reference = configs[reference_name]
    mismatches: list[str] = []
    for name, config in configs.items():
        if name == reference_name:
            continue
        for field in PROTOCOL_FIELDS:
            expected = normalized_protocol_value(field, reference.get(field))
            observed = normalized_protocol_value(field, config.get(field))
            if observed != expected:
                mismatches.append(
                    f"{name}.{field}={observed!r}, expected {reference_name}.{field}={expected!r}"
                )
    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise ValueError(f"Stage 1 runs do not use a matched evaluation protocol:\n  - {detail}")

    if not bool(reference.get("use_cell_split", False)):
        raise ValueError("Stage 1 interaction analysis requires the frozen test-cell split.")
    loss_types = {str(config.get("loss_type", "mse")).lower() for config in configs.values()}
    if "zinb" in loss_types:
        raise ValueError("ZINB runs are not supported by this scalar-target interaction analysis.")


def make_interaction_state(num_genes: int) -> dict[str, np.ndarray]:
    if num_genes <= 0:
        raise ValueError("num_genes must be positive")
    return {
        "n": np.zeros(num_genes, dtype=np.int64),
        "sum_effect": np.zeros(num_genes, dtype=np.float64),
        "sum_effect2": np.zeros(num_genes, dtype=np.float64),
        "sum_residual": np.zeros(num_genes, dtype=np.float64),
        "sum_residual2": np.zeros(num_genes, dtype=np.float64),
        "sum_effect_residual": np.zeros(num_genes, dtype=np.float64),
    }


def update_interaction_state(
    state: dict[str, np.ndarray],
    gene_indices: np.ndarray,
    promoter_predictions: np.ndarray,
    control_predictions: np.ndarray,
    expression_predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    arrays = (
        gene_indices,
        promoter_predictions,
        control_predictions,
        expression_predictions,
        targets,
    )
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("All paired prediction arrays must have the same length.")
    if len(gene_indices) == 0:
        return
    if int(np.min(gene_indices)) < 0 or int(np.max(gene_indices)) >= len(state["n"]):
        raise IndexError("gene_indices contain an out-of-range promoter row.")

    effect = np.asarray(promoter_predictions, dtype=np.float64) - np.asarray(
        control_predictions, dtype=np.float64
    )
    residual = np.asarray(targets, dtype=np.float64) - np.asarray(
        expression_predictions, dtype=np.float64
    )
    indices = np.asarray(gene_indices, dtype=np.int64)
    np.add.at(state["n"], indices, 1)
    np.add.at(state["sum_effect"], indices, effect)
    np.add.at(state["sum_effect2"], indices, effect * effect)
    np.add.at(state["sum_residual"], indices, residual)
    np.add.at(state["sum_residual2"], indices, residual * residual)
    np.add.at(state["sum_effect_residual"], indices, effect * residual)


def finalize_interaction_state(
    state: dict[str, np.ndarray], gene_ids: Iterable[str]
) -> pd.DataFrame:
    gene_id_list = [str(gene_id) for gene_id in gene_ids]
    if len(gene_id_list) != len(state["n"]):
        raise ValueError("gene_ids length does not match interaction state.")

    rows: list[dict[str, Any]] = []
    for gene_idx, gene_id in enumerate(gene_id_list):
        n = int(state["n"][gene_idx])
        if n == 0:
            continue
        mean_effect = float(state["sum_effect"][gene_idx] / n)
        mean_residual = float(state["sum_residual"][gene_idx] / n)
        effect_variance = max(
            float(state["sum_effect2"][gene_idx] / n - mean_effect * mean_effect), 0.0
        )
        residual_variance = max(
            float(state["sum_residual2"][gene_idx] / n - mean_residual * mean_residual), 0.0
        )
        covariance = float(
            state["sum_effect_residual"][gene_idx] / n - mean_effect * mean_residual
        )
        denominator = math.sqrt(effect_variance * residual_variance)
        residual_correlation = covariance / denominator if denominator > 0.0 else float("nan")
        rows.append(
            {
                "gene_id": gene_id,
                "n_cells": n,
                "sequence_effect_mean": mean_effect,
                "interaction_variance": effect_variance,
                "interaction_std": math.sqrt(effect_variance),
                "expression_only_residual_mean": mean_residual,
                "expression_only_residual_variance": residual_variance,
                "effect_residual_covariance": covariance,
                "effect_residual_correlation": residual_correlation,
            }
        )
    return pd.DataFrame(rows)


def summarize_interactions(per_gene: pd.DataFrame) -> dict[str, float | int]:
    finite_corr = per_gene["effect_residual_correlation"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "num_genes": int(len(per_gene)),
        "num_genes_with_defined_residual_correlation": int(len(finite_corr)),
        "median_interaction_variance": float(per_gene["interaction_variance"].median()),
        "mean_interaction_variance": float(per_gene["interaction_variance"].mean()),
        "median_interaction_std": float(per_gene["interaction_std"].median()),
        "median_effect_residual_correlation": float(finite_corr.median()) if len(finite_corr) else float("nan"),
        "mean_effect_residual_correlation": float(finite_corr.mean()) if len(finite_corr) else float("nan"),
        "positive_effect_residual_correlation_genes": int((finite_corr > 0.0).sum()),
        "positive_effect_residual_correlation_fraction": (
            float((finite_corr > 0.0).mean()) if len(finite_corr) else float("nan")
        ),
    }


def plot_interactions(per_gene: pd.DataFrame, output_path: Path) -> None:
    plot_data = per_gene.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["interaction_variance", "effect_residual_correlation"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    if not plot_data.empty:
        x = np.log10(plot_data["interaction_variance"].to_numpy(dtype=np.float64) + 1e-12)
        y = plot_data["effect_residual_correlation"].to_numpy(dtype=np.float64)
        ax.scatter(x, y, s=9, alpha=0.45, linewidths=0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("log10 interaction variance + 1e-12")
    ax.set_ylabel("Corr(sequence effect, expression-only residual) across cells")
    ax.set_title("Stage 1 promoter × cell-state interaction diagnostic")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_dataset(
    data_dir: Path,
    scrna_file: Path,
    test_cell_ids: np.ndarray,
    config: dict[str, Any],
    data_config: dict[str, Any],
    seed: int,
) -> MyDataset:
    panel_path = resolve_optional_path(config.get("input_gene_panel_file"))
    return MyDataset(
        promoter_file=data_dir / "promoter_test.csv",
        scrna_file=scrna_file,
        cell_ids_subset=test_cell_ids,
        mode="test",
        seed=seed,
        log1p_cpm_target=bool(data_config["log1p_cpm_target"]),
        preencode_promoters=True,
        sequence_column=str(config.get("sequence_column", "sequence")),
        sequence_length=int(config.get("sequence_length", 400)),
        expression_layer=data_config["expression_layer"],
        expression_transform=str(data_config["expression_transform"]),
        target_count_layer=data_config["target_count_layer"],
        target_value_layer=data_config["target_value_layer"],
        input_gene_panel_file=panel_path,
    )


def validate_dataset_alignment(promoter_dataset: MyDataset, control_dataset: MyDataset) -> None:
    promoter_gene_ids = promoter_dataset.promoters["gene_id"].astype(str).tolist()
    control_gene_ids = control_dataset.promoters["gene_id"].astype(str).tolist()
    if promoter_gene_ids != control_gene_ids:
        raise ValueError("Real and control sequence datasets do not have identical ordered test genes.")
    if len(set(promoter_gene_ids)) != len(promoter_gene_ids):
        raise ValueError("Interaction analysis requires one test sequence row per gene_id.")
    if promoter_dataset.C != control_dataset.C:
        raise ValueError("Real and control sequence datasets have different test-cell counts.")
    promoter_cells = promoter_dataset.scrna.obs_names[promoter_dataset.cells].astype(str).tolist()
    control_cells = control_dataset.scrna.obs_names[control_dataset.cells].astype(str).tolist()
    if promoter_cells != control_cells:
        raise ValueError("Real and control sequence datasets do not have identical ordered test cells.")


def compute_paired_interactions(
    promoter_model: torch.nn.Module,
    expression_model: torch.nn.Module,
    promoter_loader: Any,
    control_loader: Any,
    device: torch.device,
    max_samples: int,
) -> pd.DataFrame:
    promoter_dataset = promoter_loader.dataset
    control_dataset = control_loader.dataset
    if not isinstance(promoter_dataset, MyDataset) or not isinstance(control_dataset, MyDataset):
        raise TypeError("Paired interaction loaders must contain MyDataset instances.")
    validate_dataset_alignment(promoter_dataset, control_dataset)
    state = make_interaction_state(promoter_dataset.P)
    count = 0

    with torch.no_grad():
        for promoter_batch, control_batch in zip(promoter_loader, control_loader, strict=True):
            promoter_sequences, promoter_exprs, promoter_targets = promoter_batch
            control_sequences, control_exprs, control_targets = control_batch
            if not torch.equal(promoter_targets, control_targets):
                raise ValueError("Paired real/control batches contain different target values.")
            if not torch.equal(promoter_exprs, control_exprs):
                raise ValueError("Paired real/control batches contain different expression inputs.")

            if max_samples > 0:
                remaining = max_samples - count
                if remaining <= 0:
                    break
                if promoter_targets.numel() > remaining:
                    promoter_sequences = promoter_sequences[:remaining]
                    control_sequences = control_sequences[:remaining]
                    promoter_exprs = promoter_exprs[:remaining]
                    promoter_targets = promoter_targets[:remaining]

            promoter_sequences = promoter_sequences.to(device, non_blocking=True)
            control_sequences = control_sequences.to(device, non_blocking=True)
            promoter_exprs = promoter_exprs.to(device, non_blocking=True)
            promoter_targets = promoter_targets.to(device, non_blocking=True).float()

            promoter_predictions = predict_model_values(
                promoter_model, promoter_sequences, promoter_exprs
            )
            control_predictions = predict_model_values(
                promoter_model, control_sequences, promoter_exprs
            )
            expression_predictions = predict_model_values(
                expression_model, promoter_sequences, promoter_exprs
            )

            batch_size = int(promoter_targets.numel())
            flat_indices = np.arange(count, count + batch_size, dtype=np.int64)
            gene_indices = flat_indices // int(promoter_dataset.C)
            update_interaction_state(
                state=state,
                gene_indices=gene_indices,
                promoter_predictions=promoter_predictions.detach().cpu().numpy(),
                control_predictions=control_predictions.detach().cpu().numpy(),
                expression_predictions=expression_predictions.detach().cpu().numpy(),
                targets=promoter_targets.detach().cpu().numpy(),
            )
            count += batch_size
            if max_samples > 0 and count >= max_samples:
                break

    gene_ids = promoter_dataset.promoters["gene_id"].astype(str).tolist()
    return finalize_interaction_state(state, gene_ids)


def run_summary(args: argparse.Namespace) -> dict[str, Any]:
    run_dirs = {
        "promoter": resolve_run_dir(args.promoter_exp),
        "expression": resolve_run_dir(args.expression_exp),
    }
    configs = {name: load_config(run_dir) for name, run_dir in run_dirs.items()}
    validate_matched_protocol(configs)

    promoter_config = configs["promoter"]
    control_config = dict(promoter_config)
    control_config["sequence_column"] = str(args.control_sequence_column)
    data_dir = resolve_data_dir(PROJECT_ROOT, promoter_config, None).resolve()
    scrna_file = resolve_optional_path(promoter_config.get("scrna_file"))
    if scrna_file is None:
        scrna_file = data_dir / "integrated_data.h5ad"
    cell_split_value = promoter_config.get("cell_split_dir")
    cell_split_dir = resolve_cell_split_dir(
        PROJECT_ROOT,
        data_dir,
        None if cell_split_value is None else str(cell_split_value),
    ).resolve()
    test_cell_ids = read_cell_split(cell_split_dir, "test")

    data_configs = {
        name: resolve_eval_expression_data_config(argparse.Namespace(), config, str(config.get("loss_type", "mse")))
        for name, config in configs.items()
    }
    promoter_data_config = data_configs["promoter"]
    for name, data_config in data_configs.items():
        if data_config != promoter_data_config:
            raise ValueError(f"{name} resolves to a different expression/target data configuration.")

    promoter_dataset = build_dataset(
        data_dir, scrna_file, test_cell_ids, promoter_config, promoter_data_config, args.seed
    )
    control_dataset = build_dataset(
        data_dir,
        scrna_file,
        test_cell_ids,
        control_config,
        promoter_data_config,
        args.seed,
    )
    validate_dataset_alignment(promoter_dataset, control_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda" and not bool(args.no_pin_memory)
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    use_gpu_cache = device.type == "cuda" and not bool(args.no_gpu_cache_dataset)
    if use_gpu_cache:
        samples = int(args.max_samples) if int(args.max_samples) > 0 else len(promoter_dataset)
        promoter_loader = GpuCachedPairLoader(
            promoter_dataset,
            batch_size=int(args.batch_size),
            device=device,
            samples_per_epoch=samples,
            seed=int(args.seed),
            sampler_mode="sequential",
            drop_last=False,
        )
        control_loader = GpuCachedPairLoader(
            control_dataset,
            batch_size=int(args.batch_size),
            device=device,
            samples_per_epoch=samples,
            seed=int(args.seed),
            sampler_mode="sequential",
            drop_last=False,
        )
    else:
        promoter_loader = DataLoader(promoter_dataset, **loader_kwargs)
        control_loader = DataLoader(control_dataset, **loader_kwargs)

    checkpoints = {
        "promoter": resolve_checkpoint(run_dirs["promoter"], args.promoter_checkpoint),
        "expression": resolve_checkpoint(run_dirs["expression"], args.expression_checkpoint),
    }
    models = {
        name: build_test_model(
            config,
            expr_dim=int(config.get("expr_dim", promoter_dataset.expr_dim)),
            checkpoint=checkpoints[name],
            device=device,
        )
        for name, config in configs.items()
    }

    per_gene = compute_paired_interactions(
        promoter_model=models["promoter"],
        expression_model=models["expression"],
        promoter_loader=promoter_loader,
        control_loader=control_loader,
        device=device,
        max_samples=int(args.max_samples),
    )
    per_gene = per_gene.sort_values(
        ["effect_residual_correlation", "interaction_variance"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    output_dir = resolve_optional_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir cannot be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_gene_path = output_dir / "per_gene_sequence_interaction.csv"
    per_gene.to_csv(per_gene_path, index=False)
    plot_interactions(per_gene, output_dir / "sequence_interaction_scatter.png")

    summary: dict[str, Any] = summarize_interactions(per_gene)
    summary.update(
        {
            "num_pairs": int(per_gene["n_cells"].sum()),
            "test_cells": int(promoter_dataset.C),
            "variance_definition": "population variance across frozen test cells (ddof=0)",
            "sequence_effect_definition": "same promoter model: real_sequence_prediction - control_sequence_prediction",
            "residual_definition": "target - expression_only_prediction",
            "promoter_exp": str(run_dirs["promoter"]),
            "expression_exp": str(run_dirs["expression"]),
            "promoter_checkpoint": str(checkpoints["promoter"]),
            "control_sequence_column": str(args.control_sequence_column),
            "expression_checkpoint": str(checkpoints["expression"]),
            "data_dir": str(data_dir),
            "scrna_file": str(scrna_file),
            "cell_split_dir": str(cell_split_dir),
            "max_samples": int(args.max_samples),
            "device": str(device),
            "gpu_cache_dataset": bool(use_gpu_cache),
            "sequence_model_reused_for_control": True,
            "expression_only_model_is_separate": True,
        }
    )
    with open(output_dir / "sequence_interaction_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    print(f"Per-gene interaction metrics: {per_gene_path}")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> None:
    run_summary(parse_args())


if __name__ == "__main__":
    main()
