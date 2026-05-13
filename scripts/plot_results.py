# -*- coding: utf-8 -*-
"""
Standalone script to generate all plots from a completed training run.
Usage:
    python scripts/plot_results.py --exp_name my_run [--data highquality]
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from src.dataset import MyDataset
from src.model import build_model
import src.utils as utils
import src.utils as utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training results from an existing experiment.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name (outputs/<exp_name>)")
    parser.add_argument("--data", type=str, default="highquality", choices=["highquality", "processed"],
                        help="Which dataset version was used")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint. Default: outputs/<exp_name>/checkpoints/best_model.safetensors")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of val_loader passes for scatter plot (1 for highquality, >=2 recommended for processed)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = PROJECT_ROOT
    run_dir = base_dir / "outputs" / args.exp_name
    config_path = run_dir / "config.json"
    log_file = run_dir / "log" / "train_log.csv"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return

    # ---- CSV-based plots (no model needed) ----
    utils.plot_loss_curves_from_logfile(log_file, save_path=plot_dir / "loss_curve.png")
    utils.plot_zero_nonzero_loss_curves(log_file, save_path=plot_dir / "zero_nonzero_loss.png")
    utils.plot_val_metrics(log_file, save_path=plot_dir / "val_metrics.png")

    # ---- Scatter plot (needs model + val_loader) ----
    import json
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    model_name = cfg.get("model", "SimpleGeneModel")
    hidden_size = cfg.get("hidden_size", 32)
    expr_dim = cfg.get("expr_dim", None)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = run_dir / "checkpoints" / "best_model.safetensors"
    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}, skipping scatter plot.")
        return

    # Build val dataset and loader
    data_dir = base_dir / "data" / args.data
    val_dataset = MyDataset(
        promoter_file=data_dir / "promoter_val.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        mode="val",
        seed=args.seed,
    )

    if expr_dim is None:
        expr_dim = val_dataset.X.shape[1]

    if args.data == "highquality":
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        # For processed data, use a smaller subset for scatter plot to reduce noise and speed up plotting.
        subset_size = min(4*10**7, len(val_dataset))
        val_subset = torch.utils.data.Subset(val_dataset, indices=np.random.choice(len(val_dataset), subset_size, replace=False))
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, expr_dim=expr_dim, hidden_size=hidden_size)
    model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    model.to(device)
    model.eval()
    print(f"Loaded model from: {checkpoint}")

    utils.count_zero_nonzero(val_loader)
    utils.plot_pred_scatter(model, val_loader, epoch=args.epochs,
                            save_path=plot_dir / "pred_vs_true_scatter.png")
    print(f"All plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
