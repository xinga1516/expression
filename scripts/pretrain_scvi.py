# -*- coding: utf-8 -*-
"""Pretrain an scVI model on raw UMI counts to obtain a biologically
meaningful latent representation for downstream gene expression prediction.

Saves:
    outputs/scvi_{n_latent}/
        encoder.pt         -- z_encoder state_dict (for in-model loading)
        config.json        -- {n_input, n_latent, n_hidden, n_layers}
        scvi_model/        -- full scVI model directory
        latent_z.npy       -- get_latent_representation() for all cells

    outputs/pretrain_vae/
        loss_curve.png     -- train/val ELBO and components over epochs
        recon_scatter.png  -- true vs reconstructed expression (log1p)
        metrics.json       -- reconstruction Pearson r, MSE

Usage:
    python scripts/pretrain_scvi.py --n-latent 64 --epochs 200 --lr 1e-3 --lr-patience 15
    python scripts/pretrain_scvi.py --data umi_highquality --n-latent 128 --epochs 300 --lr 1e-3 --lr-patience 10
"""

import argparse
import json
from typing import Any
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_loss_curves(history: dict, save_path: Path) -> None:
    """Plot train/val loss curves from scVI training history."""
    available = list(history.keys())
    print(f"[Eval] History keys: {available}")

    # Determine which loss components are available
    has_train = any(k.startswith("elbo_train") for k in available) or \
                any(k.startswith("reconstruction_loss_train") for k in available)
    has_val = any(k.startswith("elbo_validation") for k in available) or \
              any(k.startswith("reconstruction_loss_validation") for k in available)

    if not has_train and not has_val:
        print("[Eval] No loss history found — skipping loss plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = None

    # -- ELBO --
    ax = axes[0]
    for key, label, color in [("elbo_train", "Train", "steelblue"),
                               ("elbo_validation", "Val", "darkorange")]:
        if key in history:
            vals = history[key]
            if epochs is None:
                epochs = list(range(1, len(vals) + 1))
            ax.plot(epochs[:len(vals)], vals, label=label, color=color, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO")
    ax.legend()

    # -- Reconstruction loss --
    ax = axes[1]
    for key, label, color in [("reconstruction_loss_train", "Train", "steelblue"),
                               ("reconstruction_loss_validation", "Val", "darkorange")]:
        if key in history:
            vals = history[key]
            ax.plot(epochs[:len(vals)], vals, label=label, color=color, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recon Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend()

    # -- KL divergence --
    ax = axes[2]
    for key, label, color in [("kl_local_train", "Train", "steelblue"),
                               ("kl_local_validation", "Val", "darkorange")]:
        if key in history:
            vals = history[key]
            ax.plot(epochs[:len(vals)], vals, label=label, color=color, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Loss")
    ax.set_title("KL Divergence")
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Loss curves saved to: {save_path}")


def evaluate_reconstruction(model: Any, adata: Any, save_path: Path, n_cells: int = 8000, n_points: int = 8000000, seed: int = 42) -> dict:
    """Scatter-plot true vs reconstructed expression and compute Pearson r + MSE.

    Subsets cells (keeping all genes to satisfy scVI), then randomly samples
    (cell, gene) pairs from within that slice for the scatter plot.
    """
    rng = np.random.default_rng(seed)

    total_cells = adata.n_obs
    n_cells = min(n_cells, total_cells)

    cell_idx = rng.choice(total_cells, size=n_cells, replace=False)
    adata_sub = adata[cell_idx, :].copy()

    # Normalize raw counts to library-size scale matching get_normalized_expression
    X_sub = adata_sub.X
    if hasattr(X_sub, "toarray"):
        X_sub = X_sub.toarray()
    X_sub = np.asarray(X_sub, dtype=np.float64)
    X_sub = X_sub / np.maximum(np.sum(X_sub, axis=1, keepdims=True), 1e-9)

    # Reconstructed: scVI normalized expression (same library-size scale)
    recon = model.get_normalized_expression(adata_sub,library_size=1.0)
    #recon = model.posterior_predictive_sample(adata_sub, n_samples=128)
    if hasattr(recon, "toarray"):
        recon = recon.toarray()
    elif hasattr(recon, "to_numpy"):
        recon = recon.to_numpy()
    recon = np.asarray(recon, dtype=np.float64)
    recon = np.maximum(recon, 0)

    # Randomly sample (cell, gene) pairs for scatter
    n_obs, n_vars = X_sub.shape
    total_pairs = n_obs * n_vars
    n_points = min(n_points, total_pairs)
    flat_idx = rng.choice(total_pairs, size=n_points, replace=False)
    c_idx = flat_idx // n_vars
    g_idx = flat_idx % n_vars

    t = np.log1p(X_sub[c_idx, g_idx])
    p = np.log1p(recon[c_idx, g_idx])

    # Metrics
    t_std, p_std = np.std(t), np.std(p)
    pearson = float(np.corrcoef(t, p)[0, 1]) if t_std > 0 and p_std > 0 else float("nan")
    mse = float(np.mean((t - p) ** 2))
    mae = float(np.mean(np.abs(t - p)))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(t, p, s=2, alpha=0.5, color="steelblue")
    lo, hi = float(np.min([t, p])), float(np.max([t, p]))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("True log1p(expr)")
    ax.set_ylabel("Reconstructed log1p(expr)")
    ax.set_title(f"scVI Reconstruction  (Pearson r={pearson:.4f}  MSE={mse:.4f}  MAE={mae:.4f})")
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Recon scatter saved to: {save_path}")

    return {"pearson_r": pearson, "mse": mse, "mae": mae,
            "n_cells_sampled": n_cells, "n_points_scatter": n_points}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain scVI on scRNA-seq data")
    parser.add_argument("--data", type=str, default="umi_highquality",
                        choices=["umi_processed", "umi_highquality", "processed", "log_processed"],
                        help="Data version (umi_* = raw UMI counts)")
    parser.add_argument("--n-latent", type=int, default=64,
                        help="Latent dimension")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="Hidden units in encoder/decoder")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of hidden layers")
    parser.add_argument("--dropout-rate", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate")
    parser.add_argument("--lr-patience", type=int, default=25,
                        help="Epochs of no improvement before reducing LR (0 = no reduction)")
    parser.add_argument("--lr-factor", type=float, default=0.6,
                        help="Factor by which to reduce LR on plateau")
    parser.add_argument("--lr-threshold", type=float, default=3e-3,
                        help="Relative improvement threshold for LR reduction (default 0.1%% of best ELBO)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Fraction of cells held out for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to saved scVI model dir (skip training, only evaluate + update encoder)")
    args = parser.parse_args()

    if args.load_model and not Path(args.load_model).exists():
        raise FileNotFoundError(f"Model dir not found: {args.load_model}")

    data_path = PROJECT_ROOT / "data" / args.data / "integrated_data.h5ad"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found in {PROJECT_ROOT / 'data' / args.data}")

    eval_dir = PROJECT_ROOT / "outputs" / "pretrain_vae"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {data_path}")
    adata = sc.read(data_path)
    n_cells, n_genes = adata.shape
    print(f"  Cells: {n_cells}, Genes: {n_genes}")

    from scvi.model import SCVI

    print("Setting up AnnData for scVI...")
    SCVI.setup_anndata(adata)

    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        model = SCVI.load(args.load_model, adata=adata)
        args.n_latent = model.module.n_latent
        # Extract n_hidden / n_layers from encoder structure
        enc = model.module.z_encoder
        n_hidden_loaded = enc.encoder.fc_layers[0][0].out_features
        n_layers_loaded = len(enc.encoder.fc_layers)
        args.n_hidden = n_hidden_loaded
        args.n_layers = n_layers_loaded
        print(f"  Loaded: n_latent={args.n_latent}, n_hidden={args.n_hidden}, n_layers={args.n_layers}")

    else:
        print(f"Building scVI model: n_latent={args.n_latent}, n_hidden={args.n_hidden}, "
              f"n_layers={args.n_layers}")
        model = SCVI(
            adata,
            n_latent=args.n_latent,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            dropout_rate=args.dropout_rate,
        )
        train_size = 1.0 - args.val_size
        print(f"Training for up to {args.epochs} epochs (train={train_size:.0%}, val={args.val_size:.0%})...")
        model.train(
            max_epochs=args.epochs,
            early_stopping=True,
            early_stopping_monitor="elbo_validation",
            batch_size=args.batch_size,
            accelerator="auto",
            enable_progress_bar=True,
            plan_kwargs={
                "lr": args.lr,
                "n_epochs_kl_warmup": 1000,
                "reduce_lr_on_plateau": args.lr_patience > 0,
                "lr_patience": args.lr_patience,
                "lr_factor": args.lr_factor,
                "lr_threshold": args.lr_threshold,
            },
            train_size=train_size,
            validation_size=args.val_size,
            check_val_every_n_epoch=1,
        )

    out_dir = PROJECT_ROOT / "outputs" / f"scvi_{args.n_latent}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Evaluation: loss curves + reconstruction ----
    print("\nEvaluating pretraining results...")

    history = model.history
    if history:
        plot_loss_curves(history, eval_dir / "loss_curve.png")

    metrics = evaluate_reconstruction(
        model, adata,
        save_path=eval_dir / "recon_scatter.png",
        seed=args.seed,
    )

    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v if not isinstance(v, float) or not np.isnan(v) else None
                   for k, v in metrics.items()}, f, indent=2)
    print(f"[Eval] Metrics saved to: {metrics_path}")

    # ---- Save model outputs ----
    if not args.load_model:
        scvi_dir = out_dir / "scvi_model"
        print(f"\nSaving scVI model to {scvi_dir}")
        model.save(scvi_dir, overwrite=True)

    n_latent_saved = args.n_latent
    n_hidden_saved = args.n_hidden
    n_layers_saved = args.n_layers

    encoder_path = out_dir / "encoder.pt"
    print(f"Saving encoder weights to {encoder_path}")
    torch.save(model.module.z_encoder.state_dict(), encoder_path)

    config = {
        "n_input": n_genes,
        "n_latent": n_latent_saved,
        "n_hidden": n_hidden_saved,
        "n_layers": n_layers_saved,
        "dropout_rate": args.dropout_rate,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saving config to {config_path}")

    latent_path = out_dir / "latent_z.npy"
    print(f"Computing and saving latent representations to {latent_path}")
    latent_z = model.get_latent_representation()
    np.save(latent_path, latent_z)
    print(f"  latent_z shape: {latent_z.shape}")

    print(f"\nDone.")
    print(f"  Model:   {out_dir}/")
    print(f"  Eval:    {eval_dir}/")


if __name__ == "__main__":
    main()
