# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:48:57 2026

@author: HP
"""
import argparse
from pathlib import Path
import csv
import math
import json
from datetime import datetime
import shutil
import sys

import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import copy
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MyDataset
from src.model import MODEL_REGISTRY, build_model
from src.earlystopping import EarlyStopping
import src.utils as utils
from safetensors.torch import save_file, load_file


def weighted_mse_loss(pred, target, nonzero_weight=2.0):
    """MSE with higher weight on non-zero targets."""
    weights = torch.ones_like(target)
    weights[target != 0] = nonzero_weight
    return torch.mean(weights * (pred - target) ** 2)

def train_model(
    model,
    train_loader,
    val_loader,
    exp_name,
    epochs=30,
    learning_rate=1e-4,
    nonzero_loss_weight=2.0,
    seed=42,
    patience=5,           # early stopping patience (epochs)
    min_delta=0.0,        # minimal loss improvement to reset patience
    resume_ckpt=None,      # 断点路径，如 outputs/<exp_name>/checkpoints/last.ckpt
    save_every=0,          # 每隔多少个 epoch 额外保存一个 epoch_xxxx.ckpt
):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=learning_rate * 0.01,
    )
    earlystopping = EarlyStopping(patience=patience, min_delta=min_delta)

    base_dir = Path(__file__).resolve().parent.parent
    _, ckpt_dir, _, log_dir = utils._prepare_output_dirs(base_dir, exp_name)
    log_file = log_dir / "train_log.csv"

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # Resume from checkpoint if provided
    if resume_ckpt is not None:
        resume_ckpt = Path(resume_ckpt)
        if resume_ckpt.exists():
            start_epoch, earlystopping, train_losses, val_losses = utils.resume_from_checkpoint(
                resume_ckpt, model, optimizer, scheduler, earlystopping, device
            )
            print(f"[Resume] loaded: {resume_ckpt} | start_epoch={start_epoch} | best_val={best_val_loss}")
        else:
            print(f"[Resume] checkpoint not found, start from scratch: {resume_ckpt}")

    if start_epoch >= epochs:
        print(f"[Resume] start_epoch({start_epoch}) >= epochs({epochs}), nothing to train.")
        return

    # Training loop
    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            promoters, exprs, ys = batch
            promoters = promoters.to(device, non_blocking=True)
            exprs = exprs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            out = model(promoters, exprs).squeeze(1)
            loss = weighted_mse_loss(out, ys, nonzero_weight=nonzero_loss_weight)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * ys.size(0)
            train_count += ys.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()

        avg_train_loss = train_loss_sum / max(train_count, 1)
        train_losses.append(avg_train_loss)

        # Validation loop
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    promoters, exprs, ys = batch
                    promoters = promoters.to(device, non_blocking=True)
                    exprs = exprs.to(device, non_blocking=True)
                    ys = ys.to(device, non_blocking=True).float()

                    out = model(promoters, exprs).squeeze(1)
                    loss = weighted_mse_loss(out, ys, nonzero_weight=nonzero_loss_weight)

                    val_loss_sum += loss.item() * ys.size(0)
                    val_count += ys.size(0)

            avg_val_loss = val_loss_sum / max(val_count, 1)
        else:
            avg_val_loss = float("nan")

        val_losses.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss} | Val Loss={avg_val_loss} | LR={current_lr:.3e}")

        utils.append_epoch_log(
            log_file=log_file,
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss if not math.isnan(avg_val_loss) else np.nan,
            lr=current_lr,
        )

        # Update best metric and early-stopping counter
        monitor_loss=avg_val_loss if not math.isnan(avg_val_loss) else avg_train_loss
        earlystopping(monitor_loss)
        if monitor_loss == earlystopping.best_score:
            utils.robust_save_model(model, ckpt_dir / "best_model.safetensors")

        scheduler.step()

        utils.save_checkpoint(
            checkpoint_path=ckpt_dir / "last.ckpt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            earlystopping=earlystopping,
            train_losses=train_losses,
            val_losses=val_losses,
        )

        if save_every > 0 and ((epoch + 1) % save_every == 0):
            utils.save_checkpoint(
                checkpoint_path=ckpt_dir / f"epoch_{epoch+1:04d}.ckpt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                earlystopping=earlystopping,
                train_losses=train_losses,
                val_losses=val_losses,
            )

        if patience > 0 and earlystopping.early_stop:
            print(
                f"Early stopping at epoch {epoch+1}: "
                f"no improvement for {earlystopping.counter} epochs (patience={patience})."
            )
            break

    print(f"Training done. logs: {log_file} | checkpoints: {ckpt_dir}")
        

def main():
    parser = argparse.ArgumentParser(description="Train a gene expression model.")
    parser.add_argument("--exp_name", type=str, required=True, default='default', help="Name of the experiment (used for organizing outputs)")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameter config.json")
    parser.add_argument("--model", type=str, default="SimpleGeneModel", choices=sorted(MODEL_REGISTRY.keys()), help="Model architecture to use")
    parser.add_argument("--data", type=str, default="highquality", choices=["highquality", "processed"], help="Which dataset version to use (affects data paths in config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--dryrun", action="store_true", default=False, help="Run dryrun_cpu before real training")
    parser.add_argument("--plot-loss", action="store_true", default=False, help="Plot training loss curve after training")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size for LSTM and MLP in the model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0 avoids extra memory copies)")
    parser.add_argument("--samples-per-epoch", type=int, default=90000000, help="Fixed number of unique samples to draw per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--nonzero-loss-weight", type=float, default=10.0, help="Weight multiplier for non-zero labels in MSE loss")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-100, help="Minimum loss improvement to reset patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility of validation sampling and training")
    parser.add_argument("--max-resume-snapshots", type=int, default=5, help="Max number of resume snapshots to keep (0 means keep all)")
    args = parser.parse_args()

    # # 允许 cuDNN 自动寻找最适合当前配置的算法（提高速度）
    # torch.backends.cudnn.benchmark = True 
    # # 强制 cuDNN 使用确定性算法（确保复现，但可能会稍微降低速度）
    # torch.backends.cudnn.deterministic = True

    print("start")
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / args.data

    train_dataset = MyDataset(
        promoter_file=data_dir / "promoter_train.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
    )
    val_dataset = MyDataset(
        promoter_file=data_dir / "promoter_val.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        mode="val",
        seed=args.seed,  # use the same seed to ensure val sampling is consistent with evaluation
    )

    pin_memory = torch.cuda.is_available()
    if args.data == "highquality":
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers+3,  # more workers for val_loader to speed up evaluation
            pin_memory=pin_memory,
        )
    else:
        train_sampler = utils.BalancedEpochSubsetSampler(
            train_dataset,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers+3,  # more workers for val_loader to speed up evaluation
            pin_memory=pin_memory,
        )

    expr_dim = train_dataset.X.shape[1]
    model = build_model(args.model, expr_dim=expr_dim, hidden_size=args.hidden_size)

    run_dir, ckpt_dir, plots_dir, _ = utils._prepare_output_dirs(base_dir, args.exp_name)
    has_model_backup = any(run_dir.glob("model_arch_*.py"))
    if not has_model_backup:
        utils.backup_model_architecture(run_dir, args.model)

    if args.dryrun:
        utils.dryrun_cpu(model, train_loader, steps=50, learning_rate=1e-4, save_path=plots_dir / "dryrun.png")
        # dryrun会改参数，重新初始化模型再正式训练
        model = build_model(args.model, expr_dim=expr_dim, hidden_size=args.hidden_size)

    resume = args.resume
    if resume is None:
        auto_last = base_dir / "outputs" / args.exp_name / "checkpoints" / "last.ckpt"
        resume = str(auto_last) if auto_last.exists() else None

    previous_cfg, current_cfg = utils.save_run_config(run_dir / "config.json", args, base_dir, expr_dim, resume)
    if resume is not None and Path(resume).exists():
        utils.save_resume_snapshot(
            ckpt_dir=ckpt_dir,
            resume_ckpt=Path(resume),
            max_keep=args.max_resume_snapshots,
        )
        utils.append_resume_config_history(
            run_dir / "config_history.log",
            resume_ckpt=resume,
            previous_cfg=previous_cfg,
            current_cfg=current_cfg,
        )

    train_model(
        model,
        train_loader,
        val_loader=val_loader,
        exp_name=args.exp_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        nonzero_loss_weight=args.nonzero_loss_weight,
        seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
        resume_ckpt=resume,
        save_every=0
    )
 
    if args.plot_loss:
        log_file = base_dir / "outputs" / args.exp_name / "log" / "train_log.csv"
        if log_file.exists():
            utils.plot_loss_curves_from_logfile(log_file, save_path=plots_dir / "loss_curve.png")

            best_model_path = base_dir / "outputs" / args.exp_name / "checkpoints" / "best_model.safetensors"
            if best_model_path.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.load_state_dict(load_file(str(best_model_path), device=str(device)))
                model.to(device)
                print(f"Loaded best model for scatter plot: {best_model_path}")

            utils.plot_pred_scatter(model, val_loader, max_steps=2000, save_path=plots_dir / "pred_vs_true_scatter.png")
        else:
            print(f"Log file not found for plotting: {log_file}")
# %%       
if __name__ == "__main__":
    main()

# %%


