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

from src.dataset import MyDataset
from src.model import MODEL_REGISTRY, build_model
from src.earlystopping import EarlyStopping
import src.utils as utils
from safetensors.torch import save_file, load_file

class ZINBLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_true, mu, theta, pi):
        """
        y_true: 原始 UMI Count (Batch_size, 1)
        mu:     调整过 Library Size 后的期望值 (mu_ratio * lib_size)
        theta:  离散度参数
        pi:     零膨胀概率
        """
        # 1. 计算标准负二项分布 (NB) 的概率密度部分
        log_theta_mu_eps = torch.log(theta + mu + self.eps)
        
        log_nb = (
            torch.lgamma(y_true + theta + self.eps)
            - torch.lgamma(y_true + 1.0)
            - torch.lgamma(theta + self.eps)
            + theta * (torch.log(theta + self.eps) - log_theta_mu_eps)
            + y_true * (torch.log(mu + self.eps) - log_theta_mu_eps)
        )
        nb_case = torch.exp(log_nb)

        # 2. 针对 y = 0 和 y > 0 分别处理
        zero_case = pi + (1.0 - pi) * nb_case
        non_zero_case = (1.0 - pi) * nb_case

        # 3. 使用 torch.where 组合结果
        # torch.where(condition, x, y): 满足 condition 用 x，不满足用 y
        loss = torch.where(
            y_true < self.eps,
            -torch.log(zero_case + self.eps),
            -torch.log(non_zero_case + self.eps)
        )
        
        return torch.mean(loss)

def weighted_mse_loss(pred, target, nonzero_weight=2.0):
    """MSE with higher weight on non-zero targets."""
    weights = torch.ones_like(target)
    weights[target != 0] = nonzero_weight
    sq = (pred - target) ** 2
    return (weights * sq).sum() / weights.sum().clamp_min(1e-12)


def pearson_loss(pred, target, eps=1e-8):
    """1 - Pearson correlation coefficient as loss (batch-level)."""
    pred = pred.view(-1)
    target = target.view(-1)

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    cov = (pred_centered * target_centered).sum()
    pred_var = (pred_centered ** 2).sum()
    target_var = (target_centered ** 2).sum()

    denom = (pred_var * target_var).sqrt() + eps
    r = cov / denom
    return 1.0 - r


def pearson_mse_loss(pred, target, nonzero_weight=2.0, pearson_lambda=1.0, eps=1e-8):
    """weighted MSE + lambda * (1 - Pearson)."""
    mse = weighted_mse_loss(pred, target, nonzero_weight)
    p_loss = pearson_loss(pred, target, eps)
    return mse + pearson_lambda * p_loss


def train_model(
    model,
    train_loader,
    val_loader,
    exp_name,
    epochs=30,
    learning_rate=1e-4,
    nonzero_loss_weight=2.0,
    seed=42,
    patience=5,
    min_delta=0.0,
    resume_ckpt=None,
    save_every=0,
    zero_acc_threshold=0.5,
    ema_alpha=0.9,
    loss_type="mse",
    pearson_lambda=1.0,
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
    step_log_file = log_dir / "step_train_loss.csv"
    global_step = 0
    if step_log_file.exists():
        global_step = pd.read_csv(step_log_file)["global_step"].max() + 1

    start_epoch = 0
    val_loss_ema = None
    train_losses = []
    val_losses = []
    train_losses_zero = []
    train_losses_nz = []
    val_losses_zero = []
    val_losses_nz = []
    val_pearson_nz = []
    val_zero_acc = []

    # Resume from checkpoint if provided
    if resume_ckpt is not None:
        resume_ckpt = Path(resume_ckpt)
        if resume_ckpt.exists():
            start_epoch, earlystopping, train_losses, val_losses = utils.resume_from_checkpoint(
                resume_ckpt, model, optimizer, scheduler, earlystopping, device
            )
            print(f"[Resume] loaded: {resume_ckpt} | start_epoch={start_epoch} | best_score={earlystopping.best_score}")
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
        train_loss_num = 0.0
        train_loss_den = 0.0
        train_nz_sum = 0.0
        train_nz_count = 0
        train_zero_sum = 0.0
        train_zero_count = 0
        epoch_step_losses = []

        for batch in train_loader:
            promoters, exprs, ys = batch
            promoters = promoters.to(device, non_blocking=True)
            exprs = exprs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            out = model(promoters, exprs).squeeze(1)
            if loss_type == "pearson":
                loss = pearson_loss(out, ys)
            elif loss_type == "combined":
                loss = pearson_mse_loss(out, ys, nonzero_weight=nonzero_loss_weight, pearson_lambda=pearson_lambda)
            else:
                loss = weighted_mse_loss(out, ys, nonzero_weight=nonzero_loss_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                sq = (out - ys) ** 2
                zero_mask = (ys == 0)
                nz_mask = ~zero_mask

                # weighted average aggregation (matches loss function)
                w = torch.where(ys != 0, torch.tensor(nonzero_loss_weight, device=ys.device, dtype=ys.dtype),
                                torch.ones_like(ys))
                weighted_sq_sum = (w * sq).sum().item()
                weight_sum = w.sum().item()
                train_loss_num += weighted_sq_sum
                train_loss_den += weight_sum

                nz_cnt = nz_mask.sum().item()
                if nz_cnt > 0:
                    train_nz_sum += sq[nz_mask].sum().item()
                    train_nz_count += nz_cnt

                zero_cnt = zero_mask.sum().item()
                if zero_cnt > 0:
                    train_zero_sum += sq[zero_mask].sum().item()
                    train_zero_count += zero_cnt

            if device.type == "cuda":
                torch.cuda.synchronize()

            epoch_step_losses.append(loss.item())

        if loss_type == "pearson":
            avg_train_loss = sum(epoch_step_losses) / max(len(epoch_step_losses), 1)
        else:
            avg_train_loss = train_loss_num / max(train_loss_den, 1e-12)
        train_losses.append(avg_train_loss)
        train_losses_zero.append(train_zero_sum / max(train_zero_count, 1))
        train_losses_nz.append(train_nz_sum / max(train_nz_count, 1))

        # Flush step-level train losses
        utils.append_step_log(step_log_file, epoch_step_losses, epoch + 1, global_step)
        global_step += len(epoch_step_losses)

        # Validation loop
        avg_val_loss = float("nan")
        epoch_val_pearson = float("nan")
        epoch_val_pearson_all = float("nan")
        epoch_val_zero_acc = float("nan")

        if val_loader is not None:
            model.eval()
            val_loss_num = 0.0
            val_loss_den = 0.0
            val_nz_sum = 0.0
            val_nz_count = 0
            val_zero_sum = 0.0
            val_zero_count = 0
            all_nz_true = []
            all_nz_pred = []
            all_zero_pred = []
            all_val_true = []
            all_val_pred = []

            with torch.no_grad():
                for batch in val_loader:
                    promoters, exprs, ys = batch
                    promoters = promoters.to(device, non_blocking=True)
                    exprs = exprs.to(device, non_blocking=True)
                    ys = ys.to(device, non_blocking=True).float()

                    out = model(promoters, exprs).squeeze(1)

                    sq = (out - ys) ** 2
                    zero_mask = (ys == 0)
                    nz_mask = ~zero_mask

                    w = torch.where(ys != 0, torch.tensor(nonzero_loss_weight, device=ys.device, dtype=ys.dtype),
                                    torch.ones_like(ys))
                    val_loss_num += (w * sq).sum().item()
                    val_loss_den += w.sum().item()

                    nz_cnt = nz_mask.sum().item()
                    if nz_cnt > 0:
                        val_nz_sum += sq[nz_mask].sum().item()
                        val_nz_count += nz_cnt
                        all_nz_true.append(ys[nz_mask].cpu().numpy())
                        all_nz_pred.append(out[nz_mask].cpu().numpy())

                    zero_cnt = zero_mask.sum().item()
                    if zero_cnt > 0:
                        val_zero_sum += sq[zero_mask].sum().item()
                        val_zero_count += zero_cnt
                        all_zero_pred.append(out[zero_mask].cpu().numpy())

                    if loss_type in ("pearson", "combined"):
                        all_val_true.append(ys.cpu().numpy())
                        all_val_pred.append(out.cpu().numpy())

            avg_val_loss = val_loss_num / max(val_loss_den, 1e-12)
            if loss_type in ("pearson", "combined") and all_val_true:
                val_true_all = np.concatenate(all_val_true)
                val_pred_all = np.concatenate(all_val_pred)
                if len(val_true_all) > 1 and np.std(val_true_all) > 0 and np.std(val_pred_all) > 0:
                    epoch_val_pearson_all = float(np.corrcoef(val_true_all, val_pred_all)[0, 1])
                    if loss_type == "pearson":
                        avg_val_loss = 1.0 - epoch_val_pearson_all
                    else:
                        avg_val_loss = avg_val_loss + pearson_lambda * (1.0 - epoch_val_pearson_all)
                elif loss_type == "pearson":
                    avg_val_loss = 1.0
            elif loss_type == "pearson":
                avg_val_loss = 1.0
            val_losses_zero.append(val_zero_sum / max(val_zero_count, 1))
            val_losses_nz.append(val_nz_sum / max(val_nz_count, 1))

            # Pearson correlation on non-zero samples
            if all_nz_true:
                nz_true_all = np.concatenate(all_nz_true)
                nz_pred_all = np.concatenate(all_nz_pred)
                if len(nz_true_all) > 1 and np.std(nz_true_all) > 0 and np.std(nz_pred_all) > 0:
                    epoch_val_pearson = float(np.corrcoef(nz_true_all, nz_pred_all)[0, 1])
                val_pearson_nz.append(epoch_val_pearson)

            # Accuracy on zero samples: |pred| < threshold
            if all_zero_pred:
                zero_pred_all = np.concatenate(all_zero_pred)
                epoch_val_zero_acc = float((np.abs(zero_pred_all) < zero_acc_threshold).mean())
                val_zero_acc.append(epoch_val_zero_acc)
        else:
            val_losses_zero.append(float("nan"))
            val_losses_nz.append(float("nan"))

        val_losses.append(avg_val_loss)

        # EMA-smoothed validation loss
        monitor_loss = avg_val_loss if not math.isnan(avg_val_loss) else avg_train_loss
        if val_loss_ema is None:
            val_loss_ema = monitor_loss
        else:
            val_loss_ema = ema_alpha * val_loss_ema + (1 - ema_alpha) * monitor_loss

        current_lr = optimizer.param_groups[0]["lr"]
        log_msg = (
            f"Epoch {epoch+1}/{epochs}: Train={avg_train_loss:.6f} Val={avg_val_loss:.6f} "
            f"ValEMA={val_loss_ema:.6f} LR={current_lr:.3e}"
        )
        if not math.isnan(epoch_val_pearson_all):
            log_msg += f" | Pearson(All)={epoch_val_pearson_all:.4f}"
        if not math.isnan(epoch_val_pearson):
            log_msg += f" | Pearson(NZ)={epoch_val_pearson:.4f}"
        if not math.isnan(epoch_val_zero_acc):
            log_msg += f" | ZeroAcc={epoch_val_zero_acc:.4f}"
        print(log_msg)

        utils.append_epoch_log(
            log_file=log_file,
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss if not math.isnan(avg_val_loss) else np.nan,
            lr=current_lr,
            val_loss_ema=val_loss_ema,
            train_loss_zero=train_losses_zero[-1],
            train_loss_nonzero=train_losses_nz[-1],
            val_loss_zero=val_losses_zero[-1],
            val_loss_nonzero=val_losses_nz[-1],
            val_pearson_nonzero=epoch_val_pearson,
            val_pearson_all=epoch_val_pearson_all,
            val_zero_accuracy=epoch_val_zero_acc,
        )

        # Update best metric and early-stopping counter (uses EMA-smoothed loss)
        earlystopping(val_loss_ema)
        if val_loss_ema == earlystopping.best_score:
            utils.robust_save_model(model, ckpt_dir / "best_model.safetensors")

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
    parser.add_argument("--model", type=str, default="LSTMmodel", choices=sorted(MODEL_REGISTRY.keys()), help="Model architecture to use")
    parser.add_argument("--data", type=str, default="umi_processed", choices=["highquality", "processed", "log_processed", "umi_highquality", "umi_processed"], help="Which dataset version to use (affects data paths in config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--dryrun", action="store_true", default=False, help="Run dryrun_cpu before real training")
    parser.add_argument("--plot-loss", action="store_true", default=True, help="Plot training loss curve after training")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for LSTM and MLP in the model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (0 avoids extra memory copies)")
    parser.add_argument("--samples-per-epoch", type=int, default=128000, help="Fixed number of unique samples to draw per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--nonzero-loss-weight", type=float, default=2.0, help="Weight multiplier for non-zero labels in MSE loss")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-2, help="Minimum loss improvement to reset patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility of validation sampling and training")
    parser.add_argument("--max-resume-snapshots", type=int, default=5, help="Max number of resume snapshots to keep (0 means keep all)")
    parser.add_argument("--nonzero-ratio", type=float, default=None, help="Target ratio of non-zero samples per epoch (uses ZeroNonZeroSampler). Default: natural ratio from data")
    parser.add_argument("--zero-acc-threshold", type=float, default=0.5, help="Threshold for zero-sample accuracy (|pred| < threshold counts as correct)")
    parser.add_argument("--ema-alpha", type=float, default=0.9, help="EMA smoothing factor for validation loss (0=use raw, higher=smoother)")
    parser.add_argument("--cell-ratio", type=float, default=1.0, help="Fraction of cells to randomly subsample (0-1). Useful for reducing memory usage with processed data + ZeroNonZeroSampler")
    parser.add_argument("--loss", type=str, default="combined", choices=["mse", "pearson", "combined"], help="Loss function: 'mse' (weighted MSE), 'pearson' (1 - Pearson), or 'combined' (MSE + lambda*(1-Pearson))")
    parser.add_argument("--pearson-lambda", type=float, default=10.0, help="Lambda weight for the Pearson term in combined loss (only used with --loss combined)")
    parser.add_argument("--vae-encoder", type=str, default=None, help="Path to scVI output dir (e.g., outputs/scvi_10/) containing encoder.pt and config.json")
    parser.add_argument("--vae-fine-tune", action="store_true", default=False, help="Unfreeze scVI encoder weights during training")
    args = parser.parse_args()

    # # 允许 cuDNN 自动寻找最适合当前配置的算法（提高速度）
    # torch.backends.cudnn.benchmark = True 
    # # 强制 cuDNN 使用确定性算法（确保复现，但可能会稍微降低速度）
    # torch.backends.cudnn.deterministic = True

    print("start")
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / args.data

    use_log1p = args.data.startswith("umi_")
    train_dataset = MyDataset(
        promoter_file=data_dir / "promoter_train.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        cell_ratio=args.cell_ratio,
        log1p_cpm_target=use_log1p,
    )
    val_dataset = MyDataset(
        promoter_file=data_dir / "promoter_val.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        mode="val",
        seed=args.seed,
        cell_ratio=args.cell_ratio,
        log1p_cpm_target=use_log1p,
    )

    pin_memory = torch.cuda.is_available()
    if args.data in ("processed", "log_processed", "umi_processed"):
        if args.nonzero_ratio is not None:
            train_sampler = utils.ZeroNonZeroSampler(
                train_dataset,
                nonzero_ratio=args.nonzero_ratio,
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
            val_sampler = utils.ZeroNonZeroSampler(
                val_dataset,
                nonzero_ratio=args.nonzero_ratio,
                samples_per_epoch=args.samples_per_epoch,
                seed=args.seed,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=args.num_workers,
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
            val_sampler = utils.BalancedEpochSubsetSampler(
                val_dataset,
                samples_per_epoch=args.samples_per_epoch,
                seed=args.seed,            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
    elif args.data in ("highquality", "umi_highquality"):
        if args.nonzero_ratio is not None:
            train_sampler = utils.ZeroNonZeroSampler(
                train_dataset,
                nonzero_ratio=args.nonzero_ratio,
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
        else:
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
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    expr_dim = train_dataset.X.shape[1]
    if args.vae_encoder is not None and not Path(args.vae_encoder).exists():
        raise FileNotFoundError(f"VAE encoder dir not found: {args.vae_encoder}")
    model = build_model(
        args.model,
        expr_dim=expr_dim,
        hidden_size=args.hidden_size,
        use_vae=args.vae_encoder is not None,
        vae_encoder_path=args.vae_encoder,
        vae_fine_tune=args.vae_fine_tune,
    )

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
        save_every=0,
        zero_acc_threshold=args.zero_acc_threshold,
        ema_alpha=args.ema_alpha,
        loss_type=args.loss,
        pearson_lambda=args.pearson_lambda,
    )
 
    if args.plot_loss:
        log_file = base_dir / "outputs" / args.exp_name / "log" / "train_log.csv"
        if log_file.exists():
            utils.plot_loss_curves_from_logfile(log_file, save_path=plots_dir / "loss_curve.png")
            utils.plot_zero_nonzero_loss_curves(log_file, save_path=plots_dir / "zero_nonzero_loss.png")
            utils.plot_val_metrics(log_file, save_path=plots_dir / "val_metrics.png")

            best_model_path = base_dir / "outputs" / args.exp_name / "checkpoints" / "best_model.safetensors"
            if best_model_path.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.load_state_dict(load_file(str(best_model_path), device=str(device)))
                model.to(device)
                print(f"Loaded best model for scatter plot: {best_model_path}")

            utils.count_zero_nonzero(val_loader)
            utils.plot_pred_scatter(model, val_loader, epoch=1, save_path=plots_dir / "pred_vs_true_scatter.png")
        else:
            print(f"Log file not found for plotting: {log_file}")
# %%       
if __name__ == "__main__":
    main()

# %%


