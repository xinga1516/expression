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
from typing import Any

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MyDataset
from src.model import MODEL_REGISTRY, build_model
from src.earlystopping import EarlyStopping
import src.utils as utils
from safetensors.torch import save_file, load_file


def dataloader_worker_kwargs(num_workers: int, prefetch_factor: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def resolve_cell_split_dir(base_dir: Path, data_name: str, cell_split_dir: str | None) -> Path:
    if cell_split_dir is None:
        return base_dir / "data" / data_name
    path = Path(cell_split_dir)
    if not path.is_absolute():
        path = base_dir / path
    return path


def read_cell_split(cell_split_dir: Path, split: str) -> np.ndarray:
    split_path = cell_split_dir / f"cell_{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Cell split file not found: {split_path}")
    cells = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cells:
        raise ValueError(f"Cell split file is empty: {split_path}")
    return np.asarray(cells, dtype=object)


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_batch_input_mib(batch_size: int, expr_dim: int, promoter_len: int = 400, promoter_channels: int = 5) -> float:
    bytes_per_batch = batch_size * (promoter_len * promoter_channels + expr_dim) * 4
    return bytes_per_batch / (1024 ** 2)


def print_training_resource_summary(
    model: nn.Module,
    batch_size: int,
    expr_dim: int,
    samples_per_epoch: int,
    val_samples: int,
    cell_ratio: float,
    val_cell_ratio: float,
    num_workers: int,
    prefetch_factor: int,
    amp: bool,
) -> None:
    total_params, trainable_params = count_model_parameters(model)
    input_mib = estimate_batch_input_mib(batch_size=batch_size, expr_dim=expr_dim)
    print("===== Resource Summary =====")
    print(f"params: total={total_params:,} trainable={trainable_params:,}")
    print(f"batch_size={batch_size} expr_dim={expr_dim} estimated_input={input_mib:.1f} MiB/batch")
    print(
        f"samples_per_epoch={samples_per_epoch} val_samples={val_samples} "
        f"cell_ratio={cell_ratio} val_cell_ratio={val_cell_ratio}"
    )
    print(f"num_workers={num_workers} prefetch_factor={prefetch_factor} amp={amp}")


def set_vae_trainable(model: nn.Module, trainable: bool) -> bool:
    if not hasattr(model, "vae_encoder"):
        return False
    vae_encoder = getattr(model, "vae_encoder")
    for param in vae_encoder.parameters():
        param.requires_grad = trainable
    if hasattr(model, "vae_fine_tune"):
        setattr(model, "vae_fine_tune", trainable)
    if trainable:
        vae_encoder.train()
    else:
        vae_encoder.eval()
    return trainable


def count_vae_parameters(model: nn.Module) -> tuple[int, int]:
    if not hasattr(model, "vae_encoder"):
        return 0, 0
    vae_encoder = getattr(model, "vae_encoder")
    total = sum(param.numel() for param in vae_encoder.parameters())
    trainable = sum(param.numel() for param in vae_encoder.parameters() if param.requires_grad)
    return total, trainable


def apply_vae_fine_tune_schedule(
    model: nn.Module,
    epoch: int,
    vae_fine_tune_start_epoch: int,
    force_initial_fine_tune: bool,
) -> bool:
    if force_initial_fine_tune:
        return set_vae_trainable(model, True)
    if vae_fine_tune_start_epoch < 0:
        return set_vae_trainable(model, False)
    return set_vae_trainable(model, epoch >= vae_fine_tune_start_epoch)

class ZINBLoss(nn.Module):
    '''Zero-inflated negative binomial negative log-likelihood, computed entirely
    in log-space to avoid exp() overflow and the 0*inf=NaN edge case.'''

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_true: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        eps = self.eps

        # Clamp inputs to safe ranges before any log/lgamma
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps, max=1e12)
        pi = torch.clamp(pi, min=eps, max=1.0 - eps)

        # NB log-probability:  log P(y | mu, theta)
        log_theta_plus_mu = torch.log(theta + mu)
        log_nb = (
            torch.lgamma(y_true + theta)
            - torch.lgamma(y_true + 1.0)
            - torch.lgamma(theta)
            + theta * (torch.log(theta) - log_theta_plus_mu)
            + y_true * (torch.log(mu) - log_theta_plus_mu)
        )

        # Log-space mixture to avoid ever computing exp(log_nb):
        #   P(y) = pi * I(y=0) + (1-pi) * NB(y)
        #   log P(y>0) = log(1-pi) + log_nb
        #   log P(y=0) = logaddexp(log(pi), log(1-pi) + log_nb)
        log_pi = torch.log(pi)
        log_1m_pi = torch.log(1.0 - pi)

        y_zero = y_true < eps
        log_zero_case = torch.logaddexp(log_pi, log_1m_pi + log_nb)
        log_non_zero_case = log_1m_pi + log_nb

        log_prob = torch.where(y_zero, log_zero_case, log_non_zero_case)
        return -log_prob.mean()

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


def compute_val_loss_ema(prev_ema: float | None, monitor_loss: float, ema_alpha: float) -> float:
    """Compute the validation-loss EMA used for early stopping and best-model selection."""
    if prev_ema is None:
        return float(monitor_loss)
    return float(ema_alpha * prev_ema + (1.0 - ema_alpha) * monitor_loss)


def should_save_best_model(current_monitor_loss: float, best_score: float | None) -> bool:
    """Match the training loop's current best-model decision."""
    return best_score is not None and current_monitor_loss == best_score


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
    amp: bool = False,
    vae_fine_tune_start_epoch: int = -1,
    force_vae_fine_tune: bool = False,
):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if amp and not amp_enabled:
        print("[AMP] requested but CUDA is unavailable; running in FP32.")
    elif amp_enabled:
        print("[AMP] enabled.")
    vae_trainable = apply_vae_fine_tune_schedule(
        model,
        epoch=0,
        vae_fine_tune_start_epoch=vae_fine_tune_start_epoch,
        force_initial_fine_tune=force_vae_fine_tune,
    )
    if hasattr(model, "vae_encoder"):
        status = "trainable" if vae_trainable else "frozen"
        vae_total, vae_trainable_params = count_vae_parameters(model)
        if force_vae_fine_tune:
            print(f"[VAE] fine-tune enabled from epoch 0 ({status}).")
        elif vae_fine_tune_start_epoch >= 0:
            print(f"[VAE] fine-tune scheduled at epoch {vae_fine_tune_start_epoch}; initial status={status}.")
        else:
            print(f"[VAE] fine-tune disabled; initial status={status}.")
        print(f"[VAE] encoder params: total={vae_total:,} trainable={vae_trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=max(1, epochs),
    #     eta_min=learning_rate * 0.01,
    # )
    # 1. 定义前 5 个 Epoch 的 Warmup 调度器 (从 0.001 * 0.1 线性增长到 0.001)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=5 # warm up 5 epochs
    )
    # 2. 定义后面余弦退火的调度器 (退火步数 = 总步数 - 预热步数)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(epochs - 5), 
        eta_min=1e-6
    )
    # 3. 使用 SequentialLR 将两者串联，milestones 传入切换的节点步数
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[5]
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
            earlystopping.early_stop = False
            resume_vae_trainable = apply_vae_fine_tune_schedule(
                model,
                epoch=start_epoch,
                vae_fine_tune_start_epoch=vae_fine_tune_start_epoch,
                force_initial_fine_tune=force_vae_fine_tune,
            )
            if hasattr(model, "vae_encoder") and resume_vae_trainable != vae_trainable:
                vae_trainable = resume_vae_trainable
                vae_total, vae_trainable_params = count_vae_parameters(model)
                print(f"[VAE] resume adjusted fine-tune status: {'trainable' if vae_trainable else 'frozen'} at epoch {start_epoch}.")
                print(f"[VAE] encoder params: total={vae_total:,} trainable={vae_trainable_params:,}")
        else:
            print(f"[Resume] checkpoint not found, start from scratch: {resume_ckpt}")

    if start_epoch >= epochs:
        print(f"[Resume] start_epoch({start_epoch}) >= epochs({epochs}), nothing to train.")
        return

    zinb_loss_fn = ZINBLoss() if loss_type == "zinb" else None

    # Training loop — validate first (captures initial loss at epoch 0), then train
    for epoch in range(start_epoch, epochs):
        epoch_vae_trainable = apply_vae_fine_tune_schedule(
            model,
            epoch=epoch,
            vae_fine_tune_start_epoch=vae_fine_tune_start_epoch,
            force_initial_fine_tune=force_vae_fine_tune,
        )
        if hasattr(model, "vae_encoder") and epoch_vae_trainable != vae_trainable:
            vae_trainable = epoch_vae_trainable
            vae_total, vae_trainable_params = count_vae_parameters(model)
            print(f"[VAE] fine-tune {'enabled' if vae_trainable else 'disabled'} at epoch {epoch}.")
            print(f"[VAE] encoder params: total={vae_total:,} trainable={vae_trainable_params:,}")

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        if epoch % 10 == 0:
            # Rotate cell subset each epoch so the model eventually sees all cells
            train_ds = train_loader.dataset
            if hasattr(train_ds, "resample_cells"):
                train_ds.resample_cells(seed + epoch)
                if hasattr(train_loader.sampler, "rebuild"):
                    train_loader.sampler.rebuild(train_ds)

        # ── Validation (runs before training so epoch 0 captures initial loss) ──
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
            lstm_var_sum = 0.0
            lstm_var_count = 0
            expr_var_sum = 0.0
            expr_var_count = 0
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

                    if loss_type == "zinb":
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            mu_ratio, theta, pi = model(promoters, exprs)
                        mu_ratio = mu_ratio.squeeze(1)
                        theta = theta.squeeze(1)
                        pi = pi.squeeze(1)
                        mu_ratio = mu_ratio.float()
                        theta = theta.float()
                        pi = pi.float()
                        lib_size = exprs.sum(dim=1) + ys
                        mu = mu_ratio * lib_size
                        val_loss = zinb_loss_fn(ys, mu, theta, pi)
                        val_loss_num += val_loss.item() * ys.numel()
                        val_loss_den += ys.numel()

                        # Log-space metrics for ZINB
                        y_pred_log = torch.log(mu_ratio * 1e6 + 1)
                        y_log = torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6)
                        all_val_true.append(y_log.cpu().numpy())
                        all_val_pred.append(y_pred_log.cpu().numpy())

                        sq = (y_pred_log - y_log) ** 2
                    else:
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            out = model(promoters, exprs).squeeze(1)
                        out = out.float()
                        sq = (out - ys) ** 2

                        if loss_type in ("pearson", "combined"):
                            all_val_true.append(ys.cpu().numpy())
                            all_val_pred.append(out.cpu().numpy())

                    zero_mask = (ys == 0)
                    nz_mask = ~zero_mask

                    if hasattr(model, 'last_lstm_out'):
                        lstm_var_sum += model.last_lstm_out.var(dim=0).mean().item()
                        lstm_var_count += 1
                    if hasattr(model, 'last_expr_out'):
                        expr_var_sum += model.last_expr_out.var(dim=0).mean().item()
                        expr_var_count += 1

                    w = torch.where(ys != 0, torch.tensor(nonzero_loss_weight, device=ys.device, dtype=ys.dtype),
                                    torch.ones_like(ys))
                    if loss_type != "zinb":
                        val_loss_num += (w * sq).sum().item()
                        val_loss_den += w.sum().item()

                    nz_cnt = nz_mask.sum().item()
                    if nz_cnt > 0:
                        val_nz_sum += sq[nz_mask].sum().item()
                        val_nz_count += nz_cnt
                        all_nz_true.append(ys[nz_mask].cpu().numpy() if loss_type != "zinb" else y_log[nz_mask].cpu().numpy())
                        all_nz_pred.append(out[nz_mask].cpu().numpy() if loss_type != "zinb" else y_pred_log[nz_mask].cpu().numpy())

                    zero_cnt = zero_mask.sum().item()
                    if zero_cnt > 0:
                        val_zero_sum += sq[zero_mask].sum().item()
                        val_zero_count += zero_cnt
                        all_zero_pred.append(out[zero_mask].cpu().numpy() if loss_type != "zinb" else y_pred_log[zero_mask].cpu().numpy())

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
            elif loss_type == "zinb" and all_val_true:
                val_true_all = np.concatenate(all_val_true)
                val_pred_all = np.concatenate(all_val_pred)
                if len(val_true_all) > 1 and np.std(val_true_all) > 0 and np.std(val_pred_all) > 0:
                    epoch_val_pearson_all = float(np.corrcoef(val_true_all, val_pred_all)[0, 1])
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

        # Branch output variances — monitor whether each branch learns diverse representations
        epoch_lstm_var = lstm_var_sum / max(lstm_var_count, 1) if lstm_var_count > 0 else float("nan")
        epoch_expr_var = expr_var_sum / max(expr_var_count, 1) if expr_var_count > 0 else float("nan")

        # ── Post-validation: EMA, logging, early-stopping, checkpoint ──
        prev_train = train_losses[-1] if train_losses else float("nan")
        monitor_loss = avg_val_loss if not math.isnan(avg_val_loss) else prev_train
        val_loss_ema = compute_val_loss_ema(val_loss_ema, monitor_loss, ema_alpha)

        current_lr = optimizer.param_groups[0]["lr"]
        prev_train_str = f"Train={prev_train:.6f} " if train_losses else ""
        log_msg = (
            f"Epoch {epoch}/{epochs}: {prev_train_str}Val={avg_val_loss:.6f} "
            f"ValEMA={val_loss_ema:.6f} LR={current_lr:.3e}"
        )
        if not math.isnan(epoch_val_pearson_all):
            log_msg += f" | Pearson(All)={epoch_val_pearson_all:.4f}"
        if not math.isnan(epoch_val_pearson):
            log_msg += f" | Pearson(NZ)={epoch_val_pearson:.4f}"
        if not math.isnan(epoch_val_zero_acc):
            log_msg += f" | ZeroAcc={epoch_val_zero_acc:.4f}"
        if not math.isnan(epoch_lstm_var):
            log_msg += f" | LstmVar={epoch_lstm_var:.4f}"
        if not math.isnan(epoch_expr_var):
            log_msg += f" | ExprVar={epoch_expr_var:.4f}"
        if hasattr(model, "vae_encoder"):
            vae_total, vae_trainable_params = count_vae_parameters(model)
            log_msg += f" | VAE={'on' if vae_trainable else 'off'}({vae_trainable_params}/{vae_total})"
        print(log_msg)

        utils.append_epoch_log(
            log_file=log_file,
            epoch=epoch,
            train_loss=train_losses[-1] if train_losses else float("nan"),
            val_loss=avg_val_loss if not math.isnan(avg_val_loss) else np.nan,
            lr=current_lr,
            val_loss_ema=val_loss_ema,
            train_loss_zero=train_losses_zero[-1] if train_losses_zero else float("nan"),
            train_loss_nonzero=train_losses_nz[-1] if train_losses_nz else float("nan"),
            val_loss_zero=val_losses_zero[-1],
            val_loss_nonzero=val_losses_nz[-1],
            val_pearson_nonzero=epoch_val_pearson,
            val_pearson_all=epoch_val_pearson_all,
            val_zero_accuracy=epoch_val_zero_acc,
            lstm_var=epoch_lstm_var,
            expr_var=epoch_expr_var,
            vae_fine_tune_active=int(vae_trainable) if hasattr(model, "vae_encoder") else None,
            vae_trainable_params=count_vae_parameters(model)[1] if hasattr(model, "vae_encoder") else None,
            vae_total_params=count_vae_parameters(model)[0] if hasattr(model, "vae_encoder") else None,
            vae_fine_tune_start_epoch=vae_fine_tune_start_epoch if hasattr(model, "vae_encoder") else None,
        )

        # Update best metric and early-stopping counter (uses EMA-smoothed loss)
        earlystopping(val_loss_ema)
        if should_save_best_model(val_loss_ema, earlystopping.best_score):
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

        if save_every > 0 and (epoch % save_every == 0):
            utils.save_checkpoint(
                checkpoint_path=ckpt_dir / f"epoch_{epoch:04d}.ckpt",
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
                f"Early stopping at epoch {epoch}: "
                f"no improvement for {earlystopping.counter} epochs (patience={patience})."
            )
            break

        # ── Training ──
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

            optimizer.zero_grad(set_to_none=True)
            if loss_type == "zinb":
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    mu_ratio, theta, pi = model(promoters, exprs)
                mu_ratio = mu_ratio.squeeze(1)
                theta = theta.squeeze(1)
                pi = pi.squeeze(1)
                mu_ratio = mu_ratio.float()
                theta = theta.float()
                pi = pi.float()
                lib_size = exprs.sum(dim=1) + ys  # exprs has target masked to 0
                mu = mu_ratio * lib_size
                loss = zinb_loss_fn(ys, mu, theta, pi)
            else:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    out = model(promoters, exprs).squeeze(1)
                out = out.float()
                if loss_type == "pearson":
                    loss = pearson_loss(out, ys)
                elif loss_type == "combined":
                    loss = pearson_mse_loss(out, ys, nonzero_weight=nonzero_loss_weight, pearson_lambda=pearson_lambda)
                else:
                    loss = weighted_mse_loss(out, ys, nonzero_weight=nonzero_loss_weight)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # calculate weighted MSE components for each batch
            with torch.no_grad():
                if loss_type == "zinb":
                    y_pred_log = torch.log(mu_ratio * 1e6 + 1)
                    y_log = torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6)
                    sq = (y_pred_log - y_log) ** 2
                else:
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

        scheduler.step()

        if loss_type in ("pearson", "zinb"):
            avg_train_loss = sum(epoch_step_losses) / max(len(epoch_step_losses), 1)
        else:
            avg_train_loss = train_loss_num / max(train_loss_den, 1e-12)
        train_losses.append(avg_train_loss)
        train_losses_zero.append(train_zero_sum / max(train_zero_count, 1))
        train_losses_nz.append(train_nz_sum / max(train_nz_count, 1))

        utils.append_step_log(step_log_file, epoch_step_losses, epoch, global_step)
        global_step += len(epoch_step_losses)

    print(f"Training done. logs: {log_file} | checkpoints: {ckpt_dir}")
        

def main():
    parser = argparse.ArgumentParser(description="Train a gene expression model.")
    parser.add_argument("--exp_name", type=str, required=True, default='default', help="Name of the experiment (used for organizing outputs)")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameter config.json")
    parser.add_argument("--model", type=str, default="LSTMmodel", choices=sorted(MODEL_REGISTRY.keys()), help="Model architecture to use")
    parser.add_argument("--data", type=str, default="umi_processed", choices=["highquality", "processed", "log_processed", "umi_highquality", "umi_processed","umi_E-MTAB-10519-raw","umi_E-MTAB-10519-hqcells","umi_E-MTAB-10519-hqcells_aug20","umi_E-MTAB-10519-hqcells_aug15"], help="Which dataset version to use (affects data paths in config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--dryrun", action="store_true", default=False, help="Run dryrun_cpu before real training")
    parser.add_argument("--plot-loss", action="store_true", default=True, help="Plot training loss curve after training")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for LSTM and MLP in the model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (0 avoids extra memory copies)")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Dataloader prefetch factor used when num_workers > 0")
    parser.add_argument("--preencode-promoters", action="store_true", default=False, help="Pre-encode all promoter sequences in memory. Faster, but uses much more RAM.")
    parser.add_argument("--amp", action="store_true", default=False, help="Use CUDA automatic mixed precision for model forward/backward.")
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="Number of samples per epoch. 0 = auto-select from pool sizes (no forced duplication).")
    parser.add_argument("--val-samples", type=int, default=128000, help="Number of unique samples to use for validation (uses zeroNonZeroSampler)")
    parser.add_argument("--max-duplication", type=float, default=1.0, help="Max duplication factor for auto samples_per_epoch (1.0 = no duplication, 2.0 = up to 2x)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--nonzero-loss-weight", type=float, default=2.0, help="Weight multiplier for non-zero labels in MSE loss")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum loss improvement to reset patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility of validation sampling and training")
    parser.add_argument("--max-resume-snapshots", type=int, default=5, help="Max number of resume snapshots to keep (0 means keep all)")
    parser.add_argument("--nonzero-ratio", type=float, default=None, help="Target ratio of non-zero samples per epoch (uses ZeroNonZeroSampler). Default: natural ratio from data")
    parser.add_argument("--zero-acc-threshold", type=float, default=0.5, help="Threshold for zero-sample accuracy (|pred| < threshold counts as correct)")
    parser.add_argument("--ema-alpha", type=float, default=0.9, help="EMA smoothing factor for validation loss (0=use raw, higher=smoother)")
    parser.add_argument("--cell-ratio", type=float, default=1.0, help="Fraction of cells to randomly subsample (0-1). Useful for reducing memory usage with processed data + ZeroNonZeroSampler")
    parser.add_argument("--val-cell-ratio", type=float, default=0.5, help="Fraction of cells for the validation dataset (0-1). Default 1.0 (no subsampling) since val uses fewer samples.")
    parser.add_argument("--use-cell-split", action="store_true", default=False, help="Use cell_train.txt and cell_val.txt to restrict train/validation cells.")
    parser.add_argument("--cell-split-dir", type=str, default=None, help="Directory containing cell_train.txt/cell_val.txt. Default: data/<data>.")
    parser.add_argument("--loss", type=str, default="combined", choices=["mse", "pearson", "combined", "zinb"], help="Loss function: 'mse', 'pearson', 'combined', or 'zinb' (ZINB distribution loss)")
    parser.add_argument("--fusion", type=str, default="gate", choices=["concat", "gate"], help="Fusion method for combining LSTM and expression features")
    parser.add_argument("--pearson-lambda", type=float, default=10.0, help="Lambda weight for the Pearson term in combined loss (only used with --loss combined)")
    parser.add_argument("--vae-encoder", type=str, default=None, help="Path to scVI output dir (e.g., outputs/scvi_10/) containing encoder.pt and config.json")
    parser.add_argument("--vae-fine-tune", action="store_true", default=False, help="Unfreeze scVI encoder weights during training")
    parser.add_argument("--vae-fine-tune-start-epoch", type=int, default=-1, help="Epoch to start fine-tuning the VAE encoder. -1 keeps current freeze/unfreeze behavior.")
    args = parser.parse_args()

    # # 允许 cuDNN 自动寻找最适合当前配置的算法（提高速度）
    # torch.backends.cudnn.benchmark = True 
    # # 强制 cuDNN 使用确定性算法（确保复现，但可能会稍微降低速度）
    # torch.backends.cudnn.deterministic = True

    print("start")
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / args.data
    train_cell_ids = None
    val_cell_ids = None
    if args.use_cell_split:
        cell_split_dir = resolve_cell_split_dir(base_dir, args.data, args.cell_split_dir)
        train_cell_ids = read_cell_split(cell_split_dir, "train")
        val_cell_ids = read_cell_split(cell_split_dir, "val")
        print(
            f"Using cell split from {cell_split_dir}: "
            f"train_cells={len(train_cell_ids)} val_cells={len(val_cell_ids)}"
        )

    use_log1p = args.data.startswith("umi_") and args.loss != "zinb"
    train_dataset = MyDataset(
        promoter_file=data_dir / "promoter_train.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        cell_ids_subset=train_cell_ids,
        cell_ratio=args.cell_ratio,
        log1p_cpm_target=use_log1p,
        preencode_promoters=args.preencode_promoters,
    )
    val_dataset = MyDataset(
        promoter_file=data_dir / "promoter_val.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        mode="val",
        seed=args.seed,
        cell_ids_subset=val_cell_ids,
        cell_ratio=args.val_cell_ratio,
        log1p_cpm_target=use_log1p,
        preencode_promoters=args.preencode_promoters,
    )

    pin_memory = torch.cuda.is_available()
    loader_worker_kwargs = dataloader_worker_kwargs(args.num_workers, args.prefetch_factor)
    if args.data in ("processed", "log_processed", "umi_processed","umi_E-MTAB-10519-raw","umi_E-MTAB-10519-hqcells","umi_E-MTAB-10519-hqcells_aug20","umi_E-MTAB-10519-hqcells_aug15"):
        if args.nonzero_ratio is not None:
            train_sampler = utils.ZeroNonZeroSampler(
                train_dataset,
                nonzero_ratio=args.nonzero_ratio,
                samples_per_epoch=args.samples_per_epoch,
                seed=args.seed,
                max_duplication=args.max_duplication,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                **loader_worker_kwargs,
            )
            val_sampler = utils.ZeroNonZeroSampler(
                val_dataset,
                nonzero_ratio=args.nonzero_ratio,
                samples_per_epoch=args.val_samples,
                seed=args.seed,
                max_duplication=args.max_duplication,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                **loader_worker_kwargs,
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
                drop_last=True,
                **loader_worker_kwargs,
            )
            val_sampler = utils.BalancedEpochSubsetSampler(
                val_dataset,
                samples_per_epoch=args.val_samples,
                seed=args.seed,            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                **loader_worker_kwargs,
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
                drop_last=True,
                **loader_worker_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                **loader_worker_kwargs,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            **loader_worker_kwargs,
        )

    expr_dim = train_dataset.X.shape[1]
    if args.vae_encoder is not None and not Path(args.vae_encoder).exists():
        raise FileNotFoundError(f"VAE encoder dir not found: {args.vae_encoder}")
    output_mode = "zinb" if args.loss == "zinb" else "scalar"
    model = build_model(
        args.model,
        expr_dim=expr_dim,
        hidden_size=args.hidden_size,
        use_vae=args.vae_encoder is not None,
        vae_encoder_path=args.vae_encoder,
        vae_fine_tune=args.vae_fine_tune,
        output_mode=output_mode,
        fusion=args.fusion,
    )
    print_training_resource_summary(
        model=model,
        batch_size=args.batch_size,
        expr_dim=expr_dim,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        cell_ratio=args.cell_ratio,
        val_cell_ratio=args.val_cell_ratio,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        amp=args.amp,
    )

    run_dir, ckpt_dir, plots_dir, _ = utils._prepare_output_dirs(base_dir, args.exp_name)
    has_model_backup = any(run_dir.glob("model_arch_*.py"))
    if not has_model_backup:
        utils.backup_model_architecture(run_dir, args.model)

    if args.dryrun:
        utils.dryrun_cpu(model, train_loader, steps=50, learning_rate=1e-4, save_path=plots_dir / "dryrun.png")
        # dryrun会改参数，重新初始化模型再正式训练
        model = build_model(args.model, expr_dim=expr_dim, hidden_size=args.hidden_size,
                            use_vae=args.vae_encoder is not None, vae_encoder_path=args.vae_encoder,
                            vae_fine_tune=args.vae_fine_tune, output_mode=output_mode,
                            fusion=args.fusion)

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
        amp=args.amp,
        vae_fine_tune_start_epoch=args.vae_fine_tune_start_epoch,
        force_vae_fine_tune=args.vae_fine_tune,
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
            utils.plot_pred_scatter(model, val_loader, is_umi=use_log1p, epoch=1, save_path=plots_dir / "pred_vs_true_scatter.png")
            utils.plot_per_promoter_scatter(model, val_dataset, is_umi=use_log1p, n_promoters=3, save_path=plots_dir / "per_promoter_scatter.png")
            utils.plot_per_cell_scatter(model, val_dataset, is_umi=use_log1p, n_cells=3, save_path=plots_dir / "per_cell_scatter.png")
        else:
            print(f"Log file not found for plotting: {log_file}")
# %%       
if __name__ == "__main__":
    main()

# %%


