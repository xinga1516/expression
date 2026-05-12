from pathlib import Path
from datetime import datetime
import json
import csv
import shutil
import torch
from safetensors.torch import save_file
import argparse
import sys
import subprocess
from torch import nn
from torch.utils.data import Sampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.earlystopping import EarlyStopping
from safetensors.torch import save_file, load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BalancedEpochSubsetSampler(Sampler):
    '''Yield a fixed number of unique (promoter, cell) pairs per epoch without building a full permutation.
    This sampler ensures that each promoter is sampled approximately equally in each epoch, 
    while shuffling the order of promoters and cells.'''

    def __init__(self, dataset, samples_per_epoch: int, seed: int = 42):
        self.dataset = dataset
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        self.P = int(dataset.P)
        self.C = int(dataset.C)
        self.total_len = self.P * self.C
        if self.total_len <= 0:
            raise ValueError("Dataset is empty.")
        self.samples_per_epoch = min(self.samples_per_epoch, self.total_len)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        promoter_order = rng.permutation(self.P)
        cell_perm = rng.permutation(self.C)
        offsets = rng.integers(0, self.C, size=self.P, dtype=np.int64)

        base = self.samples_per_epoch // self.P
        rem = self.samples_per_epoch % self.P

        for rank, pro_i in enumerate(promoter_order):
            k = base + (1 if rank < rem else 0) # allocate the remainder samples to the first 'rem' promoters
            if k <= 0:
                continue

            offset = int(offsets[rank])
            end = offset + k
            if end <= self.C:
                cell_ids = cell_perm[offset:end]
            else:
                wrap = end - self.C
                cell_ids = np.concatenate((cell_perm[offset:], cell_perm[:wrap]))

            base_idx = int(pro_i) * self.C
            for cell_i in cell_ids:
                yield base_idx + int(cell_i)

class ZeroNonZeroSampler(Sampler):
    '''Sample (promoter, cell) pairs with a controllable ratio of zero vs non-zero expression targets.'''

    def __init__(self, dataset, nonzero_ratio=0.5, samples_per_epoch=None, seed=42):
        self.dataset = dataset
        self.nonzero_ratio = nonzero_ratio
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else int(dataset.P * dataset.C)
        self.seed = int(seed)
        self.epoch = 0
        self.P = int(dataset.P)
        self.C = int(dataset.C)
        self.total_len = self.P * self.C
        if self.total_len <= 0:
            raise ValueError("Dataset is empty.")
        self.samples_per_epoch = min(self.samples_per_epoch, self.total_len)

        print("Precomputing zero/non-zero pools for sampler...")
        X_csc = dataset.X.tocsc()
        cells_set = set(int(c) for c in dataset.cells)
        cell_row_to_pos = {int(dataset.cells[pos]): pos for pos in range(self.C)}
        nz_indices = [] # store indices of non-zero samples in the flattened (promoter, cell) space
        zero_counts = np.empty(self.P, dtype=np.int32)
        for pro_i in range(self.P):
            col = int(dataset.promoter2expr_idx[pro_i])
            col_vec = X_csc[:, col].tocoo()
            nz_rows = {int(r) for r in col_vec.row if int(r) in cells_set}
            base = pro_i * self.C
            for cell_row in nz_rows:
                nz_indices.append(base + cell_row_to_pos[cell_row])
            zero_counts[pro_i] = self.C - len(nz_rows)

        self.nz_indices = np.array(nz_indices, dtype=np.int64)
        nonzero_cnt = len(self.nz_indices)
        zero_cnt = self.total_len - nonzero_cnt
        print(f"  Non-zero: {nonzero_cnt}, Zero: {zero_cnt}, Total: {self.total_len}")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n_nz = int(self.samples_per_epoch * self.nonzero_ratio)
        n_z = self.samples_per_epoch - n_nz

        indices = []

        if n_nz > 0 and len(self.nz_indices) > 0:
            nz_sample = self.nz_indices[rng.integers(0, len(self.nz_indices), size=n_nz)]
            indices.append(nz_sample)

        if n_z > 0:
            nz_set = set(self.nz_indices.tolist())
            zero_sample = []
            while len(zero_sample) < n_z:
                needed = n_z - len(zero_sample)
                cand = rng.integers(0, self.total_len, size=needed * 2)
                for c in cand.tolist():
                    if c not in nz_set:
                        zero_sample.append(c)
                        if len(zero_sample) >= n_z:
                            break
            indices.append(np.array(zero_sample[:n_z], dtype=np.int64))

        all_idx = np.concatenate(indices)
        rng.shuffle(all_idx)
        yield from all_idx.tolist()


def get_git_hash():
    try:
        # 执行 git 命令获取当前的 commit id
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "Not a git repository"

def _prepare_output_dirs(base_dir: Path, exp_name: str):
    '''Prepare experiment-specific output directories under base_dir/outputs/exp_name.'''
    run_dir = base_dir / "outputs" / exp_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, plot_dir, log_dir


def save_run_config(config_path: Path, args: argparse.Namespace, base_dir: Path, expr_dim: int, resume_path: str):
    '''Save run hyperparameters/config to JSON in the experiment output folder.
    Returns previous run config (if exists) and the new config.
    '''
    previous_cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            previous_cfg = json.load(f)

    cfg = {}

    if args.config is not None:
        src_cfg = Path(args.config)
        if src_cfg.exists():
            with open(src_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            print(f"[Config] file not found, fallback to CLI args only: {src_cfg}")

    cfg.update({
        "exp_name": args.exp_name,
        "resume": resume_path,
        "dryrun": args.dryrun,
        "plot_loss": args.plot_loss,
        "model": args.model,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "samples_per_epoch": args.samples_per_epoch,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "nonzero_loss_weight": args.nonzero_loss_weight,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "train_promoter_file": str(base_dir / "data" / args.data / "promoter_train.csv"),
        "val_promoter_file": str(base_dir / "data" / args.data / "promoter_val.csv"),
        "scrna_file": str(base_dir / "data" / args.data / "integrated_data.h5ad"),
        "expr_dim": int(expr_dim),
        "git_hash": get_git_hash(),
        "ema_alpha": getattr(args, "ema_alpha", 0.9),
        "cell_ratio": getattr(args, "cell_ratio", 1.0),
        "loss_type": getattr(args, "loss", "mse"),
        "pearson_lambda": getattr(args, "pearson_lambda", 1.0),
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[Config] saved: {config_path}")
    return previous_cfg, cfg


def append_resume_config_history(log_path: Path, resume_ckpt: str, previous_cfg: dict, current_cfg: dict):
    '''Append timestamp and changed hyperparameters for resumed training runs.'''
    changed = {}
    all_keys = set(previous_cfg.keys()) | set(current_cfg.keys())
    for k in sorted(all_keys):
        old_v = previous_cfg.get(k, None)
        new_v = current_cfg.get(k, None)
        if old_v != new_v:
            changed[k] = {"old": old_v, "new": new_v}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] resume_ckpt={resume_ckpt}\n")
        if changed:
            f.write("changed_params:\n")
            for k, v in changed.items():
                f.write(f"  - {k}: {v['old']} -> {v['new']}\n")
        else:
            f.write("changed_params: none\n")
        f.write("\n")

    print(f"[Config] resume history appended: {log_path}")


def save_resume_snapshot(ckpt_dir: Path, resume_ckpt: Path, max_keep: int = 5):
    '''Save a timestamped snapshot before resumed training and keep only the latest max_keep snapshots.'''
    snapshot_dir = ckpt_dir / "resume_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"resume_{timestamp}_{resume_ckpt.stem}.ckpt"
    snapshot_path = snapshot_dir / snapshot_name
    shutil.copy2(resume_ckpt, snapshot_path)
    print(f"[Resume] snapshot saved: {snapshot_path}")

    if max_keep > 0:
        snapshots = sorted(snapshot_dir.glob("resume_*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_path in snapshots[max_keep:]:
            old_path.unlink(missing_ok=True)
        if len(snapshots) > max_keep:
            print(f"[Resume] pruned {len(snapshots) - max_keep} old snapshots (keep={max_keep})")


def backup_model_architecture(run_dir: Path, model_name: str):
    '''Backup model architecture source code to the experiment output directory.'''
    src_model_path = PROJECT_ROOT / "src" / "model.py"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = run_dir / f"model_arch_{model_name}_{timestamp}.py"
    shutil.copy2(src_model_path, backup_path)
    print(f"[Model] architecture backup saved: {backup_path}")


def append_epoch_log(log_file: Path, epoch: int, train_loss: float, val_loss: float, lr: float, **extra):
    '''
    Append a row of training log for the given epoch.
    Extra keyword arguments are added as additional columns.
    If the log file does not exist, create it and write the header first.
    '''
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]
    extra_keys = sorted(k for k in extra if extra[k] is not None)
    all_keys = fieldnames + extra_keys

    write_header = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        if write_header:
            writer.writeheader()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": f"{lr:.8e}"}
        for k in extra_keys:
            row[k] = extra[k]
        writer.writerow(row)


def append_step_log(step_log_file: Path, step_losses: list, epoch: int, start_step: int):
    '''Append per-step train losses for an entire epoch to a CSV file in one write.'''
    write_header = not step_log_file.exists()
    with open(step_log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", "epoch", "train_loss"])
        if write_header:
            writer.writeheader()
        for i, loss in enumerate(step_losses):
            writer.writerow({"global_step": start_step + i, "epoch": epoch, "train_loss": f"{loss:.8f}"})


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    earlystopping: EarlyStopping,
    train_losses: list,
    val_losses: list,
):
    '''Save the training state to a checkpoint file.'''
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "earlystopping": earlystopping.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    tmp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.rename(checkpoint_path)  # atomic rename to avoid partial writes


def resume_from_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    earlystopping: EarlyStopping,
    device: torch.device,
):
    '''Load the training state from a checkpoint file and move optimizer state to device.'''
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    # Move all tensors in the optimizer state to the specified device.
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    earlystopping.load_state_dict(ckpt.get("earlystopping", {}))
    train_losses = list(ckpt.get("train_losses", []))
    val_losses = list(ckpt.get("val_losses", []))
    return start_epoch, earlystopping, train_losses, val_losses

def robust_save_model(model: nn.Module, save_path: Path):
    temp_path = save_path.with_suffix(".tmp")
    try:
        save_file(model.state_dict(), temp_path)
        # If save is successful, rename the temp file to the final path
        temp_path.rename(save_path)
    except Exception as e:
        # If save fails, remove the temp file and raise the error
        if temp_path.exists():
            temp_path.unlink()
        raise e
    
def dryrun_cpu(model,train_loader,steps=50,learning_rate=1e-4,save_path: Path | None = None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    # test if dataset and dataloader work well 
    batch = next(iter(train_loader))
    promoters, exprs, ys = batch  # (batch, 400, 5), (batch, 16300), (batch,)
    print(promoters.shape, exprs.shape, ys.shape)
    ys = ys.float()
    #记录一个 LSTM 参数训练前的值
    params_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }
    
    losses = []
    for step in range(steps):
        batch = next(iter(train_loader))
        promoters, exprs, ys = batch
        ys = ys.float()
    
        optimizer.zero_grad()
        out = model(promoters, exprs).squeeze(1)
        loss = criterion(out, ys)
    
        loss.backward()
        optimizer.step()
    
        losses.append(loss.item())
    
    # parameters and gradients after training
    for name, p in model.named_parameters():
        diff = torch.norm(p.detach() - params_before[name]).item()
        print(f"{name:30s} param_diff = {diff:.6e}")
        print(name, p.grad is None, p.grad.norm().item())
    # loss 曲线
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Dry-run loss (same batch)")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to: {save_path}")
    plt.close()
        

def count_zero_nonzero(data_loader):
    '''Count zero and non-zero targets in a data loader and print summary.'''
    zero_count = 0
    nz_count = 0
    all_y = []
    for batch in data_loader:
        _, _, ys = batch
        zero_count += (ys == 0).sum().item()
        nz_count += (ys != 0).sum().item()
        all_y.append(ys.cpu().numpy())
    y_all = np.concatenate(all_y)
    total = zero_count + nz_count
    frac_zero = zero_count / max(total, 1)
    print(f"[Data] Validation samples: zero={zero_count}, non-zero={nz_count}, "
          f"zero_frac={frac_zero:.4f}, min={y_all.min():.6f}, max={y_all.max():.6f}, "
          f"mean={y_all.mean():.6f}")
    return zero_count, nz_count


def plot_pred_scatter(model, data_loader, epoch=1, save_path: Path | None = None):
    '''Plot scatter of true vs predicted values from the model on the given data loader.
    compute Pearson correlation and show it in the title. Only use up to max_steps batches for plotting.
    highquality data with epoch = 1;
    processed data with epoch >=2 recommanded;
    '''
    device = next(model.parameters()).device
    model.eval()

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for ep in range(epoch):
            for batch in data_loader:
                promoters, exprs, ys = batch
                promoters = promoters.to(device, non_blocking=True)
                exprs = exprs.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True).float()

                preds = model(promoters, exprs).squeeze(1)
                y_true_all.append(ys.detach().cpu().numpy())
                y_pred_all.append(preds.detach().cpu().numpy())

    if not y_true_all:
        print("No samples collected for scatter plot.")
        return

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")

    print(f"Validation Pearson correlation: {corr:.6f}")

    zero_mask = y_true == 0
    nz_mask = ~zero_mask

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: all points
    ax1.scatter(y_true, y_pred, s=3, alpha=0.15)
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    ax1.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    ax1.set_xlabel("True")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"All samples (Pearson={corr:.4f})")

    # Right: non-zero samples only
    if nz_mask.any():
        nz_true = y_true[nz_mask]
        nz_pred = y_pred[nz_mask]
        ax2.scatter(nz_true, nz_pred, s=8, alpha=0.5, color="steelblue")
        nz_min = float(min(np.min(nz_true), np.min(nz_pred)))
        nz_max = float(max(np.max(nz_true), np.max(nz_pred)))
        ax2.plot([nz_min, nz_max], [nz_min, nz_max], "r--", linewidth=1)
        ax2.set_xlabel("True")
        ax2.set_ylabel("Predicted")
        if len(nz_true) > 1 and np.std(nz_true) > 0 and np.std(nz_pred) > 0:
            nz_corr = float(np.corrcoef(nz_true, nz_pred)[0, 1])
        else:
            nz_corr = float("nan")
        ax2.set_title(f"Non-zero only (n={nz_mask.sum()}, Pearson={nz_corr:.4f})")
    else:
        ax2.text(0.5, 0.5, "No non-zero samples", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Non-zero samples")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to: {save_path}")
    plt.close()

def plot_loss_curves_from_logfile(log_file: Path, save_path: Path | None = None, step_log_file: Path | None = None):
    '''Plot train loss by global_step (from step_train_loss.csv) and val loss by epoch.'''
    df_epoch = pd.read_csv(log_file)
    #df_epoch = df_epoch[1:]  # skip epoch 0 which may have very different scale due to resume
    # using max loss for normalization to keep the curve shape visible, especially when resumed training may have different absolute loss values
    loss_max = max(df_epoch["train_loss"].max(), df_epoch["val_loss"].max(), 1e-8)
    df_epoch["train_loss"] /= loss_max
    df_epoch["val_loss"] /= loss_max
    df_epoch["val_loss_ema"] /= loss_max

    if step_log_file is None:
        step_log_file = log_file.parent / "step_train_loss.csv"

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # ---- train loss by global_step ----
    if step_log_file.exists():
        df_step = pd.read_csv(step_log_file)
        #df_step = df_step[df_step["epoch"] > 1]  # skip epoch 0 which may have different scale due to resume
        loss_max = max(df_step["train_loss"].max(), loss_max, 1e-8)
        # Map each epoch to its midpoint global_step for val loss alignment
        step_bounds = df_step.groupby("epoch")["global_step"].agg(["min", "max"])
        # Normalize train loss
        train_max = df_step["train_loss"].max()
        df_step["train_loss"] /= loss_max
        ax1.plot(df_step["global_step"], df_step["train_loss"], alpha=0.3, linewidth=0.5, color="steelblue", label="Train Loss (step)")
        # Smoothed train loss: moving average over 500 steps
        if len(df_step) > 500:
            df_step["smooth"] = df_step["train_loss"].rolling(500, min_periods=1).mean()
            ax1.plot(df_step["global_step"], df_step["smooth"], linewidth=1.5, color="steelblue", label="Train Loss (smoothed)")
        ax1.set_ylabel("Train Loss", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_xlabel("Global Step")
    else:
        train_max = df_epoch["train_loss"].max()

    # ---- val loss by epoch ----
    val_color = "darkorange"
    if step_log_file.exists():
        # Map val_loss to each epoch's last step (val is computed at epoch end)
        endpoints = step_bounds["max"].values
        steps_last_epoch = 0 if df_step['epoch'].iloc[-1] == df_epoch['epoch'].iloc[-1] else len(df_step) - step_bounds["max"].iloc[-1]
        if steps_last_epoch:
            endpoints = endpoints[:-1]  # drop last point if last epoch is incomplete
        ax1.scatter(endpoints, df_epoch["val_loss"], color=val_color, s=30, zorder=5, label="Val Loss")
        ax1.plot(endpoints, df_epoch["val_loss"], color=val_color, linewidth=1, alpha=0.6)
    else:
        ax1.plot(df_epoch["epoch"], df_epoch["val_loss"], color=val_color, marker="o", linewidth=1.5, label="Val Loss")
        ax1.set_xlabel("Epoch")
    if "val_loss_ema" in df_epoch.columns:
        if step_log_file.exists():
            ax1.plot(endpoints, df_epoch["val_loss_ema"], color="red", linewidth=1.5, linestyle="--", label="Val Loss (EMA)")
        else:
            ax1.plot(df_epoch["epoch"], df_epoch["val_loss_ema"], color="red", linewidth=1.5, linestyle="--", label="Val Loss (EMA)")
    ax1.legend(loc="upper right")
    ax1.tick_params(axis="y", labelcolor=val_color)

    plt.title("Loss Curve (Train by step, Validation by epoch)")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Loss curve saved to: {save_path}")
    plt.close()


def plot_zero_nonzero_loss_curves(log_file: Path, save_path: Path | None = None):
    '''Plot separate loss curves for zero and non-zero samples from the extended log.'''
    df = pd.read_csv(log_file)
    df = df[1:] if len(df) > 1 else df
    required = {"train_loss_zero", "train_loss_nonzero", "val_loss_zero", "val_loss_nonzero"}
    if not required.issubset(df.columns):
        print("  Skipping zero/non-zero loss plot: columns not found in log.")
        return

    nz_max = max(df["train_loss_nonzero"].max(), df["val_loss_nonzero"].max(), 1e-8)
    z_max = max(df["train_loss_zero"].max(), df["val_loss_zero"].max(), 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["epoch"], df["train_loss_nonzero"] / nz_max, label="Train NonZero")
    ax1.plot(df["epoch"], df["val_loss_nonzero"] / nz_max, label="Val NonZero")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Normalized Loss")
    ax1.set_title("Non-Zero Sample Loss")
    ax1.legend()

    ax2.plot(df["epoch"], df["train_loss_zero"] / z_max, label="Train Zero")
    ax2.plot(df["epoch"], df["val_loss_zero"] / z_max, label="Val Zero")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Normalized Loss")
    ax2.set_title("Zero Sample Loss")
    ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Zero/non-zero loss curves saved to: {save_path}")
    plt.close()


def plot_val_metrics(log_file: Path, save_path: Path | None = None):
    '''Plot validation Pearson (non-zero) and accuracy (zero) over epochs.'''
    df = pd.read_csv(log_file)
    df = df[1:] if len(df) > 1 else df

    has_pearson = "val_pearson_nonzero" in df.columns
    has_acc = "val_zero_accuracy" in df.columns

    if not has_pearson and not has_acc:
        print("  Skipping metrics plot: no val_pearson_nonzero or val_zero_accuracy columns.")
        return

    n_plots = (has_pearson + has_acc)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    if has_pearson:
        axes[plot_idx].plot(df["epoch"], df["val_pearson_nonzero"], "o-", markersize=4)
        axes[plot_idx].axhline(0, color="gray", linestyle="--", linewidth=0.5)
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Pearson r")
        axes[plot_idx].set_title("Non-Zero Sample Pearson Correlation")
        plot_idx += 1

    if has_acc:
        axes[plot_idx].plot(df["epoch"], df["val_zero_accuracy"], "s-", markersize=4)
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Accuracy")
        axes[plot_idx].set_ylim(0, 1)
        axes[plot_idx].set_title("Zero Sample Accuracy ($|\\hat{y}| < \\epsilon$)")
        plot_idx += 1

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Validation metrics saved to: {save_path}")
    plt.close()