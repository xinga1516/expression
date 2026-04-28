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
    '''Yield a fixed number of unique (promoter, cell) pairs per epoch without building a full permutation.'''

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


def append_epoch_log(log_file: Path, epoch: int, train_loss: float, val_loss: float, lr: float):
    '''
    Append a row of training log for the given epoch. 
    If the log file does not exist, create it and write the header first.
    '''
    write_header = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        writer.writerow([epoch, train_loss, val_loss, f"{lr:.8e}"])


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
        

def plot_pred_scatter(model, data_loader, max_steps=20000, save_path: Path | None = None):
    '''Plot scatter of true vs predicted values from the model on the given data loader.
    compute Pearson correlation and show it in the title. Only use up to max_steps batches for plotting.
    '''
    device = next(model.parameters()).device
    model.eval()

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            promoters, exprs, ys = batch
            promoters = promoters.to(device, non_blocking=True)
            exprs = exprs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True).float()

            preds = model(promoters, exprs).squeeze(1)
            y_true_all.append(ys.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())

            if step + 1 >= max_steps:
                break

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

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.35)
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted (Pearson={corr:.4f})")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to: {save_path}")
    plt.close()

def plot_loss_curves_from_logfile(log_file: Path, save_path: Path | None = None):
    df = pd.read_csv(log_file)
    df = df[1:] # skip epoch 0 which may have very different scale due to resume
    df['train_loss'] = df['train_loss']/ max(df['train_loss'].max(), 1e-8)
    df['val_loss'] = df['val_loss']/ max(df['train_loss'].max(), 1e-8)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Loss curve saved to: {save_path}")
    plt.close()