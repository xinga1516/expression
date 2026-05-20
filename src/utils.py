from pathlib import Path
from typing import Any, Iterator, Optional
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
from torch.utils.data import DataLoader, Sampler
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

    def __init__(self, dataset: Any, samples_per_epoch: int, seed: int = 42) -> None:
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
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
    '''Sample (promoter, cell) pairs with a controllable ratio of zero vs non-zero expression targets.

    Parameters
    ----------
    replace : bool
        If False, non-zero indices are sampled without replacement (raises ValueError
        when requested n_nz exceeds the available pool). Zero-side always samples
        without replacement internally.'''

    def __init__(self, dataset: Any, nonzero_ratio: float = 0.5, samples_per_epoch: Optional[int] = None, seed: int = 42, replace: bool = True, max_duplication: float = 1.0) -> None:
        self.dataset = dataset
        self.nonzero_ratio = nonzero_ratio
        self.seed = int(seed)
        self.replace = replace
        self.max_duplication = max_duplication
        self.epoch = 0
        self.P = int(dataset.P)
        # 0 or None → auto-select after pool sizes are known
        self._auto_samples = (samples_per_epoch is None or int(samples_per_epoch) == 0)
        if not self._auto_samples:
            self.samples_per_epoch = int(samples_per_epoch)
        else:
            self.samples_per_epoch = 0  # placeholder, set in _precompute_pools
        self._precompute_pools(dataset)

    def _precompute_pools(self, dataset: Any) -> None:
        '''(Re)build nz_indices from the current dataset state.'''
        self.C = int(dataset.C)
        self.total_len = self.P * self.C
        if self.total_len <= 0:
            raise ValueError("Dataset is empty.")

        print("Precomputing zero/non-zero pools for sampler...")
        X_csr = dataset.X.tocsr()
        gene_idx_to_pro = {}
        for pro_i in range(self.P):
            gene_idx_to_pro[int(dataset.promoter2expr_idx[pro_i])] = pro_i

        nz_list: list[int] = []  # flat indices of non-zero (promoter, cell) pairs
        for cell_pos in range(self.C):
            cell_row = int(dataset.cells[cell_pos])
            row = X_csr[cell_row]
            if hasattr(row, "indices"):
                nz_gene_idx = row.indices  # sparse: column indices = non-zero gene positions
            else:
                nz_gene_idx = np.where(np.asarray(row).ravel() > 0)[0]
            for gene_idx in nz_gene_idx:
                pro_i = gene_idx_to_pro.get(int(gene_idx))
                if pro_i is not None:
                    nz_list.append(pro_i * self.C + cell_pos)
            if cell_pos % 5000 == 0:
                print(f"  processed cell {cell_pos}/{self.C}")

        self.nz_indices = np.array(nz_list, dtype=np.int64)
        nz_pool = len(self.nz_indices)
        zero_pool = self.total_len - nz_pool
        print(f"  Non-zero: {nz_pool}, Zero: {zero_pool}, Total: {self.total_len}")

        # Auto-select samples_per_epoch from pool sizes + nonzero_ratio
        if self._auto_samples or not hasattr(self, "samples_per_epoch") or self.samples_per_epoch == 0:
            max_nz = max(nz_pool, 1)
            max_z = max(zero_pool, 1)
            cap_nz = max_nz / self.nonzero_ratio if self.nonzero_ratio > 0 else float("inf")
            cap_z = max_z / (1.0 - self.nonzero_ratio) if self.nonzero_ratio < 1.0 else float("inf")
            max_unique = int(min(cap_nz, cap_z, self.total_len))
            self.samples_per_epoch = int(max_unique * self.max_duplication)
            print(f"[ZeroNonZeroSampler] Auto samples_per_epoch = {self.samples_per_epoch} "
                  f"(max_unique={max_unique}, nz_pool={nz_pool}, zero_pool={zero_pool}, "
                  f"max_duplication={self.max_duplication})")
        else:
            self.samples_per_epoch = min(self.samples_per_epoch, self.total_len)

    def rebuild(self, dataset: Any) -> None:
        '''Rebuild index pools after the dataset has been resampled (e.g. cell rotation).'''
        self.dataset = dataset
        self.P = int(dataset.P)
        self._precompute_pools(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n_nz = int(self.samples_per_epoch * self.nonzero_ratio)
        n_z = self.samples_per_epoch - n_nz

        indices = []

        if n_nz > 0 and len(self.nz_indices) > 0:
            if n_nz > len(self.nz_indices):
                if not self.replace:
                    raise ValueError(
                        f"Requested {n_nz} non-zero samples but pool only has "
                        f"{len(self.nz_indices)}. Set replace=True or reduce "
                        f"samples_per_epoch / nonzero_ratio."
                    )
                print(f"[ZeroNonZeroSampler] WARNING: n_nz({n_nz}) > "
                      f"pool({len(self.nz_indices)}), sampling with replacement — "
                      f"{n_nz - len(self.nz_indices)} duplicates guaranteed.")
            nz_sample = rng.choice(self.nz_indices, size=n_nz, replace=self.replace)
            indices.append(nz_sample)

        if n_z > 0:
            nz_set = set(self.nz_indices.tolist())
            zero_sample: list[int] = []
            zero_seen: set[int] = set()  # internal dedup
            while len(zero_sample) < n_z:
                needed = n_z - len(zero_sample)
                cand = rng.integers(0, self.total_len, size=needed * 2)
                for c in cand.tolist():
                    if c not in nz_set and c not in zero_seen:
                        zero_sample.append(c)
                        zero_seen.add(c)
                        if len(zero_sample) >= n_z:
                            break
            indices.append(np.array(zero_sample, dtype=np.int64))

        all_idx = np.concatenate(indices)
        rng.shuffle(all_idx)
        yield from all_idx.tolist()


def get_git_hash() -> str:
    try:
        # 执行 git 命令获取当前的 commit id
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "Not a git repository"

def _prepare_output_dirs(base_dir: Path, exp_name: str) -> tuple[Path, Path, Path, Path]:
    '''Prepare experiment-specific output directories under base_dir/outputs/exp_name.'''
    run_dir = base_dir / "outputs" / exp_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, plot_dir, log_dir


def save_run_config(config_path: Path, args: argparse.Namespace, base_dir: Path, expr_dim: int, resume_path: str) -> tuple[dict, dict]:
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
        "val_cell_ratio": getattr(args, "val_cell_ratio", 1.0),
        "max_duplication": getattr(args, "max_duplication", 1.0),
        "loss_type": getattr(args, "loss", "mse"),
        "pearson_lambda": getattr(args, "pearson_lambda", 1.0),
        "use_vae": getattr(args, "vae_encoder", None) is not None,
        "vae_encoder_path": getattr(args, "vae_encoder", None),
        "vae_fine_tune": getattr(args, "vae_fine_tune", False),
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[Config] saved: {config_path}")
    return previous_cfg, cfg


def append_resume_config_history(log_path: Path, resume_ckpt: str, previous_cfg: dict, current_cfg: dict) -> None:
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


def save_resume_snapshot(ckpt_dir: Path, resume_ckpt: Path, max_keep: int = 5) -> None:
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


def backup_model_architecture(run_dir: Path, model_name: str) -> None:
    '''Backup model architecture source code to the experiment output directory.'''
    src_model_path = PROJECT_ROOT / "src" / "model.py"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = run_dir / f"model_arch_{model_name}_{timestamp}.py"
    shutil.copy2(src_model_path, backup_path)
    print(f"[Model] architecture backup saved: {backup_path}")


def append_epoch_log(log_file: Path, epoch: int, train_loss: float, val_loss: float, lr: float, **extra: Any) -> None:
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


def append_step_log(step_log_file: Path, step_losses: list[float], epoch: int, start_step: int) -> None:
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
    train_losses: list[float],
    val_losses: list[float],
) -> None:
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
) -> tuple[int, EarlyStopping, list[float], list[float]]:
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

def robust_save_model(model: nn.Module, save_path: Path) -> None:
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
    
def dryrun_cpu(model: nn.Module, train_loader: DataLoader, steps: int = 50, learning_rate: float = 1e-4, save_path: Path | None = None) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    output_mode = getattr(model, "output_mode", "scalar")

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
        out_raw = model(promoters, exprs)
        if output_mode == "zinb":
            mu_ratio, _theta, _pi = out_raw
            out = mu_ratio.squeeze(1)
        else:
            out = out_raw.squeeze(1)
        loss = criterion(out, ys)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    # parameters and gradients after training
    for name, p in model.named_parameters():
        diff = torch.norm(p.detach() - params_before[name]).item()
        grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"{name:30s} param_diff = {diff:.6e}")
        print(name, p.grad is None, grad_norm)
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
        

def count_zero_nonzero(data_loader: DataLoader) -> tuple[int, int]:
    '''Count zero and non-zero targets in a data loader and print summary.'''
    zero_count = 0
    nz_count = 0
    y_sum = 0.0
    y_min = float("inf")
    y_max = float("-inf")
    for batch in data_loader:
        _, _, ys = batch
        zero_count += (ys == 0).sum().item()
        nz_count += (ys != 0).sum().item()
        y_sum += ys.sum().item()
        y_min = min(y_min, ys.min().item())
        y_max = max(y_max, ys.max().item())
    total = zero_count + nz_count
    frac_zero = zero_count / max(total, 1)
    y_mean = y_sum / max(total, 1)
    print(f"[Data] Validation samples: zero={zero_count}, non-zero={nz_count}, "
          f"zero_frac={frac_zero:.4f}, min={y_min:.6f}, max={y_max:.6f}, "
          f"mean={y_mean:.6f}")
    return zero_count, nz_count


def plot_pred_scatter(model: nn.Module, data_loader: DataLoader, epoch: int = 1, save_path: Path | None = None) -> None:
    '''Plot scatter of true vs predicted values from the model on the given data loader.
    compute Pearson correlation and show it in the title. Only use up to max_steps batches for plotting.
    highquality data with epoch = 1;
    processed data with epoch >=2 recommanded;
    '''
    device = next(model.parameters()).device
    model.eval()

    is_zinb = getattr(model, "output_mode", "scalar") == "zinb"

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for ep in range(epoch):
            for batch in data_loader:
                promoters, exprs, ys = batch
                promoters = promoters.to(device, non_blocking=True)
                exprs = exprs.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True).float()

                if is_zinb:
                    mu_ratio, _theta, _pi = model(promoters, exprs)
                    mu_ratio = mu_ratio.squeeze(1)
                    lib_size = exprs.sum(dim=1) + ys
                    y_pred_log = torch.log(mu_ratio * 10000 + 1)
                    y_log = torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6)
                    y_true_all.append(y_log.detach().cpu().numpy())
                    y_pred_all.append(y_pred_log.detach().cpu().numpy())
                else:
                    preds = model(promoters, exprs).squeeze(1)
                    y_true_all.append(ys.detach().cpu().numpy())
                    y_pred_all.append(preds.detach().cpu().numpy())

    if not y_true_all:
        print("No samples collected for scatter plot.")
        return

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        spearman = float("nan")

    print(f"Validation Pearson: {pearson:.6f}, Spearman: {spearman:.6f}")

    zero_mask = y_true == 0
    nz_mask = ~zero_mask

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: all points — use hexbin for density-aware coloring
    hb1 = ax1.hexbin(y_true, y_pred, gridsize=80, cmap="viridis", mincnt=1,
                     bins="log", linewidths=0)
    diag_min = float(max(np.min(y_true), np.min(y_pred)))
    diag_max = float(min(np.max(y_true), np.max(y_pred)))
    ax1.plot([diag_min, diag_max], [diag_min, diag_max], "r--", linewidth=1)
    ax1.set_xlabel("True")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"All samples (Pearson={pearson:.4f}, Spearman={spearman:.4f})")
    plt.colorbar(hb1, ax=ax1, label="log₁₀(count)")

    # Right: non-zero samples only — density-colored scatter
    if nz_mask.any():
        nz_true = y_true[nz_mask]
        nz_pred = y_pred[nz_mask]
        # Compute 2D density per point
        hist, xedges, yedges = np.histogram2d(nz_true, nz_pred, bins=80)
        x_bin = np.clip(np.digitize(nz_true, xedges) - 1, 0, hist.shape[0] - 1)
        y_bin = np.clip(np.digitize(nz_pred, yedges) - 1, 0, hist.shape[1] - 1)
        density = np.log1p(hist[x_bin, y_bin])
        # Sort so densest points render on top
        order = np.argsort(density)
        ax2.scatter(nz_true[order], nz_pred[order], s=8, c=density[order],
                    cmap="plasma", alpha=0.7, edgecolors="none")
        nz_diag_min = float(max(np.min(nz_true), np.min(nz_pred)))
        nz_diag_max = float(min(np.max(nz_true), np.max(nz_pred)))
        ax2.plot([nz_diag_min, nz_diag_max], [nz_diag_min, nz_diag_max], "r--", linewidth=1)
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


def plot_per_promoter_scatter(model: nn.Module, dataset: Any, n_promoters: int = 6,
                               n_cells: int = 2000, save_path: Path | None = None) -> None:
    '''Sample promoters and plot (true, predicted) across cells, one color per promoter.'''
    device = next(model.parameters()).device
    model.eval()
    is_zinb = getattr(model, "output_mode", "scalar") == "zinb"

    rng = np.random.default_rng(42)
    pro_indices = rng.choice(dataset.P, size=min(n_promoters, dataset.P), replace=False)
    cell_indices = rng.choice(dataset.C, size=min(n_cells, dataset.C), replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(pro_indices)))

    # Pre-fetch promoter tensors
    promoter_batch = dataset.promoter_tensor[pro_indices].to(device)  # (P, 400, 5)
    # Pre-fetch cell expression rows as dense tensor
    cell_rows = [int(dataset.cells[j]) for j in cell_indices]
    X_csr = dataset.X.tocsr()
    X_batch_raw = np.vstack([X_csr[r].toarray().ravel() for r in cell_rows])
    X_batch = torch.from_numpy(X_batch_raw.astype(np.float32)).to(device)  # (C, n_genes)

    fig, ax = plt.subplots(figsize=(8, 8))

    with torch.no_grad():
        for i, pro_i in enumerate(pro_indices):
            target_idx = int(dataset.promoter2expr_idx[pro_i])
            # Mask target gene for all cells in one op
            ys = X_batch[:, target_idx].clone()  # (C,)
            X_masked = X_batch.clone()
            X_masked[:, target_idx] = 0.0

            # Expand promoter to batch: (C, 400, 5)
            p_batch = promoter_batch[i].unsqueeze(0).expand(len(cell_indices), -1, -1)

            if is_zinb:
                mu_ratio, _theta, _pi = model(p_batch, X_masked)
                mu_ratio = mu_ratio.squeeze(1)
                lib_size = X_masked.sum(dim=1) + ys
                y_pred_log = torch.log(mu_ratio * 10000 + 1).cpu().numpy()
                y_log = torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6).cpu().numpy()
                yt, yp = y_log, y_pred_log
            else:
                preds = model(p_batch, X_masked).squeeze(1).cpu().numpy()
                yt, yp = ys.cpu().numpy(), preds

            ax.scatter(yt, yp, s=15, alpha=0.5, color=colors[i])

    if len(ax.collections) > 0:
        all_t = np.concatenate([np.array(ax.collections[j].get_offsets()[:, 0])
                                for j in range(len(ax.collections))])
        all_p = np.concatenate([np.array(ax.collections[j].get_offsets()[:, 1])
                                for j in range(len(ax.collections))])
        diag_lo = float(max(all_t.min(), all_p.min()))
        diag_hi = float(min(all_t.max(), all_p.max()))
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "r--", linewidth=1)
        if len(all_t) > 1 and np.std(all_t) > 0 and np.std(all_p) > 0:
            corr = float(np.corrcoef(all_t, all_p)[0, 1])
            ax.set_title(f"Per-promoter prediction ({len(pro_indices)} promoters, {n_cells} cells)\n"
                         f"Pearson r={corr:.4f}")

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Per-promoter scatter saved to: {save_path}")
    plt.close()


def plot_per_cell_scatter(model: nn.Module, dataset: Any, n_cells: int = 6,
                           n_genes: int = 500, save_path: Path | None = None) -> None:
    '''Sample cells and plot (true, predicted) across genes, one color per cell.'''
    device = next(model.parameters()).device
    model.eval()
    is_zinb = getattr(model, "output_mode", "scalar") == "zinb"

    rng = np.random.default_rng(42)
    cell_indices = rng.choice(dataset.C, size=min(n_cells, dataset.C), replace=False)
    pro_indices = rng.choice(dataset.P, size=min(n_genes, dataset.P), replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_indices)))

    # Pre-fetch promoter tensors for all sampled genes
    promoter_batch = dataset.promoter_tensor[pro_indices].to(device)  # (G, 400, 5)
    target_indices = [int(dataset.promoter2expr_idx[p]) for p in pro_indices]
    X_csr = dataset.X.tocsr()

    fig, ax = plt.subplots(figsize=(8, 8))

    with torch.no_grad():
        for i, cell_i in enumerate(cell_indices):
            cell_row = int(dataset.cells[cell_i])
            expr_vec = torch.from_numpy(
                X_csr[cell_row].toarray().ravel().astype(np.float32)
            ).to(device)  # (n_genes,)

            # Build batch: one row per gene, with that gene's target masked
            expr_batch = expr_vec.unsqueeze(0).expand(len(pro_indices), -1).clone()  # (G, n_genes)
            ys = []
            for g, tgt in enumerate(target_indices):
                ys.append(expr_batch[g, tgt].item())
                expr_batch[g, tgt] = 0.0
            ys = torch.tensor(ys, device=device)

            if is_zinb:
                mu_ratio, _theta, _pi = model(promoter_batch, expr_batch)
                mu_ratio = mu_ratio.squeeze(1)
                lib_size = expr_batch.sum(dim=1) + ys
                y_pred_log = torch.log(mu_ratio * 10000 + 1).cpu().numpy()
                y_log = torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6).cpu().numpy()
                yt, yp = y_log, y_pred_log
            else:
                preds = model(promoter_batch, expr_batch).squeeze(1).cpu().numpy()
                yt, yp = ys.cpu().numpy(), preds

            ax.scatter(yt, yp, s=15, alpha=0.5, color=colors[i])

    if len(ax.collections) > 0:
        all_t = np.concatenate([np.array(ax.collections[j].get_offsets()[:, 0])
                                for j in range(len(ax.collections))])
        all_p = np.concatenate([np.array(ax.collections[j].get_offsets()[:, 1])
                                for j in range(len(ax.collections))])
        diag_lo = float(max(all_t.min(), all_p.min()))
        diag_hi = float(min(all_t.max(), all_p.max()))
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "r--", linewidth=1)
        if len(all_t) > 1 and np.std(all_t) > 0 and np.std(all_p) > 0:
            corr = float(np.corrcoef(all_t, all_p)[0, 1])
            ax.set_title(f"Per-cell prediction ({len(cell_indices)} cells, {n_genes} genes)\n"
                         f"Pearson r={corr:.4f}")

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Per-cell scatter saved to: {save_path}")
    plt.close()


def plot_loss_curves_from_logfile(log_file: Path, save_path: Path | None = None, step_log_file: Path | None = None) -> None:
    '''Plot train loss by global_step (from step_train_loss.csv) and val loss by epoch.'''
    df_epoch = pd.read_csv(log_file)
    #df_epoch = df_epoch[1:]  # skip epoch 0 which may have very different scale due to resume
    # using max loss for normalization to keep the curve shape visible, especially when resumed training may have different absolute loss values
    train_loss_max = max(df_epoch["train_loss"].max(), 1e-8)
    val_loss_max = max(df_epoch["val_loss"].max(), 1e-8)
    df_epoch["train_loss"] /= train_loss_max
    df_epoch["val_loss"] /= val_loss_max
    df_epoch["val_loss_ema"] /= val_loss_max

    if step_log_file is None:
        step_log_file = log_file.parent / "step_train_loss.csv"

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # ---- train loss by global_step ----
    if step_log_file.exists():
        df_step = pd.read_csv(step_log_file)
        #df_step = df_step[df_step["epoch"] > 1]  # skip epoch 0 which may have different scale due to resume
        train_loss_max = max(df_step["train_loss"].max(), train_loss_max, 1e-8)
        # Map each epoch to its midpoint global_step for val loss alignment
        step_bounds = df_step.groupby("epoch")["global_step"].agg(["min", "max"])
        # Normalize train loss
        train_max = df_step["train_loss"].max()
        df_step["train_loss"] /= train_loss_max
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


def plot_zero_nonzero_loss_curves(log_file: Path, save_path: Path | None = None) -> None:
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


def plot_val_metrics(log_file: Path, save_path: Path | None = None) -> None:
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