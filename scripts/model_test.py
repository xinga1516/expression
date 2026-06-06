from pathlib import Path
import argparse
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MyDataset
from src.model import build_model
import src.utils as utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name under outputs/")
    parser.add_argument("--data", type=str, default=None, help="Fallback data directory name under data/")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Default: outputs/<exp_name>/checkpoints/best_model.safetensors")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all test samples.")
    parser.add_argument("--spearman-samples", type=int, default=1_000_000, help="Reservoir sample size for Spearman. 0 means store all samples.")
    parser.add_argument("--preencode-promoters", action="store_true", default=False, help="Pre-encode test promoters in memory.")
    return parser.parse_args()


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_dir(base_dir: Path, cfg: dict[str, Any], data_arg: str | None) -> Path:
    scrna_file = cfg.get("scrna_file", "")
    if scrna_file:
        return Path(scrna_file).parent
    if data_arg is None:
        raise ValueError("--data is required when config.json has no scrna_file")
    return base_dir / "data" / data_arg


def build_test_model(cfg: dict[str, Any], expr_dim: int, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    loss_type = cfg.get("loss_type", "mse")
    output_mode = "zinb" if loss_type == "zinb" else "scalar"
    model = build_model(
        cfg.get("model", "LSTMmodel"),
        expr_dim=expr_dim,
        hidden_size=cfg.get("hidden_size", 128),
        use_vae=cfg.get("use_vae", False),
        vae_encoder_path=cfg.get("vae_encoder_path", None),
        vae_fine_tune=cfg.get("vae_fine_tune", False),
        output_mode=output_mode,
        fusion=cfg.get("fusion", "gate"),
    )
    model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    model.to(device)
    model.eval()
    return model


def update_spearman_reservoir(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reservoir_true: list[np.ndarray],
    reservoir_pred: list[np.ndarray],
    seen: int,
    limit: int,
    rng: np.random.Generator,
) -> int:
    if limit == 0:
        reservoir_true.append(y_true.astype(np.float32, copy=False))
        reservoir_pred.append(y_pred.astype(np.float32, copy=False))
        return seen + len(y_true)

    remaining = max(limit - seen, 0)
    if remaining > 0:
        take = min(remaining, len(y_true))
        reservoir_true.append(y_true[:take].astype(np.float32, copy=False))
        reservoir_pred.append(y_pred[:take].astype(np.float32, copy=False))
        y_true = y_true[take:]
        y_pred = y_pred[take:]
        seen += take

    if len(y_true) == 0:
        return seen

    if len(reservoir_true) == 1 and len(reservoir_true[0]) == limit:
        true_arr = reservoir_true[0]
        pred_arr = reservoir_pred[0]
    else:
        true_arr = np.concatenate(reservoir_true)
        pred_arr = np.concatenate(reservoir_pred)
    for yt, yp in zip(y_true, y_pred):
        seen += 1
        j = int(rng.integers(0, seen))
        if j < limit:
            true_arr[j] = yt
            pred_arr[j] = yp
    reservoir_true[:] = [true_arr]
    reservoir_pred[:] = [pred_arr]
    return seen


def compute_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    spearman_samples: int,
    seed: int,
) -> dict[str, float | int]:
    ''' compute
    '''
    is_zinb = getattr(model, "output_mode", "scalar") == "zinb"
    rng = np.random.default_rng(seed)

    count = 0
    sum_y = 0.0
    sum_p = 0.0
    sum_yy = 0.0
    sum_pp = 0.0
    sum_yp = 0.0
    sse = 0.0
    reservoir_true: list[np.ndarray] = []
    reservoir_pred: list[np.ndarray] = []
    spearman_seen = 0

    with torch.no_grad():
        for promoters, exprs, ys in loader:
            promoters = promoters.to(device, non_blocking=True)
            exprs = exprs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True).float()

            if is_zinb:
                mu_ratio, _theta, _pi = model(promoters, exprs)
                mu_ratio = mu_ratio.squeeze(1)
                lib_size = exprs.sum(dim=1) + ys
                pred = torch.log(mu_ratio * 1e6 + 1)
                true = torch.log1p(ys / torch.clamp(lib_size, min=1.0) * 1e6)
            else:
                pred = model(promoters, exprs).squeeze(1)
                true = ys

            if max_samples > 0 and count + true.numel() > max_samples:
                keep = max_samples - count
                true = true[:keep]
                pred = pred[:keep]

            y = true.detach().cpu().numpy().astype(np.float64, copy=False)
            p = pred.detach().cpu().numpy().astype(np.float64, copy=False)

            count += int(y.size)
            diff = p - y
            sse += float(np.sum(diff * diff))
            sum_y += float(np.sum(y))
            sum_p += float(np.sum(p))
            sum_yy += float(np.sum(y * y))
            sum_pp += float(np.sum(p * p))
            sum_yp += float(np.sum(y * p))

            spearman_seen = update_spearman_reservoir(
                y, p, reservoir_true, reservoir_pred, spearman_seen, spearman_samples, rng
            )

            if max_samples > 0 and count >= max_samples:
                break

    mse = sse / max(count, 1)
    denom_y = sum_yy - sum_y * sum_y / max(count, 1)
    denom_p = sum_pp - sum_p * sum_p / max(count, 1)
    denom = float(np.sqrt(max(denom_y, 0.0) * max(denom_p, 0.0)))
    pearson = (sum_yp - sum_y * sum_p / max(count, 1)) / denom if denom > 0 else float("nan")

    spearman_true = np.concatenate(reservoir_true) if reservoir_true else np.array([], dtype=np.float32)
    spearman_pred = np.concatenate(reservoir_pred) if reservoir_pred else np.array([], dtype=np.float32)
    if len(spearman_true) > 1:
        try:
            from scipy.stats import spearmanr
            spearman = float(spearmanr(spearman_true, spearman_pred)[0])
        except Exception:
            spearman = float("nan")
    else:
        spearman = float("nan")

    return {
        "num_samples": int(count),
        "mse": float(mse),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
        "spearman_num_samples": int(len(spearman_true)),
    }


def main() -> None:
    args = parse_args()
    base_dir = PROJECT_ROOT
    run_dir = base_dir / "outputs" / args.exp_name
    test_dir = run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(run_dir)
    data_dir = resolve_data_dir(base_dir, cfg, args.data)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best_model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    loss_type = cfg.get("loss_type", "mse")
    is_umi = data_dir.name.startswith("umi_") and loss_type != "zinb"
    dataset = MyDataset(
        promoter_file=data_dir / "promoter_test.csv",
        scrna_file=data_dir / "integrated_data.h5ad",
        mode="test",
        seed=args.seed,
        log1p_cpm_target=is_umi,
        preencode_promoters=args.preencode_promoters,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expr_dim = int(cfg.get("expr_dim", dataset.X.shape[1]))
    model = build_test_model(cfg, expr_dim=expr_dim, checkpoint=checkpoint, device=device)

    metrics = compute_test_metrics(
        model=model,
        loader=loader,
        device=device,
        max_samples=args.max_samples,
        spearman_samples=args.spearman_samples,
        seed=args.seed,
    )
    metrics.update({
        "exp_name": args.exp_name,
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "loss_type": loss_type,
        "max_samples": int(args.max_samples),
    })

    metrics_path = test_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(test_dir / "test_metrics.csv", index=False)

    utils.plot_per_promoter_scatter(
        model,
        dataset,
        is_umi=is_umi,
        n_promoters=3,
        save_path=test_dir / "per_promoter_scatter.png",
    )
    utils.plot_per_cell_scatter(
        model,
        dataset,
        is_umi=is_umi,
        n_cells=3,
        save_path=test_dir / "per_cell_scatter.png",
    )

    print("========== Test ==========")
    print(f"data: {data_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"samples: {metrics['num_samples']}")
    print(f"mse: {metrics['mse']:.6f}")
    print(f"pearson_r: {metrics['pearson_r']:.6f}")
    print(f"spearman_r: {metrics['spearman_r']:.6f} (n={metrics['spearman_num_samples']})")
    print(f"outputs: {test_dir}")


if __name__ == "__main__":
    main()
