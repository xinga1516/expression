# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:49:15 2026

@author: HP
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MyDataset
from src.model import SimpleGeneModel


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	np.random.seed(seed)


def build_eval_loader(base_dir: Path, split: str, batch_size: int, num_workers: int, seed: int) -> DataLoader:
	promoter_file = base_dir / "data" / "processed" / f"promoter_{split}.csv"
	scrna_file = base_dir / "data" / "processed" / "integrated_data.h5ad"

	dataset = MyDataset(
		promoter_file=promoter_file,
		scrna_file=scrna_file,
		mode="val",
		seed=seed,
	)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	return loader


def load_model(base_dir: Path, checkpoint_path: Path | None, expr_dim: int, device: torch.device) -> SimpleGeneModel:
	if checkpoint_path is None:
		checkpoint_path = base_dir / "checkpoints" / "best_model.pth"

	model = SimpleGeneModel(expr_dim=expr_dim)
	state_dict = load_file(str(checkpoint_path), device=str(device))
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()
	return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
	mse = float(np.mean((y_pred - y_true) ** 2))
	mae = float(np.mean(np.abs(y_pred - y_true)))
	rmse = float(np.sqrt(mse))

	y_true_var = float(np.var(y_true))
	if y_true_var > 0:
		r2 = float(1.0 - np.mean((y_pred - y_true) ** 2) / y_true_var)
	else:
		r2 = float("nan")

	if len(y_true) > 1:
		corr = float(np.corrcoef(y_true, y_pred)[0, 1])
	else:
		corr = float("nan")

	return {
		"mse": mse,
		"mae": mae,
		"rmse": rmse,
		"r2": r2,
		"pearson": corr,
	}


def evaluate(model: SimpleGeneModel, loader: DataLoader, device: torch.device, max_steps: int) -> dict:
	criterion = nn.MSELoss(reduction="sum")
	y_true_list = []
	y_pred_list = []

	loss_sum = 0.0
	count = 0

	with torch.no_grad():
		for step, batch in enumerate(loader):
			promoters, exprs, ys = batch
			promoters = promoters.to(device, non_blocking=True)
			exprs = exprs.to(device, non_blocking=True)
			ys = ys.to(device, non_blocking=True).float()

			pred = model(promoters, exprs).squeeze(1)
			loss_sum += criterion(pred, ys).item()
			count += ys.numel()

			y_true_list.append(ys.detach().cpu().numpy())
			y_pred_list.append(pred.detach().cpu().numpy())

			if step + 1 >= max_steps:
				break

	y_true = np.concatenate(y_true_list, axis=0)
	y_pred = np.concatenate(y_pred_list, axis=0)

	metrics = compute_metrics(y_true, y_pred)
	metrics["avg_loss"] = float(loss_sum / count)
	metrics["num_samples"] = int(count)
	metrics["num_steps"] = int(min(max_steps, step + 1))
	return metrics


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate SimpleGeneModel on sampled val/test batches")
	parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Evaluation split")
	parser.add_argument("--checkpoint", type=str, default=None, help="Path to safetensors checkpoint")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
	parser.add_argument("--steps", type=int, default=1000, help="Number of sampled batches for evaluation")
	parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	base_dir = Path(__file__).resolve().parent.parent
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	loader = build_eval_loader(
		base_dir=base_dir,
		split=args.split,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seed=args.seed,
	)

	expr_dim = loader.dataset.X.shape[1]
	checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
	model = load_model(
		base_dir=base_dir,
		checkpoint_path=checkpoint_path,
		expr_dim=expr_dim,
		device=device,
	)

	metrics = evaluate(model=model, loader=loader, device=device, max_steps=args.steps)

	print("========== Evaluation ==========")
	print(f"split: {args.split}")
	print(f"checkpoint: {checkpoint_path if checkpoint_path else base_dir / 'checkpoints' / 'best_model.pth'}")
	print(f"device: {device}")
	print(f"steps: {metrics['num_steps']}, samples: {metrics['num_samples']}")
	print(f"avg_loss: {metrics['avg_loss']:.6f}")
	print(f"mse: {metrics['mse']:.6f}")
	print(f"mae: {metrics['mae']:.6f}")
	print(f"rmse: {metrics['rmse']:.6f}")
	print(f"r2: {metrics['r2']:.6f}")
	print(f"pearson: {metrics['pearson']:.6f}")


if __name__ == "__main__":
	main()

