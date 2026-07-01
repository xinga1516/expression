from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
import torch
from scipy import sparse

from src.dataset import MyDataset


def _to_dense_float32(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


class GpuCachedPairLoader:
    """Iterate promoter/cell pairs from a MyDataset using device-resident tensors.

    The loader keeps fixed dataset arrays on ``device`` and materializes each batch
    by gathering ``promoter_idx`` and ``cell_idx`` directly on that device. It
    intentionally returns the same ``(promoters, exprs, ys)`` tuple shape as the
    standard DataLoader path so the existing training loop can stay unchanged.
    """

    def __init__(
        self,
        dataset: MyDataset,
        batch_size: int,
        device: torch.device | str,
        samples_per_epoch: int,
        seed: int = 42,
        sampler_mode: str = "balanced",
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.sampler_mode = sampler_mode
        self.drop_last = bool(drop_last)
        self.device = torch.device(device)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.samples_per_epoch < 0:
            raise ValueError("samples_per_epoch must be non-negative.")
        if self.sampler_mode not in {"balanced", "random"}:
            raise ValueError("sampler_mode must be 'balanced' or 'random'.")

        self.P = int(dataset.P)
        self.C = int(dataset.C)
        self.total_len = self.P * self.C
        if self.total_len <= 0:
            raise ValueError("Dataset is empty.")
        if self.samples_per_epoch == 0:
            self.samples_per_epoch = min(128_000, self.total_len)
        else:
            self.samples_per_epoch = min(self.samples_per_epoch, self.total_len)
        self._cache_dataset_tensors()

    def _cache_dataset_tensors(self) -> None:
        print(
            f"[GPUCache] caching split on {self.device}: "
            f"promoters={self.P} cells={self.C} expr_dim={self.dataset.expr_dim}"
        )

        pro_indices = np.arange(self.P, dtype=np.int64)
        promoter_tensor = self.dataset.get_promoter_tensors(pro_indices).to(torch.float32)
        self.promoters_device = promoter_tensor.to(self.device, non_blocking=False)

        expr_indices = self.dataset.input_expr_indices
        if expr_indices is None:
            expr_indices = np.arange(self.dataset.expression_X.shape[1], dtype=np.int64)
        else:
            expr_indices = np.asarray(expr_indices, dtype=np.int64)

        cell_rows = np.asarray(self.dataset.cells, dtype=np.int64)
        expr_source = self.dataset.expression_X[cell_rows, :]
        target_source = self.dataset.X[cell_rows, :]
        target_value_source = self.dataset.target_value_X[cell_rows, :] if self.dataset.target_value_X is not None else target_source
        expr_panel = self.dataset._apply_expression_transform(_to_dense_float32(expr_source[:, expr_indices]))
        target_matrix = _to_dense_float32(target_value_source[:, self.dataset.promoter2expr_idx])
        raw_target_matrix = _to_dense_float32(target_source[:, self.dataset.promoter2expr_idx])
        if sparse.issparse(target_source):
            cell_totals = np.asarray(target_source.sum(axis=1), dtype=np.float32).ravel()
        else:
            cell_totals = np.asarray(target_source, dtype=np.float32).sum(axis=1)

        target_input_positions = np.full(self.P, -1, dtype=np.int64)
        if self.dataset.input_expr_indices is None:
            target_input_positions[:] = self.dataset.promoter2expr_idx.astype(np.int64)
        else:
            for pro_i, target_idx in enumerate(self.dataset.promoter2expr_idx):
                pos = self.dataset.input_gene_idx_to_position.get(int(target_idx))
                if pos is not None:
                    target_input_positions[pro_i] = int(pos)

        self.expr_panel_device = torch.as_tensor(expr_panel, dtype=torch.float32, device=self.device)
        self.target_matrix_device = torch.as_tensor(target_matrix, dtype=torch.float32, device=self.device)
        self.raw_target_matrix_device = torch.as_tensor(raw_target_matrix, dtype=torch.float32, device=self.device)
        self.cell_totals_device = torch.as_tensor(cell_totals, dtype=torch.float32, device=self.device)
        self.target_input_positions_device = torch.as_tensor(
            target_input_positions, dtype=torch.long, device=self.device
        )

        cached_mib = (
            self.promoters_device.numel()
            + self.expr_panel_device.numel()
            + self.target_matrix_device.numel()
            + self.raw_target_matrix_device.numel()
            + self.cell_totals_device.numel()
            + self.target_input_positions_device.numel()
        ) * 4 / (1024 ** 2)
        print(f"[GPUCache] cached tensors occupy approximately {cached_mib:.1f} MiB.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.samples_per_epoch // self.batch_size
        return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size

    @property
    def sampler(self) -> "GpuCachedPairLoader":
        return self

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        n_samples = self.samples_per_epoch
        if self.drop_last:
            n_samples = (n_samples // self.batch_size) * self.batch_size
        if n_samples <= 0:
            return

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + self.epoch)
        promoter_idx, cell_idx = self._sample_pair_indices(n_samples, generator)

        for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
            yield self._make_batch_from_pair_indices(promoter_idx[start:end], cell_idx[start:end])

    def _sample_pair_indices(
        self,
        n_samples: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sampler_mode == "random":
            promoter_idx = torch.randint(0, self.P, (n_samples,), generator=generator, device=self.device)
            cell_idx = torch.randint(0, self.C, (n_samples,), generator=generator, device=self.device)
            return promoter_idx, cell_idx

        base = n_samples // self.P
        rem = n_samples % self.P
        counts = torch.full((self.P,), base, dtype=torch.long, device=self.device)
        if rem > 0:
            promoter_order = torch.randperm(self.P, generator=generator, device=self.device)
            counts[promoter_order[:rem]] += 1
        promoter_idx = torch.repeat_interleave(torch.arange(self.P, device=self.device), counts)
        order = torch.randperm(n_samples, generator=generator, device=self.device)
        promoter_idx = promoter_idx.index_select(0, order)
        cell_idx = torch.randint(0, self.C, (n_samples,), generator=generator, device=self.device)
        return promoter_idx, cell_idx

    def _make_batch(self, flat_indices: Sequence[int] | np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = torch.as_tensor(flat_indices, dtype=torch.long, device=self.device)
        promoter_idx = torch.div(flat, self.C, rounding_mode="floor")
        cell_idx = flat.remainder(self.C)
        return self._make_batch_from_pair_indices(promoter_idx, cell_idx)

    def _make_batch_from_pair_indices(
        self,
        promoter_idx: torch.Tensor,
        cell_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        promoters = self.promoters_device.index_select(0, promoter_idx)
        exprs = self.expr_panel_device.index_select(0, cell_idx).clone()
        ys = self.target_matrix_device[cell_idx, promoter_idx]
        raw_ys = self.raw_target_matrix_device[cell_idx, promoter_idx]

        target_pos = self.target_input_positions_device.index_select(0, promoter_idx)
        valid = target_pos >= 0
        if bool(valid.any().item()):
            row_idx = torch.arange(exprs.shape[0], device=self.device, dtype=torch.long)
            exprs[row_idx[valid], target_pos[valid]] = 0.0

        if self.dataset.log1p_cpm_target and self.dataset.target_value_X is None:
            lib_size = torch.clamp(self.cell_totals_device.index_select(0, cell_idx) - raw_ys, min=1.0)
            ys = torch.log1p(raw_ys / lib_size * 1e6)

        return promoters, exprs, ys.float()
