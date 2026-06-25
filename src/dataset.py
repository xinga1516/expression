# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:56:24 2026

@author: HP
"""
import torch
import pandas as pd
import scanpy as sc
import numpy as np
from torch.utils.data import Dataset
from scipy import sparse
from pathlib import Path
from typing import Optional, Sequence

class PromoterOneHotEncoder:
    # One-hot encode DNA sequence (pad with N up to fixed length).
    def __init__(self, length: int = 400) -> None:
        self.length = length
        self.vocab = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "N": 4
        }
        self.num_channels = 5

    def __call__(self, seq: str) -> torch.Tensor:
        """
        seq: str, length <= self.length
        return: FloatTensor (length, 5)
        """
        # 初始化为 N
        one_hot = torch.zeros(self.length, self.num_channels)
        one_hot[:, 4] = 1.0  # 默认 N

        seq = seq.upper()

        for i, base in enumerate(seq[:self.length]):
            idx = self.vocab.get(base, 4)
            one_hot[i, :] = 0.0
            one_hot[i, idx] = 1.0

        return one_hot


class MyDataset(Dataset):
    def __init__(
        self,
        promoter_file: str | Path,
        scrna_file: str | Path,
        cell_ids_subset: Optional[np.ndarray] = None,
        mode: str = "train",
        seed: int = 42,
        cell_ratio: float = 1.0,
        log1p_cpm_target: bool = False,
        preencode_promoters: bool = False,
        sequence_column: str = "sequence",
        input_gene_ids: Optional[Sequence[str]] = None,
        input_gene_panel_file: Optional[str | Path] = None,
    ) -> None:
        self.promoter_file = Path(promoter_file)
        self.scrna_file = Path(scrna_file)
        self.promoters = pd.read_csv(promoter_file)
        self.sequence_column = sequence_column
        if self.sequence_column not in self.promoters.columns:
            raise ValueError(
                f"Sequence column {self.sequence_column!r} not found in {self.promoter_file}. "
                f"Available columns: {sorted(self.promoters.columns)}"
            )
        self.scrna = sc.read(scrna_file, sparse=True)
        # CSR: fast row access; CSC: fast column access.
        self.X = self.scrna.X.tocsr() if sparse.issparse(self.scrna.X) else self.scrna.X

        self.promoter_encoder = PromoterOneHotEncoder(length=400)
        self.preencode_promoters = preencode_promoters
        self.promoter_tensor: torch.Tensor | None = None
        if self.preencode_promoters:
            print("Pre-encoding promoter sequences...")
            self.promoter_tensor = self._preencode_promoters()
            print("Done.")
        else:
            print("Promoter pre-encoding disabled; sequences will be encoded per sample.")
        #self.gene_ids = gene_ids_subset
        if cell_ids_subset is None:
            self.cells = np.arange(self.scrna.n_obs, dtype=np.int64)
        else:
            # Support passing either obs indices (ints) or obs names (strings).
            cell_ids_subset = np.asarray(cell_ids_subset)
            if np.issubdtype(cell_ids_subset.dtype, np.integer):
                self.cells = cell_ids_subset.astype(np.int64, copy=False)
            else:
                idx = self.scrna.obs_names.get_indexer(cell_ids_subset.tolist())
                if (idx < 0).any():
                    missing = cell_ids_subset[idx < 0][:5]
                    raise ValueError(f"Some cell ids not found in scRNA obs_names, e.g. {missing!r}")
                self.cells = idx.astype(np.int64, copy=False)

        # 建 promoter index -> gene_id → scrna index 的映射
        self.gene2idx = {
            g: i for i, g in enumerate(self.scrna.var.gene_id)
        }
        self.promoter2expr_idx = np.empty(len(self.promoters), dtype=np.int32)
        for i, gene_id in enumerate(self.promoters["gene_id"]):
            try:
                self.promoter2expr_idx[i] = self.gene2idx[gene_id]
            except KeyError:
                raise ValueError(f"gene_id {gene_id} not found in scRNA var")
        #print(self.gene2idx)

        self.input_gene_ids = self._resolve_input_gene_ids(input_gene_ids, input_gene_panel_file)
        self.input_expr_indices: np.ndarray | None = None
        self.input_gene_idx_to_position: dict[int, int] = {}
        if self.input_gene_ids is not None:
            missing_input_genes = [gene_id for gene_id in self.input_gene_ids if gene_id not in self.gene2idx]
            if missing_input_genes:
                raise ValueError(
                    f"{len(missing_input_genes)} input genes not found in scRNA var, "
                    f"e.g. {missing_input_genes[:5]}"
                )
            self.input_expr_indices = np.asarray(
                [self.gene2idx[gene_id] for gene_id in self.input_gene_ids],
                dtype=np.int64,
            )
            self.input_gene_idx_to_position = {
                int(gene_idx): pos for pos, gene_idx in enumerate(self.input_expr_indices)
            }

        self._cell_ratio = cell_ratio
        self._original_cells = self.cells.copy()
        if cell_ratio < 1.0:
            rng = np.random.default_rng(seed if mode == "train" else seed + 1000)
            n_keep = max(1, int(len(self.cells) * cell_ratio))
            chosen = rng.choice(len(self.cells), size=n_keep, replace=False)
            self.cells = self.cells[chosen]

        self.P = len(self.promoters)
        self.C = len(self.cells)
        self.mode = mode
        self.seed = seed
        self.log1p_cpm_target = log1p_cpm_target
        self.expr_dim = len(self.input_expr_indices) if self.input_expr_indices is not None else self.X.shape[1]

    def _resolve_input_gene_ids(
        self,
        input_gene_ids: Optional[Sequence[str]],
        input_gene_panel_file: Optional[str | Path],
    ) -> list[str] | None:
        if input_gene_ids is not None and input_gene_panel_file is not None:
            raise ValueError("Provide only one of input_gene_ids or input_gene_panel_file.")
        if input_gene_ids is not None:
            return [str(gene_id) for gene_id in input_gene_ids]
        if input_gene_panel_file is None:
            return None
        panel_path = Path(input_gene_panel_file)
        if not panel_path.exists():
            raise FileNotFoundError(f"Input gene panel file not found: {panel_path}")
        genes = [
            line.strip().split("\t")[0].split(",")[0]
            for line in panel_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not genes:
            raise ValueError(f"Input gene panel file is empty: {panel_path}")
        return genes

    def resample_cells(self, seed: int) -> None:
        '''Re-select a random subset of cells using cell_ratio with a new seed.
        Call between epochs to expose the model to different cells over time.'''
        if self._cell_ratio >= 1.0:
            return
        rng = np.random.default_rng(seed if self.mode == "train" else seed + 1000)
        n_keep = max(1, int(len(self._original_cells) * self._cell_ratio))
        chosen = rng.choice(len(self._original_cells), size=n_keep, replace=False)
        self.cells = self._original_cells[chosen]
        self.C = len(self.cells)

    def _preencode_promoters(self) -> torch.Tensor:
        sequences = self.promoters[self.sequence_column].values
        n = len(sequences)

        # (N, 400, 5)
        promoter_tensor = torch.zeros(
            n, 400, 5, dtype=torch.float32
        )

        for i, seq in enumerate(sequences):
            promoter_tensor[i] = self.promoter_encoder(seq)

            if i % 1000 == 0:
                print(f"  encoded {i}/{n}")

        return promoter_tensor
    
    def __len__(self) -> int:
        return self.P * self.C

    def get_promoter_tensor(self, pro_i: int) -> torch.Tensor:
        if self.promoter_tensor is not None:
            return self.promoter_tensor[pro_i]
        seq = str(self.promoters[self.sequence_column].iloc[pro_i])
        return self.promoter_encoder(seq)

    def get_promoter_tensors(self, pro_indices: np.ndarray | list[int]) -> torch.Tensor:
        if self.promoter_tensor is not None:
            return self.promoter_tensor[pro_indices]
        tensors = [self.get_promoter_tensor(int(pro_i)) for pro_i in pro_indices]
        return torch.stack(tensors, dim=0)

    def in_getitem(self, pro_i: int, cell_i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        promoter = self.get_promoter_tensor(pro_i)
        #gene_id = (self.promoters["gene_id"]).iloc[pro_i]
        cell_id = self.cells[cell_i]
        expr_all = self.X[cell_id]

        if hasattr(expr_all, "toarray"):
            expr_all = expr_all.toarray().astype("float32").squeeze()     # (16300,)
        expr_all_np = np.asarray(expr_all, dtype=np.float32).ravel()
        
        target_idx = self.promoter2expr_idx[pro_i]
        y_value = float(expr_all_np[target_idx])
        expr_input_np = self.make_masked_expression_input(expr_all_np, int(target_idx))
        expr_input = torch.from_numpy(expr_input_np).float()
        y = torch.tensor(y_value, dtype=torch.float32)

        if self.log1p_cpm_target:
            lib_size = max(float(expr_all_np.sum() - y_value), 1.0)
            cpm = y / max(float(lib_size), 1.0) * 1e6
            y = torch.log1p(cpm)

        return promoter, expr_input, y

    def make_masked_expression_input(self, full_expr: np.ndarray, target_idx: int) -> np.ndarray:
        full_expr = np.asarray(full_expr, dtype=np.float32).ravel()
        if self.input_expr_indices is None:
            expr_input = full_expr.copy()
            expr_input[target_idx] = 0.0
            return expr_input

        expr_input = full_expr[self.input_expr_indices].copy()
        target_pos = self.input_gene_idx_to_position.get(int(target_idx))
        if target_pos is not None:
            expr_input[target_pos] = 0.0
        return expr_input

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pro_i = idx // self.C
        cell_i = idx % self.C
        
        return self.in_getitem(pro_i,cell_i)

