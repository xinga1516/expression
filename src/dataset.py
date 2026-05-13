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
from typing import Optional

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
    ) -> None:
        self.promoters = pd.read_csv(promoter_file)
        self.scrna = sc.read(scrna_file, sparse=True)
        # CSR: fast row access; CSC: fast column access.
        self.X = self.scrna.X.tocsr() if sparse.issparse(self.scrna.X) else self.scrna.X

        self.promoter_encoder = PromoterOneHotEncoder(length=400)
        print("Pre-encoding promoter sequences...")
        self.promoter_tensor = self._preencode_promoters()
        print("Done.")
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

        if cell_ratio < 1.0:
            rng = np.random.default_rng(seed if mode == "train" else seed + 1000)
            n_keep = max(1, int(len(self.cells) * cell_ratio))
            chosen = rng.choice(len(self.cells), size=n_keep, replace=False)
            self.cells = self.cells[chosen]

        self.P = len(self.promoters)
        self.C = len(self.cells)
        self.mode = mode
        self.seed = seed

    def _preencode_promoters(self) -> torch.Tensor:
        sequences = self.promoters["sequence"].values
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

    def in_getitem(self, pro_i: int, cell_i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        promoter = self.promoter_tensor[pro_i]
        #gene_id = (self.promoters["gene_id"]).iloc[pro_i]
        cell_id = self.cells[cell_i]
        expr_all = self.X[cell_id]

        if hasattr(expr_all, "toarray"):
            expr_all = expr_all.toarray().astype("float32").squeeze()     # (16300,)
        expr_all = torch.from_numpy(expr_all).float()
        
        target_idx = self.promoter2expr_idx[pro_i]
        y = expr_all[target_idx].clone()
        expr_all[target_idx] = 0.0 # mask the promoter expression
        
        return promoter, expr_all, y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pro_i = idx // self.C
        cell_i = idx % self.C
        
        return self.in_getitem(pro_i,cell_i)

