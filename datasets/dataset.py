# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:56:24 2026

@author: HP
"""
import torch
import pandas as pd
import scanpy as sc
import numpy as np
from torch.utils.data import IterableDataset

class PromoterOneHotEncoder:
    def __init__(self, length=400):
        self.length = length
        self.vocab = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "N": 4
        }
        self.num_channels = 5

    def __call__(self, seq: str):
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


class MyDataset(IterableDataset):
    def __init__(
        self,
        promoter_file,
        scrna_file,
        cell_ids_subset=None
    ):
        self.promoters = pd.read_csv(promoter_file)
        self.scrna = sc.read(scrna_file, sparse=True)

        self.promoter_encoder = PromoterOneHotEncoder(length=400)
        print("Pre-encoding promoter sequences...")
        self.promoter_tensor = self._preencode_promoters()
        print("Done.")
        #self.gene_ids = gene_ids_subset
        self.use_direct_index = cell_ids_subset is None
        self.cells = cell_ids_subset or np.arange(self.scrna.n_obs)

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

        self.P = len(self.promoters)
        self.C = len(self.cells)

    def _preencode_promoters(self):
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
    
    def __len__(self):
        return self.P * self.C

    def in_getitem(self, pro_i, cell_i):
        promoter = self.promoter_tensor[pro_i]
        #gene_id = (self.promoters["gene_id"]).iloc[pro_i]
        if self.use_direct_index:
            expr_all = self.scrna[cell_i].X
        else:
            cell_id = self.cells[cell_i]
            expr_all = self.scrna[cell_id].X

        if hasattr(expr_all, "toarray"):
            expr_all = expr_all.toarray().astype("float32").squeeze()     # (16300,)
        expr_all = torch.from_numpy(expr_all).float()
        
        target_idx = self.promoter2expr_idx[pro_i]
        y = expr_all[target_idx]
        expr_all[target_idx] = 0.0 # mask the promoter expression
        
        return promoter, expr_all, y

    def __getitem__(self, idx):
        pro_i = idx // self.C
        cell_i = idx % self.C
        
        return self.in_getitem(pro_i,cell_i)

    def __iter__(self):
        while True:
            pro_i = np.random.randint(len(self.promoters))
            cell_i = np.random.randint(len(self.cells))
            yield self.in_getitem(pro_i,cell_i)
