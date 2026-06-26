from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import MyDataset, PromoterOneHotEncoder
from src.gpu_cache import GpuCachedPairLoader


pytestmark = pytest.mark.unit


def test_promoter_one_hot_encoder_encodes_pads_and_truncates() -> None:
    encoder = PromoterOneHotEncoder(length=6)
    encoded = encoder("ACGTNXZ")

    assert encoded.shape == (6, 5)
    assert torch.equal(encoded[0], torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32))
    assert torch.equal(encoded[1], torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32))
    assert torch.equal(encoded[2], torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32))
    assert torch.equal(encoded[3], torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32))
    assert torch.equal(encoded[4], torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32))
    assert torch.equal(encoded[5], torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32))


def test_dataset_masks_target_and_returns_target(tiny_data_dir) -> None:
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=False,
    )

    promoter, expr, y = dataset.in_getitem(pro_i=0, cell_i=0)
    target_idx = int(dataset.promoter2expr_idx[0])

    assert promoter.shape == (400, 5)
    assert y.item() == pytest.approx(10.0)
    assert expr[target_idx].item() == 0.0
    assert expr.sum().item() == pytest.approx(6.0)


def test_dataset_log1p_cpm_target_uses_masked_library_size(tiny_data_dir) -> None:
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        log1p_cpm_target=True,
    )

    _promoter, _expr, y = dataset.in_getitem(pro_i=0, cell_i=0)
    expected = math.log1p(10.0 / 6.0 * 1e6)
    assert y.item() == pytest.approx(expected)


def test_dynamic_and_preencoded_promoter_tensors_match(tiny_data_dir) -> None:
    dynamic = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=False,
    )
    preencoded = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=True,
    )

    assert dynamic.promoter_tensor is None
    assert preencoded.promoter_tensor is not None
    assert torch.equal(dynamic.get_promoter_tensor(1), preencoded.get_promoter_tensor(1))
    assert torch.equal(dynamic[0][0], preencoded[0][0])


def test_resample_cells_updates_subset_size(tiny_data_dir) -> None:
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        mode="train",
        cell_ratio=0.5,
        seed=1,
    )
    before = dataset.cells.copy()

    dataset.resample_cells(seed=2)

    assert dataset.C == max(1, int(len(dataset._original_cells) * 0.5))
    assert len(dataset.cells) == dataset.C
    assert not np.array_equal(before, dataset.cells)


def test_dataset_uses_control_sequence_and_fixed_input_panel(tiny_data_dir) -> None:
    promoter_path = tiny_data_dir / "promoter_train.csv"
    promoters = pd.read_csv(promoter_path)
    promoters["control_sequence"] = ["T" * 400 for _ in range(len(promoters))]
    promoters.to_csv(promoter_path, index=False)
    panel_path = tiny_data_dir / "input_gene_panel_train.txt"
    panel_path.write_text("g1\ng2\n", encoding="utf-8")

    dataset = MyDataset(
        promoter_file=promoter_path,
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        sequence_column="control_sequence",
        input_gene_panel_file=panel_path,
    )

    promoter, expr, y = dataset.in_getitem(pro_i=0, cell_i=0)

    assert dataset.expr_dim == 2
    assert promoter.shape == (400, 5)
    assert torch.all(promoter[:, 3] == 1.0)
    assert y.item() == pytest.approx(10.0)
    assert expr.tolist() == pytest.approx([0.0, 5.0])


def test_dataset_masks_target_inside_fixed_input_panel(tiny_data_dir) -> None:
    panel_path = tiny_data_dir / "input_gene_panel_train.txt"
    panel_path.write_text("g0\ng1\ng2\n", encoding="utf-8")
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        input_gene_panel_file=panel_path,
    )

    _promoter, expr, y = dataset.in_getitem(pro_i=0, cell_i=0)

    assert dataset.expr_dim == 3
    assert y.item() == pytest.approx(10.0)
    assert expr.tolist() == pytest.approx([0.0, 0.0, 5.0])


def test_gpu_cached_pair_loader_matches_dataset_fixed_panel_and_masking(tiny_data_dir) -> None:
    panel_path = tiny_data_dir / "input_gene_panel_train.txt"
    panel_path.write_text("g0\ng1\ng2\n", encoding="utf-8")
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        input_gene_panel_file=panel_path,
        log1p_cpm_target=True,
        preencode_promoters=True,
    )
    flat_indices = [0, dataset.C + 1, 2 * dataset.C + 2]
    loader = GpuCachedPairLoader(
        dataset=dataset,
        batch_size=3,
        device=torch.device("cpu"),
        samples_per_epoch=3,
        seed=11,
        drop_last=False,
    )

    promoters, exprs, ys = loader._make_batch(flat_indices)

    expected = [dataset[idx] for idx in flat_indices]
    assert torch.equal(promoters, torch.stack([item[0] for item in expected]))
    assert torch.allclose(exprs, torch.stack([item[1] for item in expected]))
    assert torch.allclose(ys.cpu(), torch.stack([item[2] for item in expected]))


def test_gpu_cached_pair_loader_balanced_sampler_covers_promoters_evenly(tiny_data_dir) -> None:
    dataset = MyDataset(
        promoter_file=tiny_data_dir / "promoter_train.csv",
        scrna_file=tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=True,
    )
    loader = GpuCachedPairLoader(
        dataset=dataset,
        batch_size=2,
        device=torch.device("cpu"),
        samples_per_epoch=dataset.P * 2 + 1,
        seed=13,
        sampler_mode="balanced",
        drop_last=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(13)

    promoter_idx, cell_idx = loader._sample_pair_indices(dataset.P * 2 + 1, generator)
    counts = torch.bincount(promoter_idx.cpu(), minlength=dataset.P)

    assert counts.min().item() == 2
    assert counts.max().item() == 3
    assert int(cell_idx.min()) >= 0
    assert int(cell_idx.max()) < dataset.C
