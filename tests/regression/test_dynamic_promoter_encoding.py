from __future__ import annotations

import pytest
import torch

from src.dataset import MyDataset


pytestmark = pytest.mark.regression


def test_dynamic_encoding_path_does_not_preencode_but_matches_preencoded(tiny_data_dir) -> None:
    dynamic = MyDataset(
        tiny_data_dir / "promoter_train.csv",
        tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=False,
    )
    preencoded = MyDataset(
        tiny_data_dir / "promoter_train.csv",
        tiny_data_dir / "integrated_data.h5ad",
        preencode_promoters=True,
    )

    assert dynamic.promoter_tensor is None
    assert preencoded.promoter_tensor is not None
    for pro_i in range(dynamic.P):
        assert torch.equal(dynamic.get_promoter_tensor(pro_i), preencoded.get_promoter_tensor(pro_i))
