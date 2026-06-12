from __future__ import annotations

import pytest

from scripts.pretrain_scvi import (
    build_scvi_config,
    default_out_name,
    read_cell_split,
    resolve_cell_split_dir,
    resolve_pretrain_data_path,
    subset_adata_to_cells,
)


pytestmark = pytest.mark.unit


def test_resolve_pretrain_data_path_supports_hqcells_name(tmp_path) -> None:
    path = resolve_pretrain_data_path(tmp_path, "umi_E-MTAB-10519-hqcells")

    assert path == tmp_path / "data" / "umi_E-MTAB-10519-hqcells" / "integrated_data.h5ad"
    assert default_out_name("umi_E-MTAB-10519-hqcells") == "scvi_umi_E-MTAB-10519-hqcells"
    assert default_out_name("emtab") == "E-MTAB10519_VAE"


def test_read_cell_split_and_subset_adata(tiny_data_dir, tiny_adata) -> None:
    (tiny_data_dir / "cell_train.txt").write_text("cell0\ncell2\ncell4\n", encoding="utf-8")

    cells = read_cell_split(tiny_data_dir, "train")
    subset = subset_adata_to_cells(tiny_adata, cells)

    assert cells == ["cell0", "cell2", "cell4"]
    assert subset.n_obs == 3
    assert subset.obs_names.tolist() == ["cell0", "cell2", "cell4"]


def test_subset_adata_to_cells_raises_for_missing_cells(tiny_adata) -> None:
    with pytest.raises(ValueError, match="not found"):
        subset_adata_to_cells(tiny_adata, ["cell0", "missing"])


def test_resolve_cell_split_dir_defaults_to_data_dir(tmp_path) -> None:
    split_dir = resolve_cell_split_dir(tmp_path, "umi_E-MTAB-10519-hqcells", None)

    assert split_dir == tmp_path / "data" / "umi_E-MTAB-10519-hqcells"


def test_build_scvi_config_records_cell_split_metadata() -> None:
    config = build_scvi_config(
        n_genes=13956,
        n_latent=128,
        n_hidden=512,
        n_layers=2,
        dropout_rate=0.1,
        data_name="umi_E-MTAB-10519-hqcells",
        use_cell_split=True,
        cell_split_dir="data/umi_E-MTAB-10519-hqcells",
        train_cell_count=17845,
    )

    assert config["n_input"] == 13956
    assert config["data"] == "umi_E-MTAB-10519-hqcells"
    assert config["use_cell_split"] is True
    assert config["cell_split_dir"] == "data/umi_E-MTAB-10519-hqcells"
    assert config["train_cell_count"] == 17845
