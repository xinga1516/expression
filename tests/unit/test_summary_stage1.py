from __future__ import annotations

import numpy as np
import pytest

from scripts.summary_stage1 import (
    finalize_interaction_state,
    make_interaction_state,
    summarize_interactions,
    update_interaction_state,
    validate_matched_protocol,
)


def test_interaction_statistics_use_cellwise_effect_and_expression_residual() -> None:
    state = make_interaction_state(2)
    update_interaction_state(
        state=state,
        gene_indices=np.asarray([0, 0, 0, 1, 1, 1]),
        promoter_predictions=np.asarray([2.0, 4.0, 6.0, 3.0, 4.0, 5.0]),
        control_predictions=np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        expression_predictions=np.asarray([9.0, 8.0, 7.0, 4.0, 4.0, 4.0]),
        targets=np.asarray([10.0, 10.0, 10.0, 5.0, 6.0, 7.0]),
    )

    result = finalize_interaction_state(state, ["g0", "g1"]).set_index("gene_id")

    assert result.loc["g0", "interaction_variance"] == pytest.approx(2.0 / 3.0)
    assert result.loc["g0", "effect_residual_correlation"] == pytest.approx(1.0)
    assert result.loc["g1", "interaction_variance"] == pytest.approx(0.0)
    assert np.isnan(result.loc["g1", "effect_residual_correlation"])
    summary = summarize_interactions(result.reset_index())
    assert summary["num_genes"] == 2
    assert summary["num_genes_with_defined_residual_correlation"] == 1
    assert summary["positive_effect_residual_correlation_fraction"] == pytest.approx(1.0)


def test_validate_matched_protocol_rejects_different_seed() -> None:
    base = {
        "seed": 7,
        "git_hash": "abc",
        "train_promoter_file": "/tmp/train.csv",
        "val_promoter_file": "/tmp/val.csv",
        "scrna_file": "/tmp/data.h5ad",
        "expr_dim": 4,
        "use_cell_split": True,
        "cell_split_dir": "/tmp/splits",
        "input_gene_panel_file": "/tmp/panel.txt",
        "expression_layer": "logcpm",
        "expression_transform": "none",
        "target_count_layer": "counts",
        "target_value_layer": "logcpm",
        "target_transform": "none",
        "sequence_length": 400,
        "checkpoint_metric": "val_rmse",
        "loss_type": "mse",
    }
    configs = {
        "promoter": dict(base),
        "expression": {**base, "seed": 42},
    }

    with pytest.raises(ValueError, match="expression.seed"):
        validate_matched_protocol(configs)
