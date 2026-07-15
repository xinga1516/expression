from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.summarize_stage1_bootstrap import (
    bootstrap_paired_delta,
    hierarchical_seed_bootstrap,
    summarize_violin_data,
)


pytestmark = pytest.mark.unit


def test_bootstrap_paired_delta_reports_positive_interval() -> None:
    deltas = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    result = bootstrap_paired_delta(
        deltas=deltas,
        repeats=2_000,
        confidence=0.95,
        rng=np.random.default_rng(7),
    )

    assert result["mean_delta"] == pytest.approx(0.25)
    assert result["mean_ci_low"] > 0
    assert result["mean_ci_high"] >= result["mean_delta"]
    assert result["win_fraction"] == 1.0


def test_hierarchical_seed_bootstrap_uses_equal_seed_weighting() -> None:
    deltas_by_seed = {
        1: np.asarray([0.1, 0.2], dtype=np.float64),
        7: np.asarray([0.3, 0.4], dtype=np.float64),
        42: np.asarray([0.5, 0.6], dtype=np.float64),
    }
    result = hierarchical_seed_bootstrap(
        deltas_by_seed=deltas_by_seed,
        repeats=2_000,
        confidence=0.95,
        rng=np.random.default_rng(11),
    )

    assert result["seed_count"] == 3
    assert result["equal_seed_mean_delta"] == pytest.approx(0.35)
    assert result["mean_ci_low"] > 0
    assert result["all_seed_means_positive"] is True



def test_summarize_violin_data_selects_both_extremes() -> None:
    frame = pd.DataFrame(
        {
            "level": ["per_gene"] * 6,
            "model_group": ["promoter"] * 6,
            "model_label": ["Real promoter"] * 6,
            "seed": [1, 1, 7, 7, 42, 42],
            "sample_id": [f"g{i}" for i in range(6)],
            "pearson_r": [-0.4, -0.1, 0.0, 0.2, 0.5, 0.8],
        }
    )

    stats, extremes = summarize_violin_data(frame, extreme_count=2)

    assert len(stats) == 1
    assert int(stats.iloc[0]["n_valid"]) == 6
    assert int(stats.iloc[0]["n_extreme_points"]) == 4
    assert set(extremes["sample_id"]) == {"g0", "g1", "g4", "g5"}
    assert set(extremes["extreme_side"]) == {"low", "high"}


def test_summarize_violin_data_keeps_fifty_extreme_points_per_distribution() -> None:
    frame = pd.DataFrame(
        {
            "level": ["per_cell"] * 60,
            "model_group": ["promoter"] * 60,
            "model_label": ["Real promoter"] * 60,
            "seed": [7] * 60,
            "sample_id": [f"c{i}" for i in range(60)],
            "pearson_r": np.linspace(-0.5, 0.8, 60),
        }
    )

    stats, extremes = summarize_violin_data(frame, extreme_count=25)

    assert int(stats.iloc[0]["n_extreme_points"]) == 50
    assert len(extremes) == 50
    assert set(extremes["extreme_side"]) == {"low", "high"}
