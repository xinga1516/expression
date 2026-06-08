from __future__ import annotations

import pandas as pd
import pytest


pytestmark = pytest.mark.regression


def test_augmented_dataset_keeps_val_and_test_equal_to_original(tmp_path, tiny_promoters_df) -> None:
    original = tmp_path / "original"
    augmented = tmp_path / "augmented"
    original.mkdir()
    augmented.mkdir()

    val = tiny_promoters_df.iloc[:2].copy()
    test = tiny_promoters_df.iloc[2:].copy()
    val.to_csv(original / "promoter_val.csv", index=False)
    test.to_csv(original / "promoter_test.csv", index=False)
    val.to_csv(augmented / "promoter_val.csv", index=False)
    test.to_csv(augmented / "promoter_test.csv", index=False)

    pd.testing.assert_frame_equal(
        pd.read_csv(original / "promoter_val.csv"),
        pd.read_csv(augmented / "promoter_val.csv"),
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(original / "promoter_test.csv"),
        pd.read_csv(augmented / "promoter_test.csv"),
    )
