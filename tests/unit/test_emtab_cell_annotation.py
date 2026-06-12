from __future__ import annotations

import pandas as pd

from scripts.annotate_emtab_cells import (
    annotate_cell_ids,
    build_match_summary,
    load_sdrf_sample_extract_map,
    normalize_barcode,
    write_cell_splits,
)


def test_annotate_cell_ids_maps_sdrf_and_loom_annotations() -> None:
    metadata = annotate_cell_ids(
        ["SAMEA1-AAAC-1", "SAMEA2-TTGG-1"],
        sample_to_extract={"SAMEA1": "extract_a", "SAMEA2": "extract_b"},
        loom_annotations={("extract_a", "AAAC"): ("fat body", "fat cell")},
        sample_to_tissue={"SAMEA1": "Head"},
    )

    assert normalize_barcode("AAAC-1") == "AAAC"
    assert metadata.loc[0, "annotation"] == "fat body"
    assert metadata.loc[0, "annotation_broad"] == "fat cell"
    assert metadata.loc[0, "is_unknown"] == False
    assert metadata.loc[1, "annotation"] == "unknown"
    assert metadata.loc[1, "is_unknown"] == True


def test_load_sdrf_sample_extract_map_prefers_biosd_sample(tmp_path) -> None:
    sdrf_path = tmp_path / "test.sdrf.txt"
    sdrf_path.write_text(
        "Source Name\tComment[BioSD_SAMPLE]\tExtract Name\n"
        "source_a\tSAMEA1\textract_a\n",
        encoding="utf-8",
    )

    mapping = load_sdrf_sample_extract_map(sdrf_path)

    assert mapping == {"SAMEA1": "extract_a"}


def test_match_summary_contains_cell_type_tissue_and_sample_rates() -> None:
    metadata = pd.DataFrame({
        "cell_id": ["c1", "c2", "c3"],
        "sample_id": ["s1", "s1", "s2"],
        "annotation": ["a", "unknown", "b"],
        "annotation_broad": ["A", "unknown", "B"],
        "tissue": ["head", "head", "gut"],
        "is_unknown": [False, True, False],
    })

    summary = build_match_summary(metadata)

    levels = set(summary["summary_level"])
    assert {"overall", "cell_type", "cell_type_broad", "tissue", "sample"}.issubset(levels)
    overall = summary[summary["summary_level"] == "overall"].iloc[0]
    assert overall["matched_cells"] == 2
    assert overall["unknown_cells"] == 1
    assert overall["match_rate"] == 2 / 3


def test_write_cell_splits_drops_unknown_and_holds_out_labels(tmp_path) -> None:
    metadata = pd.DataFrame({
        "cell_id": ["c1", "c2", "c3", "c4", "c5"],
        "annotation": ["a", "b", "c", "unknown", "unannotated"],
        "annotation_broad": ["A", "B", "C", "unknown", "unannotated"],
        "is_unknown": [False, False, False, True, False],
    })

    split_summary = write_cell_splits(
        metadata,
        output_dir=tmp_path,
        split_column="annotation_broad",
        val_label="A",
        test_label="B",
    )

    assert (tmp_path / "cell_train.txt").read_text(encoding="utf-8").strip() == "c3"
    assert (tmp_path / "cell_val.txt").read_text(encoding="utf-8").strip() == "c1"
    assert (tmp_path / "cell_test.txt").read_text(encoding="utf-8").strip() == "c2"
    assert set((tmp_path / "cell_unknown.txt").read_text(encoding="utf-8").splitlines()) == {"c4", "c5"}
    assert set(split_summary["split"]) == {"train", "val", "test", "unknown"}
