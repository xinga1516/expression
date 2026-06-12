from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNKNOWN_LABEL = "unknown"
DEFAULT_EXCLUDED_LABELS = ("unknown", "unannotated", "artefact", "artifact", "none", "nan", "")


def decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ])
    return values.astype(str)


def normalize_barcode(barcode: str) -> str:
    return str(barcode).split("-")[0]


def split_emtab_cell_id(cell_id: str) -> tuple[str, str]:
    sample_id, sep, barcode = str(cell_id).partition("-")
    if not sep:
        raise ValueError(f"Cell id does not contain SAMPLE-barcode format: {cell_id}")
    return sample_id, normalize_barcode(barcode)


def load_sdrf_sample_extract_map(sdrf_path: Path) -> dict[str, str]:
    sdrf = pd.read_csv(sdrf_path, sep="\t", dtype=str)
    sample_column = next(
        (column for column in ("Comment[BioSD_SAMPLE]", "BioSD_SAMPLE", "Source Name") if column in sdrf.columns),
        None,
    )
    if sample_column is None:
        raise ValueError("SDRF missing BioSD sample column")
    required = {sample_column, "Extract Name"}
    missing = required - set(sdrf.columns)
    if missing:
        raise ValueError(f"SDRF missing columns: {sorted(missing)}")
    mapping = (
        sdrf[[sample_column, "Extract Name"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={sample_column: "sample_id", "Extract Name": "extract_name"})
    )
    duplicated = mapping[mapping.duplicated("sample_id", keep=False)]
    if not duplicated.empty:
        bad = duplicated["sample_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Ambiguous SDRF Source Name -> Extract Name mapping, e.g. {bad}")
    return dict(zip(mapping["sample_id"], mapping["extract_name"]))


def load_sample_tissue_map(tissue_path: Path) -> dict[str, str]:
    if not tissue_path.exists():
        return {}
    with open(tissue_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(key): str(value) for key, value in raw.items()}


def _read_loom_col_attr(handle: h5py.File, name: str) -> np.ndarray:
    path = f"col_attrs/{name}"
    if path not in handle:
        raise ValueError(f"Loom file missing {path}")
    return decode_array(handle[path][:])


def load_loom_annotation_map(loom_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    with h5py.File(loom_path, "r") as handle:
        sample_ids = _read_loom_col_attr(handle, "sample_id")
        barcodes = np.asarray([normalize_barcode(value) for value in _read_loom_col_attr(handle, "Barcode")])
        annotations = _read_loom_col_attr(handle, "annotation")
        broad = _read_loom_col_attr(handle, "annotation_broad")

    result: dict[tuple[str, str], tuple[str, str]] = {}
    for sample_id, barcode, annotation, annotation_broad in zip(sample_ids, barcodes, annotations, broad):
        result[(str(sample_id), str(barcode))] = (str(annotation), str(annotation_broad))
    return result


def annotate_cell_ids(
    cell_ids: Iterable[str],
    sample_to_extract: dict[str, str],
    loom_annotations: dict[tuple[str, str], tuple[str, str]],
    sample_to_tissue: dict[str, str] | None = None,
) -> pd.DataFrame:
    sample_to_tissue = sample_to_tissue or {}
    rows: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        sample_id, barcode = split_emtab_cell_id(str(cell_id))
        extract_name = sample_to_extract.get(sample_id, "")
        annotation, annotation_broad = loom_annotations.get((extract_name, barcode), (UNKNOWN_LABEL, UNKNOWN_LABEL))
        is_unknown = annotation == UNKNOWN_LABEL or annotation_broad == UNKNOWN_LABEL
        rows.append({
            "cell_id": str(cell_id),
            "sample_id": sample_id,
            "barcode": barcode,
            "extract_name": extract_name,
            "tissue": sample_to_tissue.get(sample_id, UNKNOWN_LABEL),
            "annotation": annotation,
            "annotation_broad": annotation_broad,
            "is_unknown": bool(is_unknown),
        })
    return pd.DataFrame(rows)


def build_match_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_rows(level: str, frame: pd.DataFrame, label_column: str | None = None) -> None:
        if label_column is None:
            groups = [(level, frame)]
        else:
            groups = list(frame.groupby(label_column, dropna=False, sort=False))
        for label, group in groups:
            total = int(len(group))
            unknown = int(group["is_unknown"].sum())
            matched = total - unknown
            rows.append({
                "summary_level": level,
                "label": str(label),
                "total_cells": total,
                "matched_cells": matched,
                "unknown_cells": unknown,
                "match_rate": matched / total if total else 0.0,
            })

    add_rows("overall", metadata)
    add_rows("cell_type", metadata, "annotation")
    add_rows("cell_type_broad", metadata, "annotation_broad")
    add_rows("tissue", metadata, "tissue")
    add_rows("sample", metadata, "sample_id")
    return pd.DataFrame(rows)


def valid_split_mask(metadata: pd.DataFrame, split_column: str, excluded_labels: Iterable[str]) -> pd.Series:
    excluded = {label.lower() for label in excluded_labels}
    labels = metadata[split_column].fillna(UNKNOWN_LABEL).astype(str).str.lower()
    return (~metadata["is_unknown"]) & (~labels.isin(excluded))


def choose_holdout_labels(
    metadata: pd.DataFrame,
    split_column: str,
    valid_mask: pd.Series,
    val_label: str | None,
    test_label: str | None,
) -> tuple[str, str]:
    counts = metadata.loc[valid_mask, split_column].value_counts()
    if val_label is None or test_label is None:
        if len(counts) < 2:
            raise ValueError(f"Need at least two valid {split_column} labels for val/test holdout")
        defaults = counts.index.tolist()
        val_label = val_label or str(defaults[0])
        next_labels = [str(label) for label in defaults if str(label) != val_label]
        test_label = test_label or next_labels[0]
    if val_label == test_label:
        raise ValueError("Validation and test cell labels must be different")
    return val_label, test_label


def write_cell_splits(
    metadata: pd.DataFrame,
    output_dir: Path,
    split_column: str = "annotation_broad",
    val_label: str | None = None,
    test_label: str | None = None,
    excluded_labels: Iterable[str] = DEFAULT_EXCLUDED_LABELS,
) -> pd.DataFrame:
    if split_column not in metadata.columns:
        raise ValueError(f"Unknown split column: {split_column}")
    valid_mask = valid_split_mask(metadata, split_column, excluded_labels)
    val_label, test_label = choose_holdout_labels(metadata, split_column, valid_mask, val_label, test_label)

    labels = metadata[split_column].astype(str)
    val_mask = valid_mask & (labels == val_label)
    test_mask = valid_mask & (labels == test_label)
    train_mask = valid_mask & (~val_mask) & (~test_mask)
    unknown_mask = ~valid_mask

    split_masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
        "unknown": unknown_mask,
    }
    rows: list[dict[str, Any]] = []
    for split, mask in split_masks.items():
        cell_ids = metadata.loc[mask, "cell_id"].astype(str)
        (output_dir / f"cell_{split}.txt").write_text("\n".join(cell_ids.tolist()) + "\n", encoding="utf-8")
        rows.append({
            "split": split,
            "cell_count": int(mask.sum()),
            "fraction_of_all_cells": float(mask.mean()),
            "split_column": split_column,
            "heldout_label": val_label if split == "val" else test_label if split == "test" else "",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "cell_split_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate E-MTAB hqcells with loom cell types and create cell splits.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "umi_E-MTAB-10519-hqcells")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "umi_E-MTAB-10519-raw")
    parser.add_argument("--loom", type=Path, default=PROJECT_ROOT / "data" / "raw" / "s_fca_biohub_all_wo_blood_10x.loom")
    parser.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "data" / "emtab_hqcells_annotation_match_summary.csv")
    parser.add_argument("--split-column", type=str, default="annotation_broad", choices=["annotation", "annotation_broad"])
    parser.add_argument("--val-cell-type", type=str, default=None, help="Cell type label to hold out as validation.")
    parser.add_argument("--test-cell-type", type=str, default=None, help="Cell type label to hold out as test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scrna_path = args.data_dir / "integrated_data.h5ad"
    sdrf_path = args.raw_dir / "E-MTAB-10519.sdrf.txt"
    tissue_path = args.raw_dir / "sample_tissue.json"

    adata = sc.read(scrna_path, sparse=True)
    sample_to_extract = load_sdrf_sample_extract_map(sdrf_path)
    sample_to_tissue = load_sample_tissue_map(tissue_path)
    loom_annotations = load_loom_annotation_map(args.loom)

    metadata = annotate_cell_ids(
        adata.obs_names.astype(str),
        sample_to_extract=sample_to_extract,
        loom_annotations=loom_annotations,
        sample_to_tissue=sample_to_tissue,
    )
    args.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.data_dir / "cell_annotation.csv"
    metadata.to_csv(metadata_path, index=False)

    summary = build_match_summary(metadata)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_path, index=False)

    split_summary = write_cell_splits(
        metadata,
        output_dir=args.data_dir,
        split_column=args.split_column,
        val_label=args.val_cell_type,
        test_label=args.test_cell_type,
    )

    overall = summary.loc[summary["summary_level"] == "overall"].iloc[0]
    print(
        f"matched={int(overall.matched_cells)}/{int(overall.total_cells)} "
        f"({overall.match_rate:.2%}); unknown={int(overall.unknown_cells)}"
    )
    print(f"metadata: {metadata_path}")
    print(f"summary: {args.summary_path}")
    print("splits:")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
