from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def log(message: str) -> None:
    print(message, flush=True)


def load_fasta(path: Path) -> dict[str, str]:
    genome: dict[str, list[str]] = {}
    current_name: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_name = line[1:].split()[0]
                genome[current_name] = []
            elif current_name is not None:
                genome[current_name].append(line.upper())
    return {name: "".join(parts) for name, parts in genome.items()}


def reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(table)[::-1].upper()


def oriented_sequence(chrom_sequence: str, start0: int, end0: int, strand: str) -> str:
    sequence = chrom_sequence[int(start0) : int(end0)].upper()
    if str(strand) == "-":
        return reverse_complement(sequence)
    return sequence


def centered_window(start0: int, end0: int, target_length: int) -> tuple[int, int]:
    center = (int(start0) + int(end0)) // 2
    new_start = center - int(target_length) // 2
    return new_start, new_start + int(target_length)


def extract_centered(
    genome: dict[str, str],
    contig: str,
    start0: int,
    end0: int,
    strand: str,
    target_length: int,
) -> tuple[str, int, int, str]:
    chrom_sequence = genome.get(contig)
    new_start, new_end = centered_window(start0, end0, target_length)
    if chrom_sequence is None:
        return "", new_start, new_end, "missing_contig"
    if new_start < 0 or new_end > len(chrom_sequence):
        return "", new_start, new_end, "out_of_bounds"
    sequence = oriented_sequence(chrom_sequence, new_start, new_end, strand)
    if len(sequence) != target_length:
        return sequence, new_start, new_end, "bad_length"
    if "N" in sequence:
        return sequence, new_start, new_end, "contains_N"
    return sequence, new_start, new_end, "ok"


def extract_shifted(
    genome: dict[str, str],
    contig: str,
    start0: int,
    target_length: int,
    strand: str,
    shift_bp: int,
) -> tuple[str, int, int, str]:
    chrom_sequence = genome.get(contig)
    shifted_start = int(start0) + int(shift_bp)
    shifted_end = shifted_start + int(target_length)
    if chrom_sequence is None:
        return "", shifted_start, shifted_end, "missing_contig"
    if shifted_start < 0 or shifted_end > len(chrom_sequence):
        return "", shifted_start, shifted_end, "out_of_bounds"
    sequence = oriented_sequence(chrom_sequence, shifted_start, shifted_end, strand)
    if len(sequence) != target_length:
        return sequence, shifted_start, shifted_end, "bad_length"
    if "N" in sequence:
        return sequence, shifted_start, shifted_end, "contains_N"
    return sequence, shifted_start, shifted_end, "shifted"


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    shutil.copy2(source, destination)
    return True


def load_source_promoters(source_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = source_dir / f"promoter_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)
        log(f"[StageReuse] loaded {split}: rows={len(frame)}")
    promoters = pd.concat(frames, ignore_index=True)
    return promoters.drop_duplicates(subset=["gene_id", "split"], keep="first").reset_index(drop=True)


def reextract_row(row: Any, genome: dict[str, str], sequence_window_length: int, positive_shift_bp: int) -> dict[str, Any]:
    contig = str(getattr(row, "chrom"))
    strand = str(getattr(row, "strand", "+"))
    sequence, start, end, sequence_status = extract_centered(
        genome,
        contig,
        int(getattr(row, "start")),
        int(getattr(row, "end")),
        strand,
        sequence_window_length,
    )
    if sequence_status != "ok":
        sequence = str(getattr(row, "sequence"))
        start = int(getattr(row, "start"))
        end = int(getattr(row, "end"))

    control_contig = str(getattr(row, "control_chrom", contig))
    control_strand = str(getattr(row, "control_strand", strand))
    control_start0 = int(getattr(row, "control_start"))
    control_end0 = int(getattr(row, "control_end"))
    control_sequence, control_start, control_end, control_status = extract_centered(
        genome,
        control_contig,
        control_start0,
        control_end0,
        control_strand,
        sequence_window_length,
    )
    if control_status != "ok":
        control_sequence = str(getattr(row, "control_sequence"))
        control_start = control_start0
        control_end = control_end0

    positive_sequence, positive_start, positive_end, positive_status = extract_shifted(
        genome,
        contig,
        start,
        sequence_window_length,
        strand,
        positive_shift_bp,
    )
    effective_shift = int(positive_shift_bp)
    if positive_status != "shifted":
        positive_sequence = sequence
        positive_start = start
        positive_end = end
        effective_shift = 0
        positive_status = f"{positive_status}_fallback_center"

    record = row._asdict()
    record.update(
        {
            "start": start,
            "end": end,
            "sequence": sequence,
            "length": int(sequence_window_length),
            "sequence_status": sequence_status,
            "control_start": control_start,
            "control_end": control_end,
            "control_sequence": control_sequence,
            "control_length": len(control_sequence),
            "control_reextract_status": control_status,
            "positive_sequence": positive_sequence,
            "positive_shift_bp": effective_shift,
            "positive_start": positive_start,
            "positive_end": positive_end,
            "positive_status": positive_status,
        }
    )
    return record


def build_assets(args: argparse.Namespace) -> None:
    source_dir = PROJECT_ROOT / "data" / args.source_data
    output_dir = PROJECT_ROOT / "data" / args.output_data
    genome_fasta = Path(args.genome_fasta)
    if not genome_fasta.is_absolute():
        genome_fasta = PROJECT_ROOT / genome_fasta
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[StageReuse] loading genome: {genome_fasta}")
    genome = load_fasta(genome_fasta)
    log(f"[StageReuse] loaded contigs={len(genome)} total_bp={sum(len(seq) for seq in genome.values()):,}")

    promoters = load_source_promoters(source_dir)
    log(f"[StageReuse] total rows after dedupe={len(promoters)}")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(promoters.itertuples(index=False), start=1):
        rows.append(
            reextract_row(
                row=row,
                genome=genome,
                sequence_window_length=args.sequence_window_length,
                positive_shift_bp=args.positive_shift_bp,
            )
        )
        if index % 1000 == 0:
            log(f"[StageReuse] processed {index}/{len(promoters)}")

    output = pd.DataFrame(rows)
    output.to_csv(output_dir / "promoter_windows.tsv", sep="\t", index=False)
    output.to_csv(output_dir / "control_windows.tsv", sep="\t", index=False)
    for split in ("train", "val", "test"):
        split_output = output.loc[output["split"] == split].copy()
        split_output.to_csv(output_dir / f"promoter_{split}.csv", index=False)
        log(f"[StageReuse] wrote {split}: rows={len(split_output)}")

    copied_files: dict[str, bool] = {}
    for name in (
        "genes.tsv",
        "gene_splits.tsv",
        "cells.tsv",
        "frozen_eval_cells.tsv",
        "cell_train.txt",
        "cell_val.txt",
        "cell_test.txt",
        "input_gene_panel_train.txt",
    ):
        copied_files[name] = copy_if_exists(source_dir / name, output_dir / name)
    if (source_dir / "integrated_data.h5ad").exists() and not (output_dir / "integrated_data.h5ad").exists():
        (output_dir / "integrated_data.h5ad").symlink_to(Path("../") / args.source_data / "integrated_data.h5ad")

    report = {
        "source_data": args.source_data,
        "output_data": args.output_data,
        "sequence_window_length": int(args.sequence_window_length),
        "positive_shift_bp": int(args.positive_shift_bp),
        "n_rows": int(len(output)),
        "split_counts": output["split"].value_counts().to_dict(),
        "sequence_status_counts": output["sequence_status"].value_counts().to_dict(),
        "control_reextract_status_counts": output["control_reextract_status"].value_counts().to_dict(),
        "positive_status_counts": output["positive_status"].value_counts().to_dict(),
        "sequence_length_min": int(output["sequence"].astype(str).str.len().min()),
        "sequence_length_max": int(output["sequence"].astype(str).str.len().max()),
        "control_length_min": int(output["control_sequence"].astype(str).str.len().min()),
        "control_length_max": int(output["control_sequence"].astype(str).str.len().max()),
        "reused_gene_splits": bool(copied_files.get("gene_splits.tsv")),
        "reused_cell_splits": all(copied_files.get(name, False) for name in ("cell_train.txt", "cell_val.txt", "cell_test.txt")),
        "reused_input_gene_panel": bool(copied_files.get("input_gene_panel_train.txt")),
        "copied_files": copied_files,
    }
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build wider sequence assets while reusing an existing gene/cell split and input panel."
    )
    parser.add_argument("--source-data", type=str, default="promoter_stage1_v1")
    parser.add_argument("--output-data", type=str, default="promoter_stage2_v1")
    parser.add_argument("--genome-fasta", type=str, default="data/raw/dmel-all-chromosome-r6.54.fasta")
    parser.add_argument("--sequence-window-length", type=int, default=440)
    parser.add_argument("--positive-shift-bp", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    build_assets(parse_args())
