from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from Bio import SeqIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_promoter_stage1_assets import (
    Interval,
    build_interval_index,
    choose_gene_splits,
    file_sha256,
    gc_fraction,
    overlaps_any,
    parse_gtf_attributes,
    read_gtf_gene_table,
    read_source_promoters,
    reverse_complement,
    resolve_cell_ids,
    select_input_gene_panel,
    write_cell_panels,
)


def read_stop_landmarks(gtf_path: Path) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    with gtf_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            contig, _source, feature, start, end, _score, strand, _frame, attr_text = parts
            attrs = parse_gtf_attributes(attr_text)
            gene_id = attrs.get("gene_id")
            if not gene_id:
                continue
            record = records.setdefault(
                gene_id,
                {
                    "gene_id": gene_id,
                    "contig": contig,
                    "strand": strand,
                    "stop_intervals": [],
                    "coding_intervals": [],
                },
            )
            if feature == "stop_codon":
                record["stop_intervals"].append((int(start) - 1, int(end)))
            if feature in {"CDS", "stop_codon"}:
                record["coding_intervals"].append((int(start) - 1, int(end)))

    rows: list[dict[str, Any]] = []
    for record in records.values():
        stop_intervals = list(record["stop_intervals"])
        if not stop_intervals:
            continue
        strand = str(record["strand"])
        if strand == "+":
            stop_start0, stop_end0 = max(stop_intervals, key=lambda item: item[1])
            anchor0 = stop_end0
        else:
            stop_start0, stop_end0 = min(stop_intervals, key=lambda item: item[0])
            anchor0 = stop_start0
        rows.append(
            {
                "gene_id": record["gene_id"],
                "contig": record["contig"],
                "strand": strand,
                "stop_start0": int(stop_start0),
                "stop_end0": int(stop_end0),
                "stop_anchor0": int(anchor0),
                "coding_intervals": list(record["coding_intervals"]),
            }
        )
    return pd.DataFrame(rows)


def interval_overlaps_any(intervals: list[tuple[int, int]], start0: int, end0: int) -> bool:
    for interval_start, interval_end in intervals:
        if interval_end > start0 and interval_start < end0:
            return True
    return False


def extract_utr_windows(
    genes: pd.DataFrame,
    stop_landmarks: pd.DataFrame,
    genome: dict[str, Any],
    sequence_length: int,
) -> pd.DataFrame:
    stop_by_gene = stop_landmarks.set_index("gene_id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for row in genes.itertuples(index=False):
        gene_id = str(row.gene_id)
        landmark = stop_by_gene.get(gene_id)
        base = {
            "gene_id": gene_id,
            "gene_class": getattr(row, "gene_class", ""),
            "split": getattr(row, "split", ""),
            "split_strategy": getattr(row, "split_strategy", ""),
        }
        if landmark is None:
            rows.append({**base, "utr_status": "missing_stop_codon"})
            continue
        contig = str(landmark["contig"])
        strand = str(landmark["strand"])
        if contig not in genome:
            rows.append({**base, "utr_status": "missing_contig", "utr_chrom": contig, "utr_strand": strand})
            continue
        chrom_seq = str(genome[contig].seq).upper()
        if strand == "+":
            start0 = int(landmark["stop_end0"])
            end0 = start0 + sequence_length
            genomic_seq = chrom_seq[start0:end0]
            sequence = genomic_seq
        else:
            end0 = int(landmark["stop_start0"])
            start0 = end0 - sequence_length
            genomic_seq = chrom_seq[start0:end0] if start0 >= 0 else ""
            sequence = reverse_complement(genomic_seq) if genomic_seq else ""
        if start0 < 0 or end0 > len(chrom_seq) or len(sequence) != sequence_length:
            rows.append(
                {
                    **base,
                    "utr_status": "out_of_bounds",
                    "utr_chrom": contig,
                    "utr_start": start0,
                    "utr_end": end0,
                    "utr_strand": strand,
                    "stop_start0": int(landmark["stop_start0"]),
                    "stop_end0": int(landmark["stop_end0"]),
                    "stop_anchor0": int(landmark["stop_anchor0"]),
                }
            )
            continue
        if "N" in sequence:
            status = "contains_N"
        elif interval_overlaps_any(list(landmark["coding_intervals"]), start0, end0):
            status = "overlaps_cds_or_stop"
        else:
            status = "extracted"
        rows.append(
            {
                **base,
                "utr_id": f"{contig}:{start0}-{end0}:{strand}",
                "utr_chrom": contig,
                "utr_start": start0,
                "utr_end": end0,
                "utr_strand": strand,
                "stop_start0": int(landmark["stop_start0"]),
                "stop_end0": int(landmark["stop_end0"]),
                "stop_anchor0": int(landmark["stop_anchor0"]),
                "utr_sequence": sequence,
                "utr_gc": gc_fraction(sequence),
                "utr_length": len(sequence),
                "utr_status": status,
            }
        )
    return pd.DataFrame(rows)


def find_utr_control(
    utr_row: pd.Series,
    genome: dict[str, Any],
    intervals_by_contig: dict[str, list[Interval]],
    rng: np.random.Generator,
    attempts: int,
) -> dict[str, Any]:
    contig = str(utr_row["utr_chrom"])
    if contig not in genome:
        return {"utr_control_status": "missing_contig"}
    chrom_seq = str(genome[contig].seq).upper()
    length = int(utr_row["utr_length"])
    if len(chrom_seq) < length:
        return {"utr_control_status": "contig_too_short"}
    target_gc = gc_fraction(str(utr_row["utr_sequence"]))
    intervals = intervals_by_contig.get(contig, [])
    best: dict[str, Any] | None = None
    for _ in range(attempts):
        start0 = int(rng.integers(0, len(chrom_seq) - length + 1))
        end0 = start0 + length
        if overlaps_any(intervals, start0, end0):
            continue
        seq = chrom_seq[start0:end0]
        if "N" in seq:
            continue
        seq_gc = gc_fraction(seq)
        gc_diff = abs(seq_gc - target_gc)
        if best is None or gc_diff < float(best["utr_control_gc_diff"]):
            best = {
                "utr_control_id": f"{contig}:{start0}-{end0}",
                "utr_control_kind": "matched_downstream_intergenic",
                "utr_control_chrom": contig,
                "utr_control_start": start0,
                "utr_control_end": end0,
                "utr_control_strand": "+",
                "utr_control_sequence": seq,
                "utr_control_gc": seq_gc,
                "utr_control_gc_diff": gc_diff,
                "utr_control_status": "matched",
            }
    if best is None:
        return {"utr_control_status": "no_nonoverlap_candidate"}
    if str(utr_row.get("utr_strand", "+")) == "-":
        best["utr_control_strand"] = "-"
        best["utr_control_sequence"] = reverse_complement(str(best["utr_control_sequence"]))
    return best


def add_utr_controls(
    utr_windows: pd.DataFrame,
    genome: dict[str, Any],
    gtf_genes: pd.DataFrame,
    attempts: int,
    seed: int,
) -> pd.DataFrame:
    blocker_windows = utr_windows.loc[utr_windows["utr_status"] == "extracted"].rename(
        columns={"utr_chrom": "chrom", "utr_start": "start", "utr_end": "end"}
    )
    intervals_by_contig = build_interval_index(gtf_genes, blocker_windows)
    rng = np.random.default_rng(seed)
    controls = [
        find_utr_control(row, genome, intervals_by_contig, rng, attempts)
        for _, row in utr_windows.iterrows()
    ]
    return pd.concat([utr_windows.reset_index(drop=True), pd.DataFrame(controls)], axis=1)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.write_bytes(src.read_bytes())
        return True
    return False


def prepare_gene_splits_and_panels(
    args: argparse.Namespace,
    promoter_data_dir: Path,
    output_dir: Path,
    gtf_genes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_path = promoter_data_dir / "gene_splits.tsv"
    if split_path.exists():
        gene_splits = pd.read_csv(split_path, sep="\t")
        copied = {}
        for name in [
            "input_gene_panel_train.txt",
            "cell_train.txt",
            "cell_val.txt",
            "cell_test.txt",
            "cells.tsv",
            "frozen_eval_cells.tsv",
        ]:
            copied[name] = copy_if_exists(promoter_data_dir / name, output_dir / name)
        return gene_splits, {"split_source": str(split_path), "copied_panels": copied}

    source_data_dir = PROJECT_ROOT / "data" / args.source_data
    adata = sc.read_h5ad(source_data_dir / "integrated_data.h5ad")
    source_promoters = read_source_promoters(source_data_dir)
    var_gene_ids = set(adata.var["gene_id"].astype(str))
    promoter_gene_ids = set(source_promoters["gene_id"].astype(str))
    eligible_ids = set(gtf_genes.loc[gtf_genes["is_stage1_promoter_eligible"], "gene_id"].astype(str))
    usable_gene_ids = var_gene_ids & promoter_gene_ids & eligible_ids
    included = gtf_genes.loc[gtf_genes["gene_id"].isin(usable_gene_ids)].copy()
    gene_splits = choose_gene_splits(
        included,
        val_contig=args.val_contig,
        test_contig=args.test_contig,
        min_contig_genes=args.min_contig_genes,
        seed=args.seed,
    )
    gene_splits[["gene_id", "split", "split_strategy", "contig", "strand", "tss", "gene_class"]].to_csv(
        output_dir / "gene_splits.tsv",
        sep="\t",
        index=False,
    )
    cells = write_cell_panels(
        source_data_dir=source_data_dir,
        output_dir=output_dir,
        adata=adata,
        max_eval_cells=args.max_eval_cells,
        seed=args.seed,
    )
    train_genes = gene_splits.loc[gene_splits["split"] == "train", "gene_id"].astype(str).tolist()
    train_cells = resolve_cell_ids(source_data_dir, adata, "train")
    input_panel = select_input_gene_panel(
        adata,
        train_genes,
        max_genes=args.input_gene_panel_size,
        train_cell_ids=train_cells,
        method=args.input_gene_panel_method,
        hvg_flavor=args.hvg_flavor,
    )
    (output_dir / "input_gene_panel_train.txt").write_text("\n".join(input_panel) + "\n", encoding="utf-8")
    return gene_splits, {
        "split_source": "generated_from_source_data",
        "source_data": args.source_data,
        "n_expression_genes": int(len(var_gene_ids)),
        "n_source_promoter_genes": int(len(promoter_gene_ids)),
        "n_stage1_eligible_genes": int(len(eligible_ids)),
        "input_gene_panel_size": int(len(input_panel)),
        "cell_panel_counts": cells["panel"].value_counts().to_dict(),
    }


def build_assets(args: argparse.Namespace) -> None:
    promoter_data_dir = PROJECT_ROOT / "data" / args.promoter_data
    output_dir = PROJECT_ROOT / "data" / args.output_data
    output_dir.mkdir(parents=True, exist_ok=True)

    gtf_path = Path(args.gtf)
    genome_fasta = Path(args.genome_fasta)
    if not gtf_path.is_absolute():
        gtf_path = PROJECT_ROOT / gtf_path
    if not genome_fasta.is_absolute():
        genome_fasta = PROJECT_ROOT / genome_fasta

    genome = SeqIO.to_dict(SeqIO.parse(str(genome_fasta), "fasta"))
    gtf_genes = read_gtf_gene_table(gtf_path)
    gene_splits, split_report = prepare_gene_splits_and_panels(args, promoter_data_dir, output_dir, gtf_genes)
    stop_landmarks = read_stop_landmarks(gtf_path)
    utr_windows = extract_utr_windows(
        gene_splits,
        stop_landmarks=stop_landmarks,
        genome=genome,
        sequence_length=args.sequence_length,
    )
    utr_windows = add_utr_controls(
        utr_windows.loc[utr_windows["utr_status"] == "extracted"].copy(),
        genome=genome,
        gtf_genes=gtf_genes,
        attempts=args.control_attempts,
        seed=args.seed,
    )
    final = utr_windows.loc[utr_windows["utr_control_status"] == "matched"].copy()
    final["sequence"] = final["utr_sequence"]
    final["control_sequence"] = final["utr_control_sequence"]
    final["chrom"] = final["utr_chrom"]
    final["start"] = final["utr_start"]
    final["end"] = final["utr_end"]
    final["strand"] = final["utr_strand"]

    final.to_csv(output_dir / "utr_windows.tsv", sep="\t", index=False)
    final.to_csv(output_dir / "control_windows.tsv", sep="\t", index=False)
    split_cols = ["gene_id", "split", "split_strategy", "utr_chrom", "utr_strand", "stop_anchor0", "gene_class"]
    final[split_cols].rename(columns={"utr_chrom": "contig", "utr_strand": "strand"}).to_csv(
        output_dir / "gene_splits.tsv",
        sep="\t",
        index=False,
    )
    for split in ("train", "val", "test"):
        final.loc[final["split"] == split].to_csv(output_dir / f"promoter_{split}.csv", index=False)

    report = {
        "promoter_data": args.promoter_data,
        "output_data": args.output_data,
        "sequence_kind": "utr_downstream",
        "sequence_length": int(args.sequence_length),
        "n_stage1_split_genes": int(gene_splits.shape[0]),
        "n_stop_landmark_genes": int(stop_landmarks.shape[0]),
        "n_extracted_utr_genes": int(utr_windows.shape[0]),
        "n_matched_utr_genes": int(final.shape[0]),
        "split_counts": final["split"].value_counts().to_dict(),
        "utr_control_match_counts": utr_windows["utr_control_status"].value_counts().to_dict(),
        **split_report,
        "gtf_sha256": file_sha256(gtf_path),
        "genome_fasta_sha256": file_sha256(genome_fasta),
    }
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3'UTR/downstream assets from frozen promoter Stage 1 splits.")
    parser.add_argument("--promoter-data", type=str, default="promoter_stage1_v1")
    parser.add_argument("--source-data", type=str, default="umi_E-MTAB-10519-hqcells")
    parser.add_argument("--output-data", type=str, default="utr_stage1_v1")
    parser.add_argument("--gtf", type=str, default="data/raw/dmel-all-r6.54.gtf")
    parser.add_argument("--genome-fasta", type=str, default="data/raw/dmel-all-chromosome-r6.54.fasta")
    parser.add_argument("--sequence-length", type=int, default=801)
    parser.add_argument("--val-contig", type=str, default="2L")
    parser.add_argument("--test-contig", type=str, default="3L")
    parser.add_argument("--min-contig-genes", type=int, default=200)
    parser.add_argument("--input-gene-panel-size", type=int, default=4096)
    parser.add_argument("--input-gene-panel-method", choices=["hvg", "variance"], default="hvg")
    parser.add_argument("--hvg-flavor", type=str, default="cell_ranger")
    parser.add_argument("--max-eval-cells", type=int, default=2048)
    parser.add_argument("--control-attempts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_assets(parse_args())
