from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from Bio import SeqIO
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXCLUDED_TRANSCRIPT_FEATURES = {
    "tRNA",
    "rRNA",
    "miRNA",
    "pre_miRNA",
    "snoRNA",
    "snRNA",
    "pseudogene",
}
CODING_EVIDENCE_FEATURES = {"CDS", "start_codon", "stop_codon"}


@dataclass(frozen=True)
class Interval:
    start0: int
    end0: int


def parse_gtf_attributes(attr_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in re.findall(r'(\S+)\s+"([^"]+)"', attr_text):
        attrs[key] = value
    return attrs


def read_gtf_gene_table(gtf_path: Path) -> pd.DataFrame:
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
                    "gene_symbol": attrs.get("gene_symbol", ""),
                    "contig": contig,
                    "strand": strand,
                    "start0": int(start) - 1,
                    "end0": int(end),
                    "feature_set": set(),
                    "symbol_set": set(),
                },
            )
            record["feature_set"].add(feature)
            for symbol_key in ("gene_symbol", "transcript_symbol"):
                symbol = attrs.get(symbol_key)
                if symbol:
                    record["symbol_set"].add(symbol)
            if feature == "gene":
                record["contig"] = contig
                record["strand"] = strand
                record["start0"] = int(start) - 1
                record["end0"] = int(end)
                record["gene_symbol"] = attrs.get("gene_symbol", record["gene_symbol"])

    rows: list[dict[str, Any]] = []
    for record in records.values():
        features = set(record["feature_set"])
        symbols = set(record["symbol_set"])
        has_excluded = bool(features & EXCLUDED_TRANSCRIPT_FEATURES)
        is_protein_coding = "mRNA" in features and bool(features & CODING_EVIDENCE_FEATURES)
        is_lnc = "ncRNA" in features and any(symbol.startswith("lncRNA:") for symbol in symbols)

        if (is_protein_coding or is_lnc) and has_excluded:
            gene_class = "ambiguous"
            eligible = False
            reason = "mixed_stage1_and_excluded_features"
        elif is_protein_coding:
            gene_class = "protein_coding"
            eligible = True
            reason = ""
        elif is_lnc:
            gene_class = "lncRNA_candidate"
            eligible = False
            reason = "lncRNA_candidate_not_used_in_stage1"
        elif has_excluded:
            gene_class = sorted(features & EXCLUDED_TRANSCRIPT_FEATURES)[0]
            eligible = False
            reason = f"excluded_{gene_class}"
        else:
            gene_class = "other_or_unknown"
            eligible = False
            reason = "not_stage1_protein_coding"

        start0 = int(record["start0"])
        end0 = int(record["end0"])
        strand = str(record["strand"])
        tss = start0 if strand == "+" else end0 - 1
        rows.append(
            {
                "gene_id": record["gene_id"],
                "gene_symbol": record["gene_symbol"],
                "contig": record["contig"],
                "strand": strand,
                "start0": start0,
                "end0": end0,
                "tss": tss,
                "gene_class": gene_class,
                "is_stage1_promoter_eligible": bool(eligible),
                "excluded_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def read_source_promoters(source_data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = source_data_dir / f"promoter_{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["source_split"] = split
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No promoter_*.csv files found in {source_data_dir}")
    promoters = pd.concat(frames, ignore_index=True)
    if "augment_offset" in promoters.columns:
        promoters = promoters.loc[promoters["augment_offset"].fillna(0).astype(int) == 0].copy()
    required = {"gene_id", "chrom", "start", "end", "strand", "sequence"}
    missing = required - set(promoters.columns)
    if missing:
        raise ValueError(f"Source promoter CSV missing columns: {sorted(missing)}")
    promoters["length"] = promoters["sequence"].astype(str).str.len()
    return promoters.drop_duplicates(subset=["gene_id"], keep="first").reset_index(drop=True)


def choose_gene_splits(
    eligible_genes: pd.DataFrame,
    val_contig: str,
    test_contig: str,
    min_contig_genes: int,
    seed: int,
) -> pd.DataFrame:
    genes = eligible_genes.copy()
    contig_counts = genes["contig"].value_counts()
    if (
        val_contig != test_contig
        and contig_counts.get(val_contig, 0) >= min_contig_genes
        and contig_counts.get(test_contig, 0) >= min_contig_genes
    ):
        genes["split"] = "train"
        genes.loc[genes["contig"] == val_contig, "split"] = "val"
        genes.loc[genes["contig"] == test_contig, "split"] = "test"
        genes["split_strategy"] = "contig_holdout"
        return genes

    rng = np.random.default_rng(seed)
    shuffled = genes["gene_id"].drop_duplicates().to_numpy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(round(n * 0.1)))
    n_test = max(1, int(round(n * 0.1)))
    val_genes = set(shuffled[:n_val])
    test_genes = set(shuffled[n_val:n_val + n_test])
    genes["split"] = "train"
    genes.loc[genes["gene_id"].isin(val_genes), "split"] = "val"
    genes.loc[genes["gene_id"].isin(test_genes), "split"] = "test"
    genes["split_strategy"] = "seeded_gene_holdout"
    return genes


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    valid = [base for base in seq if base in {"A", "C", "G", "T"}]
    if not valid:
        return float("nan")
    return float(sum(base in {"G", "C"} for base in valid) / len(valid))


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def build_interval_index(gtf_genes: pd.DataFrame, promoters: pd.DataFrame) -> dict[str, list[Interval]]:
    index: dict[str, list[Interval]] = defaultdict(list)
    for row in gtf_genes.itertuples(index=False):
        index[str(row.contig)].append(Interval(int(row.start0), int(row.end0)))
    for row in promoters.itertuples(index=False):
        index[str(row.chrom)].append(Interval(int(row.start), int(row.end)))
    for contig in index:
        index[contig].sort(key=lambda item: item.start0)
    return index


def overlaps_any(intervals: list[Interval], start0: int, end0: int) -> bool:
    for interval in intervals:
        if interval.start0 >= end0:
            return False
        if interval.end0 > start0 and interval.start0 < end0:
            return True
    return False


def find_intergenic_control(
    promoter_row: pd.Series,
    genome: dict[str, Any],
    intervals_by_contig: dict[str, list[Interval]],
    rng: np.random.Generator,
    attempts: int,
) -> dict[str, Any]:
    contig = str(promoter_row["chrom"])
    if contig not in genome:
        return {"match_status": "missing_contig"}
    chrom_seq = str(genome[contig].seq).upper()
    length = int(promoter_row.get("length", len(str(promoter_row["sequence"]))))
    if len(chrom_seq) < length:
        return {"match_status": "contig_too_short"}

    target_gc = gc_fraction(str(promoter_row["sequence"]))
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
        if best is None or gc_diff < float(best["gc_diff"]):
            best = {
                "control_id": f"{contig}:{start0}-{end0}",
                "control_kind": "matched_intergenic",
                "control_chrom": contig,
                "control_start": start0,
                "control_end": end0,
                "control_strand": "+",
                "control_sequence": seq,
                "promoter_gc": target_gc,
                "control_gc": seq_gc,
                "gc_diff": gc_diff,
                "match_status": "matched",
            }
    if best is None:
        return {"match_status": "no_nonoverlap_candidate"}
    if str(promoter_row.get("strand", "+")) == "-":
        best["control_strand"] = "-"
        best["control_sequence"] = reverse_complement(str(best["control_sequence"]))
    return best


def add_intergenic_controls(
    promoters: pd.DataFrame,
    genome_fasta: Path,
    gtf_genes: pd.DataFrame,
    seed: int,
    attempts: int,
) -> pd.DataFrame:
    genome = SeqIO.to_dict(SeqIO.parse(str(genome_fasta), "fasta"))
    intervals_by_contig = build_interval_index(gtf_genes, promoters)
    rng = np.random.default_rng(seed)
    controls = [
        find_intergenic_control(row, genome, intervals_by_contig, rng, attempts)
        for _, row in promoters.iterrows()
    ]
    return pd.concat([promoters.reset_index(drop=True), pd.DataFrame(controls)], axis=1)


def resolve_cell_ids(source_data_dir: Path, adata: Any, split: str) -> np.ndarray:
    split_path = source_data_dir / f"cell_{split}.txt"
    if split_path.exists():
        cells = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cells = [cell for cell in cells if cell in adata.obs_names]
        if cells:
            return np.asarray(cells, dtype=object)
    return np.asarray(adata.obs_names.astype(str), dtype=object)


def write_cell_panels(source_data_dir: Path, output_dir: Path, adata: Any, max_eval_cells: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_cells = resolve_cell_ids(source_data_dir, adata, "train")
    val_cells = resolve_cell_ids(source_data_dir, adata, "val")
    test_cells = resolve_cell_ids(source_data_dir, adata, "test")

    if len(val_cells) > max_eval_cells:
        val_cells = rng.choice(val_cells, size=max_eval_cells, replace=False)
    if len(test_cells) > max_eval_cells:
        test_cells = rng.choice(test_cells, size=max_eval_cells, replace=False)

    (output_dir / "cell_train.txt").write_text("\n".join(map(str, train_cells)) + "\n", encoding="utf-8")
    (output_dir / "cell_val.txt").write_text("\n".join(map(str, val_cells)) + "\n", encoding="utf-8")
    (output_dir / "cell_test.txt").write_text("\n".join(map(str, test_cells)) + "\n", encoding="utf-8")

    panel_map = {str(cell): "train" for cell in train_cells}
    panel_map.update({str(cell): "validation" for cell in val_cells})
    panel_map.update({str(cell): "test" for cell in test_cells})
    cells = pd.DataFrame({"cell_id": adata.obs_names.astype(str)})
    cells["panel"] = cells["cell_id"].map(panel_map).fillna("unused")
    for col in adata.obs.columns:
        cells[col] = adata.obs[col].astype(str).to_numpy()
    cells.to_csv(output_dir / "cells.tsv", sep="\t", index=False)
    cells.loc[cells["panel"].isin(["validation", "test"]), ["cell_id", "panel"]].to_csv(
        output_dir / "frozen_eval_cells.tsv",
        sep="\t",
        index=False,
    )
    return cells


def rank_input_genes_by_variance(adata: Any, gene_indices: np.ndarray, cell_indices: np.ndarray) -> list[str]:
    var_gene_ids = adata.var["gene_id"].astype(str).to_numpy()
    X = adata.X[cell_indices, :][:, gene_indices]
    if sparse.issparse(X):
        X_csr = X.tocsr()
        means = np.asarray(X_csr.mean(axis=0)).ravel()
        sq_means = np.asarray(X_csr.power(2).mean(axis=0)).ravel()
        variances = sq_means - means * means
        detection = np.asarray((X_csr > 0).mean(axis=0)).ravel()
    else:
        X_arr = np.asarray(X)
        variances = X_arr.var(axis=0)
        detection = (X_arr > 0).mean(axis=0)
    candidates = pd.DataFrame(
        {
            "gene_id": var_gene_ids[gene_indices],
            "variance": variances,
            "detection": detection,
        }
    ).drop_duplicates(subset=["gene_id"], keep="first")
    candidates = candidates.sort_values(["variance", "detection", "gene_id"], ascending=[False, False, True])
    return candidates["gene_id"].astype(str).tolist()


def select_input_gene_panel(
    adata: Any,
    train_gene_ids: list[str],
    max_genes: int,
    train_cell_ids: list[str] | np.ndarray | None = None,
    method: str = "hvg",
    hvg_flavor: str = "cell_ranger",
) -> list[str]:
    train_gene_set = set(train_gene_ids)
    var_gene_ids = adata.var["gene_id"].astype(str).to_numpy()
    candidate_indices = np.asarray([i for i, gene_id in enumerate(var_gene_ids) if gene_id in train_gene_set], dtype=np.int64)
    if len(candidate_indices) == 0:
        raise ValueError("No train genes found in AnnData var for input panel.")

    if train_cell_ids is None:
        cell_indices = np.arange(adata.n_obs, dtype=np.int64)
    else:
        obs_names = pd.Index(adata.obs_names.astype(str))
        resolved = obs_names.get_indexer(pd.Index([str(cell_id) for cell_id in train_cell_ids]))
        cell_indices = resolved[resolved >= 0].astype(np.int64)
        if len(cell_indices) == 0:
            raise ValueError("No train cells found in AnnData obs for input panel.")

    fallback_ranked = rank_input_genes_by_variance(adata, candidate_indices, cell_indices)
    if method == "variance":
        return fallback_ranked[:max_genes]
    if method != "hvg":
        raise ValueError(f"Unsupported input gene panel method: {method}")

    n_top_genes = min(max_genes, len(candidate_indices))
    try:
        hvg_adata = adata[cell_indices, :][:, candidate_indices].copy()
        sc.pp.highly_variable_genes(
            hvg_adata,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            inplace=True,
        )
        hvg_var = hvg_adata.var.copy()
        hvg_var["gene_id"] = hvg_var["gene_id"].astype(str)
        if "highly_variable" in hvg_var.columns:
            hvg_var = hvg_var.loc[hvg_var["highly_variable"].astype(bool)].copy()
        else:
            hvg_var = hvg_var.iloc[0:0].copy()
        if "highly_variable_rank" in hvg_var.columns:
            hvg_var = hvg_var.sort_values(["highly_variable_rank", "gene_id"], ascending=[True, True])
        elif "dispersions_norm" in hvg_var.columns:
            hvg_var = hvg_var.sort_values(["dispersions_norm", "gene_id"], ascending=[False, True])
        elif "variances_norm" in hvg_var.columns:
            hvg_var = hvg_var.sort_values(["variances_norm", "gene_id"], ascending=[False, True])
        else:
            hvg_var = hvg_var.sort_values("gene_id")
        selected = hvg_var["gene_id"].drop_duplicates().astype(str).tolist()
    except Exception as exc:
        print(f"Warning: scanpy HVG selection failed ({exc}); falling back to variance ranking.")
        selected = []

    selected_set = set(selected)
    selected.extend([gene_id for gene_id in fallback_ranked if gene_id not in selected_set])
    return selected[:max_genes]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_assets(args: argparse.Namespace) -> None:
    source_data_dir = PROJECT_ROOT / "data" / args.source_data
    output_dir = PROJECT_ROOT / "data" / args.output_data
    output_dir.mkdir(parents=True, exist_ok=True)

    gtf_path = Path(args.gtf)
    genome_fasta = Path(args.genome_fasta)
    if not gtf_path.is_absolute():
        gtf_path = PROJECT_ROOT / gtf_path
    if not genome_fasta.is_absolute():
        genome_fasta = PROJECT_ROOT / genome_fasta

    scrna_file = source_data_dir / "integrated_data.h5ad"
    adata = sc.read_h5ad(scrna_file)
    gtf_genes = read_gtf_gene_table(gtf_path)
    source_promoters = read_source_promoters(source_data_dir)

    var_gene_ids = set(adata.var["gene_id"].astype(str))
    promoter_gene_ids = set(source_promoters["gene_id"].astype(str))
    eligible_ids = set(
        gtf_genes.loc[gtf_genes["is_stage1_promoter_eligible"], "gene_id"].astype(str)
    )
    usable_gene_ids = var_gene_ids & promoter_gene_ids & eligible_ids

    genes = gtf_genes.copy()
    genes["in_expression"] = genes["gene_id"].isin(var_gene_ids)
    genes["has_promoter"] = genes["gene_id"].isin(promoter_gene_ids)
    genes["included"] = genes["gene_id"].isin(usable_gene_ids)
    genes.loc[genes["is_stage1_promoter_eligible"] & ~genes["in_expression"], "excluded_reason"] = "missing_expression"
    genes.loc[genes["is_stage1_promoter_eligible"] & genes["in_expression"] & ~genes["has_promoter"], "excluded_reason"] = "missing_promoter"

    included = genes.loc[genes["included"]].copy()
    promoters = source_promoters.loc[source_promoters["gene_id"].isin(included["gene_id"])].copy()
    promoters = promoters.merge(included[["gene_id", "gene_class"]], on="gene_id", how="left")
    promoters = add_intergenic_controls(
        promoters,
        genome_fasta=genome_fasta,
        gtf_genes=gtf_genes,
        seed=args.seed,
        attempts=args.control_attempts,
    )
    matched_gene_ids = set(promoters.loc[promoters["match_status"] == "matched", "gene_id"].astype(str))
    matched_genes = included.loc[included["gene_id"].isin(matched_gene_ids)].copy()
    split_genes = choose_gene_splits(
        matched_genes,
        val_contig=args.val_contig,
        test_contig=args.test_contig,
        min_contig_genes=args.min_contig_genes,
        seed=args.seed,
    )
    final_split_genes = split_genes.copy()

    genes.loc[:, "included"] = genes["gene_id"].isin(final_split_genes["gene_id"])
    genes.loc[genes["is_stage1_promoter_eligible"] & genes["has_promoter"] & genes["in_expression"] & ~genes["gene_id"].isin(matched_gene_ids), "excluded_reason"] = "missing_matched_intergenic_control"
    genes = genes.merge(
        final_split_genes[["gene_id", "split", "split_strategy"]],
        on="gene_id",
        how="left",
    )
    genes["split"] = genes["split"].fillna("excluded")
    genes["split_strategy"] = genes["split_strategy"].fillna("")

    promoters = promoters.merge(final_split_genes[["gene_id", "split"]], on="gene_id", how="left")
    promoters["split"] = promoters["split"].fillna("excluded")

    genes.to_csv(output_dir / "genes.tsv", sep="\t", index=False)
    final_split_genes[["gene_id", "split", "split_strategy", "contig", "strand", "tss", "gene_class"]].to_csv(
        output_dir / "gene_splits.tsv",
        sep="\t",
        index=False,
    )
    promoters.to_csv(output_dir / "promoter_windows.tsv", sep="\t", index=False)
    promoters.loc[promoters["match_status"] == "matched"].to_csv(output_dir / "control_windows.tsv", sep="\t", index=False)
    for split in ("train", "val", "test"):
        split_df = promoters.loc[(promoters["split"] == split) & (promoters["match_status"] == "matched")].copy()
        split_df.to_csv(output_dir / f"promoter_{split}.csv", index=False)

    cells = write_cell_panels(
        source_data_dir=source_data_dir,
        output_dir=output_dir,
        adata=adata,
        max_eval_cells=args.max_eval_cells,
        seed=args.seed,
    )
    train_genes = final_split_genes.loc[final_split_genes["split"] == "train", "gene_id"].astype(str).tolist()
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

    report = {
        "source_data": args.source_data,
        "output_data": args.output_data,
        "scrna_file": str(scrna_file),
        "n_cells": int(adata.n_obs),
        "n_expression_genes": int(len(var_gene_ids)),
        "n_source_promoter_genes": int(len(promoter_gene_ids)),
        "n_stage1_eligible_genes": int(len(eligible_ids)),
        "n_control_match_candidate_genes": int(included.shape[0]),
        "n_genes_after_control_match_before_split": int(matched_genes.shape[0]),
        "n_included_genes": int(final_split_genes.shape[0]),
        "gene_class_counts": genes["gene_class"].value_counts().to_dict(),
        "included_gene_class_counts": final_split_genes["gene_class"].value_counts().to_dict(),
        "split_counts": final_split_genes["split"].value_counts().to_dict(),
        "control_match_counts": promoters["match_status"].value_counts().to_dict(),
        "input_gene_panel_method": args.input_gene_panel_method,
        "hvg_flavor": args.hvg_flavor,
        "input_gene_panel_size": int(len(input_panel)),
        "cell_panel_counts": cells["panel"].value_counts().to_dict(),
        "gtf_sha256": file_sha256(gtf_path),
        "genome_fasta_sha256": file_sha256(genome_fasta),
    }
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build promoter Stage 1 assets with Pol II gene filtering.")
    parser.add_argument("--source-data", type=str, default="umi_E-MTAB-10519-hqcells")
    parser.add_argument("--output-data", type=str, default="promoter_stage1_v1")
    parser.add_argument("--gtf", type=str, default="data/raw/dmel-all-r6.54.gtf")
    parser.add_argument("--genome-fasta", type=str, default="data/raw/dmel-all-chromosome-r6.54.fasta")
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
