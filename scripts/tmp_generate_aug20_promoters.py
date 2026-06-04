from pathlib import Path
import argparse
import os
import shutil
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.process_data import augment_promoter_windows


def link_or_copy_h5ad(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        print(f"Hard-linked {dst}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"Copied {dst}")


def write_augmented_split(src_dir: Path, dst_dir: Path, genome_fasta: Path, split: str, shift_bp: int) -> None:
    src_csv = src_dir / f"promoter_{split}.csv"
    dst_csv = dst_dir / f"promoter_{split}.csv"
    tmp_csv = dst_dir / f"promoter_{split}.csv.tmp"

    promoters = pd.read_csv(src_csv)
    augmented = augment_promoter_windows(promoters, genome_fasta_path=genome_fasta, shift_bp=shift_bp)
    augmented.to_csv(tmp_csv, index=False)
    tmp_csv.replace(dst_csv)
    print(f"{split}: {len(promoters)} -> {len(augmented)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift-bp", type=int, default=20)
    parser.add_argument("--src-dir", type=Path, default=PROJECT_ROOT / "data" / "umi_E-MTAB-10519-hqcells")
    parser.add_argument("--dst-dir", type=Path, default=None)
    args = parser.parse_args()

    src_dir = PROJECT_ROOT / "data" / "umi_E-MTAB-10519-hqcells"
    src_dir = args.src_dir
    dst_dir = args.dst_dir or PROJECT_ROOT / "data" / f"umi_E-MTAB-10519-hqcells_aug{args.shift_bp}"
    genome_fasta = PROJECT_ROOT / "data" / "raw" / "dmel-all-chromosome-r6.54.fasta"

    dst_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_augmented_split(src_dir, dst_dir, genome_fasta, split, args.shift_bp)

    for filename in ("data_sanity_summary.json", "gene_check.csv", "sample_tissue.json"):
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)

    link_or_copy_h5ad(src_dir / "integrated_data.h5ad", dst_dir / "integrated_data.h5ad")


if __name__ == "__main__":
    main()
