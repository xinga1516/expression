# %%
import loompy
import scanpy as sc
import h5py
import anndata as ad
import pandas as pd
import numpy as np
from Bio import SeqIO
from scipy import sparse
from scipy.io import mmread
import re
import pyranges as pr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Any


def hdf5_matrix_to_sparse(
    mat: Any,
    chunk_size: int = 5000,
    transpose: bool = True,
    dtype: type = np.float32,
) -> sparse.csr_matrix:
    """
    Convert an HDF5 dense-looking dataset (genes × cells)
    into a scipy sparse matrix safely using chunking.

    Parameters
    ----------
    mat : h5py.Dataset
        HDF5 dataset, shape (n_genes, n_cells)
    chunk_size : int
        Number of cells to read per chunk
    transpose : bool
        If True, return cell × gene matrix (AnnData convention)
    dtype : numpy dtype
        Data type for sparse matrix values

    Returns
    -------
    scipy.sparse.csr_matrix
    """

    n_genes, n_cells = mat.shape

    rows = []
    cols = []
    data = []

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)

        block = mat[:, start:end]          # gene × chunk
        block = block.astype(dtype, copy=False)

        nz = np.nonzero(block)
        if nz[0].size == 0:
            continue

        rows.append(nz[0])
        cols.append(nz[1] + start)
        data.append(block[nz])

    if len(data) == 0:
        raise ValueError("Matrix contains no non-zero entries.")

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    X = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_genes, n_cells),
    )

    if transpose:
        X = X.T.tocsr()   # cell × gene
    else:
        X = X.tocsr()     # gene × cell

    return X


def augment_promoter_windows(
    promoter_df: pd.DataFrame,
    genome_fasta_path: Path,
    shift_bp: int = 20,
    include_original: bool = True,
) -> pd.DataFrame:
    """
    Build shifted 400bp promoter windows from genome sequence.

    Offsets are genomic-coordinate shifts of the existing window. With
    shift_bp=20 and include_original=True, each original promoter can produce
    up to 41 rows with augment_offset from -20 to +20.
    """
    required_cols = {"gene_id", "chrom", "start", "end", "strand"}
    missing = required_cols - set(promoter_df.columns)
    if missing:
        raise ValueError(f"augment_promoter_windows missing required columns: {sorted(missing)}")
    if shift_bp < 0:
        raise ValueError("shift_bp must be >= 0")
    if not genome_fasta_path.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {genome_fasta_path}")

    genome = SeqIO.to_dict(SeqIO.parse(str(genome_fasta_path), "fasta"))
    offsets = list(range(-shift_bp, shift_bp + 1))
    if not include_original:
        offsets = [offset for offset in offsets if offset != 0]

    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in promoter_df.itertuples(index=False):
        row_dict = row._asdict()
        chrom = str(row_dict["chrom"])
        if chrom not in genome:
            skipped += len(offsets)
            continue

        chrom_seq = genome[chrom].seq
        start = int(row_dict["start"])
        end = int(row_dict["end"])
        strand = str(row_dict["strand"])

        for offset in offsets:
            shifted_start = start + offset
            shifted_end = end + offset
            if shifted_start < 0 or shifted_end > len(chrom_seq):
                skipped += 1
                continue

            seq = chrom_seq[shifted_start:shifted_end]
            if strand == "-":
                seq = seq.reverse_complement()

            augmented = dict(row_dict)
            augmented["start"] = shifted_start
            augmented["end"] = shifted_end
            augmented["sequence"] = str(seq).upper()
            augmented["length"] = shifted_end - shifted_start
            augmented["augment_offset"] = offset
            rows.append(augmented)

    augmented_df = pd.DataFrame(rows)
    print(
        f"  Augmented promoters: {len(promoter_df)} -> {len(augmented_df)} rows "
        f"(shift_bp={shift_bp}, skipped={skipped})"
    )
    return augmented_df


def add_gene_annotation(rawdata: ad.AnnData, synonym_path: Optional[str] = None, gtf_path: Optional[str] = None) -> ad.AnnData:
    """
    Add FlyBase annotation for genes in an AnnData object.

    Adds to `rawdata.var`:
    - `gene_symbol` (from `rawdata.var.index`)
    - `gene_id` (FlyBase primary FBid)
    - `chr`
    - `mt` (mitochondrion_genome)
    - `gene_length` (End - Start from GTF gene coordinates)
    """
    if not isinstance(rawdata, ad.AnnData):
        raise TypeError(f"`rawdata` must be an AnnData, got: {type(rawdata)!r}")

    base_dir = Path(__file__).resolve().parent.parent
    synonym_path = Path(synonym_path) if synonym_path is not None else (base_dir / "data" / "raw" / "fb_synonym_fb_2025_05.tsv")
    gtf_path = Path(gtf_path) if gtf_path is not None else (base_dir / "data" / "raw" / "dmel-all-r6.54.gtf")

    # --- symbol -> FlyBase gene_id ---
    synonym = pd.read_csv(synonym_path, sep="\t", skiprows=5, dtype=str)
    if "organism_abbreviation" in synonym.columns:
        synonym = synonym.loc[synonym["organism_abbreviation"] == "Dmel"].copy()

    required_cols = {"##primary_FBid", "current_symbol", "symbol_synonym(s)"}
    missing = required_cols - set(synonym.columns)
    if missing:
        raise ValueError(f"Synonym table missing columns: {sorted(missing)}")

    # CHANGED: mapping priority to avoid synonym ambiguity:
    # 1) exact match on `current_symbol`
    # 2) fallback to synonym *only if* that synonym maps to exactly one gene_id
    current_symbol_to_fbid = {}
    synonym_symbol_to_fbids = {}

    for _, row in synonym.iterrows():
        fbid = row.get("##primary_FBid")
        if pd.isna(fbid) or str(fbid).strip() == "":
            continue
        fbid = str(fbid).strip()

        cur = row.get("current_symbol")
        if pd.notna(cur):
            cur = str(cur).strip()
            if cur:
                # Current symbols should be unique; keep first-seen deterministically.
                current_symbol_to_fbid.setdefault(cur, fbid)

        syns = row.get("symbol_synonym(s)")
        if pd.notna(syns):
            for sym in str(syns).split("|"):
                sym = sym.strip()
                if sym:
                    synonym_symbol_to_fbids.setdefault(sym, set()).add(fbid)

    synonym_symbol_to_fbid = {
        sym: next(iter(fbids))
        for sym, fbids in synonym_symbol_to_fbids.items()
        if len(fbids) == 1
    }

    if "gene_symbol" not in rawdata.var.columns:
        rawdata.var["gene_symbol"] = rawdata.var.index.astype(str)

    # Prioritize current_symbol; only use synonym when unambiguous.
    gene_id = rawdata.var["gene_symbol"].map(current_symbol_to_fbid)
    gene_id = gene_id.fillna(rawdata.var["gene_symbol"].map(synonym_symbol_to_fbid))
    rawdata.var["gene_id"] = gene_id

    # --- gene_id -> chr/length from GTF ---
    gtf = pr.read_gtf(str(gtf_path))
    df = gtf.df
    if "gene_id" not in df.columns:
        raise ValueError("GTF parsed table has no `gene_id` column.")

    if "Feature" in df.columns:
        df = df.loc[df["Feature"] == "gene"].copy()

    cols = [c for c in ["gene_id", "Chromosome", "Start", "End"] if c in df.columns]
    if set(cols) != {"gene_id", "Chromosome", "Start", "End"}:
        raise ValueError(f"GTF parsed table missing required columns; got columns: {sorted(df.columns)}")

    df_gene = (
        df[["gene_id", "Chromosome", "Start", "End"]]
        .groupby("gene_id", as_index=True)
        .agg({"Chromosome": "first", "Start": "min", "End": "max"})
    )
    df_gene["gene_length"] = (df_gene["End"] - df_gene["Start"]).astype(np.int64)

    chr_mapped = rawdata.var["gene_id"].map(df_gene["Chromosome"])
    rawdata.var["chr"] = pd.Series(chr_mapped, index=rawdata.var.index, dtype="object").fillna("")
    rawdata.var["mt"] = rawdata.var["chr"] == "mitochondrion_genome"
    gene_length_mapped = rawdata.var["gene_id"].map(df_gene["gene_length"])
    rawdata.var["gene_length"] = pd.Series(gene_length_mapped, index=rawdata.var.index, dtype=np.float32).fillna(0)

    return rawdata

def _safe_col_attr_to_array(handle: h5py.File, key: str) -> np.ndarray | None:
    """Safely read a 1D column attribute from a loom file.

    Skips structured/compound dtypes (e.g. ClusterMarkers) and multi-
    dimensional arrays, which are not suitable for obs columns.
    """
    try:
        arr = handle["col_attrs"][key]
        if arr.ndim != 1:
            return None
        # Structured (compound) dtypes have named fields — skip.
        if arr.dtype.names is not None:
            return None
        return arr[:]
    except Exception:
        return None


def build_integrated_data() -> ad.AnnData:
    """Build integrated single-cell RNA sequencing data from a .loom file.

    Converts the dense HDF5 matrix to a sparse CSR AnnData, preserving
    cell-level metadata (cell type, tissue, batch, etc.) in ``.obs``.
    """
    base_dir = Path(__file__).resolve().parent.parent
    loom_path = base_dir / "data" / "raw" / "s_fca_biohub_all_wo_blood_10x.loom"

    f = h5py.File(str(loom_path), "r")
    print("Top-level keys:", list(f.keys()))

    X = f["matrix"]
    X_sparse = hdf5_matrix_to_sparse(X)
    print("Is sparse:", sparse.issparse(X_sparse))
    print("Max UMI count:", X_sparse.data.max())

    gene_names = f["row_attrs"]["Gene"][:].astype(str)
    cell_ids = f["col_attrs"]["CellID"][:].astype(str)

    # Build obs with selected cell metadata from loom col_attrs
    obs_columns: dict[str, np.ndarray] = {}
    col_attrs = list(f["col_attrs"].keys())
    print(f"\ncol_attrs ({len(col_attrs)}): {sorted(col_attrs)}")

    # Columns worth carrying into AnnData.obs (biologically informative).
    keep_cols = [
        "annotation",
        "annotation_broad",
        "tissue",
        "sex",
        "age",
        "batch",
        "batch_id",
        "sample_id",
        "dissection_lab",
        "fly_genetics",
        "n_counts",
        "n_genes",
        "percent_mito",
        "scrublet__predicted_doublets",
        "fca_id",
    ]

    skipped: list[str] = []
    for key in keep_cols:
        arr = _safe_col_attr_to_array(f, key)
        if arr is not None:
            # Decode bytes to str for object columns
            if arr.dtype.kind in ("S", "O"):
                arr = arr.astype(str)
            obs_columns[key] = arr
        else:
            skipped.append(key)

    if skipped:
        print(f"Skipped {len(skipped)} col_attrs (non-1D or structured dtype): {skipped}")

    obs_df = pd.DataFrame(obs_columns, index=cell_ids)
    print(f"\nobs columns ({len(obs_df.columns)}): {sorted(obs_df.columns)}")
    print(f"obs shape: {obs_df.shape}")

    rawdata = ad.AnnData(
        X=X_sparse,
        obs=obs_df,
        var=pd.DataFrame(index=gene_names),
    )

    f.close()
    return rawdata

def filter_cells(rawdata: ad.AnnData, top_total_fraction: float = 0.10, plot_qc: bool = True) -> ad.AnnData:
    # filter cells based on quality control metrics
    # AnnData here stores raw UMI counts in X, so make sure QC uses counts explicitly.
    if 'counts' not in rawdata.layers:
        rawdata.layers['counts'] = rawdata.X

    sc.pp.calculate_qc_metrics(
        rawdata,
        qc_vars=['mt'],
        layer='counts',
        percent_top=None,
        log1p=False,
        inplace=True
    )

    n_keep = max(1, int(np.ceil(rawdata.n_obs * top_total_fraction)))
    selected_cells = rawdata.obs['total_counts'].nlargest(n_keep).index
    cutoff = rawdata.obs.loc[selected_cells, 'total_counts'].min()

    if plot_qc:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(rawdata.obs['total_counts'], bins=50, color='steelblue', alpha=0.85)
        axes[0].axvline(cutoff, color='red', linestyle='--', linewidth=1.5, label=f'top {int(top_total_fraction * 100)}% cutoff')
        axes[0].set_xlabel('Total gene expression (total counts per cell)')
        axes[0].set_ylabel('Number of cells')
        axes[0].set_title('Total counts distribution')
        axes[0].legend()

        axes[1].scatter(rawdata.obs['total_counts'], rawdata.obs['n_genes_by_counts'], s=7, alpha=0.6)
        axes[1].set_xlabel('Total gene expression (total counts per cell)')
        axes[1].set_ylabel('Detected genes per cell')
        axes[1].set_title('Total counts vs detected genes')

        axes[2].scatter(rawdata.obs['total_counts'], rawdata.obs['pct_counts_mt'], s=7, alpha=0.6)
        axes[2].set_xlabel('Total gene expression (total counts per cell)')
        axes[2].set_ylabel('Mitochondrial counts fraction (%)')
        axes[2].set_title('Total counts vs mitochondrial fraction')

        plt.tight_layout()
        plt.show()

    filtered_data = rawdata[selected_cells].copy()
    print(rawdata.shape, '->', filtered_data.shape)

    return filtered_data

def split_train_val(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    output_dir: Optional[Path] = None,
    by_gene: bool = False,
) -> tuple:
    """
    按每条染色体上的物理位置排序后，按比例切分为 train/val/test。

    input:
        promoter dataframe with columns:
        ["gene_id", "chrom", "start", "end", "strand", "sequence", "length"]

    by_gene: bool
        If True, split at gene level (one gene's all promoters go to the same
        split, using the gene's median genomic position).  If False (default),
        split at individual promoter row level.
    """
    required_cols = {"gene_id", "chrom", "start", "end"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"split_train_val 缺少必要列: {sorted(missing)}")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    data = df.copy()
    data["start"] = data["start"].astype(int)
    data["end"] = data["end"].astype(int)
    data["mid"] = ((data["start"] + data["end"]) // 2).astype(int)

    train_genes, val_genes, test_genes = set(), set(), set()
    block_chrom = ['2R', '3R']

    if by_gene:
        # ── gene-level split: use median position per gene ──
        gene_mid = data.groupby("gene_id")["mid"].median()
        for chrom, sub in data.groupby("chrom", sort=True):
            gene_ids_chrom = sub["gene_id"].unique()
            if chrom not in block_chrom:
                train_genes.update(gene_ids_chrom)
                print(f"{chrom}: {len(gene_ids_chrom)} genes -> all train")
                continue

            chrom_genes = gene_mid.loc[gene_mid.index.isin(gene_ids_chrom)].sort_values()
            gene_list = chrom_genes.index.tolist()
            n = len(gene_list)
            n_train = int(np.floor(n * train_ratio))
            n_val = int(np.floor(n * val_ratio))
            n_test = n - n_train - n_val
            if n_test < 0:
                n_test = 0
                n_val = n - n_train

            train_genes.update(gene_list[:n_train])
            val_genes.update(gene_list[n_train:n_train + n_val])
            test_genes.update(gene_list[n_train + n_val:])
            print(f"{chrom}: {n} genes -> train={n_train} val={n_val} test={n_test}")

    else:
        # ── row-level split (original behaviour) ──
        for chrom, sub in data.groupby("chrom", sort=True):
            if chrom not in block_chrom:
                train_genes.update(sub["gene_id"].tolist())
                print(f"{chrom}: total={len(sub)}, span ignored -> all train")
                continue

            sub = sub.sort_values("mid").copy()
            n = len(sub)
            if n == 0:
                continue

            n_train = int(np.floor(n * train_ratio))
            n_val = int(np.floor(n * val_ratio))
            n_test = n - n_train - n_val

            if n_test < 0:
                n_test = 0
                n_val = n - n_train

            train_genes.update(sub.iloc[:n_train]["gene_id"].tolist())
            val_genes.update(sub.iloc[n_train:n_train + n_val]["gene_id"].tolist())
            test_genes.update(sub.iloc[n_train + n_val:]["gene_id"].tolist())
            print(f"{chrom}: total={n}, train={n_train}, val={n_val}, test={n_test}")

    # 去重并确保互斥（优先级：train > val > test）
    val_genes -= train_genes
    test_genes -= train_genes
    test_genes -= val_genes

    train_genes = sorted(train_genes)
    val_genes = sorted(val_genes)
    test_genes = sorted(test_genes)

    print("\nFinal counts (gene level):")
    print(f"Train: {len(train_genes)}")
    print(f"Val:   {len(val_genes)}")
    print(f"Test:  {len(test_genes)}")
    print("Train ∩ Val:", set(train_genes) & set(val_genes))
    print("Val ∩ Test:", set(val_genes) & set(test_genes))
    print("Train ∩ Test:", set(train_genes) & set(test_genes))

    base_dir = Path(__file__).resolve().parent.parent
    outdir = Path(output_dir) if output_dir is not None else (base_dir / "data" / "highquality")
    outdir.mkdir(parents=True, exist_ok=True)

    data[data["gene_id"].isin(train_genes)].drop(columns=["mid"]).to_csv(outdir / "promoter_train.csv", index=False)
    data[data["gene_id"].isin(val_genes)].drop(columns=["mid"]).to_csv(outdir / "promoter_val.csv", index=False)
    data[data["gene_id"].isin(test_genes)].drop(columns=["mid"]).to_csv(outdir / "promoter_test.csv", index=False)

    print("\nFinal counts (promoter rows):")
    print(f"Train: {int(data['gene_id'].isin(train_genes).sum())}")
    print(f"Val:   {int(data['gene_id'].isin(val_genes).sum())}")
    print(f"Test:  {int(data['gene_id'].isin(test_genes).sum())}")
    print(f"\nPromoter CSV files written to: {outdir}")

    return train_genes, val_genes, test_genes

def filter_genes(
    rawdata: ad.AnnData,
    promoter_fa_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    genome_fasta_path: Optional[Path] = None,
    augment_shift_bp: int = 20,
) -> ad.AnnData:
    '''
    Remove genes without promoter sequences, and split into train/val/test.

    Parameters
    ----------
    rawdata : AnnData
    promoter_fa_path : Path or None
        Path to promoters.fa.  Default: data/processed/promoters.fa
    output_dir : Path or None
        Directory for promoter CSV files.  Default: data/processed/
    genome_fasta_path : Path or None
        Genome FASTA used to rebuild shifted promoter windows.
        Default: data/raw/dmel-all-chromosome-r6.54.fasta
    augment_shift_bp : int
        Genomic-coordinate shift range. Default 20 means offsets -20..+20.
        Set to 0 to keep only the original centered window.
    '''
    base_dir = Path(__file__).resolve().parent.parent
    if promoter_fa_path is None:
        promoter_fa_path = base_dir / "data" / "processed" / "promoters.fa"
    if genome_fasta_path is None:
        genome_fasta_path = base_dir / "data" / "raw" / "dmel-all-chromosome-r6.54.fasta"

    promoters = SeqIO.parse(str(promoter_fa_path), 'fasta')
    rows = []
    for record in promoters:
        m = re.match(r"(.+)::(.+):(\d+)-(\d+)\(([+-])\)", record.id)
        if not m:
            continue
        gene, chrom, start, end, strand = m.groups()
        rows.append({
            "gene_id": gene,
            "chrom": chrom,
            "start": int(start),
            "end": int(end),
            "strand": strand,
            "sequence": str(record.seq),
            "length": len(record.seq)
        })
    promoters = pd.DataFrame(rows)
    gene_promoters = set(promoters['gene_id'])
    print(f"  Parsed promoters.fa: {len(promoters)} promoters, {len(gene_promoters)} genes")

    # Remove duplicate gene_ids in h5ad
    print(rawdata.shape)
    df = rawdata.var
    filtered_data = rawdata[:, df.duplicated(subset="gene_id", keep=False) == False]
    gene_singlecell = set((filtered_data.var)['gene_id'])

    insect = gene_singlecell - gene_promoters
    insect1 = gene_promoters - gene_singlecell
    eligible_genes = gene_singlecell & gene_promoters
    print("{0} genes only in {1} single cell data : ".format(len(insect), len(gene_singlecell)), insect)
    print("{0} genes only in {1} promoters data : ".format(len(insect1), len(gene_promoters)), insect1)

    before = filtered_data.shape
    flag = [(i in eligible_genes) for i in filtered_data.var.gene_id]
    filtered_data = filtered_data[:, flag]
    print("genes from {0} to {1}".format(before[1], filtered_data.shape[1]))

    filtered_promoters = promoters[promoters["gene_id"].isin(eligible_genes)]
    print(f"promoters number: {len(filtered_promoters)}")
    promoter_df = filtered_promoters.copy()
    promoter_df["start"] = promoter_df["start"].astype(int)
    train_genes, _, _ = split_train_val(promoter_df, by_gene=True, output_dir=output_dir)

    if augment_shift_bp > 0:
        base_dir = Path(__file__).resolve().parent.parent
        outdir = Path(output_dir) if output_dir is not None else (base_dir / "data" / "highquality")
        train_promoters = promoter_df[promoter_df["gene_id"].isin(train_genes)].copy()
        augmented_train = augment_promoter_windows(
            train_promoters,
            genome_fasta_path=genome_fasta_path,
            shift_bp=augment_shift_bp,
        )
        augmented_train.to_csv(outdir / "promoter_train.csv", index=False)
        print(f"  Wrote augmented train promoters only: {outdir / 'promoter_train.csv'}")

    return filtered_data

def compute_cpm(rawdata: ad.AnnData) -> ad.AnnData:
    sc.pp.normalize_total(rawdata, target_sum=1e6) # 将每个细胞的总表达量归一化到 1,000,000（CPM）
    sc.pp.log1p(rawdata) # 对归一化后的数据进行 log1p 转换（log(x + 1)），以减小数据范围并处理零值。
    # # using gene length to compute RPKM/FPKM
    # gene_length_kb = rawdata.var["gene_length"] / 1e3
    # rawdata.X = rawdata.X / gene_length_kb.values[np.newaxis, :]
    return rawdata

def draw_high_quality_samples(rawdata: ad.AnnData, top_total_fraction: float = 0.10) -> ad.AnnData:
    ''' delete genes with zero expression across all cells, 
    and select highly variable genes'''
    sc.pp.highly_variable_genes(rawdata, flavor="seurat_v3", n_top_genes=2000)
    rawdata = rawdata[:, rawdata.var['highly_variable']]
    return rawdata

def _get_mito_gene_ids(gtf_path: Path) -> set[str]:
    """从 GTF 中提取 mitochondrion_genome 上的 gene_id 集合"""
    gtf = pr.read_gtf(str(gtf_path))
    df = gtf.df
    if "Feature" in df.columns:
        df = df[df["Feature"] == "gene"]
    return set(df[df["Chromosome"] == "mitochondrion_genome"]["gene_id"])


def build_from_mtx(
    mtx_dir: Path,
    out_dir: Path,
    top_total_fraction: float = 0.10,
    seed: int = 42,
) -> ad.AnnData:
    """从 E-MTAB-10519 raw MTX 构建 AnnData h5ad，含 QC 绘图和细胞过滤。

    Parameters
    ----------
    mtx_dir : Path
        MTX 文件所在目录 (e.g. data/E-MTAB-10519-raw/)
    out_dir : Path
        输出目录 (e.g. data/emtab_processed/)
    promoter_dir : Path or None
        promoter CSV 所在目录，None 则跳过复制
    top_total_fraction : float
        top 细胞比例 (default 10%%)
    seed : int
        随机种子
    """
    import shutil

    mtx_path = mtx_dir / "E-MTAB-10519.aggregated_filtered_counts.mtx"
    rows_path = mtx_dir / "E-MTAB-10519.aggregated_filtered_counts.mtx_rows"
    cols_path = mtx_dir / "E-MTAB-10519.aggregated_filtered_counts.mtx_cols"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "qc_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read MTX → cell × gene CSR ──
    print("Reading MTX (may take a few minutes) ...")
    X = mmread(str(mtx_path))                # COO: gene × cell (float from MatrixMarket)
    X.data = np.round(X.data).astype(np.int64)  # enforce integer UMI counts
    n_genes, n_cells = X.shape
    print(f"  Loaded COO: {n_genes} genes × {n_cells} cells  |  {X.nnz:,} non-zeros")
    X = X.T.tocsr()                  # CSR: cell × gene
    import gc; gc.collect()
    print(f"  Transposed to CSR: {X.shape}")

    # ── 2. Gene IDs ──
    with open(rows_path) as f:
        gene_ids = [line.strip().split('\t')[0] for line in f]
    print(f"  Gene IDs: {len(gene_ids)} (sample: {gene_ids[:3]})")

    # ── 3. Cell IDs ──
    with open(cols_path) as f:
        cell_ids = [line.strip() for line in f]
    sample_ids = [c.split('-')[0] for c in cell_ids]

    # ── 4. Gene symbol mapping ──
    base_dir = Path(__file__).resolve().parent.parent
    gene_symbols: list[str] = []
    fca_csv = base_dir / "data" / "FCA" / "FCA_expression_logCPM.csv"
    if fca_csv.exists():
        fca = pd.read_csv(fca_csv, usecols=["flybase_id", "gene_symbol"])
        id2sym = dict(zip(fca["flybase_id"], fca["gene_symbol"]))
        gene_symbols = [id2sym.get(g, "") for g in gene_ids]
        print(f"  Gene symbols mapped: {sum(1 for s in gene_symbols if s)} / {len(gene_ids)}")
    else:
        gene_symbols = [""] * len(gene_ids)
        print("  FCA CSV not found — gene_symbol left blank")

    # ── 5. Mitochondrial gene annotation ──
    gtf_path = base_dir / "data" / "raw" / "dmel-all-r6.54.gtf"
    mito_genes = _get_mito_gene_ids(gtf_path) if gtf_path.exists() else set()
    mt_flags = [g in mito_genes for g in gene_ids]
    print(f"  Mitochondrial genes: {sum(mt_flags)} / {len(gene_ids)}")

    # ── 6. Build AnnData ──
    var_df = pd.DataFrame({
        "gene_id": gene_ids,
        "gene_symbol": gene_symbols,
        "mt": mt_flags,
    }, index=gene_ids)
    obs_df = pd.DataFrame({
        "sample_id": sample_ids,
        "total_counts": np.asarray(X.sum(axis=1)).ravel().astype(np.int64),
    }, index=cell_ids)

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.layers["counts"] = adata.X.copy()

    # ── 7. QC plots on full data ──
    print("Generating QC plots on full data ...")
    qc_fig_path = plot_dir / "qc_summary.png"
    adata = filter_cells(adata, top_total_fraction=1.0, plot_qc=True)
    # filter_cells with top_total_fraction=1.0 keeps all cells, only adds QC metrics + plots
    if plt.get_fignums():
        plt.gcf().savefig(str(qc_fig_path), dpi=150, bbox_inches="tight")
        print(f"  QC plots saved to: {qc_fig_path}")
    plt.close("all")

    # ── 8. Promoter match + gene-level split ──
    # Let filter_genes write CSVs, then apply the gene mask to our adata directly
    # (filter_genes's return may drop obs/layers due to AnnData copy semantics)
    sc.pp.filter_genes(adata, min_counts=10)
    _adata_filtered = filter_genes(adata, output_dir=out_dir)
    keep_genes = set(_adata_filtered.var["gene_id"])
    keep_mask = [g in keep_genes for g in adata.var["gene_id"]]
    adata = adata[:, keep_mask].copy()
    del _adata_filtered
    adata.layers["counts"] = adata.X.copy()  # ensure counts layer matches filtered X
    print(f"  After promoter match: {adata.shape}")

    # ── 9. Top cells count filter ──
    n_keep = max(1, int(np.ceil(adata.n_obs * top_total_fraction)))
    cell_totals = np.asarray(adata.X.sum(axis=1)).ravel()
    top_cells = np.argsort(cell_totals)[::-1][:n_keep]
    adata_hq = adata[top_cells].copy()
    print(f"  HQ: {adata_hq.shape}  (top {int(top_total_fraction*100)}% cells + min_counts=3)")


    # ── 10. Gene check ──
    gene_check_path = out_dir / "gene_check.csv"
    rng = np.random.default_rng(seed)
    hq_n = adata_hq.n_vars
    check_idx = rng.choice(hq_n, size=min(10, hq_n), replace=False)
    check_rows = []
    for gi in check_idx:
        col = adata_hq.X[:, gi]
        gene_expr = col.toarray().ravel() if hasattr(col, "toarray") else col
        top_cell_idx = int(np.argmax(gene_expr))
        check_rows.append({
            "gene_id": adata_hq.var["gene_id"].iloc[gi],
            "gene_symbol": adata_hq.var["gene_symbol"].iloc[gi],
            "mt": adata_hq.var["mt"].iloc[gi],
            "mean_expr": float(np.mean(gene_expr)),
            "max_expr": float(np.max(gene_expr)),
            "top_cell": adata_hq.obs_names[top_cell_idx],
        })
    pd.DataFrame(check_rows).to_csv(gene_check_path, index=False)
    print(f"  Gene check saved to: {gene_check_path}")

    # ── 11. Save ──
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "integrated_data_full.h5ad"
    hq_path = out_dir / "integrated_data.h5ad"
    adata.write_h5ad(full_path)
    adata_hq.write_h5ad(hq_path)
    print(f"\nSaved:")
    print(f"  Full: {full_path}  ({adata.shape})")
    print(f"  HQ:   {hq_path}  ({adata_hq.shape})")

    return adata



def build_h5ad_from_mtx(mtx_dir: Path) -> ad.AnnData:
    """从 E-MTAB-10519 raw MTX 构建 AnnData h5ad，含 QC 绘图和细胞过滤。

    Parameters
    ----------
    mtx_dir : Path
        MTX 文件所在目录 (e.g. data/E-MTAB-10519-raw/)
    out_path : Path
        输出 h5ad 文件路径 (e.g. data/emtab_processed/integrated_data.h5ad)
    """
    base_dir = Path(__file__).resolve().parent.parent
    if mtx_dir.exists():
        print("\n" + "=" * 60)
        print("  Processing E-MTAB-10519 raw MTX ...")
        print("=" * 60)
        build_from_mtx(
            mtx_dir=mtx_dir,
            out_dir=base_dir / "data" / "emtab_processed",
        )
    return

def build_h5ad_from_loom(loom_dir: Path) -> ad.AnnData:
    """从 Loom 文件构建 AnnData h5ad，含 QC 绘图,细胞过滤,基因过滤和高变基因选择。

    Parameters
    ----------
    loom_dir : Path
        Loom 文件所在目录 (e.g. data/raw/)
    out_path : Path
        输出 h5ad 文件路径 (e.g. data/emtab_processed/integrated_data.h5ad)
    """
    base_dir = Path(__file__).resolve().parent.parent
    # rawdata = build_integrated_data()
    # # %%
    # rawdata = add_gene_annotation(rawdata)
    # # %%
    # filtered_data = filter_cells(rawdata)
    # # %%
    # filtered_data = filter_genes(filtered_data)
    # # %%
    filtered_data = sc.read_h5ad(base_dir / "data" / "processed" / "integrated_data.h5ad")
    filtered_data = compute_cpm(filtered_data)
    high_quality_data = draw_high_quality_samples(filtered_data)
    print("high quality data shape: ", high_quality_data.shape)
    high_quality_data = filter_genes(high_quality_data)
    # %%
    #filtered_data.write_h5ad(base_dir / "data" / "processed" / "log_integrated_data.h5ad")
    high_quality_data.write_h5ad(base_dir / "data" / "processed" / "log_integrated_data_hvg.h5ad")
    return

# %%
def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    build_h5ad_from_loom(loom_dir = base_dir / "data" / "raw")

    # E-MTAB-10519 raw MTX processing
    mtx_dir = base_dir / "data" / "E-MTAB-10519-raw"
    build_h5ad_from_mtx(mtx_dir=mtx_dir)

if __name__ == "__main__":
    main()

# %%
