# %%
import loompy
import scanpy as sc
import h5py
import anndata as ad
import pandas as pd
import numpy as np
from Bio import SeqIO
from scipy import sparse
import re
import pyranges as pr
import matplotlib.pyplot as plt
from pathlib import Path

def hdf5_matrix_to_sparse(
    mat,
    chunk_size=5000,
    transpose=True,
    dtype=np.float32,
):
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


def add_gene_annotation(rawdata, synonym_path=None, gtf_path=None):
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

def build_integrated_data():
    '''
    Build integrated single-cell RNA sequencing data.
    '''
    # check the file structure
    f = h5py.File("raw/s_fca_biohub_all_wo_blood_10x.loom", "r")
    print(list(f.keys())) # output ['attrs', 'col_attrs', 'col_graphs', 'layers', 'matrix', 'row_attrs', 'row_graphs']

    # reorganize data structure
    X = f["matrix"]  # shape: genes × cells
    # change the dense matrix to sparse matrix
    #print(sparse.issparse(X));print(X)
    X_sparse = hdf5_matrix_to_sparse(X)
    print(sparse.issparse(X))
    print(X_sparse.data.max()) # check if the date is the UMI count: yes
    # build eligible anndata with sparse matrix, var, and obs. 
    # add the flybase gene id and gene length to the anndata
    row_attrs = list(f['row_attrs'].keys())
    col_attrs = list(f['col_attrs'].keys())
    attrs = list(f['attrs'].keys())
    gene_names = f['row_attrs']['Gene'][:].astype(str)
    cell_ids = f['col_attrs']['CellID'][:].astype(str)
    # print(row_attrs)
    # print(col_attrs)
    print(gene_names)
    print(cell_ids)
    rawdata = ad.AnnData(
        X = X_sparse,
        obs = pd.DataFrame(index=cell_ids),
        var = pd.DataFrame(index=gene_names)
    )
    return rawdata

def filter_cells(rawdata, top_total_fraction=0.10, plot_qc=True):
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
    df,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    output_dir=None,
):
    """
    按每条染色体上的物理位置排序后，按样本个数比例切分为连续区间。
    这样 train/val/test 都是位置连续块，同时数量由比例控制。

    input:
        promoter dataframe with columns:
        ["gene_id", "chrom", "start", "end", "strand", "sequence", "length"]
    output:
        写出 promoter_train.csv / promoter_val.csv / promoter_test.csv
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
    block_chrom = ['2R','3R']

    for chrom, sub in data.groupby("chrom", sort=True):
        if chrom not in block_chrom:
            train_genes.update(sub["gene_id"].tolist())
            print(f"{chrom}: total={len(sub)}, span ignored -> all train")
            continue

        sub = sub.sort_values("mid").copy()
        n = len(sub)
        if n == 0:
            continue

        # 按“样本个数比例”切分（连续索引块）
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        # 极小染色体时保证总数一致且不报错
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_idx_end = n_train
        val_idx_end = n_train + n_val

        train_part = sub.iloc[:train_idx_end]
        val_part = sub.iloc[train_idx_end:val_idx_end]
        test_part = sub.iloc[val_idx_end:]

        train_genes.update(train_part["gene_id"].tolist())
        val_genes.update(val_part["gene_id"].tolist())
        test_genes.update(test_part["gene_id"].tolist())

        min_pos = int(sub["mid"].min())
        max_pos = int(sub["mid"].max())
        print(
            f"{chrom}: total={n}, "
            f"train={len(train_part)}, val={len(val_part)}, test={len(test_part)}, "
            f"range=[{min_pos}, {max_pos}]"
        )

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
    outdir = Path(output_dir) if output_dir is not None else (base_dir / "data" / "highquality")#(base_dir / "data" / "processed")
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

def filter_genes(rawdata):
    '''
    For anndata: remove genes without promoter sequence, and output file as integrated_data.h5ad
    and split the data into train/val/test sets by gene, and output promoter csv files for each set.
    '''
    base_dir = Path(__file__).resolve().parent.parent
    promoters = SeqIO.parse(base_dir / "data" / "processed" / "promoters.fa", 'fasta')
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

    # remove genes that have the same flybase gene_id
    print(rawdata.shape)
    df = rawdata.var
    filtered_data = rawdata[:,df.duplicated(subset="gene_id", keep=False) == False]
    gene_singlecell = set((filtered_data.var)['gene_id'])

    # check if genes are comprehensive in promoters.fa
    insect = gene_singlecell - gene_promoters
    insect1 = gene_promoters - gene_singlecell
    eligible_genes = gene_singlecell & gene_promoters
    print("{0} genes only in {1} single cell data : ".format(len(insect), len(gene_singlecell)), insect)
    print("{0} genes only in {1} promoters data : ".format(len(insect1), len(gene_promoters)), insect1)

    before = filtered_data.shape
    flag = [(i in eligible_genes) for i in filtered_data.var.gene_id]
    filtered_data = filtered_data[:,flag]
    print("genes from {0} to {1}".format(before[1], filtered_data.shape[1]))

    # merged_df = pd.merge(filtered_data.var,promoters,how='left',on='gene_id')
    # print(merged_df['sequence'])
    # filtered_data.var["seq"] = merged_df['sequence']
    # filtered_data.var["seq"] = (
    #     filtered_data.var["gene_id"]
    #     .map(promoters.set_index("gene_id")["sequence"])
    # )
    # filtered_data.write_h5ad("processed/integrated_data.h5ad")

    filtered_promoters = promoters[
        promoters["gene_id"].isin(eligible_genes)
    ]
    print(f"promoters number: {len(filtered_promoters)}")
    df = filtered_promoters
    df["start"] = df["start"].astype(int)
    split_train_val(df)

    return filtered_data

def compute_tpm(rawdata):
    sc.pp.normalize_total(rawdata, target_sum=1e6) # 将每个细胞的总表达量归一化到 1,000,000（TPM）
    sc.pp.log1p(rawdata) # 对归一化后的数据进行 log1p 转换（log(x + 1)），以减小数据范围并处理零值。
    # # using gene length to compute RPKM/FPKM
    # gene_length_kb = rawdata.var["gene_length"] / 1e3
    # rawdata.X = rawdata.X / gene_length_kb.values[np.newaxis, :]
    return rawdata

def draw_high_quality_samples(rawdata, top_total_fraction=0.10):
    ''' delete genes with zero expression across all cells, 
    and select highly variable genes'''
    sc.pp.highly_variable_genes(rawdata, flavor="seurat_v3", n_top_genes=2000)
    rawdata = rawdata[:, rawdata.var['highly_variable']]
    return rawdata

# %%
def main():
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
    filtered_data = compute_tpm(filtered_data)
    high_quality_data = draw_high_quality_samples(filtered_data)
    print("high quality data shape: ", high_quality_data.shape)
    high_quality_data = filter_genes(high_quality_data)
    # %%
    #filtered_data.write_h5ad(base_dir / "data" / "processed" / "log_integrated_data.h5ad")
    high_quality_data.write_h5ad(base_dir / "data" / "processed" / "log_integrated_data_hvg.h5ad")

if __name__ == "__main__":
    main()

# %%
