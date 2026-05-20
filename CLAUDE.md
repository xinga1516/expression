# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Train a model (standard MSE/combined loss)
python scripts/train.py --exp_name my_run --data highquality

# Train with ZINB loss on UMI count data (auto samples_per_epoch)
python scripts/train.py --exp_name my_run --data umi_processed --loss zinb --cell-ratio 0.1 --nonzero-ratio 0.5

# Train with ZINB loss + manual samples_per_epoch
python scripts/train.py --exp_name my_run --data umi_processed --loss zinb --cell-ratio 0.1 --nonzero-ratio 0.5 --samples-per-epoch 500000

# Train with custom config
python scripts/train.py --exp_name my_run --data highquality --model SimpleGeneModel --batch-size 128 --epochs 30 --learning-rate 1e-4 --hidden-size 32

# Resume training from last checkpoint
python scripts/train.py --exp_name my_run --data highquality --resume outputs/my_run/checkpoints/last.ckpt

# Dry-run (test data pipeline + gradient flow on 50 batches)
python scripts/train.py --exp_name my_run --data highquality --dryrun

# Evaluate on val/test split
python scripts/evaluate.py --split val --checkpoint outputs/my_run/checkpoints/best_model.safetensors

# Plot results from a completed experiment (loss curves, metrics, scatter)
python scripts/plot_results.py --exp_name my_run --data highquality

# Plot loss curves and scatter plot after training
python scripts/train.py --exp_name my_run --data highquality --plot-loss

# Data processing pipeline
python scripts/process_data.py

# Check data sanity (sparsity, distributions)
python scripts/data_sanity.py

# Submit to HPC cluster (SLURM)
sbatch hpc/command.sh
```

## Project Structure

Gene expression prediction from promoter sequences and scRNA-seq data. The model takes a one-hot encoded promoter sequence (400bp × 5 channels ACGTN) and a cell's full expression profile (all genes except the target), then predicts the target gene's expression level.

### Data Pipeline (`scripts/process_data.py`)
- Loom file → h5ad (AnnData) with FlyBase gene annotations
- Cell filtering (top 10% by total counts) and QC
- Gene filtering (keep only genes present in both scRNA-seq and promoter FASTA)
- TPM normalization + log1p + highly variable gene selection (Seurat v3, top 2000)
- Train/val/test split by genomic position (80/10/10 on chromosomes 2R/3R, all others → train)
- Output: `data/highquality/` (integrated_data.h5ad, promoter_train/val/test.csv)

### Dataset (`src/dataset.py`)
- `PromoterOneHotEncoder`: encodes DNA sequences to (400, 5) one-hot tensors
- `MyDataset`: yields `(promoter_tensor, all_expr_except_target, target_expr)` tuples. The full combinatorial dataset size is `n_promoters × n_cells`. Uses pre-encoded promoter tensors and CSR sparse matrix for efficient expression lookups
- `--cell-ratio`: subsample fraction of cells (e.g., 0.1 for 10%). Cells are **rotated per epoch** via `resample_cells(seed + epoch)`, so the model eventually sees all cells over multiple epochs. Controlled by `--cell-ratio`
- `--val-cell-ratio`: separate subsample ratio for validation set (default 1.0 = no subsampling)
- `resample_cells(seed)`: re-selects a random subset of cells with a new seed — called between epochs to implement cell rotation

### Model (`src/model.py`)
- `SimpleGeneModel`: LSTM over promoter (takes last hidden state) + linear MLP over expression profile → concatenate → linear output. Models auto-register via `MODEL_REGISTRY` for CLI selection via `--model`
- `LSTMmodel` / `ConvAttentionModel`: support dual output modes — `scalar` (single expression value) or `zinb` (mu_ratio, theta, pi for zero-inflated negative binomial). Output mode auto-selected from `--loss`: `zinb` → `output_mode="zinb"`, others → `output_mode="scalar"`
- ZINB heads: `mu_ratio = exp(clamp(linear(x), max=10))`, `theta = exp(clamp(linear(x), max=10))`, `pi = sigmoid(linear(x))` — clamping prevents overflow

### Training (`scripts/train.py`)

**Loss functions:**
- `weighted_mse_loss`: higher weight on non-zero expression targets
- `pearson_loss`: 1 - Pearson correlation coefficient (batch-level, select via `--loss pearson`)
- `ZINBLoss`: zero-inflated negative binomial NLL computed entirely in log-space (avoids `exp` overflow and `0*inf=NaN`). Select via `--loss zinb`. Automatically clamps model inputs and applies gradient clipping (`max_norm=1.0`)
- `pearson_mse_loss` (combined): weighted MSE + lambda × (1 - Pearson) — select via `--loss combined`
- `--pearson-lambda`: lambda weight for the Pearson term in combined loss (default 10.0)

**Data loading & samplers:**
- Three data modes: `highquality` (shuffled DataLoader), `processed` / `umi_processed` (use samplers for fixed-size epochs)
- `BalancedEpochSubsetSampler`: ensures each promoter is sampled equally per epoch
- `ZeroNonZeroSampler`: controls the ratio of zero vs non-zero expression targets via `--nonzero-ratio`. Supports `replace` parameter and auto `samples_per_epoch` from pool sizes
- `--samples-per-epoch`: number of samples per epoch. Default `0` = auto-select from pool sizes (no forced duplication). Set explicitly to override
- `--max-duplication`: when auto-selecting, allows up to N× duplication of the unique pool (default 1.0 = no duplication)
- Cell rotation: when `--cell-ratio < 1.0`, a different random subset of cells is selected each epoch via `resample_cells(seed + epoch)`. The `ZeroNonZeroSampler` rebuilds its index pools accordingly via `rebuild()`

**Training loop:**
- Cosine annealing LR scheduler, early stopping, periodic checkpointing
- Gradient clipping (`max_norm=1.0`) applied after `loss.backward()` for numerical stability
- Checkpoints saved as `.ckpt` (full state) in `outputs/<exp_name>/checkpoints/`
- Best model saved as `best_model.safetensors`
- Resume support with snapshot backup and config diff logging
- `--loss`: choose loss function (`mse`, `pearson`, `combined`, or `zinb`)

### Evaluation (`scripts/evaluate.py`)
- Computes MSE, MAE, RMSE, R2, Pearson correlation

### HPC (`hpc/`)
- SLURM submission via `command.sh` (GPU partition, conda env `promodel`)

### Coding Conventions
- All function parameters and return types must have type annotations (e.g., `def foo(x: int, y: str = "") -> bool:`)

### Dependencies
- Python 3.10, PyTorch 2.5.1, scanpy 1.11.5, anndata 0.11.4
- Full conda env in `hpc/server_env.yml`
