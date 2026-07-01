# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Runtime Environment

Use WSL for this project. The default runtime is the WSL conda environment named `promodel_wsl`.

First, running "wsl" in windows powershell.

Before running Python project commands, activate the environment in WSL:

```bash
conda activate promodel_wsl
```

Do not use Windows-side Python, Spyder Python, or Windows conda environments for `scanpy`, `anndata`, `.h5ad`, training, evaluation, mutation tests, or data-building commands. Windows-side tools may be used only for lightweight text/config inspection when WSL is unavailable.

If WSL or `promodel_wsl` is not visible from the current session, report that explicitly and avoid silently substituting another Python environment for biology/data/model checks. This prevents repeated confusion around `.h5ad` contents, AnnData layers, UMI counts, CPM/logCPM transforms, and GPU/training behavior.

## AnnData Expression Layers

For E-MTAB / UMI h5ad files, code should treat expression matrices by semantic role, not by directory name:

- `adata.layers["counts"]`: raw UMI counts. Use this for VAE/scVI input, ZINB targets, zero/non-zero sampler pools, and library-size denominators.
- precomputed logCPM layer: preferred name `logcpm`. The code also accepts `log_cpm`, `logCPM`, `log1p_cpm`, and, for backward compatibility, `cpm` as a precomputed logCPM layer. Do not apply another `log1p` transform to this layer.
- `adata.X`: fallback only when explicitly requested with `X`/`none`; do not rely on `adata.X` implicitly for current E-MTAB training.

Default training/evaluation behavior:

- `--vae-encoder ...` or `--loss zinb`: expression input uses UMI counts.
- no VAE with scalar losses (`mse`, `pearson`, `combined`): expression input uses precomputed logCPM.
- scalar targets use the precomputed logCPM target layer by default.
- ZINB targets remain raw UMI counts.

Relevant CLI controls:

```bash
--expression-layer auto|counts|logcpm|cpm|X
--expression-transform auto|none|log1p
--target-count-layer auto|counts|X
--target-value-layer auto|logcpm|cpm|none
--target-transform auto|none|log1p_cpm
```

Use `--target-transform log1p_cpm --target-value-layer none` only for legacy experiments that must recompute logCPM from counts at runtime.

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

## Project Memory and Change Records

Maintain these repository-level project memory files:

- `TODO.md`: current task list. Update it when tasks are added, completed, blocked, or reprioritized.
- `CHANGELOG.md`: human-readable change record. Every code, data-processing, configuration, experiment-script, or documentation change should add an entry describing what changed and why.
- `LOG.md`: Codex operation log. Record meaningful Codex actions such as files inspected, files changed, commands/tests run, generated artifacts, and known follow-up work.
- `project_overview.md`: concise project overview and current workflow. Update it periodically, especially after changes to data flow, training/evaluation workflow, model architecture, or major experiment conventions.
- `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`: project guide and source of truth for intended biological/modeling assumptions, parameter choices, and workflow requirements.

Before making changes, compare the planned change against `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md` and call out any conflict, missing requirement, or parameter mismatch. If a requested change intentionally deviates from the guide, record the deviation in `CHANGELOG.md` and `LOG.md`.

For every completed repository change:

- Update `CHANGELOG.md` with a dated entry.
- Update `LOG.md` with a short operation record, including tests or commands run.
- Update `TODO.md` if task status changed.
- Update `project_overview.md` when the change affects project workflow, data/model assumptions, or recommended commands.

These requirements are project-level standing instructions. They should be followed automatically in this repository; the user should not need to repeat them in each prompt.

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
