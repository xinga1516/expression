# CHANGELOG

## 2026-07-02

- Refreshed `outputs/stage1_shift420_summary.csv` directly from nine run-local `config.json` and `test_metrics.json` files, including sample counts, test-cell/gene counts, checkpoint metric, and zero/nonzero RMSE fields.
- Added Stage 1 shift420 interpretation to `docs/RUN_RESULTS_SUMMARY.md`, registered the nine completed full-test runs in `records/registry.tsv`, and marked unrelated E-MTAB/VAE artifacts as excluded from the current leaderboard.
- Added GPU cached runtime cropping for 420 bp promoter assets so Stage 1 can use `--gpu-cache-dataset` while preserving train-time random 400 bp crops from wider windows.
- Changed CUDA `model_test.py` evaluation to use GPU sequential cache by default.
- Cleared old Stage 1/Stage 2 output directories and regenerated the Stage 1 nine-run comparison on 420 bp assets.
- Added `outputs/stage1_shift420_summary.csv` with full-test metrics for expression-matched, real-promoter, and matched-intergenic controls across seeds 1/7/42.

- Added `--promoter-window-length` to `scripts/build_promoter_stage1_assets.py` so Stage 1 promoter assets can be regenerated with wider-than-model-input sequence windows.
- Changed asset-builder default `--control-attempts` from 2000 to 200 for faster matched intergenic control generation.
- Optimized intergenic overlap checks with a sorted interval index and prefix max-end lookup.
- Updated `data/promoter_stage1_v1` promoter split files in place to 420 bp `sequence`, `positive_sequence`, and `control_sequence` windows while preserving existing gene split, cell split, input gene panel, and matched control identities.

Human-readable record of repository changes.

## 2026-07-01

- Added explicit expression/target matrix source controls to training and model testing:
  - `--expression-layer`, `--expression-transform`, `--target-count-layer`, and `--target-transform`.
- Added `--target-value-layer` and changed defaults to use precomputed logCPM directly: VAE/ZINB expression input uses UMI counts; no-VAE scalar-loss expression input uses precomputed logCPM; scalar targets use precomputed logCPM; ZINB targets use raw counts.
- Updated `MyDataset`, GPU cached loading, model testing, mutagenesis helpers, and scatter plotting so expression inputs, scalar target values, and target/count sources are separated consistently.
- Updated tiny AnnData fixtures and unit tests for `counts` and `cpm` layers.
- Updated `AGENTS.md` with AnnData layer conventions and CLI usage.
- Documented the project runtime requirement in `AGENTS.md`: run project Python work in WSL with conda environment `promodel_wsl`; do not substitute Windows Python for `.h5ad`, AnnData, training, evaluation, or data-building checks.
- Added project memory rules to `AGENTS.md`.
- Added `TODO.md`, `LOG.md`, and `project_overview.md` as standing project-tracking files.
- Documented that future changes must be checked against `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`.
