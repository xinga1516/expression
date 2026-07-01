# CHANGELOG

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
