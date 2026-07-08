# CHANGELOG

## 2026-07-08

- Added remote server operating rules to `AGENTS.md` for the `sulab7g-zxy` SSH target, including approved `/PROJ5/liangn_zxy/` directory purposes, transfer conventions, explicit GPU selection, and account/filesystem safety boundaries.
- Added the remote training infrastructure layout to `project_overview.md`.

## 2026-07-08

- Added `scripts/summary_stage1.py` to apply one promoter checkpoint to both
  real promoter and matched `control_sequence` inputs on frozen test cells;
  only the expression-only residual uses a separate checkpoint.
- Added per-gene real-minus-control sequence-effect variance and correlation
  with expression-only residuals, matched-protocol validation, streaming
  statistics, GPU cached inference, CSV/JSON outputs, a diagnostic plot, and
  unit coverage for the statistical calculations.

## 2026-07-06

- Replaced the post-warmup cosine decay in `scripts/train.py` with a constant learning rate: five linear warmup epochs from 10% of the configured rate, followed by the configured rate unchanged.
- Added `--warmup-epochs` (default 5), scheduler unit coverage, and a separate Stage 1 promoter experiment configuration named `stage1_shift420_combined_fixedlr_seed7` with LR `5e-4` and `ema_alpha=0.9`.
- Recorded `warmup_epochs` in run-local configs and completed the fixed-LR seed-7 promoter run plus full 5,543,936-pair test (RMSE 2.255851, Pearson 0.289195, Spearman 0.294108).
- Added a same-protocol full test for the prior `stage1_shift420_combined_seed7` cosine run (RMSE 2.297328, Pearson 0.313015, Spearman 0.308607) to support direct outcome comparison.
- Replaced the `ema_alpha=0.9` fixed-LR output directory with an `ema_alpha=0.9999` rerun under the same experiment name. Training completed at epoch 62; full testing is pending because the GPU entered an unrecoverable CUDA launch/driver error state after training.

## 2026-07-05

- Replaced `configs/config.json` with a `train.py --prior_config`-compatible Stage 1 combined-loss configuration using `data/promoter_stage1_v1`, 420 bp shift-compatible assets, a 400 bp model crop, shift20 augmentation, seed 7, EMA 0.9999, nonzero weight 2, and the historical shift420 checkpoint/evaluation settings.
- Set the matched combined-loss run to 80 epochs as requested. This is an intentional deviation from the guide's preference for no hard epoch cap; early stopping remains enabled with patience 32 for comparability with the shift420 screening runs.

## 2026-07-03

- Recorded creation and GPU validation of the HPC-defined `promodel` conda environment at `/home/jovyan/.conda/envs/promodel`; no project code, data, model assumptions, or experiment configuration changed.

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

## 2026-07-06

- Integrated the seed-7 MSE/combined/fixed-learning-rate ablation into the Stage 1 summary pipeline, including global metrics, training-trajectory metadata, paired per-cell/per-gene Pearson bootstrap intervals, and a four-panel figure.
- Added a Chinese ablation interpretation record and a unit test for strict ID-paired delta construction.

- Added `scripts/summarize_stage1_bootstrap.py` for paired promoter-vs-baseline Pearson bootstrap analysis at per-cell and per-gene levels.
- Added deterministic per-seed and hierarchical three-seed confidence intervals with raw paired delta records under `outputs/stage1/summary/`.
- Added unit tests for paired and hierarchical bootstrap helpers.

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
