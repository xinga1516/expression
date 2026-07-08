# Project Overview

This repository develops models for predicting Drosophila gene expression from promoter sequence features and single-cell RNA-seq expression context.

## Current Workflow

- Primary data focus: `umi_E-MTAB-10519-hqcells` and `umi_E-MTAB-10519-hqcells_aug15`.
- Current split direction: use promoter gene split plus cell split files derived from E-MTAB hqcells annotations.
- Current VAE direction: pretrain scVI/VAE on `umi_E-MTAB-10519-hqcells` train cells only, then use the encoder in downstream expression-prediction models.
- Current fine-tune direction: freeze VAE at training start and enable VAE fine-tune at epoch 60 to test whether validation/test metrics improve after loss stabilization.
- Expression matrix convention: VAE or ZINB paths use UMI count input (`counts` layer by default). No-VAE scalar-loss paths use precomputed logCPM expression input (`logcpm` by default, with `cpm` accepted as a backward-compatible logCPM layer name). Scalar-loss targets read the precomputed logCPM target layer by default; ZINB targets remain raw counts.
- Evaluation should include standard test metrics and input ablation for promoter/expression contribution.
- Stage 1 sequence-interaction evaluation uses `scripts/summary_stage1.py` to
  apply the same promoter checkpoint to real promoter and matched
  `control_sequence` inputs. It reports per-gene cell-wise variance of the
  real-minus-control prediction effect and its correlation with residuals from
  a separate expression-only checkpoint.
- Current Stage 1 correlation evidence: paired hierarchical bootstrap across seeds 1/7/42 shows promoter-minus-intergenic Pearson deltas of `0.1093 [0.0960, 0.1187]` per cell and `0.0259 [0.0194, 0.0346]` per gene; full records are under `outputs/stage1/summary/`.
- Promoter Stage 1 assets in `data/promoter_stage1_v1` now store 420 bp `sequence`, `positive_sequence`, and `control_sequence` windows; model input remains 400 bp via `--sequence-length 400`, with train-time random crop enabled by `--promoter-shift-max 20` and centered val/test crops.
- Training now uses linear LR warmup for `--warmup-epochs` (default 5) followed by a constant configured LR; the previous post-warmup cosine decay has been removed. The current fixed-LR promoter experiment uses LR `5e-4` and `ema_alpha=0.9`.
- `stage1_shift420_combined_fixedlr_seed7` now refers to the replacement `ema_alpha=0.9999` fixed-LR run, which completed at epoch 62. Its full test is pending GPU recovery; the superseded `ema_alpha=0.9` full-test result remains historical only.
- With EMA matched at 0.9999, fixed LR and cosine have close validation outcomes, but cosine is slightly better: minimum RMSE 1.839322 vs 1.844529, maximum Pearson 0.267689 vs 0.267607, and maximum Spearman 0.252792 vs 0.250772.
- `configs/config.json` is the matched seed-7 combined-loss prior for comparing against `stage1_shift420_promoter_seed7`. It uses the same Stage 1 v1 data/split, sampling, EMA, nonzero weighting, checkpoint, and evaluation settings, with combined loss (`pearson_lambda=5`) and an explicitly requested 80-epoch cap. Run it with `python scripts/train.py --exp_name stage1_shift420_combined_seed7 --prior_config configs/config.json` from the project conda environment.

- Latest Stage 1 shift420 comparison outputs are `outputs/stage1_shift420_*`; `outputs/stage1_shift420_summary.csv` summarizes the nine full-test runs from run-local config/metrics, and `docs/RUN_RESULTS_SUMMARY.md` plus `records/registry.tsv` hold the interpreted current leaderboard. Real promoter outperforms expression-matched and matched-intergenic controls on Pearson/Spearman; matched intergenic has slightly lower mean RMSE and should be reported as a caveat.

- Seed-7 training-strategy ablation is recorded under outputs/stage1/summary/stage1_training_ablation_*. Combined loss improves paired per-cell/per-gene Pearson over MSE; fixed LR has the best global metrics and per-cell Pearson, but lower per-gene Pearson than cosine-decayed combined. This remains a single-seed result, confounded by the MSE run's shorter training budget.

## Remote Training Infrastructure

- Remote access uses only the `sulab7g-zxy` SSH alias.
- Server code and small inputs live under `/PROJ5/liangn_zxy/work/`; environments under `envs/`; outputs/checkpoints under `runs/`; caches under `cache/`; temporary files under `scratch/`.
- Project data, environments, models, and outputs must not be written under `/home`.
- GPU jobs must explicitly set `CUDA_VISIBLE_DEVICES` and should not occupy all eight GPUs by default.
- `/ssd` is unavailable unless an administrator explicitly authorizes its use.
## Standing Project Guide

Before changing data processing, split logic, training parameters, model architecture, or evaluation workflow, compare the proposed change with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`.

Record any intentional deviation from that guide in `CHANGELOG.md` and `LOG.md`.
