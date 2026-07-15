# Project Overview

This repository develops models for predicting Drosophila gene expression from promoter sequence features and single-cell RNA-seq expression context.

## Current Workflow

- Primary data focus: `umi_E-MTAB-10519-hqcells` and `umi_E-MTAB-10519-hqcells_aug15`.
- Current split direction: use promoter gene split plus cell split files derived from E-MTAB hqcells annotations.
- Current VAE direction: pretrain scVI/VAE on `umi_E-MTAB-10519-hqcells` train cells only, then use the encoder in downstream expression-prediction models.
- Current fine-tune direction: freeze VAE at training start and enable VAE fine-tune at epoch 60 to test whether validation/test metrics improve after loss stabilization.
- Expression matrix convention: VAE or ZINB paths use UMI count input (`counts` layer by default). No-VAE scalar-loss paths use precomputed logCPM expression input (`logcpm` by default, with `cpm` accepted as a backward-compatible logCPM layer name). Scalar-loss targets read the precomputed logCPM target layer by default; ZINB targets remain raw counts.
- Evaluation should include standard test metrics and input ablation for promoter/expression contribution.
- Stage 1 sequence-interaction evaluation uses `scripts/model_compare.py stage1-sequence-interaction` to
  apply the same promoter checkpoint to real promoter and matched
  `control_sequence` inputs. It reports per-gene cell-wise variance of the
  real-minus-control prediction effect and its correlation with residuals from
  a separate expression-only checkpoint.
- Current Stage 1 correlation evidence: paired hierarchical bootstrap across seeds 1/7/42 shows promoter-minus-intergenic Pearson deltas of `0.1093 [0.0960, 0.1187]` per cell and `0.0259 [0.0194, 0.0346]` per gene; full records are under `outputs/stage1/summary/`.
- Promoter Stage 1 assets in `data/promoter_stage1_v1` now store 420 bp `sequence`, `positive_sequence`, and `control_sequence` windows; model input remains 400 bp via `--sequence-length 400`, with train-time random crop enabled by `--promoter-shift-max 20` and centered val/test crops.
- Stage2 and later promoter-derived sequence assets should reuse the Stage1 `gene_splits.tsv`, `cell_*.txt`, `frozen_eval_cells.tsv`, and `input_gene_panel_train.txt`; use `scripts/build_sequence_assets.py reuse` to re-extract wider promoter/control/positive sequence windows without rerunning HVG, rebuilding the full gene table, or remaking splits.
- Stage 2 contrastive sweeps use the active remote `configs/config.json` base profile: combined loss with `pearson_lambda=5`, gated fusion, explicit logCPM scalar inputs/targets with count denominators, fixed LR `5e-4` after warmup, 80-epoch cap, EMA `0.9`, and full cached 128,000-pair train/validation passes. Only contrastive weights, positive/control sequence columns, and Stage 2 random crops are sweep-specific.
- The GPU-cache/config-aligned Stage 2 rerun uses experiment IDs prefixed `stage2_cfgcache_` under `outputs/stage2/`, preserving the earlier incomplete `stage2_cw*` attempts for audit rather than overwriting them.
- Every new Stage 2 run now performs the standard test followed by top-1000 sequence mutagenesis with the Stage 1-comparable 2%/20-pair gene cap; position importance and de novo motif files are written under that run's `test/sequence_mutagenesis/`, and motif claims require `support_genes >= 5` in the Stage 2 summary gate.
- The active Stage 2 rerun runs in remote tmux session `promoter_stage2`, overwrites the user-approved incomplete `stage2_cw*` directories, and writes the matched 12-run weight/seed grid to `outputs/stage2/`.
- If the no-projection grid has no clear matched improvement, `hpc/stage2_projection_sweep.sh` provides a three-seed `projection_dim=64`, two-layer fallback using the identical Stage 2 base configuration.
- Stage 2 completed a 15-run grid including matched `cw=0` and weights `0.05/0.10/0.20/0.40`. Mean Pearson/Spearman were `0.3220/0.3197` at `cw=0` and `0.3349/0.3270` at `cw=0.40`; mean MSE increased from `5.0508` to `5.1048`. The high-weight contrastive branch improves correlation but has a residual MSE tradeoff.
- Historical Stage 2 mutation testing used a top-1000 selection and a 10%/100-pair gene cap. The standardized current and future contract is 2%/20 pairs per gene; the historical maximum de novo motif support was `support_genes=1`, so no motif claim is accepted before the standardized seed-7 rerun completes.
- A gene-balanced sensitivity analysis, keeping at most one important mutation window per gene, raised the observed maximum support only to `2`; the `support_genes >= 5` gate still failed across all 15 runs. The diagnostic is written by `scripts/summarize_gene_balanced_motifs.py` to `stage2_gene_balanced_motif_summary.csv` and per-run gene-balanced motif files.
- The projection-head fallback checkpoints were created successfully, but automatic post-training testing initially failed because `model_test.py` did not rebuild the configured projection head. The loader and regression test now cover this checkpoint format; the post-fix three-seed retest completed with mean MSE/Pearson/Spearman `5.054070/0.317593/0.315024`, below matched `cw=0` correlations, while gene-balanced motif support remained at most `2`. No motif claim passes the `support_genes >= 5` gate.
- The Stage 1-style seed-7 Stage 2 ablation compares matched `cw=0.40` against `cw=0`: global Pearson improves from `0.326239` to `0.333287`, and paired per-cell Pearson improves by `0.006881 [0.006352, 0.007408]`, but paired per-gene Pearson is `-0.002591 [-0.006683, 0.001327]`. Treat this as a cell-context benefit, not uniform gene-level evidence.
- The same Stage 2 ablation writes per-cell/per-gene Pearson violin PNG/SVG files plus distribution/statistics/extreme-point CSVs under its summary directory; the visual comparison follows the Stage 1 reporting convention.
- New script organization exposes `scripts/build_sequence_assets.py` as the public asset-builder workflow (`full`, `reuse`, `utr`) and `scripts/model_compare.py` as the public comparison workflow (Stage 1 bootstrap/sequence interaction and Stage 2 summary/gene-balanced motifs/ablation). Legacy implementation filenames remain compatibility-only.
- `scripts/model_compare.py report` is the consolidated reporting workflow. It validates only manifest-registered Stage 1/2 summaries, paired statistics, and PNG figures; audits checkpoint/test provenance; groups configurations differing only by seed; and embeds stage figures in one workbook. Existing stage comparison commands remain independently runnable. Use `report --refresh` only to explicitly rerun the manifest-declared comparisons before export; raw paired per-gene/per-cell rows stay outside the workbook.
- `scripts/data_sanity.py` is the single h5ad/promoter integrity checker; temporary augmentation, fixed-path TPM-reversal, raw-matrix inspection, and the former `project_test.py` entry point have been removed.
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
- Remote environment: `/PROJ5/liangn_zxy/envs/promodel`, created with micromamba 2.8.1 and currently using Python 3.10 plus PyTorch 2.5.1 CUDA 12.1 for compatibility with the NVIDIA 535 driver.
## Standing Project Guide

Before changing data processing, split logic, training parameters, model architecture, or evaluation workflow, compare the proposed change with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`.

Record any intentional deviation from that guide in `CHANGELOG.md` and `LOG.md`.

