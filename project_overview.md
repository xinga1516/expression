# Project Overview

This repository develops models for predicting Drosophila gene expression from promoter sequence features and single-cell RNA-seq expression context.

## Current Workflow

- Primary data focus: `umi_E-MTAB-10519-hqcells` and `umi_E-MTAB-10519-hqcells_aug15`.
- Current split direction: use promoter gene split plus cell split files derived from E-MTAB hqcells annotations.
- Current VAE direction: pretrain scVI/VAE on `umi_E-MTAB-10519-hqcells` train cells only, then use the encoder in downstream expression-prediction models.
- Current fine-tune direction: freeze VAE at training start and enable VAE fine-tune at epoch 60 to test whether validation/test metrics improve after loss stabilization.
- Expression matrix convention: VAE or ZINB paths use UMI count input (`counts` layer by default). No-VAE scalar-loss paths use precomputed logCPM expression input (`logcpm` by default, with `cpm` accepted as a backward-compatible logCPM layer name). Scalar-loss targets read the precomputed logCPM target layer by default; ZINB targets remain raw counts.
- Evaluation should include standard test metrics and input ablation for promoter/expression contribution.
- Promoter Stage 1 assets in `data/promoter_stage1_v1` now store 420 bp `sequence`, `positive_sequence`, and `control_sequence` windows; model input remains 400 bp via `--sequence-length 400`, with train-time random crop enabled by `--promoter-shift-max 20` and centered val/test crops.

- Latest Stage 1 shift420 comparison outputs are `outputs/stage1_shift420_*`; `outputs/stage1_shift420_summary.csv` summarizes the nine full-test runs from run-local config/metrics, and `docs/RUN_RESULTS_SUMMARY.md` plus `records/registry.tsv` hold the interpreted current leaderboard. Real promoter outperforms expression-matched and matched-intergenic controls on Pearson/Spearman; matched intergenic has slightly lower mean RMSE and should be reported as a caveat.

## Standing Project Guide

Before changing data processing, split logic, training parameters, model architecture, or evaluation workflow, compare the proposed change with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`.

Record any intentional deviation from that guide in `CHANGELOG.md` and `LOG.md`.
