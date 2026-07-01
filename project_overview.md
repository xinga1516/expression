# Project Overview

This repository develops models for predicting Drosophila gene expression from promoter sequence features and single-cell RNA-seq expression context.

## Current Workflow

- Primary data focus: `umi_E-MTAB-10519-hqcells` and `umi_E-MTAB-10519-hqcells_aug15`.
- Current split direction: use promoter gene split plus cell split files derived from E-MTAB hqcells annotations.
- Current VAE direction: pretrain scVI/VAE on `umi_E-MTAB-10519-hqcells` train cells only, then use the encoder in downstream expression-prediction models.
- Current fine-tune direction: freeze VAE at training start and enable VAE fine-tune at epoch 60 to test whether validation/test metrics improve after loss stabilization.
- Expression matrix convention: VAE or ZINB paths use UMI count input (`counts` layer by default). No-VAE scalar-loss paths use precomputed logCPM expression input (`logcpm` by default, with `cpm` accepted as a backward-compatible logCPM layer name). Scalar-loss targets read the precomputed logCPM target layer by default; ZINB targets remain raw counts.
- Evaluation should include standard test metrics and input ablation for promoter/expression contribution.

## Standing Project Guide

Before changing data processing, split logic, training parameters, model architecture, or evaluation workflow, compare the proposed change with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`.

Record any intentional deviation from that guide in `CHANGELOG.md` and `LOG.md`.
