# TODO

Current project task list.

## Active

- [x] Compare seed-7 MSE, combined loss, and combined fixed-LR promoter runs and add global plus paired per-cell/per-gene results to the Stage 1 summary.

- [x] Re-run promoter Stage 1 v2 nine-model comparison after 420 bp asset update.
- [x] Record Stage 1 shift420 nine-run metrics in summary, registry, and result documents.
- [x] Use `--sequence-length 400 --promoter-shift-max 20` so train-time random crop uses the new 420 bp windows while val/test use centered crops.
- [x] Add a Stage2+ sequence-asset builder that reuses Stage1 gene/cell/input-panel splits and only re-extracts wider promoter/control/positive sequences.
- [x] Align the Stage 2 contrastive sweep launcher with the active remote `configs/config.json` profile before restarting the queue.
- [x] Monitor the Stage 2 contrastive sweep, add the matched `cw=0` baseline, and summarize metrics across all weights/seeds.
- [x] Run the Stage 2 mutagenesis/motif gate for all 15 runs; no run produced a candidate with `support_genes >= 5`, so no motif evidence is claimed.
- [x] Run the gene-balanced motif sensitivity check; maximum support remained `2`, confirming the motif gate failure is not only caused by repeated pairs from one gene.
- [x] Complete projection-head fallback evaluation at `cw=0.40` after fixing projection-head checkpoint reconstruction; it did not improve matched predictive metrics or gene-balanced motif support.
- [x] Add and run a Stage 1-style paired seed-7 ablation for Stage 2 `cw=0.40` versus `cw=0`.
- [ ] Run the standardized Stage 1-comparable mutation retest for each Stage 2 parameter group using only seed 7.
- [ ] Design a follow-up motif-specific de-biasing experiment; current Stage 2 improves correlation but does not satisfy the cross-gene motif-support gate.
- [ ] Re-run the E-MTAB workflow after deleting old `outputs/` results:
  - Pretrain VAE on `umi_E-MTAB-10519-hqcells` using only `cell_train.txt`.
  - Train `umi_E-MTAB-10519-hqcells` with cell split and delayed VAE fine-tune at epoch 60.
  - Train `umi_E-MTAB-10519-hqcells_aug15` with the same cell split and delayed VAE fine-tune at epoch 60.
  - Evaluate both models with standard test metrics and input ablation.
- [ ] Compare whether VAE fine-tune after loss stabilization improves validation/test performance.
- [ ] Compare `hqcells` vs `hqcells_aug15` model performance under the same split and training profile.

## Maintenance

- [x] Add a manifest-driven two-layer model-comparison report: retain standalone Stage 1/2 comparisons, and use `model_compare.py report` or `report --refresh` to create one validated workbook.
- [x] Consolidate public sequence-asset and model-comparison script entry points while preserving historical command compatibility.
- [x] Remove obsolete one-off scripts and consolidate project data checks under `scripts/data_sanity.py`.
- [ ] Keep `project_overview.md` current after workflow, dataset, model, or experiment convention changes.
- [ ] Keep `CHANGELOG.md` and `LOG.md` updated for every repository change.

