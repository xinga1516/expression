# TODO

Current project task list.

## Active

- [x] Re-run promoter Stage 1 v2 nine-model comparison after 420 bp asset update.
- [x] Record Stage 1 shift420 nine-run metrics in summary, registry, and result documents.
- [x] Use `--sequence-length 400 --promoter-shift-max 20` so train-time random crop uses the new 420 bp windows while val/test use centered crops.
- [ ] Re-run the E-MTAB workflow after deleting old `outputs/` results:
  - Pretrain VAE on `umi_E-MTAB-10519-hqcells` using only `cell_train.txt`.
  - Train `umi_E-MTAB-10519-hqcells` with cell split and delayed VAE fine-tune at epoch 60.
  - Train `umi_E-MTAB-10519-hqcells_aug15` with the same cell split and delayed VAE fine-tune at epoch 60.
  - Evaluate both models with standard test metrics and input ablation.
- [ ] Compare whether VAE fine-tune after loss stabilization improves validation/test performance.
- [ ] Compare `hqcells` vs `hqcells_aug15` model performance under the same split and training profile.

## Maintenance

- [ ] Keep `project_overview.md` current after workflow, dataset, model, or experiment convention changes.
- [ ] Keep `CHANGELOG.md` and `LOG.md` updated for every repository change.
