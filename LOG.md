# LOG

## 2026-07-08

- Compared the requested Stage 1 interaction diagnostic with
  `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`; it implements the guide's
  matched real/control-sequence comparison on unseen genes and frozen test
  cells, with no intentional protocol deviation.
- Added `scripts/summary_stage1.py` and `tests/unit/test_summary_stage1.py`.
  The script validates matched run provenance, applies the same promoter model
  to real promoter and `control_sequence`, applies the separate expression-only
  model for residuals, and writes per-gene interaction variance and residual
  correlation.
- Confirmed that `promodel_wsl` is not visible in the current session (available
  environments are `base`, `codex`, `genome`, and `promodel`). Per repository
  policy, no Python tests, AnnData loading, or full model inference were run.
- Ran `git diff --check` for the new Python files; it passed.

## 2026-07-06

- User requested replacing the `ema_alpha=0.9` fixed-LR output directory with an EMA-matched `ema_alpha=0.9999` rerun under the same experiment name; the prior full-test metrics remain recorded below for provenance.
- Deleted `outputs/stage1_shift420_combined_fixedlr_seed7`, changed `configs/config.json` to `ema_alpha=0.9999`, and completed the same-name fixed-LR retraining in `promodel`; early stopping occurred at epoch 62.
- EMA-matched validation comparison: fixed LR minimum RMSE 1.844529, maximum Pearson 0.267607, maximum Spearman 0.250772; cosine minimum RMSE 1.839322, maximum Pearson 0.267689, maximum Spearman 0.252792.
- The automatic full test failed at its first GPU-cached batch with `CUDA error: unspecified launch failure`. A clean-process retry found CUDA unavailable and was stopped to avoid a full CPU evaluation; `nvidia-smi` then reported GPU0 `Unknown Error`. Full test remains pending a GPU/driver reset or host restart.
- Compared the requested LR schedule with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`; fixed LR `5e-4` matches the promoter starting default and introduces no guide conflict.
- Replaced the five-epoch-warmup-plus-cosine scheduler with a five-epoch linear warmup followed by constant LR, exposed the warmup duration as `--warmup-epochs`, and added `tests/unit/test_lr_scheduler.py`.
- Kept `ema_alpha=0.9` as explicitly requested and assigned the independent experiment name `stage1_shift420_combined_fixedlr_seed7` so the existing cosine run is not resumed or overwritten.
- Ran `conda run --no-capture-output -n promodel pytest -q tests/unit/test_lr_scheduler.py tests/smoke/test_cli_help.py`: 8 tests passed.
- Ran `conda run --no-capture-output -n promodel python scripts/train.py --exp_name stage1_shift420_combined_fixedlr_seed7 --prior_config configs/config.json`; training stopped normally at epoch 38 after 32 non-improving epochs, and LR stayed at `5e-4` from epoch 5 onward.
- Completed the frozen full test on 2,048 cells x 2,707 genes (5,543,936 pairs): MSE 5.088862, RMSE 2.255851, Pearson 0.289195, Spearman 0.294108, nonzero RMSE 3.261428, zero RMSE 1.770706.
- Added the fixed-LR run to `records/registry.tsv` and `docs/RUN_RESULTS_SUMMARY.md`.
- Ran the prior `stage1_shift420_combined_seed7` checkpoint through the same frozen full-test protocol in `promodel`: 5,543,936 pairs, RMSE 2.297328, Pearson 0.313015, Spearman 0.308607, nonzero RMSE 3.056439, zero RMSE 1.961314.
- Compared with the prior combined/cosine run, fixed LR reduced overall RMSE by 0.041478 and zero-target RMSE by 0.190608, but reduced Pearson by 0.023820 and Spearman by 0.014500 while increasing nonzero RMSE by 0.204989. Because EMA differs (`0.9999` prior vs `0.9` fixed-LR), this is not an isolated scheduler ablation.

## 2026-07-05

- Compared the requested prior configuration with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md` and `outputs/stage1_shift420_promoter_seed7/config.json`.
- Found that the previous `configs/config.json` mixed saved-run keys with argparse keys (`loss_type` instead of `loss`, `vae_encoder_path` instead of `vae_encoder`) and set `data` to the old E-MTAB promoter directory; `train.py` ignores saved `train_promoter_file`/`val_promoter_file` fields and derives promoter CSV paths from `data`.
- Rewrote `configs/config.json` as a valid combined-loss prior config for `data/promoter_stage1_v1`, aligned key training/evaluation parameters with the shift420 seed-7 MSE run, and set epochs to 80.
- Validated the JSON against every argparse destination in `scripts/train.py`; no unknown keys were found. Confirmed the h5ad, train/val/test promoter CSVs, cell split directory, and input gene panel all exist. No training was started.

## 2026-07-03

- Compared the requested local environment setup with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`, `hpc/server_env.yml`, and `hpc/install_promodel.sh`; the biological/model guide has no conflicting environment requirement, while the repository runtime note names `promodel_wsl` and the HPC files name `promodel`.
- Created the HPC-defined conda environment at the explicit prefix `/home/jovyan/.conda/envs/promodel` using `hpc/server_env.yml`; stopped and removed an initial `/opt/conda/envs/promodel` attempt after detecting the default-prefix mismatch.
- Set the environment `LD_LIBRARY_PATH`, installed `torch==2.11.0+cu128` from the CUDA 12.8 PyTorch wheel index, and verified Python 3.10.20, Scanpy 1.11.5, AnnData 0.11.4, CUDA availability, and a 2048 x 2048 GPU matrix multiplication on the NVIDIA GeForce RTX 5090 D v2.
- Ran `pytest -q tests/smoke/test_imports.py tests/smoke/test_cli_help.py` in the new environment: 7 tests passed.

## 2026-07-02

- Read `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md` before recording the Stage 1 shift420 results; no guide conflict was found for the summary/registry update.
- Parsed all nine `outputs/stage1_shift420_*_seed*/config.json` and `test/test_metrics.json` files to regenerate `outputs/stage1_shift420_summary.csv` and compute group means.
- Updated `docs/RUN_RESULTS_SUMMARY.md`, `records/registry.tsv`, `records/run_interpretation_exclusions.tsv`, `TODO.md`, `CHANGELOG.md`, and `project_overview.md` for the completed Stage 1 shift420 formal-run summary.
- Cleared old `outputs/stage1*`, `outputs/stage2*`, `outputs/smoke_stage1*`, and `outputs/smoke_stage2*` result directories before the Stage 1 420 bp rerun.
- Added GPU-cache support for 420 bp cached sequence windows with GPU-side 400 bp cropping: train split uses random crop when `--promoter-shift-max > 0`; validation/test use centered crops.
- Updated `scripts/model_test.py` so CUDA evaluation uses GPU sequential cache by default, including centered 420 bp to 400 bp crop for test.
- Re-ran Stage 1 shift420 nine-model comparison (`exprmatched`, real `promoter`, matched `intergenic`; seeds 1, 7, 42) with fixed gene/cell/input-panel split, logCPM scalar targets, `--sequence-length 400`, and `--promoter-shift-max 20`.
- Wrote the Stage 1 rerun metrics to `outputs/stage1_shift420_summary.csv`. Test Pearson/Spearman means: expression matched 0.129/0.132, intergenic 0.171/0.173, real promoter 0.269/0.273.

- Checked existing `data/promoter_stage1_v1` promoter split files and confirmed `sequence`, `positive_sequence`, and `control_sequence` were 400 bp before the update.
- Updated `scripts/build_promoter_stage1_assets.py` with `--promoter-window-length`, default `--control-attempts 200`, and faster overlap checking for future full rebuilds.
- Attempted a full 420 bp rebuild with rematched controls; stopped it because even 200 attempts remained slow in the current Python matching loop.
- Performed a deterministic in-place 420 bp update of `promoter_train.csv`, `promoter_val.csv`, and `promoter_test.csv`: promoter and control windows were extended by 10 bp on each side from existing coordinates; `positive_sequence` was regenerated as the +20 bp shifted 420 bp promoter window.
- Rewrote `promoter_windows.tsv`, `control_windows.tsv`, and `audit_report.json` to reflect the 420 bp sequence windows.
- Validated all split files: every `sequence`, `positive_sequence`, and `control_sequence` is 420 bp; train/val/test genes remain mutually exclusive; `input_gene_panel_train.txt` remains train-gene-only; `sequence == positive_sequence` and `sequence == control_sequence` fractions are both 0.0 in all splits.

Codex operation log for meaningful repository work.

## 2026-07-01

- Checked the requested expression-data rule against `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`; the change refines data semantics and does not alter the promoter/3'UTR staged modeling plan.
- Updated `src/dataset.py` to keep UMI count target source separate from expression input source, with selectable AnnData layers and optional `log1p` expression transform.
- Updated `src/gpu_cache.py` so GPU cached batches use the same expression/target source split as `MyDataset`.
- Updated `scripts/train.py` with explicit expression/target CLI parameters and automatic defaults: VAE/ZINB use counts input, no-VAE scalar losses use precomputed logCPM expression input, scalar targets use precomputed logCPM, and ZINB targets use counts.
- Updated `scripts/model_test.py` so standard test, input ablation, mutagenesis, and train-after-test calls reuse the same expression/target config.
- Updated `src/utils.py` scatter helpers to build expression inputs from the expression source and true targets/library sizes from the count source.
- Updated tiny fixtures/tests to include `counts`, `cpm`, and `logcpm` AnnData layers and added tests for the new defaults.
- Updated `AGENTS.md` with current AnnData layer conventions and CLI usage guidance.
- Tried `wsl` directly from PowerShell, but this Codex session still returned the system WSL-not-installed/no-distribution message. No WSL/`promodel_wsl` pytest run was possible from this session.
- Ran read-only in-memory Python syntax compilation for changed Python files; all compiled successfully. A normal `py_compile` attempt failed because writing `__pycache__` was denied.
- Inspected the current `AGENTS.md`, `CHANGELOG.md`, `LOG.md`, and `TODO.md`.
- Added a runtime environment section to `AGENTS.md` requiring WSL conda env `promodel_wsl` for project Python work, especially `.h5ad`/AnnData, training, evaluation, mutation tests, and data-building checks.
- Recorded the documentation-only runtime update in `CHANGELOG.md`.
- Attempted to check git status, but `git` was not available in the current PowerShell PATH.
- No tests were run; documentation-only change.
- Inspected `AGENTS.md` and confirmed `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md` exists.
- Checked the guide's core contract; this documentation/governance change does not alter model parameters, data split logic, masking behavior, or evaluation requirements.
- Added project memory and change-record rules to `AGENTS.md`.
- Created initial `TODO.md`, `CHANGELOG.md`, `LOG.md`, and `project_overview.md`.
- No tests were run; documentation-only change.
