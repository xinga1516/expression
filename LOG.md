# LOG

## 2026-07-12
- Compared the two-layer reporting change with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`; it is reporting-only and preserves the guide's validation/test, matched-baseline, and frozen-panel requirements.
- Added `configs/model_compare_report.json`, `scripts/model_compare_report.py`, and manifest-driven `model_compare.py report` / `report --refresh`. Existing stage-specific comparison commands remain intact.
- Validated the current Stage 1/2 manifest collection locally: 29 completed runs, 11 strict seed-only configuration groups, 24 registered paired-statistic rows, and two stage payloads.
- Rendered and reopened a local report check workbook: 9 expected sheets and 7 embedded Stage 1/2 PNG drawings were verified. The bundled local Python lacks pytest, so the focused pytest suite remains to be run in the project `promodel` environment.

## 2026-07-11
- Compared the reporting request with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`: the change is audit/reporting-only and preserves the guide's validation-on-unseen-genes, frozen test-panel, matched-baseline, and fair-comparison requirements.
- Added the local `model_compare_workbook.mjs` artifact-tool exporter plus the `model_compare.py workbook` dispatcher and documentation. It reads local run `config.json`, training logs, checkpoint directories, and test metric files; no remote server connection, synchronization, training, or output regeneration was performed.
- Added the workbook-dispatch unit test. Per the request, stopped before creating the xlsx; a prior incomplete local export was terminated and produced no output file.

## 2026-07-10
- Checked the active remote Stage 2 sweep. The first four jobs were still in epoch 0 with low GPU utilization, consistent with CPU-side dynamic sequence encoding in contrastive training.
- Implemented local GPU-cache contrastive sequence support in `src/gpu_cache.py`, updated `scripts/train.py` to consume cached positive/control tensors when present, and added `tests/unit/test_gpu_cache.py`.
- Local syntax-only compile passed for `src/gpu_cache.py`, `scripts/train.py`, and `tests/unit/test_gpu_cache.py` using in-memory `compile(...)` because Windows pycache writing was denied.
- Remote sync was interrupted by usage limits after an earlier copy of `scripts/train.py`; before any new remote launch, re-sync the repaired local `scripts/train.py`, `src/gpu_cache.py`, Stage 2 launchers, and `tests/unit/test_gpu_cache.py`, then run remote py_compile and pytest.

## 2026-07-10
- Compared the reused Stage2 asset plan with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`; it is consistent with the guide's requirement to keep model comparisons on the same gene split, feature panel, and frozen eval-cell panel.
- Added the reusable promoter sequence-window derivation script and a focused unit test.
- Updated `scripts/build_utr_stage_assets.py` so matched downstream/intergenic controls try at most 200 random positions by default.

## 2026-07-08
- Installed micromamba 2.8.1 at `/PROJ5/liangn_zxy/envs/bin/micromamba` and created the remote project environment at `/PROJ5/liangn_zxy/envs/promodel`; package caches and temporary files remain under the approved `/PROJ5/liangn_zxy/` paths.
- Installed PyTorch `2.5.1+cu121` instead of the repository install script's CUDA 12.8 wheel because sulab7g uses NVIDIA driver 535.113.01/CUDA 12.2. Validated CUDA on GPU 0 (RTX 3090), Scanpy 1.11.5, and AnnData 0.11.4.
- Tested Stage 1 seed-7 promoter mutagenesis. The full promoter candidate scan remained CPU-bound after 10 minutes and was stopped; `top_n` limits selected pairs but not the preliminary promoter/cell scan.
- Completed a mutation smoke test using 10 formal test promoters, all 2,048 frozen test cells, the 4,096-gene input panel, and the formal seed-7 best checkpoint. It selected two pairs from two genes with cap=1 and generated all expected mutation/motif outputs under `/PROJ5/liangn_zxy/runs/stage1_mutation_smoke_small_seed7/`.

- Successfully tested read-only SSH access through `sulab7g-zxy`. Confirmed user `liangn_zxy`, host `sulab7g`, project directory `/PROJ5/liangn_zxy/work/expression`, and a clean `main...origin/main` git status. No remote files were changed and no GPU task was started.

- Compared the remote-server documentation change with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md`; it changes infrastructure guidance only and does not conflict with the guide's gene-split, masking, baseline, or evaluation requirements.
- Updated `AGENTS.md` with the required `sulab7g-zxy` connection, `/PROJ5/liangn_zxy/` directory layout, `rsync` usage, explicit `CUDA_VISIBLE_DEVICES`, and prohibited operations.
- Updated `project_overview.md` with the remote execution layout.
- No server connection or tests were run; this was a documentation-only change.

## 2026-07-08
- Installed micromamba 2.8.1 at `/PROJ5/liangn_zxy/envs/bin/micromamba` and created the remote project environment at `/PROJ5/liangn_zxy/envs/promodel`; package caches and temporary files remain under the approved `/PROJ5/liangn_zxy/` paths.
- Installed PyTorch `2.5.1+cu121` instead of the repository install script's CUDA 12.8 wheel because sulab7g uses NVIDIA driver 535.113.01/CUDA 12.2. Validated CUDA on GPU 0 (RTX 3090), Scanpy 1.11.5, and AnnData 0.11.4.
- Tested Stage 1 seed-7 promoter mutagenesis. The full promoter candidate scan remained CPU-bound after 10 minutes and was stopped; `top_n` limits selected pairs but not the preliminary promoter/cell scan.
- Completed a mutation smoke test using 10 formal test promoters, all 2,048 frozen test cells, the 4,096-gene input panel, and the formal seed-7 best checkpoint. It selected two pairs from two genes with cap=1 and generated all expected mutation/motif outputs under `/PROJ5/liangn_zxy/runs/stage1_mutation_smoke_small_seed7/`.

- Successfully tested read-only SSH access through `sulab7g-zxy`. Confirmed user `liangn_zxy`, host `sulab7g`, project directory `/PROJ5/liangn_zxy/work/expression`, and a clean `main...origin/main` git status. No remote files were changed and no GPU task was started.

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

## 2026-07-10

- Added a default `stage2_cfgcache` run prefix because four earlier `stage2_cw*` directories already exist remotely; the new queue remains under `outputs/stage2/` while retaining those historical partial runs.
- Extended the Stage 2 launcher with a post-training `model_test.py --run-mutagenesis` step and a separate mutation log; configured the top-1000 gene-capped mutation protocol so motif support can be evaluated at the required `support_genes >= 5` gate.
- Prepared a four-GPU backfill queue for all twelve legacy Stage 2 checkpoints; it waits for active training and uses GPUs 4-7 so the active training batch on GPUs 0-3 is not interrupted.
- Reworked `select_top_expressed_pairs()` to group promoter rows by gene and retain each gene's top capped candidates with NumPy partitioning; existing gene-cap regression tests remain the validation target.
- Identified that the completed 12-run sweep lacked a same-config `cw=0` baseline; prepared a three-seed `stage2_cw000` run using the launcher weight override before attributing gains to contrastive loss.
- Completed the matched `cw=0` baseline and regenerated remote `runs/expression/stage2/stage2_summary.csv` for 15 runs. Confirmed all standard/mutation outputs exist, maximum `support_genes=1`, and decided not to launch projection fallback solely for prediction because `cw=0.40` improves Pearson/Spearman; motif-specific follow-up remains open.
- Added and prepared a gene-balanced motif diagnostic to distinguish genuine cross-gene support from single-gene pair multiplicity; it writes `gene_balanced_motif_windows.csv`, `gene_balanced_de_novo_motifs.csv`, and a Stage 2 summary table.
- Ran the diagnostic remotely: 15 runs processed, maximum gene-balanced `support_genes=2`, zero motifs passed the threshold 5. This confirms the Stage 2 motif limitation is not explained solely by the 100-pair single-gene cap.
- Prepared the projection-head fallback at `contrastive_weight=0.40`, projection dimension 64, two layers, three seeds, with automatic standard test and mutation outputs.

## 2026-07-10

- Read `/PROJ5/liangn_zxy/work/expression/configs/config.json` through the approved `sulab7g-zxy` SSH alias and compared it with `hpc/stage2_sweep.sh` and the Stage 2 guide defaults.
- Updated the local Stage 2 sweep launcher to use the remote config's combined-loss, fusion, expression-layer, training-budget, EMA, validation, checkpoint, and test parameters explicitly; retained only Stage 2 contrastive sequence settings as sweep-specific values.
- The guide's EMA `0.9999` starting prior is intentionally overridden by the active remote configuration's `0.9`, as requested; no training job was launched or stopped by this change.
- Synced `src/gpu_cache.py`, `scripts/train.py`, and `tests/unit/test_gpu_cache.py` to `/PROJ5/liangn_zxy/work/expression`; remote `py_compile` passed and `tests/unit/test_gpu_cache.py` passed (`1 passed`).
- Started the four-GPU Stage 2 contrastive sweep in detached tmux session `promoter_stage2` with weights `0.05/0.10/0.20/0.40` and seeds `1/7/42`. The first four runs (`cw005` seeds 1/7/42 and `cw010` seed1) use GPUs 0--3. Earlier incomplete run directories are overwritten with explicit user approval.
- Added a deterministic unit assertion for random intergenic-negative crop positions so the Stage 2 requirement is covered directly rather than inferred from cached tensor shapes.
- Audited the deferred projection-head fallback launcher and aligned its loss, expression layers, training budget, EMA, validation, checkpoint, and test parameters with the active Stage 2 configuration before any projection run is considered.

## 2026-07-06

- 2026-07-11: Added the Stage 1-comparable Stage 2 mutation parameter contract to all launchers and created a seed-7 retest launcher. Ran the reusable Stage 1-style Stage 2 seed-7 contrastive ablation for cw=0.40 vs cw=0: global Pearson increased 0.326239 to 0.333287, per-cell paired Pearson increased 0.006881 (95% CI [0.006352, 0.007408]), while per-gene delta was -0.002591 (95% CI [-0.006683, 0.001327]).
- 2026-07-11: Added and generated Stage 2 seed-7 per-cell/per-gene Pearson violin plots for cw=0 vs cw=0.40, with 9,510 distribution rows and 200 displayed extreme points.
- 2026-07-11: Fixed standalone Stage 2 violin output directory creation and ran the focused ablation test module in remote promodel: 3 passed.

- 2026-07-11: Confirmed the new remote Stage 2 projection sweep started on GPUs 0/1/2. Training completed and checkpoints were written, but the automatic post-training test failed because `model_test.py` omitted the projection head when rebuilding the model. Patched the loader and added a focused unit test; the completed checkpoints can now be evaluated without retraining.
- 2026-07-11: Added the projection checkpoint retest launcher to regenerate standard test, mutation position, and motif outputs without retraining.
- 2026-07-11: Fixed projection checkpoint loading, reran all three completed projection checkpoints, and regenerated standard/mutation outputs. The projection mean was MSE 5.054070, Pearson 0.317593, Spearman 0.315024; gene-balanced motif support max was 2, so the biological motif gate still failed.

- Compared stage1_shift420_promoter_seed7, stage1_shift420_combined_seed7, and stage1_shift420_combined_fixedlr_seed7 on the same frozen 5,543,936 test pairs.
- Generated outputs/stage1/summary/stage1_training_ablation_seed7.csv, paired deltas/bootstrap CSVs, PNG/SVG summary, and a Chinese README using 10,000 bootstrap repeats.
- Fixed LR achieved the best global test RMSE/Pearson/Spearman (2.245754/0.320310/0.315785); versus ordinary combined it improved per-cell Pearson by 0.010415 but reduced per-gene Pearson by 0.006758.
- Ran the focused Stage 1 summary tests in WSL promodel_wsl: 4 passed.

- Generated `outputs/stage1/summary/paired_bootstrap_summary.png`, a two-panel forest plot of per-cell/per-gene promoter Pearson deltas with individual seed estimates and hierarchical 95% CIs.
- Added and ran the Stage 1 paired Pearson bootstrap summary with 10,000 repeats, 95% percentile intervals, and random seed 42.
- Wrote 23,557 valid paired records plus per-seed and hierarchical three-seed summaries to `outputs/stage1/summary/`.
- Promoter vs intergenic hierarchical deltas were positive with intervals excluding zero: per-cell `0.1093 [0.0960, 0.1187]`; per-gene `0.0259 [0.0194, 0.0346]`.
- Ran `pytest tests/unit/test_stage1_bootstrap_summary.py -q` in WSL `promodel_wsl`: 2 passed.

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


- 2026-07-11: Organized script entry points into `build_sequence_assets.py` and `model_compare.py`; added dispatcher unit tests and `scripts/README.md` documenting the public workflows. This does not alter asset construction, split reuse, or comparison calculations.
- 2026-07-11: Fixed asset dispatcher support for legacy `build_assets(parse_args())` backends and preserved unspecified promoter window length behavior. Remote syntax/help checks passed; focused asset/comparison/ablation suite: 20 passed.
- 2026-07-11: Added and tested the standalone Stage 1 ablation model-comparison subcommand. Final focused asset/comparison/ablation suite: 21 passed.
- 2026-07-11: Local-only cleanup: removed obsolete temporary/one-off scripts and normalized `project_test.py` to `data_sanity.py`; updated smoke and sanity test imports and renamed the corresponding sanity test file. No remote synchronization was performed after the user's Git-only instruction.
