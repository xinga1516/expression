# CHANGELOG

## 2026-07-11
- Replaced the direct run-scanning workbook workflow with the manifest-driven `model_compare.py report` workflow. Standalone Stage 1/2 comparison commands remain public; `report` consumes only registered canonical summaries, paired statistics, and PNG figures, while `report --refresh` explicitly reruns the manifest-declared comparisons first.
- Added `configs/model_compare_report.json` for Stage 1/2 run roots, summary tables, paired statistics, figures, and refresh commands. The workbook now includes Input Inventory, Run Audit, Seed Summary, Paired Summary, Stage 1/2 raw all-variable run tables, and Stage 1/2 overview sheets with embedded figures.
- Removed raw per-gene/per-cell paired records from the workbook. Paired detail remains owned by the comparison scripts; report sheets contain only registered bootstrap/statistics results. Violin outputs retain 25 low plus 25 high points per distribution.

- Added the initial workbook renderer and public comparison entry point; it was subsequently superseded by the manifest-driven `model_compare.py report` workflow above.
- The workbook records the configured validation checkpoint monitor, each diagnostic checkpoint's best validation epoch/value, the checkpoint actually used for test, and MSE/RMSE/Pearson/Spearman test metrics. It does not change training, checkpoint selection, or evaluation calculations.
- Added an entry-point unit test for Node workbook dispatch. Workbook generation is intentionally deferred until the target run set is finalized.

## 2026-07-10
- Added the default `stage2_cfgcache` experiment prefix to the Stage 2 launcher so config-aligned GPU-cache reruns do not overwrite the earlier incomplete `stage2_cw*` directories.
- Added an automatic post-training mutation position test to every Stage 2 sweep run: top 1000 test pairs, 10% per-gene ratio with an absolute cap of 100, and de novo motif outputs under `test/sequence_mutagenesis/`. The mutation step runs only after training and standard test complete.
- Added `hpc/stage2_mutagenesis_pending.sh` to backfill the same mutation protocol for all legacy Stage 2 checkpoints; it waits for active training, skips duplicate mutation processes, and uses GPUs 4-7.
- Vectorized top-pair mutation candidate selection with per-gene NumPy `argpartition`, preserving the gene cap while removing the previous per-nonzero Python heap bottleneck.
- Added `STAGE2_WEIGHTS` launcher override so a matched `contrastive_weight=0` baseline can be run with the identical Stage 2 configuration and evaluation protocol.
- Completed and documented the 15-run Stage 2 contrastive comparison, including matched `cw=0`, all mutation outputs, and the `support_genes >= 5` motif gate. `cw=0.40` improved mean correlation but the motif gate failed with maximum support of one gene.
- Added `scripts/summarize_gene_balanced_motifs.py`, a secondary diagnostic that gives each gene at most one important mutation window before motif aggregation while preserving the original pair-weighted motif outputs.
- Ran the gene-balanced motif diagnostic across all 15 Stage 2 runs; maximum support was `2`, so the required cross-gene motif gate remains unsatisfied even after removing pair-multiplicity bias.
- Updated the projection-head fallback launcher to run the same post-training mutation protocol and use an isolated experiment prefix.
- Added local GPU-cache support for Stage 2 contrastive batches: `GpuCachedPairLoader` can cache positive and matched-control sequence columns and return cached tensors with promoter/cell indices, avoiding per-batch CPU one-hot encoding for triplet loss.
- Updated local Stage 2 sweep launchers to request `--gpu-cache-dataset`, `--val-cell-ratio 1.0`, and explicit `--samples-per-epoch 128000` / `--val-samples 128000` for comparable, runnable contrastive sweeps.
- Fixed a local `scripts/train.py` dryrun reinitialization syntax issue detected by syntax-only compilation.

## 2026-07-10
- Aligned `hpc/stage2_sweep.sh` with the remote `configs/config.json` training profile: combined loss (`pearson_lambda=5`), gated fusion, explicit logCPM/count layer choices, fixed-LR 80-epoch budget, EMA `0.9`, validation cadence, and checkpoint/test settings. The Stage 2 contrastive columns, weights, and shift settings remain the only intentional sweep-specific overrides.
- This intentionally uses EMA `0.9` rather than the guide's initial `0.9999` prior because the active remote Stage 2 configuration is the requested source of truth; the deviation is recorded for fair comparison with its existing runs.
- Restarted the Stage 2 contrastive sweep on `sulab7g-zxy` after syncing the GPU-cached contrastive path; it overwrites the prior incomplete `stage2_cw*` directories as approved and writes results to `outputs/stage2/`.
- Strengthened the GPU-cache unit coverage with a deterministic assertion that matched intergenic negatives honor a nonzero contrastive crop shift instead of falling back to the centered window.
- Aligned `hpc/stage2_projection_sweep.sh` with the same remote config base so a future projection-head fallback differs from the completed Stage 2 grid only by its projection head.
- Added `scripts/build_reused_split_sequence_assets.py` for Stage2+ promoter-derived assets that reuse Stage1 gene splits, cell splits, frozen eval cells, and input gene panel while only re-extracting wider `sequence`, `control_sequence`, and `positive_sequence` windows from the genome.
- Changed the 3'UTR/downstream asset builder default `--control-attempts` from 2000 to 200 to match the current matched-control search budget.
- Added unit coverage for the reused-split sequence builder so future stages do not accidentally rerun HVG, rebuild the full gene table, or regenerate split/panel files.

## 2026-07-08
- Configured the remote `promodel` environment under `/PROJ5/liangn_zxy/envs/promodel` with Python 3.10 and PyTorch 2.5.1 CUDA 12.1, and validated GPU execution on sulab7g.
- Completed a Stage 1 promoter mutation smoke test and documented that full top-pair selection is currently CPU-bound because `top_n` does not limit the preliminary candidate scan.

- Added remote server operating rules to `AGENTS.md` for the `sulab7g-zxy` SSH target, including approved `/PROJ5/liangn_zxy/` directory purposes, transfer conventions, explicit GPU selection, and account/filesystem safety boundaries.
- Added the remote training infrastructure layout to `project_overview.md`.

## 2026-07-08
- Configured the remote `promodel` environment under `/PROJ5/liangn_zxy/envs/promodel` with Python 3.10 and PyTorch 2.5.1 CUDA 12.1, and validated GPU execution on sulab7g.
- Completed a Stage 1 promoter mutation smoke test and documented that full top-pair selection is currently CPU-bound because `top_n` does not limit the preliminary candidate scan.

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

## 2026-07-11

- Removed obsolete one-off scripts: the temporary augment generator, fixed-path TPM reversal tool, and raw E-MTAB inspector. Replaced the `project_test.py`/exploration-style `data_sanity.py` split with one focused `data_sanity.py` h5ad/promoter integrity checker.
- Consolidated public script entry points: `build_sequence_assets.py` now groups full/reused/3'UTR asset construction, and `model_compare.py` groups Stage 1/Stage 2 comparisons, motif summaries, and ablations. Historical module names remain compatibility backends.
- Preserved direct function compatibility for full asset construction when `promoter_window_length` is absent; only explicit CLI/config requests widen promoter windows.
- Added `model_compare.py stage1-ablation` as the standalone counterpart to the Stage 1 bootstrap-integrated ablation output.
- Standardized all Stage 2 mutation launchers to the Stage 1-comparable contract: top 1,000 pairs, 2%/20 per-gene cap, mutation batch 512, and 9 bp / 200 / 20 / 3 motif settings.
- Added a seed-7-only Stage 2 mutation retest launcher and a reusable two-run ablation helper derived from the Stage 1 paired per-cell/per-gene bootstrap workflow.
- Ran the Stage 2 seed-7 `cw=0.40` versus `cw=0` ablation: contrastive training improved per-cell Pearson by 0.006881 (95% CI [0.006352, 0.007408]) but did not improve per-gene Pearson (-0.002591, 95% CI [-0.006683, 0.001327]).
- Added Stage 1-style per-cell and per-gene Pearson violin outputs to the Stage 2 ablation, including raw distributions, summary statistics, and sampled extreme points.
- Made the Stage 2 violin writer create its output directory when called independently, with a focused regression test.
- Fixed `scripts/model_test.py` checkpoint reconstruction for Stage 2 projection-head models by restoring the configured contrastive projection dimensions/layers before loading weights.
- Added a unit regression test covering projection-head checkpoint loading; this fixes the post-training automatic test path that previously skipped mutation analysis with `Unexpected key(s)`.
- Added `hpc/stage2_projection_retest.sh` to re-run standard metrics and mutation analysis on already completed projection-head checkpoints after loader fixes.
- Completed the three-seed projection-head fallback: mean MSE/Pearson/Spearman were `5.054070/0.317593/0.315024`, below the matched `cw=0` correlation baseline, and gene-balanced motif support remained at most `2`; no cross-gene motif evidence passes the `support_genes>=5` gate.

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


