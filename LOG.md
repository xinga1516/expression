# LOG

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
