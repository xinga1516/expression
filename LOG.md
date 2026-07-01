# LOG

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
