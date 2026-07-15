# Script Entry Points

Use these public entry points for new work:

- `build_sequence_assets.py full`: construct the complete Stage 1 promoter asset bundle, including gene/cell splits and input panel.
- `build_sequence_assets.py reuse`: reuse a Stage 1 split/panel bundle and re-extract wider promoter/control/positive sequence windows.
- `build_sequence_assets.py utr`: construct 3'UTR/downstream sequence assets.
- `model_compare.py stage1-bootstrap`: paired Stage 1 bootstrap, violin, and training-ablation summary.
- `model_compare.py stage1-ablation`: standalone Stage 1 seed-7 training-strategy ablation.
- `model_compare.py stage1-sequence-interaction`: real-promoter versus matched-control sequence interaction analysis.
- `model_compare.py stage2-summary`: Stage 2 grid metric/motif summary.
- `model_compare.py stage2-gene-balanced-motifs`: gene-balanced de novo motif support analysis.
- `model_compare.py stage2-ablation`: matched two-run Stage 2 ablation with paired bootstrap and violin outputs.
- `model_compare.py report`: validate registered stage summaries, paired statistics,
  and PNG figures, then write one auditable Stage 1/Stage 2 xlsx report.

The older implementation filenames remain callable only to preserve historic
commands, test imports, and remote job logs. New documentation and launchers
should use the public entry points above.

`data_sanity.py` is the single data-integrity entry point for h5ad/promoter
alignment checks. Historical one-off inspection, TPM-reversal, and temporary
augmentation scripts were removed.

## Consolidated report

Create a single workbook from already confirmed stage outputs:

```bash
python scripts/model_compare.py report
```

Use `--refresh` only when the manifest-declared Stage 1/2 comparison outputs
should be recomputed before report generation. The report manifest is
`configs/model_compare_report.json`; use `--manifest` or `--output` to
override it. The xlsx renderer requires a Node runtime with
`@oai/artifact-tool`; set `MODEL_COMPARE_NODE` when Node is not on `PATH`.

The report does not infer new ablations and does not include raw per-gene or
per-cell paired rows. It consumes only paired statistics registered in the
manifest. Violin figures retain 25 lower and 25 higher extreme points per
distribution, for 50 displayed extreme points.
