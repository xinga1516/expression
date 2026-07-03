# Run Results Summary

## Stage 1 shift420 gate, 2026-07-02

Source table: `outputs/stage1_shift420_summary.csv`. Each row was regenerated from the run-local `config.json` and `test/test_metrics.json` files, not copied from training logs.

Fair-comparison checks across all nine runs:

- Data: `data/promoter_stage1_v1` with the same `promoter_train.csv`, `promoter_val.csv`, and cell split directory.
- scRNA-seq file: `data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad`.
- Input panel: `data/promoter_stage1_v1/input_gene_panel_train.txt`.
- Frozen full test panel: 2,048 test cells x 2,707 held-out test genes = 5,543,936 evaluated samples per run.
- Expression/target source: `expression_layer=logcpm`, `expression_transform=none`, scalar `target_value_layer=logcpm`, `target_transform=none`; count layer retained as `counts`.
- Training/eval settings: `loss_type=mse`, `checkpoint_metric=val_rmse`, `run_test_after_train=true`, `sequence_length=400`, `promoter_shift_max=20`, `gpu_cache_dataset=true`, `contrastive_weight=0.0`.
- Shared code snapshot in config: `git_hash=3f6582107bc40554a411c4d7d3dee74674d0163c`.

No conflict with `DROSOPHILA_CELL_TYPE_PROMOTER_3UTR_MODEL_GUIDE.md` was found for this result-recording change. The comparison matches the guide's Stage 1 contract: expression-matched baseline, real promoter, and matched intergenic control are evaluated on the same held-out genes and frozen test-cell panel.

### Group Means

| Group | RMSE mean +/- sd | Pearson mean +/- sd | Spearman mean +/- sd |
|---|---:|---:|---:|
| Expression matched | 2.442161 +/- 0.010037 | 0.129182 +/- 0.004616 | 0.132203 +/- 0.001494 |
| Real promoter | 2.357046 +/- 0.086650 | 0.269109 +/- 0.017437 | 0.272538 +/- 0.014231 |
| Matched intergenic | 2.334850 +/- 0.023524 | 0.171254 +/- 0.021358 | 0.173291 +/- 0.018305 |

### Per-Run Metrics

| Group | Seed | Model | Sequence column | RMSE | Pearson | Spearman | Nonzero RMSE | Zero RMSE |
|---|---:|---|---|---:|---:|---:|---:|---:|
| exprmatched | 1 | MatchedExpressionBaseline | `sequence` | 2.436360 | 0.133493 | 0.133653 | 3.056909 | 2.176143 |
| exprmatched | 7 | MatchedExpressionBaseline | `sequence` | 2.453750 | 0.129739 | 0.132289 | 3.014432 | 2.222922 |
| exprmatched | 42 | MatchedExpressionBaseline | `sequence` | 2.436372 | 0.124312 | 0.130669 | 3.071256 | 2.169038 |
| promoter | 1 | CNNFlattenPromoterModel | `sequence` | 2.404821 | 0.251374 | 0.257832 | 2.857013 | 2.223824 |
| promoter | 7 | CNNFlattenPromoterModel | `sequence` | 2.257024 | 0.286233 | 0.286241 | 3.223022 | 1.797290 |
| promoter | 42 | CNNFlattenPromoterModel | `sequence` | 2.409293 | 0.269719 | 0.273539 | 2.814648 | 2.249246 |
| intergenic | 1 | CNNFlattenPromoterModel | `control_sequence` | 2.307720 | 0.152061 | 0.157507 | 3.573749 | 1.644326 |
| intergenic | 7 | CNNFlattenPromoterModel | `control_sequence` | 2.347250 | 0.194263 | 0.193358 | 3.194387 | 1.963835 |
| intergenic | 42 | CNNFlattenPromoterModel | `control_sequence` | 2.349579 | 0.167438 | 0.169009 | 3.306932 | 1.901003 |

### Interpretation

The real promoter condition is the current Stage 1 shift420 winner by correlation: Pearson mean 0.269109 and Spearman mean 0.272538, above expression matched (0.129182/0.132203) and matched intergenic (0.171254/0.173291). The best single real-promoter run by Pearson is `stage1_shift420_promoter_seed7` with Pearson 0.286233 and Spearman 0.286241.

RMSE is more mixed: matched intergenic has the lowest mean RMSE (2.334850) versus real promoter (2.357046). Because the Stage 1 biological question is whether matched promoter sequence adds held-out-gene signal beyond expression state and matched non-regulatory sequence, the correlation metrics support the promoter signal, while RMSE should be reported as a residual caveat rather than ignored.

Best single-run Pearson by group:

- Expression matched: `stage1_shift420_exprmatched_seed1` Pearson 0.133493.
- Real promoter: `stage1_shift420_promoter_seed7` Pearson 0.286233.
- Matched intergenic: `stage1_shift420_intergenic_seed7` Pearson 0.194263.
