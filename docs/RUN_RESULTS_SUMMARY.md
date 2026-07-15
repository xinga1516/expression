# Run Results Summary

## Stage 2 contrastive sweep, 2026-07-11

The remote Stage 2 grid completed 15 runs on the same `promoter_stage2_v1` gene/cell/input-panel split and the same frozen test panel: 2,048 cells x 2,707 held-out genes = 5,543,936 pairs per run. All runs used the combined loss configuration from the remote `configs/config.json`, `CNNFlattenPromoterModel`, gated fusion, EMA `0.9`, train-time 440-to-400 random crops, and matched `control_sequence` negatives with `negative_shift_max=20`.

The original 12-run contrastive grid had weights `0.05/0.10/0.20/0.40`; a matched three-seed `cw=0` baseline was then added because no clean contrastive control was present in the first grid. Group means from the remote `outputs/stage2/stage2_summary.csv` are:

| Contrastive weight | MSE mean | Pearson mean | Spearman mean | Interpretation |
|---:|---:|---:|---:|---|
| 0.00 | 5.050813 | 0.322006 | 0.319676 | matched no-contrastive baseline |
| 0.05 | 5.091621 | 0.320933 | 0.316587 | no independent gain |
| 0.10 | 5.052150 | 0.322646 | 0.318900 | essentially tied with baseline |
| 0.20 | 4.980238 | 0.323774 | 0.316547 | lower MSE, no Spearman gain |
| 0.40 | 5.104766 | 0.334921 | 0.326970 | best correlation, higher MSE |

Relative to the matched `cw=0` baseline, `cw=0.40` improves mean Pearson by `+0.012915` and mean Spearman by `+0.007294`, while increasing mean MSE by `+0.053953`. This supports a correlation-specific Stage 2 improvement, but not a uniform improvement across all metrics. The contrastive negative crop invariant is confirmed in each run config/audit (`negative_crop_ready=True`).

Every run also completed top-1000 mutation testing with a 10% per-gene ratio and absolute cap of 100. The motif gate failed: maximum `support_genes=1` across the original 15 runs, so no de novo motif is accepted as cross-gene evidence. For example, a motif can have 98 supporting pairs while all 98 come from one gene. A three-seed projection-head fallback (`projection_dim=64`, two layers, `cw=0.40`) was then trained and retested after fixing checkpoint reconstruction. Its means were MSE `5.054070`, Pearson `0.317593`, and Spearman `0.315024`, below the matched `cw=0` correlations; its gene-balanced motif support was at most `2`. The Stage 2 result therefore supports a correlation-specific improvement for the non-projection `cw=0.40` model, but does not support a claim that the encoder learned a motif shared by at least five promoters.

As a sensitivity check, `scripts/summarize_gene_balanced_motifs.py` was run on all 15 mutation effect tables. It retained at most one important window per gene before aggregation. The maximum gene-balanced support was still only `2`, with zero motifs passing the `support_genes >= 5` gate. This rules out single-gene pair multiplicity as the only explanation for the failed motif evidence.

## Stage 1 fixed-LR promoter run, 2026-07-06

The original `stage1_shift420_combined_fixedlr_seed7` used five linear warmup epochs followed by constant LR `5e-4`, combined loss with `pearson_lambda=5`, and `ema_alpha=0.9`. It stopped at epoch 38 and completed the frozen 2,048-cell x 2,707-gene test (5,543,936 pairs). These results are historical: the output directory was subsequently replaced by an EMA-matched `ema_alpha=0.9999` run at the user's request.

| RMSE | Pearson | Spearman | Nonzero RMSE | Zero RMSE |
|---:|---:|---:|---:|---:|
| 2.255851 | 0.289195 | 0.294108 | 3.261428 | 1.770706 |

The prior `stage1_shift420_combined_seed7` checkpoint was subsequently evaluated on the identical full-test panel:

| Run | Post-warmup schedule | EMA alpha | Stop epoch | RMSE | Pearson | Spearman | Nonzero RMSE | Zero RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `stage1_shift420_combined_seed7` | cosine decay | 0.9999 | 62 | 2.297328 | 0.313015 | 0.308607 | 3.056439 | 1.961314 |
| `stage1_shift420_combined_fixedlr_seed7` | fixed `5e-4` | 0.9 | 38 | 2.255851 | 0.289195 | 0.294108 | 3.261428 | 1.770706 |

Fixed LR reduced overall RMSE by 0.041478 and zero-target RMSE by 0.190608, but Pearson decreased by 0.023820, Spearman by 0.014500, and nonzero RMSE worsened by 0.204989. The fixed-LR run also stopped 24 epochs earlier. The result suggests a stronger fit to the dominant zero targets but weaker expressed-gene ranking. It is not an isolated scheduler result because EMA also changed from 0.9999 to 0.9; a strict scheduler comparison requires rerunning one schedule with matched EMA.

The replacement fixed-LR run matched `ema_alpha=0.9999` and stopped at epoch 62. Its validation result is close to, but slightly below, cosine: minimum RMSE 1.844529 vs 1.839322, maximum Pearson 0.267607 vs 0.267689, and maximum Spearman 0.250772 vs 0.252792. Its automatic full test failed after training due a CUDA launch failure, and a clean retry confirmed the GPU driver was unavailable (`nvidia-smi`: GPU0 Unknown Error). The full-test scheduler comparison remains pending GPU/driver reset.

Relative to the existing seed-7 MSE promoter run, fixed-LR combined loss improved RMSE by 0.001174, Pearson by 0.002961, and Spearman by 0.007866. This second comparison is also not isolated because the loss functions differ.

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


## Stage 1 seed 7 训练策略消融，2026-07-06

比较对象为同一 CNNFlattenPromoterModel、同一 gene/cell split、同一 420 bp promoter 资产和 400 bp 随机 crop 设置：

| 策略 | 训练日志 epochs | Test RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| MSE | 30 | 2.257024 | 0.286233 | 0.286241 |
| Combined (Pearson lambda=5) | 63 | 2.297328 | 0.313015 | 0.308607 |
| Combined + fixed LR | 63 | 2.245754 | 0.320310 | 0.315785 |

严格按相同 cell_id/gene_id 配对后，combined 相对 MSE 的 Pearson mean delta 为 per-cell +0.024373，95% CI [0.023682, 0.025048]；per-gene +0.030655，[0.026570, 0.034860]。fixed LR 相对 MSE 为 per-cell +0.034788，[0.034046, 0.035542]；per-gene +0.023898，[0.019101, 0.028843]。

fixed LR 相对普通 combined 的 per-cell Pearson 提升 +0.010415，[0.009850, 0.010968]，但 per-gene mean delta 为 -0.006758，[-0.010280, -0.003177]。因此 fixed LR 当前是全局相关性、全局 RMSE 和 per-cell Pearson 最好的 seed 7 策略，但不能宣称其 per-gene 表现优于普通 combined。

这是单 seed 训练策略消融，不替代三 seed Stage 1 正式比较。MSE run 仅训练 30 epochs，而 combined runs 配置为 80 epochs 并记录到 63 epochs，因此 combined 与 MSE 的差异同时包含 loss 和训练预算效应。详细 CSV、配对 bootstrap 和图位于 outputs/stage1/summary/stage1_training_ablation_*。
