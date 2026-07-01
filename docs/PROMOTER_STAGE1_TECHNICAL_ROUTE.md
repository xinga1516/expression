# Promoter Stage 1 技术路线文档

本文档记录当前 Drosophila 表达预测项目中 Promoter Stage 1 clean gate 的技术路线。Stage 1 的目标刻意保持收敛：在加入 contrastive loss、3'UTR/downstream sequence、shift augmentation 或 generation 之前，先证明 promoter sequence 分支在 held-out genes 上确实提供了额外预测信息。

## 1. 目标与范围

Stage 1 的一个监督样本对应一个 `(cell, target_gene)` 组合，任务是预测该细胞中目标基因的表达值：

```text
input = (target masked cell expression state, target gene sequence)
output = scalar target expression
```

clean gate 比较三个模型组。三组模型必须使用相同的数据 split、input gene panel、frozen evaluation cells、loss、checkpoint 策略和 test 流程。

| 模型组 | 模型类 | 序列列 | 作用 |
|---|---|---|---|
| Expression baseline | `ExpressionBaseline` | 传入 `sequence`，但模型忽略序列 | 测量仅用 cell state 能预测到什么程度。 |
| Real promoter | `CNNFlattenPromoterModel` | `sequence` | 测试 TSS-proximal promoter sequence 是否提升 held-out gene 预测。 |
| Matched intergenic control | `CNNFlattenPromoterModel` | `control_sequence` | 控制 sequence branch 容量和 matched non-promoter genomic background 的影响。 |

Stage 1 不包含：

- 3'UTR/downstream sequence；
- contrastive/triplet loss；
- promoter shift augmentation 作为默认训练方案；
- generation 或候选序列设计；
- 大规模 architecture search。

## 2. 数据资产

Stage 1 数据资产位于：

```text
data/promoter_stage1_v1/
```

原始表达矩阵不会复制到该目录中。训练和测试仍指向：

```text
data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad
```

当前 `data/promoter_stage1_v1/audit_report.json` 中记录的数据资产数量如下：

| 项目 | 数量 |
|---|---:|
| source atlas 细胞数 | 52,753 |
| expression genes | 13,956 |
| source promoter genes | 13,956 |
| Stage 1 eligible genes | 13,986 |
| expression/promoter/control matching 后纳入的 protein-coding genes | 13,680 |
| train genes | 8,430 |
| validation genes | 2,543 |
| test genes | 2,707 |
| input expression panel size | 4,096 |
| train cells | 17,845 |
| frozen validation cells | 2,048 |
| frozen test cells | 2,048 |

### 2.1 Gene Universe

Stage 1 的 gene universe 由 `scripts/build_promoter_stage1_assets.py` 构建。当前规则比较保守：

- 只纳入 protein-coding genes；
- 必须存在于 expression matrix；
- 必须有 promoter sequence row；
- 必须成功生成 matched intergenic control；
- train/validation/test target genes 必须互斥。

protein-coding eligibility 从 GTF feature 推断：

- 存在 `mRNA` feature；
- 同时至少存在一个 coding evidence feature：`CDS`、`start_codon` 或 `stop_codon`。

lncRNA-like 记录目前会被标记为 `lncRNA_candidate`，但不进入 Stage 1。原因是当前仅凭 symbol prefix 不能严格证明其 Pol II promoter 身份，因此先不混入 clean gate。

### 2.2 Promoter 与 Matched Intergenic Control

source promoter CSV 为每个 gene 提供一个 400 bp 的 `sequence`。Stage 1 使用非 augment 的中心 promoter window。

matched intergenic control 在最终 gene split 之前生成：

- 在 promoter 所在的同一 contig 上随机采样候选窗口；
- 候选窗口长度与 promoter window 相同；
- 候选窗口不能与任意 GTF gene interval 或任意 promoter window 重叠；
- 候选窗口不能包含 `N`；
- 在随机候选中选择 GC fraction 与 promoter 最接近的窗口；
- 如果目标 promoter 位于负链，则对 control sequence 做 reverse-complement，使输入方向与目标基因方向一致。

只有 `match_status == "matched"` 的 gene 会进入最终 split 和最终 CSV 文件。

关键输出文件：

| 文件 | 作用 |
|---|---|
| `genes.tsv` | 完整 gene annotation/audit 表，包含纳入状态和排除原因。 |
| `gene_splits.tsv` | 最终 gene-level train/validation/test split。 |
| `promoter_windows.tsv` | promoter rows 以及 control matching metadata。 |
| `control_windows.tsv` | 只包含 matched control rows。 |
| `promoter_train.csv`, `promoter_val.csv`, `promoter_test.csv` | 模型直接读取的 split CSV，同时包含 `sequence` 和 `control_sequence`。 |
| `input_gene_panel_train.txt` | train-derived 4,096 个 expression input features。 |
| `cell_train.txt`, `cell_val.txt`, `cell_test.txt` | 训练和评估使用的 cell panels。 |
| `frozen_eval_cells.tsv` | validation/test frozen cell panel 记录。 |
| `audit_report.json` | 数据资产 provenance 和数量统计。 |

### 2.3 Input Gene Panel

expression branch 使用固定 4,096-gene input panel。该 panel 只从 train genes 中选择，当前选择方法是 Scanpy HVG：

```text
input_gene_panel_method = hvg
hvg_flavor = cell_ranger
```

所有模型组使用同一个 input panel。如果 target gene 出现在 input panel 中，该位置会在每个样本中动态置零。由于 test target genes 与 train genes 互斥，test target genes 通常不在 train-derived input panel 中；但 masking 逻辑仍然保留，用于安全性和兼容性。

## 3. Dataset 与 Dataloader

运行时 dataset 为 `src.dataset.MyDataset`。

### 3.1 Dataset 输入

Stage 1 相关构造参数：

| 参数 | Stage 1 取值 |
|---|---|
| `promoter_file` | `data/promoter_stage1_v1/promoter_<split>.csv` |
| `scrna_file` | `data/umi_E-MTAB-10519-hqcells/integrated_data.h5ad` |
| `sequence_column` | real promoter 使用 `sequence`；intergenic control 使用 `control_sequence` |
| `sequence_length` | 400 |
| `input_gene_panel_file` | `data/promoter_stage1_v1/input_gene_panel_train.txt` |
| `cell_ids_subset` | `cell_train.txt`、`cell_val.txt` 或 `cell_test.txt` |
| `log1p_cpm_target` | UMI/log target 模式下为 true，ZINB 除外 |
| `preencode_promoters` | 当前 GPU runs 中为 true |

### 3.2 样本形状

一个 dataset index 对应一个展平后的 `(promoter_index, cell_index)` pair：

```text
pro_i = idx // C
cell_i = idx % C
```

模型接收的张量：

| Tensor | Shape | 含义 |
|---|---|---|
| `promoter` | `(400, 5)` | A/C/G/T/N one-hot sequence。 |
| `expr_input` | `(4096,)` | 固定 input panel 上的 cell-state expression，target 若存在则被 mask。 |
| `y` | scalar | 当前 cell 中目标 gene 的表达值。 |

one-hot encoder 会用 `N` padding 短序列，并截断超过 400 bp 的序列。通道顺序为：

```text
A, C, G, T, N
```

### 3.3 Target 构建

对每个 `(gene, cell)` pair：

1. 读取该 cell 的完整 expression row。
2. 使用 `promoter2expr_idx` 找到 target gene 的 expression column。
3. 将原始 target value 保存为 `y_value`。
4. 根据固定 input panel 构建 `expr_input`。
5. 如果 target gene 在 input panel 中，将对应 input position 置零。
6. 如果 `log1p_cpm_target=True`，使用排除 target count 后的 library size 将 target 转为 `log1p(CPM)`。

### 3.4 GPU Cached Loader

当前 Stage 1 GPU runs 使用 `GpuCachedPairLoader`：

```text
gpu_cache_dataset = true
gpu_sampler = balanced
batch_size = 2048
samples_per_epoch = 128000
```

该 loader 将以下张量缓存到 GPU：

- 当前 split 的全部 promoter tensors；
- 选定 cells 的 dense expression input panel；
- 选定 cells × target promoters 的 target matrix；
- cell total counts；
- target 在 input panel 中的位置，用于动态 masking。

balanced sampler 每个 epoch 为每个 promoter 分配近似相同数量的 pairs，然后为每个 promoter 随机采样 cells。这样可以避免 materialize 完整的 `P * C` permutation，同时保持每个 epoch 的样本数固定。

非 GPU fallback 路径：

- 未设置 `nonzero_ratio` 时使用 `BalancedEpochSubsetSampler`；
- 设置 `nonzero_ratio` 时使用 `ZeroNonZeroSampler`；
- 当前 Stage 1 clean runs 使用 balanced sampling，不使用 zero/nonzero balancing。

## 4. 模型结构

### 4.1 ExpressionBaseline

`ExpressionBaseline` 忽略 promoter tensor，仅从 masked cell state 预测 target expression：

```text
expr_input (4096)
  -> Linear(4096, 512)
  -> ReLU
  -> Dropout(0.3)
  -> Linear(512, 256)
  -> ReLU
  -> Dropout(0.3)
  -> Linear(256, 1)
  -> scalar prediction
```

由于 held-out test genes 不在 train-derived input panel 中，对固定 test cell 而言，expression-only 输入在不同 test target genes 之间通常几乎没有 target-specific 变化。因此 expression-only baseline 很难在固定 cell 内对 held-out genes 做有效排序，per-cell Pearson 预期会是 undefined 或较弱。

### 4.2 CNNFlattenPromoterModel

`CNNFlattenPromoterModel` 包含一个 promoter branch、一个 expression branch 和一个 fusion regressor。

当 `hidden_size=128` 时，promoter branch 为：

```text
promoter (400, 5)
  -> permute to (5, 400)
  -> Conv1d(5, 64, kernel=9, padding=4)
  -> GELU
  -> Dropout(0.1)
  -> Conv1d(64, 64, kernel=7, padding=3)
  -> GELU
  -> flatten
  -> Linear(400 * 64, 128)
  -> LayerNorm(128)
  -> GELU
  -> Dropout(0.1)
  -> promoter embedding (128)
```

expression branch 为：

```text
expr_input (4096)
  -> Linear(4096, 256)
  -> LayerNorm(256)
  -> GELU
  -> Dropout(0.1)
  -> Linear(256, 128)
  -> LayerNorm(128)
  -> GELU
  -> expression embedding (128)
```

fusion 和 output 为：

```text
concat(promoter_embedding, expression_embedding) -> 256
  -> Linear(256, 128)
  -> GELU
  -> Dropout(0.1)
  -> Linear(128, 1)
  -> scalar prediction
```

real promoter 和 intergenic control 使用完全相同的模型结构。唯一目标差异是读取的 sequence column：

```text
real promoter:       sequence
intergenic control:  control_sequence
```

## 5. 训练协议

当前 Stage 1 GPU run 配置记录在 `outputs/promoter_stage1/stage1_promoter_seed7_gpu/config.json` 以及对应 run configs 中。

| 参数 | 取值 |
|---|---|
| Seeds | 1, 7, 42 |
| Epochs | 30 |
| Batch size | 2048 |
| Samples per epoch | 128,000 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-2 |
| Loss | weighted MSE |
| Nonzero target weight | 2.0 |
| Scheduler | 5-epoch LinearLR warmup + CosineAnnealingLR |
| Scheduler min LR | 1e-6 |
| Gradient clipping | max norm 1.0 |
| AMP | true |
| Pre-encode promoters | true |
| GPU cached dataset | true |
| Cell split | `data/promoter_stage1_v1` |
| Train cell ratio | 1.0 |
| Validation cell ratio | 1.0 |
| Checkpoint metric | `val_rmse` |
| Patience | 32 |
| EMA alpha | 0.9999 |
| Test after train | true |

validation 在每个 epoch 的训练前执行，因此 epoch 0 记录的是初始化模型在 validation 上的表现。训练日志记录：

- train weighted loss；
- validation weighted loss；
- validation loss EMA；
- validation RMSE；
- validation Pearson 和 Spearman；
- zero/nonzero loss components；
- zero accuracy；
- branch embedding variance diagnostics。

当前 checkpoint 输出：

| 文件 | 含义 |
|---|---|
| `best_model.safetensors` | 根据 `--checkpoint-metric` 选择的最佳 checkpoint。 |
| `best_val_rmse.safetensors` | 根据 validation RMSE 保存的诊断最佳 checkpoint。 |
| `best_val_pearson.safetensors` | 根据 validation Pearson 保存的诊断最佳 checkpoint。 |
| `best_val_spearman.safetensors` | 根据 validation Spearman 保存的诊断最佳 checkpoint。 |
| `last.ckpt` | 可 resume 的完整训练状态。 |

## 6. 测试与评价协议

当训练命令启用 `--run-test-after-train` 时，`scripts/model_test.py` 会在训练结束后运行标准 test 流程。

Stage 1 test 设置：

| 设置 | 取值 |
|---|---|
| Test split | `data/promoter_stage1_v1/promoter_test.csv` |
| Test cells | frozen `cell_test.txt` panel |
| Test cell count | 2,048 |
| Test target genes | 当前持久化 test metrics 中为 128 rows |
| Total evaluated pairs | 262,144 |
| Spearman sample limit | 0，即使用全部 evaluated pairs |

每个 run 的标准输出：

| 文件 | 作用 |
|---|---|
| `test_metrics.json` / `test_metrics.csv` | pooled MSE/RMSE/Pearson/Spearman 以及 zero/nonzero RMSE。 |
| `per_gene_metrics.csv` | 每个 test gene 一行；计算该 gene 在 frozen test cells 上的 Pearson。 |
| `per_cell_metrics.csv` | 每个 frozen test cell 一行；计算该 cell 在 held-out genes 上的 Pearson。 |
| `per_promoter_scatter.png` | sampled promoters 的诊断 scatter。 |
| `per_cell_scatter.png` | sampled cells 的诊断 scatter。 |

pooled MSE/RMSE 会保留，但不能单独作为 Stage 1 结论依据。原因是 pair distribution 中 zero/low-expression samples 占比较高，pooled error 容易被低表达背景主导。Stage 1 解释应优先同时报告：

- pooled Pearson/Spearman；
- per-cell Pearson across held-out genes；
- per-gene Pearson across frozen test cells；
- 在同一 panel 下的 real promoter vs matched intergenic control。

## 7. 当前 Stage 1 结果

本节使用的结果文件：

- `outputs/promoter_stage1/summary/stage1_seed_metrics.csv`
- `outputs/promoter_stage1/summary/stage1_model_summary_compact.csv`
- `outputs/promoter_stage1/summary/violin_plots/stage1_pearson_compare_violin_stats.csv`
- `outputs/promoter_stage1/summary/violin_plots/stage1_pearson_extreme_points.csv`
- `outputs/promoter_stage1/summary/violin_plots/stage1_per_gene_pearson_violin_compare.svg`
- `outputs/promoter_stage1/summary/violin_plots/stage1_per_cell_pearson_violin_compare.svg`

### 7.1 Pooled Test Metrics

下表由当前 `outputs/promoter_stage1` 中已经落盘的最新 test JSON 文件重新汇总得到。三个模型均使用 seed 1、7、42；汇总文件位于 `outputs/promoter_stage1/summary/`。

| 模型组 | Seeds | MSE | RMSE | Pearson r | Spearman r | Nonzero RMSE | Zero RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ExpressionBaseline | 3 | 6.040079 ± 0.048450 | 2.457644 ± 0.009863 | 0.109547 ± 0.017836 | 0.122628 ± 0.015677 | 3.883948 ± 0.195641 | 1.597382 ± 0.187170 |
| Intergenic control | 3 | 6.281595 ± 0.128141 | 2.506224 ± 0.025616 | 0.160625 ± 0.008025 | 0.180791 ± 0.023760 | 3.469956 ± 0.080671 | 2.018300 ± 0.097668 |
| Real promoter | 3 | 5.744617 ± 0.118859 | 2.396708 ± 0.024750 | 0.238405 ± 0.003799 | 0.237267 ± 0.003624 | 3.019932 ± 0.058837 | 2.133636 ± 0.066351 |

### 7.2 Per-Gene 与 Per-Cell Pearson

violin comparison 已使用当前落盘的全部三个 seeds。每个 violin 上叠加的黑点表示该模型/source 中最低 25 和最高 25 个有效 Pearson samples。对应的 gene/cell id、seed 和 Pearson 值已导出到 `stage1_pearson_extreme_points.csv`。

图表输出：

- `outputs/promoter_stage1/summary/violin_plots/stage1_per_gene_pearson_violin_compare.svg`
- `outputs/promoter_stage1/summary/violin_plots/stage1_per_cell_pearson_violin_compare.svg`

| Source | 模型组 | Valid / total | Mean Pearson | Median Pearson | Q25 | Q75 |
|---|---|---:|---:|---:|---:|---:|
| per-gene | ExpressionBaseline | 381 / 384 | 0.130138 | 0.099958 | 0.035625 | 0.222571 |
| per-gene | Intergenic control | 381 / 384 | 0.109130 | 0.060869 | -0.002662 | 0.236803 |
| per-gene | Real promoter | 7830 / 8121 | 0.123200 | 0.081076 | 0.002045 | 0.261248 |
| per-cell | ExpressionBaseline | 0 / 6144 | NaN | NaN | NaN | NaN |
| per-cell | Intergenic control | 6144 / 6144 | 0.141371 | 0.145509 | 0.090001 | 0.196840 |
| per-cell | Real promoter | 6144 / 6144 | 0.214686 | 0.205487 | 0.177694 | 0.250762 |

per-cell Pearson 是当前 Stage 1 最强的证据：

- expression-only 的 per-cell Pearson 全部为 NaN。原因是固定 test cell 时，它对不同 held-out genes 缺乏 target-gene-specific 输入变化；
- matched intergenic control 有非零 per-cell signal，可能来自 sequence branch 容量和 matched genomic background；
- real promoter 的 per-cell median Pearson 明显高于 matched intergenic control。

per-gene Pearson 当前需要谨慎横向比较：promoter 的 per-gene 输出已经覆盖完整 test gene set，而 expression-only 与 intergenic control 当前落盘结果仍是较小 gene 子集。因此 per-gene violin 是对现有文件的真实汇总，但不是最严格的同 gene universe 对照。

### 7.3 结果解释

当前 Stage 1 证据支持 clean gate：

1. real promoter 的 pooled Pearson/Spearman 高于 expression-only 和 matched intergenic control；
2. 在当前持久化汇总中，real promoter 的 pooled MSE/RMSE 也优于两个 baseline；
3. real promoter 最清晰的优势来自 per-cell Pearson，该指标直接测试模型是否能在同一个 frozen cell 内对 held-out genes 做正确排序。

per-gene Pearson 的提升较弱，需要谨慎解释。该指标测试的是固定一个 held-out gene 后，模型是否能跨 cells 追踪表达变化；在这个问题中，cell-state expression branch 预期会占主导。

### 7.4 当前 Caveats

- pooled metrics 当前三个模型均为 3 seeds，可作为 Stage 1 的主汇总表。
- per-cell Pearson 当前三个模型的 cell 数一致，适合作为 promoter 分支有效性的主要补充证据。
- per-gene Pearson 当前 promoter 与两个 baseline 的 gene 覆盖范围不完全一致；最终冻结图表前，建议把 expression-only 与 intergenic control 也重新输出全 test gene set 的 per-gene metrics。
- 每个 Stage 1 run 都应报告 per-cell Pearson 与 pooled MSE/RMSE；不能只看 pooled MSE。
- Stage 1 成立不意味着可以直接进入 generation。下一步应是 confirmatory clean reruns 或按既定路线进入 contrastive enhancement branch。
