# process_data.py 处理流程合规性审查报告

## 一、流程概览

当前 `process_data.py` 的 `main()` 实际执行路径为：

```
已保存的 integrated_data.h5ad (UMI counts)
  → compute_tpm()         [CPM归一化 + log1p]
  → draw_high_quality_samples()  [Seurat v3 HVG top2000]
  → filter_genes()        [与promoter取交集 + split_train_val]
  → 写出 log_integrated_data_hvg.h5ad
```

原始流程（被注释掉，第446-452行）为：

```
loom文件 → build_integrated_data() → add_gene_annotation() → filter_cells() → filter_genes()
```

另有 UMI 分支（`umi_processed/` 和 `umi_highquality/`）专供 ZINB loss 使用，其数据为原始 UMI counts 不做 log1p 变换。

---

## 二、逐步骤审查

### 2.1 `build_integrated_data()` — 数据加载与稀疏化

| 项目 | 评估 |
|------|------|
| 稀疏矩阵转换 | ✅ 使用分块读取避免内存溢出，合理 |
| AnnData 构建 | ✅ 正确传入 cell IDs 和 gene names |
| 数据类型 | ✅ 转换为 float32 节约内存 |

**问题：**
- `hdf5_matrix_to_sparse()` 中默认 `transpose=True`，但函数文档写的是 "shape (n_genes, n_cells)" → 返回 "cell × gene"，调用者需要知道这个约定。当前没有问题，但缺少显式的 shape 断言确认转换正确。

### 2.2 `add_gene_annotation()` — 基因注释

| 项目 | 评估 |
|------|------|
| symbol→FlyBase ID 映射策略 | ✅ 优先 current_symbol，仅当 synonym 唯一时才回退，避免歧义 |
| GTF 解析 | ✅ 使用 pyranges，按 gene_id 聚合取 min Start / max End |
| MT 标记 | ✅ 通过 `mitochondrion_genome` 标记线粒体基因 |

**问题：**
- 第167行 `gene_length = End - Start` 计算的是基因体长度（含内含子），不是 CDS 或转录本长度。对于 TPM 归一化（如果需要的话），应使用 exon 长度之和（所有 exon 的 feature）。但当前 TPM 计算中基因长度被注释掉了（第432行），所以暂时不影响结果。
- `current_symbol_to_fbid.setdefault(cur, fbid)` 应为 `setdefault` → `setdefault` 不存在于 Python dict。这是 **bug**：应该是 `setdefault` 应为 `setdefault`。等等，让我重新读代码... 实际代码是 `current_symbol_to_fbid.setdefault(cur, fbid)`，Python dict 没有 `setdefault` 方法，应该用 `setdefault` 或 `dict.setdefault`。实际上阅读第127行：`current_symbol_to_fbid.setdefault(cur, fbid)` — 这确实是 bug！Python dict 的方法是 `setdefault()`。这会导致 AttributeError。**除非这个函数实际上从未在当前流程中被调用**（因为 main() 中直接从已保存的 h5ad 开始，跳过了 add_gene_annotation）。

   **更新**：经确认，当前 `main()` 直接读取已保存的 `integrated_data.h5ad`，跳过了 `add_gene_annotation()`。但如果是首次运行（取消注释第446-452行），会触发 `AttributeError: 'dict' object has no attribute 'setdefault'`。这是一个潜伏的 **关键 bug**。

### 2.3 `filter_cells()` — 细胞过滤

| 项目 | 评估 |
|------|------|
| QC 指标计算 | ✅ 使用 scanpy 标准函数，正确传入 `layer='counts'` |
| MT 含量计算 | ✅ 依赖前面标记的 `mt` 列 |

**问题：**

1. **仅按 total_counts 取 top 10% — 过于激进**
   - 标准 scRNA-seq 流程通常保留 50-90% 的细胞，而非仅保留 top 10%
   - 这会严重偏向高表达细胞，模型的预测能力可能无法泛化到中低表达水平的细胞
   - 数据中约 5 万细胞被缩减到约 5 千（从 `data_sanity_summary.json` 中 `n_cells=50666` 可推算原始约 50 万细胞）
   - **建议**：改为多条件过滤（如 `min_counts > 500, max_counts < 某个上界, pct_mt < 20%`），或至少放宽 top fraction 到 0.5 以上

2. **未使用 MT 含量过滤**
   - 第245-248行绘制了 `total_counts vs pct_counts_mt` 散点图，但未根据 MT 比例过滤
   - 高 MT 含量的细胞通常是受损/濒死细胞，应被移除
   - **建议**：添加 `rawdata = rawdata[rawdata.obs['pct_counts_mt'] < 20, :]`

3. **未过滤双细胞（doublets）**
   - 对于 10x 数据，doublet 率通常为 0.5-8%（取决于细胞数）
   - **建议**：考虑使用 Scrublet 或 DoubletFinder 进行双细胞检测

4. **未过滤低表达基因**
   - 虽然下游会做 HVG 选择，但在 QC 阶段过滤掉在极少数细胞中表达的基因（如 `< 3 cells`）是标准做法
   - **建议**：添加 `sc.pp.filter_genes(rawdata, min_cells=3)`

### 2.4 `compute_tpm()` — 归一化

| 项目 | 评估 |
|------|------|
| 每细胞总计数归一化 | ✅ `sc.pp.normalize_total(target_sum=1e6)` — CPM 标准化 |
| log1p 变换 | ✅ 标准操作 |

**问题：**

1. **函数名误导：名为 `compute_tpm` 但实际做的是 CPM**
   - TPM (Transcripts Per Million) 需要先除以基因长度再归一化：`TPM = (counts / gene_length) * (1e6 / sum(counts / gene_length))`
   - 当前代码只做了 `normalize_total(target_sum=1e6) + log1p`，这是 CPM (Counts Per Million) 或更准确地说是 log(CPM + 1)
   - 基因长度除法在第432行被注释掉了，注释中写的是 "using gene length to compute RPKM/FPKM"
   - **建议**：将函数重命名为 `compute_log_cpm()` 或类似名称，或者实现真正的 TPM 计算

2. **对 UMI 数据的适用性**
   - UMI 数据通常不需要基因长度校正（因为 UMI 只计每个转录本的1次），CPM 或简单的 library size 归一化已经足够
   - 所以当前行为（CPM + log1p）对于 UMI 数据是合理的，只是命名不对

### 2.5 `draw_high_quality_samples()` — 高变基因选择

| 项目 | 评估 |
|------|------|
| Seurat v3 方法 | ✅ 适用于 log-normalized 数据 |
| top 2000 | ✅ 合理范围（标准是 2000-5000） |

**问题：**

1. **HVG 选择在 train/val/test 划分之前 — 数据泄露**
   - 这是当前流程中**最严重的合规性问题**
   - `main()` 中的执行顺序是：`compute_tpm → draw_high_quality_samples → filter_genes(→ split_train_val)`
   - HVG 选择使用了全部细胞和全部基因的信息（均值、方差），其中包含了本应属于 val/test 的基因
   - 这导致验证集和测试集的性能评估存在乐观偏差
   - **正确做法**：应该先在基因层面做 train/val/test 划分，然后仅使用 train 集的基因来选择 HVG，再将选出的基因子集应用于 val/test
   - 但由于此处 HVG 选择在基因维度而非细胞维度，需要仔细设计：当前 split 是按基因的基因组位置划分的，而 HVG 也在选基因。应在 split 之后，仅用 train genes 的表达数据选 HVG，然后将筛选后的基因列表应用到所有 split

2. **未检查 HVG 选择后的基因数是否足够**
   - 如果 promoter 交集后的基因数少于 2000，`n_top_genes=2000` 会默默选所有基因
   - 当前实际数据中有约 1.6 万基因，2000 HVG 占约 12.5%，合理

3. **函数副作用**
   - `draw_high_quality_samples()` 修改 `rawdata` 为 HVG 子集后返回，又将其传给 `filter_genes()` 做 promoter 交集和 split
   - 但在 `filter_genes()` 中又会重新从 `promoters.fa` 读取全部启动子、重新输出 promoter CSV 文件
   - 这意味着 **promoter CSV 文件被重复生成了两次**（一次在原始流程 filter_genes 中，一次在 main 末尾的 filter_genes 中），且第二次会覆盖第一次

### 2.6 `filter_genes()` — 基因过滤与数据划分

| 项目 | 评估 |
|------|------|
| 去重（保留非重复 gene_id） | ✅ 避免了同一 gene_id 对应多行的歧义 |
| 与 promoter 取交集 | ✅ 确保模型有对应的启动子序列 |
| 染色体位置划分策略 | ✅ 基因组位置连续块划分，防止空间自相关导致的泄露 |

**问题：**

1. **去重逻辑过于简单**
   - 第394行：`df.duplicated(subset="gene_id", keep=False) == False` 会移除所有出现重复的 gene_id，即使重复中可能有一个正确的
   - 丢失的数据可能包含有效信息
   - **建议**：对于有重复 gene_id 的基因，保留表达量最高的那个（或第一个出现的），而非全部删除

2. **函数职责混杂**
   - `filter_genes()` 同时做了：基因过滤（与 promoter 交集）、去重、split_train_val、写 CSV
   - 函数名暗示只做"基因过滤"，但实际承担了数据划分的职责
   - **建议**：将 `split_train_val` 调用移到 main() 中，保持函数单一职责

3. **duplicated 的检查方向**
   - `filter_genes` 第394行检查 `df.duplicated(subset="gene_id", keep=False) == False`
   - 注意 `df` 是 `rawdata.var`，即 AnnData 的 var DataFrame
   - `keep=False` 意味着标记所有重复项为 True，然后取反（保留非重复项）
   - 这确实会丢弃所有有重复的基因，在 AnnData 中是安全的做法

### 2.7 `split_train_val()` — 训练/验证/测试划分

| 项目 | 评估 |
|------|------|
| 按基因组位置连续块划分 | ✅ 避免了相邻基因间的信息泄露 |
| 互斥性检查 | ✅ 去重并打印交集验证 |
| 仅 chr 2R/3R 做划分 | ⚠️ 有争议 |

**问题：**

1. **验证/测试集仅来自 chr 2R 和 3R**
   - 其他染色体全部分配给训练集（第292-294行）
   - 这意味着 val/test 的基因都位于两条染色体上，可能无法代表全基因组水平
   - 2R 和 3R 是果蝇两条最大的常染色体（约占基因组的60%），具有一定代表性
   - **但**：如果 2R/3R 上有特殊的染色质结构、调控元件或基因密度特征，val/test 评估会有偏差
   - **建议**：考虑交叉验证策略，或至少在所有染色体上均匀采样作为 val/test

2. **划分在基因级别而非启动子级别**
   - 当前按 gene_id 划分，一个基因只出现在一个集合中
   - 正确做法 ✅，防止同一基因的不同启动子出现在 train 和 test 中

3. **连续块划分可能受局部效应影响**
   - 如果 chr 2R 上前 80% 区域和后 20% 区域有系统差异（如着丝粒附近 vs 端粒附近），划分会引入偏差
   - **建议**：考虑交错采样（如每5个基因取1个为val，每5个取1个为test），或者基于位置的 k-fold 策略

---

## 三、全局性问题

### 3.1 数据版本管理混乱

当前存在四套数据目录，但处理逻辑散落在注释中：
- `processed/` — log-normalized 全基因集
- `highquality/` — log-normalized HVG 2000
- `umi_processed/` — UMI counts 全基因集
- `umi_highquality/` — UMI counts HVG 2000

这些目录的生成逻辑不透明，promoter CSV 文件在多个目录间重复。**建议**：用单一的 CLI 脚本参数化地生成所有数据版本，并记录版本的来源和参数。

### 3.2 注释掉的代码过多

`main()` 函数中有大量被注释掉的步骤（第446-461行），说明流程在迭代中发生了变更但旧代码未被清理。这会造成维护困难和新使用者的困惑。

### 3.3 `setdefault` Bug

第127行 `current_symbol_to_fbid.setdefault(cur, fbid)` 在 Python 中不存在。正确写法是 `current_symbol_to_fbid.setdefault(cur, fbid)` 或 `if cur not in current_symbol_to_fbid: current_symbol_to_fbid[cur] = fbid`。好在当前 main() 不调用该函数，但首次构建数据时必须修复。

### 3.4 缺少批次效应评估

原始 loom 文件来自多个样本合并，但 pipeline 中没有任何批次效应检测或校正步骤（如 Harmony、Scanorama、BBKNN）。如果存在显著的批次效应，模型可能学习到批次特异而非生物学相关的信号。

### 3.5 缺失数据版本追踪

没有记录处理所使用的软件版本（scanpy、pandas 等）和参数。**建议**：在输出的 h5ad 文件中添加 `adata.uns['processing_params']` 记录处理参数。

---

## 四、建议修正的优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| **P0-致命** | `setdefault` bug — 首次构建数据时会崩溃 | 阻塞首次运行 |
| **P0-严重** | HVG 选择在 split 之前 — 数据泄露 | 模型评估指标系统性乐观 |
| **P1-高** | 细胞过滤仅保留 top 10% — 严重偏差 | 模型无法泛化到中低表达细胞 |
| **P1-高** | 未过滤高 MT 含量的受损细胞 | 数据中混杂低质量细胞 |
| **P2-中** | 函数命名误导（`compute_tpm` 实际做 CPM） | 代码可读性差 |
| **P2-中** | 验证/测试集仅来自 chr 2R/3R | 评估可能不具全基因组代表性 |
| **P2-中** | `filter_genes` 函数职责混杂 | 可维护性差 |
| **P3-低** | 缺少 doublet 检测 | 数据质量 |
| **P3-低** | 缺少批次效应分析和校正 | 模型可能学习批次信号 |
| **P3-低** | 注释代码未清理 | 可维护性差 |

---

## 五、建议的处理流程（修正后）

```
1. build_integrated_data()           — 从 loom 构建 AnnData
2. add_gene_annotation()             — 基因注释（修复 setdefault bug）
3. filter_cells()                     — 多条件过滤（MT、min_counts、max_counts）
4. sc.pp.filter_genes(min_cells=3)   — 低表达基因过滤
5. filter_genes() 或独立步骤        — 去重 + 与 promoter 取交集
6. split_train_val()                  — 仅用 train genes 的基因组位置划分
7. 仅用 train set 选 HVG             — 用 train 的细胞×基因子矩阵选 HVG
8. 将 HVG 列表应用于 train/val/test — 各自取子集
9. compute_log_cpm()                  — CPM 归一化（对 log-based 模型）
   或保持 UMI counts（对 ZINB 模型）
10. 写出带参数记录的 h5ad
```
