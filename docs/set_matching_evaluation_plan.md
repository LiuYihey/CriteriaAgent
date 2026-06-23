# Set Matching 一致性评估方案

## 1. 目标

用 **embedding-based set matching** 衡量 AI 生成的 eligibility criteria 与专家原文之间的一致性，替代 LLM-based agreement scoring 和 BERTScore。对 CriteriaAgent (Pipeline) 和 Direct Generation 两种方法分别计算 Recall / Precision / F1 及对应的 Soft 版本，进行对比。

## 2. 数据来源

| 数据 | 路径 | 说明 |
|------|------|------|
| Expert criteria | `CriteriaBench/final_bench_filtered/trials/*.json` → `eligibilityModule.eligibilityCriteria` | 专家原文，含 Inclusion / Exclusion |
| Pipeline output | `outputs/criteria_v0_bench/{nct_id}/criteria_final.md` | CriteriaAgent 优化后最终版 |
| Direct generation | `CriteriaBench/final_bench_filtered/generated_criteria.jsonl` → `generated_criteria` 字段 | 直接生成（无graph/optimizer） |

> **注意**：当前脚本 `PIPELINE_OUT` 指向 `criteria_pipeline_bench`（仅2条），需修正为 `criteria_v0_bench`（64条）。

## 3. 评估指标体系

### 3.1 硬指标 (Hard Metrics)

基于阈值 τ = 0.5 的二值匹配：

$$\text{Recall} = \frac{n_{\text{matched}}}{n_{\text{expert}}}, \quad \text{Precision} = \frac{n_{\text{matched}}}{n_{\text{AI}}}, \quad F1 = \frac{2 \cdot P \cdot R}{P + R}$$

### 3.2 软指标 (Soft Metrics)

以实际 cosine similarity 加权，不做阈值二值化：

$$\text{Soft Recall} = \frac{\sum_{i=1}^{n_{\text{matched}}} \text{sim}_i}{n_{\text{expert}}}, \quad \text{Soft Precision} = \frac{\sum_{i=1}^{n_{\text{matched}}} \text{sim}_i}{n_{\text{AI}}}, \quad \text{Soft F1} = \frac{2 \cdot sP \cdot sR}{sP + sR}$$

软指标反映"覆盖了多少语义信息量"，而不仅是"匹配了几条"。

## 4. 核心算法

### 4.1 Embedding 编码器

- **模型**: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
- **理由**: 在临床 SNLI 上微调的 BioBERT，对医学文本语义理解最强
- **依赖**: `sentence-transformers` 库

### 4.2 文本预处理

```
原始 criteria 文本
    ↓ parse_criteria_bullets()
[(bullet_text, "inclusion"|"exclusion"), ...]
```

1. 正则匹配 `### Inclusion` / `### Exclusion` section headers
2. 用 `^\s*(?:\d+\.|[-*])\s*(.+?)$` 提取编号/列表条目
3. 每条 bullet 打上 section label

### 4.3 贪心二分匹配 (Greedy Bipartite Matching)

```
Expert bullets (E)  ───┐
                        ├─→ cosine similarity matrix S (|E| × |A|)
AI bullets (A)     ───┘

For each expert bullet (按最佳匹配分降序):
    1. 找到 cos_sim 最高的未匹配 AI bullet
    2. If cos_sim ≥ τ (0.5):
       a. Negation flip 检查
          - 同 section + flip → 匹配失败（语义矛盾）
          - 跨 section + flip → 匹配成功（合法反转，如 inclusion↔exclusion）
       b. 标记为已匹配，记录 sim score
```

**为什么用贪心而非匈牙利算法**：
- 每个 bullet 语义独立，不需要全局最优
- 贪心按置信度排序，优先匹配最确定的对
- 复杂度 O(n²) 足够（n 通常 5–20 条）

### 4.4 Negation Flip 检测

用关键词极性计数判断语义翻转：

| 极性词集 | 示例 |
|---------|------|
| NEG_WORDS | no, not, without, never, exclude, absence, denied, refused, discontinue |
| POS_WORDS | currently, active, present, history of, has, receiving, diagnosed with |

```python
e_pol = count_neg(expert_bullet) - count_pos(expert_bullet)
a_pol = count_neg(ai_bullet) - count_pos(ai_bullet)
flip = (e_pol > 0 and a_pol < 0) or (e_pol < 0 and a_pol > 0)
```

## 5. 执行流程

```
┌─────────────────────────────────────────────────┐
│ 1. 加载 embedding model (BiomedNLI-BioBERT)     │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 2. 加载数据                                      │
│    - expert criteria from trial JSONs            │
│    - pipeline criteria_final.md                  │
│    - direct generation JSONL                     │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 3. 逐 trial 评估                                 │
│    For each NCT (64 trials):                     │
│      a. parse_criteria_bullets() × 3             │
│      b. compute_set_matching(expert, pipeline)   │
│      c. compute_set_matching(expert, direct)     │
│      d. 写入 consistency_scores.jsonl            │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 4. 汇总 summary.json                             │
│    - Per-method mean/std for all 6 metrics       │
│    - Delta (pipeline - direct) per metric        │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 5. 绘图 (plot_bench_comparison.py)               │
│    - Grouped bar: R / P / F1 / sR / sP / sF1    │
│    - Pipeline vs Direct                          │
└─────────────────────────────────────────────────┘
```

## 6. 输出文件

| 文件 | 内容 |
|------|------|
| `outputs/bench_consistency/consistency_scores.jsonl` | 逐 trial 的 6 指标 + bullet 计数 |
| `outputs/bench_consistency/summary.json` | 全局均值、标准差、delta |
| `CriteriaAgent_EMNLP2026/figures/bench_consistency_bars.png/.pdf` | 对比柱状图 |

## 7. 脚本修改清单

### 7.1 修复路径 (`scripts/run_consistency_eval.py`)

```python
# 修改前
PIPELINE_OUT = ROOT / "outputs" / "criteria_pipeline_bench"

# 修改后
PIPELINE_OUT = ROOT / "outputs" / "criteria_v0_bench"
```

### 7.2 运行命令

```powershell
python scripts/run_consistency_eval.py
```

可选参数：
- `--model`: 替换 embedding 模型
- `--threshold`: 调整匹配阈值（默认 0.5）
- `--limit N`: 仅评估前 N 个 trial（调试用）

### 7.3 绘图

已集成在 `Analyze/plot_bench_comparison.py` 的 `plot_consistency_bars()` 函数中，运行一致性评估后会自动生成图表。

## 8. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 匹配策略 | 全局池化（inclusion+exclusion混合） | 允许跨section匹配，捕获inclusion↔exclusion的合法语义反转 |
| 阈值 | τ = 0.5 | 平衡严格性与宽容度；太低会引入噪声匹配，太高会遗漏同义替换 |
| 编码器 | BiomedNLI-BioBERT | 临床NLI微调，比普通sentence-BERT更适合医学criteria文本 |
| Negation处理 | 关键词极性检测 + section约束 | 避免同section内的语义矛盾被误判为匹配 |
| 匹配算法 | 贪心（非匈牙利） | bullet语义独立、规模小，贪心更简单且效果等价 |
