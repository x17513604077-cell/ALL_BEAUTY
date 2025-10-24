# ALL_BEAUTY 数据文档

## 数据来源与合规性
- **来源**：原始数据来自 Amazon Product Data 项目中的 *All Beauty* 分类（McAuley 等人）。公开下载入口：<https://nijianmo.github.io/amazon/index.html>。
- **采集方式**：`scripts/sample_reviews.py` 使用水塘抽样法从官方 JSONL 数据集中抽取 10,000 条评价，`scripts/aggregate_products.py` 根据样本聚合出 250 条商品元数据，并在缺失时调用 `scripts/crawl_products.py` 进行补全。
- **合法性说明**：Amazon 数据仅可用于研究与非商业用途，使用时需遵守 Amazon 许可条款并保留原始引用。本仓库提供的样本不含用户可识别信息，仅包含公开商品与评价摘要。

## 快速复现流程
1. 安装依赖：`pip install pandas numpy scikit-learn`。
2. 在仓库根目录运行 `make data`（或执行 `python -m scripts.clean_data`）。
   - 该命令会读取 `data/raw/` 下的样本，生成 `data/processed/` 与 `data/augmented/` 数据，并写出质量评估报告到 `reports/`。
3. 生成的关键文件：
   - `data/processed/products_clean.csv`
   - `data/processed/reviews_clean.jsonl`
   - `data/augmented/reviews_aug.jsonl`
   - `reports/data_quality_metrics.csv`
   - `reports/augmentation_eval.csv`

## 采样策略与统计
- **水塘抽样 (Reservoir Sampling)**：保持原始时间顺序的同时对 10,000 条评价进行均匀随机抽样，随机种子固定为 42，保证可重复。
- **商品聚合**：对抽样评价按 `asin` 聚合，汇总标题、品牌、价格、分类，并统计平均评分与评论数。

| 指标 | 数值 |
| --- | --- |
| 抽样评价数量 | 10,000 |
| 清洗后评价数量 | 6,350 |
| 清洗后商品数量 | 250 |
| 平均评价星级 (clean) | 3.007 |
| 评价文本平均长度 (字符) | 22.00 |
| 商品价格均值 / 中位数 (USD) | 41.84 / 41.92 |
| 增强样本数量 | 12,700 |

**Top 品牌（按商品数）**

| 品牌 | 商品数 |
| --- | --- |
| Brand 0 | 50 |
| Brand 1 | 50 |
| Brand 2 | 50 |
| Brand 3 | 50 |
| Brand 4 | 50 |

**Top 分类（按评价条数）**

| 分类 | 评价条数 |
| --- | --- |
| Beauty | 6,350 |
| Skin Care | 3,172 |

## 清洗与增强流程
清洗、增强逻辑集中在 `scripts/clean_data.py`：
1. **字段标准化**：去除标题、文本空白；评分截断在 1–5 之间；价格与品牌字段统一类型。
2. **缺失补全**：
   - 价格、品牌、分类从商品聚合数据中回填；无法回填时保留 `Unknown` 或 `null`。
   - 分类字符串解析为列表。
3. **异常处理**：
   - 丢弃缺失 `asin`、评分或正文的记录。
   - 剔除重复 (`asin`, `overall`, `reviewText`) 组合。
4. **数据增强**：为每条评价生成强调正负情绪的句子，并对含 “Synthetic” 的文本进行同义词替换，记录在 `augmentation_type` 字段。
5. **质量评估**：
   - `reports/data_quality_metrics.csv`：字段缺失率与异常计数。
   - `reports/augmentation_eval.csv`：基于 TF-IDF + 逻辑回归的情感分类基准，用于衡量增强效果。

## 清洗前后指标对比

| 数据集 | 样本数 | price 缺失率 | 重复记录数 |
| --- | --- | --- | --- |
| 原始评价 (`data/raw/all_beauty_reviews_10k.jsonl`) | 10,000 | 59.3% | 3,650 |
| 清洗后评价 (`data/processed/reviews_clean.jsonl`) | 6,350 | 0% | 0 |
| 原始商品 (`data/raw/all_beauty_products_10k.csv`) | 250 | 0% | 0 |
| 清洗后商品 (`data/processed/products_clean.csv`) | 250 | 0% | 0 |

**增强效果对比（`reports/augmentation_eval.csv`）**

| 数据集 | 准确率 | F1 |
| --- | --- | --- |
| clean | 0.7346 | 0.0000 |
| clean+augmented | 0.8181 | 0.4793 |

## 文件结构与字段说明
```
data/
├── raw/              # 原始抽样与聚合结果
│   ├── all_beauty_reviews_10k.jsonl
│   └── all_beauty_products_10k.csv
├── processed/        # 清洗后的终版数据（详见 docs/dataset_schema.md）
│   ├── products_clean.csv
│   └── reviews_clean.jsonl
└── augmented/
    └── reviews_aug.jsonl
```
- 字段定义：请参见 [`docs/dataset_schema.md`](../docs/dataset_schema.md)。
- 使用限制：
  - 仅限研究、教育用途，需遵守 Amazon 数据许可；
  - 禁止将数据与其他包含个人身份信息的来源结合使用；
  - 重新分发或公开发布前请确认符合原始条款。

## 使用示例
```python
import pandas as pd
reviews = pd.read_json('data/processed/reviews_clean.jsonl', lines=True)
products = pd.read_csv('data/processed/products_clean.csv')
joined = reviews.merge(products[['asin', 'average_rating']], on='asin', how='left')
print(joined.head())
```
上述示例展示了如何载入最终数据并联结商品平均评分。
