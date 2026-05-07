# 缓存配置详解

## 📁 缓存层级架构

你的数据缓存分为两层：

### 1. HuggingFace 原始数据缓存 (HF_DATASETS_CACHE)
**位置**: `/data/HF_Cache_dataevo/datasets/` (已有 **374GB**)

**用途**: 
- `datasets.load_dataset()` 下载的原始数据集
- HuggingFace 自动管理
- **跨项目共享**：所有使用 HF datasets 的项目都复用这个缓存

**优点**:
- ✅ 自动去重（相同数据集只下载一次）
- ✅ 支持增量更新
- ✅ 已经配置了 HF Mirror 加速

**示例内容**:
```
EleutherAI___hendrycks_math/
openai___gsm8k/
qwedsacf___competition_math/
...
```

### 2. 已物化数据集缓存 (MATH_PT_SHARED_CACHE)
**位置**: `/data/yaxindu/datascientist/math_posttrain_cache/materialized_datasets/`

**用途**:
- 存储已经处理过的 `.jsonl` 文件
- 包含了你的数据选择和过滤逻辑
- **实验间共享**：避免重复处理相同的数据集配置

**命名规则**: `{dataset}_{config}_{max_rows}.jsonl`
```
openai_gsm8k_main_5000.jsonl
EleutherAI_hendrycks_math_algebra_10000.jsonl
qwedsacf_competition_math_default_28572.jsonl
```

**优点**:
- ✅ 避免重复下载和转换
- ✅ 不同实验可以复用相同的预处理数据
- ✅ 减少磁盘占用（相比每个实验都有自己的 `_cache`）

---

## 🚀 使用方法

### 方案 A: 使用共享缓存（推荐）

在运行实验前设置环境变量：

```bash
# 一次性设置
source playground/math_posttrain_datatree/setup_cache.sh

# 或者手动设置
export MATH_PT_SHARED_CACHE=/data/yaxindu/datascientist/math_posttrain_cache
```

**效果**:
```python
# 第一次运行某个配置
materialize_dataset_entry({"source_id": "openai/gsm8k", ...}, cache_dir, max_rows=5000)
# → 下载数据到 /data/HF_Cache_dataevo/datasets/openai___gsm8k/
# → 物化到 /data/yaxindu/datascientist/math_posttrain_cache/materialized_datasets/openai_gsm8k_main_5000.jsonl

# 第二次运行相同配置（任何实验）
materialize_dataset_entry({"source_id": "openai/gsm8k", ...}, cache_dir, max_rows=5000)
# → 直接从 .../openai_gsm8k_main_5000.jsonl 读取，秒开！
```

### 方案 B: 每个实验独立缓存（默认行为）

不设置 `MATH_PT_SHARED_CACHE`，每个实验会在自己的目录创建 `_cache/`：

```
runs/math_posttrain_datatree_20260408/
  └── workspaces/task_0/artifacts/train_packs/
      └── abc123.../_cache/
          ├── openai_gsm8k.jsonl
          └── EleutherAI_hendrycks_math.jsonl
```

**缺点**: 多次实验会重复存储相同数据

---

## 📊 缓存管理

### 查看缓存使用情况

```bash
# HF 原始数据缓存
du -sh /data/HF_Cache_dataevo/datasets
# 当前: 374GB

# 共享物化数据缓存
du -sh /data/yaxindu/datascientist/math_posttrain_cache
```

### 清理缓存

```bash
# 清理特定数据集的 HF 缓存
rm -rf /data/HF_Cache_dataevo/datasets/openai___gsm8k

# 清理所有物化数据缓存
rm -rf /data/yaxindu/datascientist/math_posttrain_cache/materialized_datasets/*

# 清理某个实验的独立缓存
rm -rf runs/math_posttrain_datatree_*/workspaces/*/artifacts/train_packs/*/_cache
```

### 预热缓存（可选）

如果你知道会用到哪些数据集，可以提前下载：

```python
from datasets import load_dataset

# 预下载常用数据集到 HF cache
datasets_to_preload = [
    "openai/gsm8k",
    "EleutherAI/hendrycks_math",
    "qwedsacf/competition_math",
]

for dataset_id in datasets_to_preload:
    print(f"Preloading {dataset_id}...")
    load_dataset(dataset_id, split="train[:100]")  # 下载并缓存
```

---

## ⚙️ 环境变量总结

| 变量 | 位置 | 用途 | 当前值 |
|-----|------|------|--------|
| `HF_HOME` | 全局 | HF 总缓存目录 | `/data/HF_Cache_dataevo` |
| `HF_ENDPOINT` | 全局 | 镜像地址 | `https://hf-mirror.com` |
| `HF_DATASETS_CACHE` | 全局 | datasets 原始数据 | `/data/HF_Cache_dataevo/datasets` (374GB) |
| `MATH_PT_SHARED_CACHE` | 项目 | 物化数据共享缓存 | `/data/yaxindu/datascientist/math_posttrain_cache` |

---

## 🎯 推荐配置

将以下内容添加到 `~/.bashrc` 或项目启动脚本：

```bash
# HuggingFace 配置
export HF_HOME=/data/HF_Cache_dataevo
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/data/HF_Cache_dataevo/datasets

# Math PostTrain 共享缓存
export MATH_PT_SHARED_CACHE=/data/yaxindu/datascientist/math_posttrain_cache
```

然后：
```bash
source ~/.bashrc
```

---

## 🔍 验证配置

```bash
# 检查环境变量
echo "HF_HOME: $HF_HOME"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "MATH_PT_SHARED_CACHE: $MATH_PT_SHARED_CACHE"

# 测试缓存功能
python -c "
import os
os.environ['MATH_PT_SHARED_CACHE'] = '/data/yaxindu/datascientist/math_posttrain_cache'
from playground.math_posttrain_datatree.core.utils.data import materialize_dataset_entry
from pathlib import Path
import tempfile

entry = {'source_id': 'openai/gsm8k', 'config': 'main'}
result = materialize_dataset_entry(entry, tempfile.gettempdir(), max_rows=100)
print(f'✓ Cached at: {result}')
"
```

---

## 💡 常见问题

### Q: 为什么需要两层缓存？
**A**: 
- **HF 缓存**存储原始数据（Parquet/Arrow 格式），所有项目共享
- **物化缓存**存储转换后的 JSONL，包含你的业务逻辑（过滤、格式化等）

### Q: 如果不设置 `MATH_PT_SHARED_CACHE` 会怎样？
**A**: 每个实验都会在自己的 `_cache/` 目录创建副本，浪费磁盘空间但不影响功能。

### Q: 缓存占用太多空间怎么办？
**A**: 
```bash
# 只保留最近使用的缓存
find /data/HF_Cache_dataevo/datasets -type f -atime +30 -delete
find /data/yaxindu/datascientist/math_posttrain_cache -type f -atime +7 -delete
```

### Q: 如何强制重新下载某个数据集？
**A**:
```bash
# 删除 HF 缓存
rm -rf /data/HF_Cache_dataevo/datasets/openai___gsm8k

# 删除物化缓存
rm /data/yaxindu/datascientist/math_posttrain_cache/materialized_datasets/openai_gsm8k_*.jsonl
```
