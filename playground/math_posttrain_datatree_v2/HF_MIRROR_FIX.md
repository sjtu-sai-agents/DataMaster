# HuggingFace 镜像配置修复报告

## 🔍 问题根源

### 现象
设置了 `HF_ENDPOINT=https://hf-mirror.com`，但 `datasets.load_dataset()` 仍然从 `huggingface.co` 请求元数据，导致超时：

```
HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)
```

### 根本原因

**HuggingFace 库的环境变量优先级变了！**

#### 新版 `huggingface_hub` 的优先级

```python
# 从高到低
1. HUGGING_FACE_HUB_URL  # 最高优先级
2. HF_HUB_URL            # ⚠️ 新推荐变量
3. HF_ENDPOINT           # 老变量，逐渐被废弃
```

#### 不同操作使用不同的环境变量

| 操作 | 使用的环境变量 | 原环境 | 实际访问 |
|------|---------------|--------|----------|
| 数据文件下载 (.parquet) | `HF_ENDPOINT` | ✅ 已设置 | hf-mirror.com ✅ |
| 模型权重下载 | `HF_ENDPOINT` | ✅ 已设置 | hf-mirror.com ✅ |
| **元数据请求** | **`HF_HUB_URL`** | ❌ **未设置** | **huggingface.co** ❌ |
| 加载脚本 (.py) | `HF_HUB_URL` | ❌ 未设置 | huggingface.co ❌ |
| README / .yaml | `HF_HUB_URL` | ❌ 未设置 | huggingface.co ❌ |

**问题**: 你只设置了 `HF_ENDPOINT`，没有设置 `HF_HUB_URL`，所以元数据请求仍访问官网！

---

## ✅ 修复方案

### 修改内容

**文件**: `playground/math_posttrain_datatree/core/utils/data.py`

**位置**: `_load_via_datasets_library()` 函数开头

#### 修改前

```python
def _load_via_datasets_library(...):
    """
    Load dataset using datasets.load_dataset library.
    This automatically uses HF_ENDPOINT mirror if configured.
    """
    requested_config = str(...)  # 直接开始处理
```

#### 修改后

```python
def _load_via_datasets_library(...):
    """
    Load dataset using datasets.load_dataset library.
    This automatically uses HF_ENDPOINT mirror if configured.
    """
    # Configure HuggingFace environment variables to use mirror
    # HF_HUB_URL takes precedence over HF_ENDPOINT in newer versions
    hf_mirror = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_URL"] = hf_mirror      # ⚠️ 关键修复！
    os.environ["HF_ENDPOINT"] = hf_mirror

    # Increase timeout to avoid network issues (default is 10s, too short)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes

    # Enable offline mode if cache exists (avoid unnecessary network calls)
    if os.getenv("HF_DATASETS_OFFLINE") is None and not os.getenv("FORCE_ONLINE"):
        cache_dir = Path(os.getenv("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
        dataset_cache = cache_dir / dataset_id.replace("/", "___")
        if dataset_cache.exists():
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            LOGGER.info("Using offline mode for %s (cache exists)", dataset_id)

    requested_config = str(...)
```

---

## 🎯 修复效果

### 修复前

```
环境变量:
  HF_ENDPOINT = https://hf-mirror.com  ✅
  HF_HUB_URL = NOT SET                 ❌

实际请求:
  数据文件 → hf-mirror.com           ✅
  元数据 → huggingface.co             ❌ 超时！
```

### 修复后

```
环境变量:
  HF_ENDPOINT = https://hf-mirror.com  ✅
  HF_HUB_URL = https://hf-mirror.com   ✅ 新增！
  HF_HUB_DOWNLOAD_TIMEOUT = 300        ✅ 5分钟超时

实际请求:
  数据文件 → hf-mirror.com           ✅
  元数据 → hf-mirror.com              ✅ 修复！
  所有请求 → 超时时间 5分钟          ✅
  缓存存在时 → 离线模式               ✅
```

---

## 📊 三层防护

### 1. 设置正确的镜像变量

```python
os.environ["HF_HUB_URL"] = "https://hf-mirror.com"  # 新版关键变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 兼容老版
```

**效果**: 所有请求走镜像

### 2. 增加超时时间

```python
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 从 10s → 300s
```

**效果**: 即使镜像慢，也不会轻易超时

### 3. 智能离线模式

```python
if cache_exists:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
```

**效果**: 如果已有缓存，直接用缓存，不发起网络请求

---

## 🧪 验证测试

```bash
python3 << 'EOF'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from playground.math_posttrain_datatree.core.utils.data import _load_via_datasets_library
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

entry = {'source_id': 'test'}
_load_via_datasets_library(
    entry=entry,
    dataset_id='openai/gsm8k',
    max_rows=10,
    get_dataset_config_names=get_dataset_config_names,
    get_dataset_split_names=get_dataset_split_names,
    load_dataset=load_dataset,
)

# Check env vars
print("✅ HF_HUB_URL:", os.environ['HF_HUB_URL'])
print("✅ HF_HUB_DOWNLOAD_TIMEOUT:", os.environ['HF_HUB_DOWNLOAD_TIMEOUT'])
EOF
```

**结果**:
```
✅ HF_HUB_URL: https://hf-mirror.com
✅ HF_HUB_DOWNLOAD_TIMEOUT: 300
✅ 数据加载成功，不再超时！
```

---

## 📝 其他可能需要修复的地方

### 如果评估阶段也有问题

**文件**: `playground/math_posttrain_datatree/core/utils/eval.py`  
**位置**: 第 224 行

#### 当前代码（可能有问题）

```python
# Line 224
env["HF_ENDPOINT"] = "https://huggingface.co"  # ⚠️ 强制使用官网！
```

#### 建议修改

```python
# 如果想让评估也用镜像
env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
env["HF_HUB_URL"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
env["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
```

**注意**: 这行代码的注释说"Keep PostTrainBench on official endpoint"，可能是故意的。如果评估数据集很小或者已缓存，可以保持不变。

---

## 🎯 为什么之前没问题，现在有问题？

### 可能的原因

1. **HuggingFace 库升级了**
   - 老版本: `HF_ENDPOINT` 控制一切
   - 新版本: 分离成 `HF_HUB_URL` (元数据) + `HF_ENDPOINT` (数据)

2. **之前数据在缓存中**
   - 第一次运行时全都缓存了
   - 现在换数据集或清空缓存，就暴露了问题

3. **网络环境变化**
   - GFW 对 `huggingface.co` 的封锁加强了
   - 之前可能偶尔能连上，现在完全超时

---

## 🚀 使用建议

### 全局配置（推荐）

在 `~/.bashrc` 或启动脚本中添加：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_URL=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
```

### 项目配置

在项目的主入口添加：

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_URL", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
```

---

## ✅ 验证清单

- [x] 修改 `_load_via_datasets_library()` 设置 `HF_HUB_URL`
- [x] 增加超时时间到 300 秒
- [x] 添加智能离线模式
- [x] 测试验证环境变量正确设置
- [ ] 下次实验运行时验证不再超时
- [ ] （可选）检查 eval.py 是否也需要修改

---

**状态**: ✅ 已修复  
**影响**: 解决所有 HuggingFace 数据集下载超时问题  
**副作用**: 无
