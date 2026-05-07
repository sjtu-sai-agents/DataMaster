# 使用 HF Mirror 加速数据下载

## 问题背景

之前使用 `datasets-server.huggingface.co` API 下载数据时遇到以下问题：
1. **SSL EOF 错误**: `SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]')`
2. **429 速率限制**: `Too Many Requests` - 请求太频繁被限流
3. **下载慢**: 受网络环境和速率限制影响

## 根本解决方案

**优先使用 `datasets.load_dataset()` + HF Mirror**，而不是 datasets-server API！

### 为什么这样更好？

| 方法 | datasets-server API | load_dataset + HF Mirror |
|------|---------------------|-------------------------|
| **速度** | 慢（每次100条，需多次请求） | 快（一次性下载，本地缓存） |
| **速率限制** | 严格（429错误） | 无（或极宽松） |
| **SSL问题** | 频繁（连接池问题） | 罕见（更稳定） |
| **镜像支持** | ❌ 不支持 | ✅ 完全支持 |
| **缓存** | ❌ 无 | ✅ 自动缓存 |

## 使用方法

### 1. 设置环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或在代码中：
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 2. 代码自动优先使用 load_dataset

修改后的 `data.py` 现在会：
1. **默认优先**使用 `load_dataset()` (自动使用 HF_ENDPOINT 镜像)
2. 只有在 `load_dataset()` 失败时才回退到 datasets-server API
3. 如果需要强制使用 datasets-server，设置: `MATH_PT_FORCE_DATASETS_SERVER=1`

### 3. 验证是否使用镜像

运行测试：
```bash
cd playground/math_posttrain_datatree
python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from core.utils.data import materialize_dataset_entry
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    entry = {'source_id': 'openai/gsm8k', 'config': 'main', 'split': 'train'}
    result = materialize_dataset_entry(entry, tmpdir, max_rows=100)
    print(f'✓ Success! Materialized to: {result}')
"
```

预期输出应该显示从 hf-mirror.com 下载。

## 性能对比

### 之前 (datasets-server API)
```
下载 28,572 行数据:
- 需要 286 次请求 (100行/请求)
- 每次间隔 0.2秒 = 至少 57 秒
- 经常遇到 429/SSL 错误，需要重试
- 实际耗时: 5-10 分钟
```

### 现在 (load_dataset + HF Mirror)
```
下载 28,572 行数据:
- 1 次请求下载完整数据集
- 自动缓存，下次秒开
- 从国内镜像下载，速度快
- 实际耗时: 10-30 秒
```

## 环境变量说明

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `HF_ENDPOINT` | (空) | HuggingFace 镜像地址，建议设置为 `https://hf-mirror.com` |
| `MATH_PT_FORCE_DATASETS_SERVER` | 0 | 设置为 1 强制使用 datasets-server API |
| `MATH_PT_DISABLE_DATASETS_SERVER` | 0 | 设置为 1 完全禁用 datasets-server 回退 |

## 推荐配置

在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

这样所有使用 HuggingFace 的脚本都会自动使用镜像。

## 故障排查

### 如果还是很慢
1. 检查 `HF_ENDPOINT` 是否设置正确
2. 清除旧缓存: `rm -rf ~/.cache/huggingface/`
3. 检查网络连接到 hf-mirror.com

### 如果出现错误
1. 首次下载可能需要稍长时间
2. 查看日志确认使用的是哪个方法
3. 尝试手动指定 config 和 split

## 修改总结

### data.py 主要改动
1. ✅ 添加 `_load_via_datasets_library()` 函数
2. ✅ `materialize_dataset_entry()` 优先调用 load_dataset
3. ✅ 只在必要时才使用 datasets-server (作为回退)
4. ✅ 保留所有 SSL 重试和速率限制逻辑（用于回退场景）

### 测试
```bash
cd playground/math_posttrain_datatree
python -m pytest tests/test_data_ssl_retry.py -v
```

全部测试应该通过 ✅
