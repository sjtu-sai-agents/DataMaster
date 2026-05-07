# 为什么 1.7B 小模型却需要 60+ GB 显存？

## 🤯 令人困惑的现象

**Qwen3-1.7B** 模型只有 **3.3 GB** 大小，但 vLLM 配置 `gpu_memory_utilization: 0.8` 却需要 **63 GB** 显存！

这不是 bug，而是 **KV Cache 的锅**！

---

## 📊 显存占用分解

| 组件 | 大小 | 占比 | 说明 |
|------|------|------|------|
| **模型权重** | 3.17 GB | 1.1% | 1.7B 参数 × 2 字节 (BF16) |
| **KV Cache** | 280 GB | 98% | 🔥 罪魁祸首！ |
| 激活值/缓冲 | 1.5 GB | 0.5% | 临时计算用 |
| CUDA 开销 | 1 GB | 0.3% | 内核/驱动开销 |
| **总计** | **285.67 GB** | 100% | 理论最大值 |

---

## 🔍 KV Cache 为什么这么大？

### KV Cache 计算公式

```python
KV_Cache_Size = (
    num_layers              # 28 层
    × 2                     # K 和 V 两个矩阵
    × hidden_size           # 2048 维
    × max_seq_len           # 40,960 tokens (⚠️ 超长！)
    × max_num_seqs          # 32 并发请求 (⚠️ 很高！)
    × dtype_size            # 2 字节 (BF16)
)

= 28 × 2 × 2048 × 40960 × 32 × 2
= 280 GB
```

### 三个关键因素

1. **超长上下文 (40,960 tokens)** 🔥
   - Qwen3-1.7B 支持最长 40K context
   - 数学推理需要长上下文（题目 + 推理过程）
   - 每个 token 的 KV: 224 KB
   - **40K tokens → 8.75 GB per request**

2. **高并发 (32 requests)** 🔥
   - `max_num_seqs: 32` 允许同时处理 32 个请求
   - 提高吞吐量，但显存成倍增长
   - **32 请求 × 8.75 GB = 280 GB**

3. **深层网络 (28 layers)**
   - 每层都要存储 KV Cache
   - 越深的网络，KV 越大

---

## 💡 为什么 vLLM 能在 40GB 上运行？

### 动态分配 + PagedAttention

vLLM 不会真的分配 280 GB！

**实际机制**:
1. **按需分配**: 只在有请求时才分配 KV 块
2. **PagedAttention**: 像虚拟内存一样分页管理
3. **实际序列长度**: 大多数请求 < 8K tokens，不会用满 40K
4. **并发数动态**: 显存不够时自动降低并发

**`gpu_memory_utilization` 的作用**:
- `0.8` = 预留 63 GB 给 KV Cache 池
- vLLM 在这个池子里按需分配
- 不是一次性分配满

### 实际运行时的显存使用

```
启动时:
  模型加载: 3.17 GB
  预留 KV 池: 60 GB (0.8 × 79 GB)
  实际使用: ~10-30 GB (取决于实际请求)
```

---

## 🛠️ 配置优化建议

### 选项 1: 降低 `gpu_memory_utilization` (已采用)

```yaml
gpu_memory_utilization: 0.5  # 从 0.8 降到 0.5
```

**效果**:
- KV Cache 池: 63 GB → 40 GB
- ✅ 可以与其他任务共存
- ⚠️ 降低最大并发能力

### 选项 2: 降低 `max_num_seqs`

```yaml
max_num_seqs: 16  # 从 32 降到 16
```

**效果**:
- 理论 KV 需求: 280 GB → 140 GB
- vLLM 实际预留会减少
- ⚠️ 吞吐量减半

### 选项 3: 限制 `max_tokens`

```yaml
max_tokens: 4000  # 从 8000 降到 4000
```

**效果**:
- 每个请求最多生成 4K tokens
- 但模型的 `max_seq_len` 仍是 40K
- ⚠️ 对 KV Cache 影响有限

### 选项 4: 使用专用 GPU (推荐)

```yaml
device: 1  # 使用空闲的 GPU
auto_select_device: false
gpu_memory_utilization: 0.8  # 可以恢复高利用率
```

---

## 📊 不同配置的性能对比

| 配置 | KV 池大小 | 最大并发 | 吞吐量 | 适用场景 |
|------|----------|----------|--------|----------|
| `0.8 util, 32 seqs` | 63 GB | ~32 | 高 | GPU 独占 |
| `0.5 util, 32 seqs` | 40 GB | ~20 | 中 | GPU 共享 (当前) |
| `0.5 util, 16 seqs` | 40 GB | ~16 | 中低 | 保守设置 |
| `0.3 util, 16 seqs` | 24 GB | ~10 | 低 | 高度共享 |

---

## 🎯 为什么 AIME 评估需要这么大显存？

AIME 2025 数学题特点:
- **题目长**: 几百 tokens
- **推理长**: CoT 推理可达 2-4K tokens
- **答案长**: 详细步骤 + 最终答案
- **需要批处理**: 30 道题，希望并发处理加速

**实际需求**:
```
平均每题: 2000 tokens (题目) + 3000 tokens (推理) = 5000 tokens
并发处理: 6-8 题
KV Cache: 5000 × 8 × (层数/头数开销) ≈ 15-20 GB
加上模型: 3 GB
总计: ~20-25 GB (实际)
```

所以 `0.5 util` (40 GB 预留) 是足够的！

---

## 🔑 关键结论

1. **模型小 ≠ 推理省显存**
   - 模型: 3 GB
   - KV Cache: 可达 280 GB (理论最大)
   - 实际: 20-40 GB (取决于配置和并发)

2. **长上下文是主要消耗**
   - 40K context → 每个请求 8.75 GB KV
   - 这是为什么小模型也能"吃"很多显存

3. **`gpu_memory_utilization` 是预留，不是实际使用**
   - 类似虚拟内存
   - 按需分配
   - 但启动时要检查够不够

4. **当前配置 (0.5) 对于 AIME 评估是充足的**
   - 实际使用: 20-30 GB
   - 预留: 40 GB
   - 剩余: 10-20 GB 安全边际

---

## 📚 延伸阅读

- [vLLM PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [KV Cache 优化技术](https://huggingface.co/blog/kv-cache-quantization)
- [长上下文模型的显存优化](https://arxiv.org/abs/2305.14314)
