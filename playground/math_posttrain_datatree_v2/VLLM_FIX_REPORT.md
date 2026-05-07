# vLLM 评估失败问题修复报告

## 🔍 问题诊断

### 错误现象
```
ValueError: Free memory on device (46.07/79.15 GiB) on startup is less than 
desired GPU memory utilization (0.8, 63.32 GiB).
```

### 根本原因

**GPU 显存不足**:
- 可用显存: 46.07 GB / 79.15 GB (58%)
- vLLM 期望: 63.32 GB (80% × 79.15 GB)
- **缺口: 17.25 GB**

**占用显存的进程**:
```
PID 3591257: OpenVLA 训练任务
- 占用所有 8 张 GPU
- 每张 GPU ~33 GB
- 运行时间: 15+ 小时（从昨天开始）
```

## ✅ 解决方案

### 修改配置文件

**文件**: `configs/math_posttrain_datatree/config.yaml`

```yaml
# 修改前
gpu_memory_utilization: 0.8  # 需要 63.32 GB

# 修改后
gpu_memory_utilization: 0.5  # 只需要 39.58 GB ✅
```

### 验证修复

手动测试 vLLM 启动：
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  <merged_model_path> \
  --gpu-memory-utilization 0.5 \
  --port 32936
```

结果：
```
✅ INFO: Started server process [4038122]
✅ INFO: Application startup complete.
✅ Listening on 0.0.0.0:32936
```

**vLLM 成功启动！**

## 📊 内存使用对比

| 配置 | 需求显存 | 可用显存 | 状态 |
|------|---------|----------|------|
| `0.8` (原配置) | 63.32 GB | 46.07 GB | ❌ 失败 (缺17GB) |
| `0.5` (新配置) | 39.58 GB | 46.07 GB | ✅ 成功 (余6.5GB) |

## 🔧 其他修复建议

### 方案 1: 使用空闲 GPU（推荐）

如果其他 GPU 完全空闲，可以指定使用：

```yaml
evaluation:
  posttrainbench:
    device: 1  # 或其他空闲GPU编号
    auto_select_device: false  # 禁用自动选择
    gpu_memory_utilization: 0.8  # 可以恢复到0.8
```

### 方案 2: 降低其他参数

如果评估太慢，可以同时调整：

```yaml
evaluation:
  posttrainbench:
    max_num_seqs: 16  # 从32降到16（减少并行请求）
    max_connections: 3  # 从6降到3
    gpu_memory_utilization: 0.6  # 中间值
```

### 方案 3: 错峰评估

在其他训练任务完成后再评估：
```bash
# 等待 OpenVLA 训练完成
# 或者暂停/停止其他训练任务
```

## 📝 关于 `gpu_memory_utilization` 参数

| 值 | 适用场景 | 优点 | 缺点 |
|----|---------|------|------|
| `0.9` | GPU 独占使用 | 最大吞吐 | 易OOM |
| `0.8` | GPU 主要用途 | 高吞吐，留余量 | 多任务易冲突 |
| `0.6` | GPU 中度共享 | 平衡性能/共享 | 吞吐下降 |
| `0.5` | GPU 轻度使用 | 兼容性好 | 性能打折扣 |
| `0.3` | GPU 多任务 | 高兼容 | 性能大幅下降 |

**当前设置 0.5 的影响**:
- ✅ 可以与其他训练任务共存
- ✅ 评估稳定性高
- ⚠️ 推理吞吐量降低约 30-40%
- ⚠️ AIME 2025 评估时间可能从 5 分钟增加到 7-8 分钟

## 🚀 下次运行建议

1. **提前检查 GPU 使用情况**:
   ```bash
   nvidia-smi
   # 或者
   watch -n 1 nvidia-smi
   ```

2. **添加预检查脚本**:
   ```python
   # 在评估前检查可用显存
   import torch
   free_memory = torch.cuda.mem_get_info()[0] / 1e9
   required_memory = 39.58  # GB for 0.5 utilization
   
   if free_memory < required_memory:
       logger.warning(f"GPU memory may be insufficient: {free_memory:.1f} GB available, {required_memory:.1f} GB required")
   ```

3. **考虑使用队列系统**:
   - 训练任务完成后自动触发评估
   - 避免显存冲突

## 📄 相关文件

- **配置文件**: `configs/math_posttrain_datatree/config.yaml` (已修改)
- **评估代码**: `playground/math_posttrain_datatree/core/utils/eval.py`
- **测试日志**: `/tmp/vllm_test_0.5.log`

## ✅ 验证清单

- [x] 诊断出根本原因（GPU 显存不足）
- [x] 修改配置文件（0.8 → 0.5）
- [x] 手动验证 vLLM 可以启动
- [x] 服务器正常监听端口
- [ ] 下次实验运行时验证评估成功

## 🎯 预期效果

下次运行 math_posttrain_datatree 实验时：
1. 训练阶段：正常完成 ✅
2. 评估阶段：vLLM 正常启动 ✅（之前失败）
3. AIME 2025 评估：成功完成（略慢但稳定）
4. 最终结果：得到准确率指标

---

**状态**: ✅ 已修复
**修改时间**: 2026-04-08 07:00
**影响范围**: 仅评估阶段，训练不受影响
