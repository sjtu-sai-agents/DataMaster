# 训练日志改进说明

## 问题
之前训练开始后，主日志只显示：
```
starting training base_model=/data/public_model/Qwen3-1.7B output_dir=...
```

之后就没有输出了，看起来像"卡住"，实际上训练已经在后台运行。

## 解决方案

### 1. 增强日志输出

修改了 `core/utils/llama_factory.py` 的 `run_llama_factory_sft()` 函数，添加了详细的训练状态日志：

**训练开始时的日志**：
```
================================================================================
🚀 TRAINING STARTED
================================================================================
Command: llamafactory-cli train /path/to/recipe.json
Model: /data/public_model/Qwen3-1.7B
Dataset: /path/to/alpaca_train.jsonl
Output: /path/to/checkpoints/xxx
Recipe: /path/to/base_recipe.json
Log file: /path/to/train.log
================================================================================
Training is running in background. Monitor progress with:
  tail -f /path/to/checkpoints/xxx/trainer_log.jsonl
  watch -n 1 nvidia-smi
================================================================================
```

**训练完成时的日志**：
```
================================================================================
✅ TRAINING COMPLETED
================================================================================
Return code: 0
Log written to: /path/to/train.log
================================================================================
```

或失败时：
```
================================================================================
❌ TRAINING FAILED
================================================================================
Return code: 1
Log written to: /path/to/train.log
================================================================================
```

### 2. 实时监控脚本

创建了 `monitor_training.sh` 脚本，提供友好的实时监控界面：

#### 使用方法

```bash
# 监控当前训练
./playground/math_posttrain_datatree/monitor_training.sh \
  /path/to/checkpoints/d28cad738e2648368ac53046832b001e
```

#### 监控界面显示

```
========================================================================
🚀 Training Progress - 2026-04-08 06:10:23
========================================================================
📊 Progress: [████████████████████░░░░░░░░░░] 42.4%

   Step:       100 /  236
   Epoch:     0.4237
   Loss:      1.2345
   LR:        7.5e-05

   ⏱️  Elapsed:   0:08:12
   ⏳ Remaining: 0:11:08

========================================================================
💻 GPU Status
========================================================================
GPU 0: 100% util | 74851 / 81920 MB
GPU 1: 100% util | 74195 / 81920 MB
GPU 2: 100% util | 80114 / 81920 MB
GPU 3: 100% util | 72237 / 81920 MB
GPU 4:  91% util | 76363 / 81920 MB
GPU 5: 100% util | 52633 / 81920 MB
GPU 6: 100% util | 64132 / 81920 MB
GPU 7: 100% util | 56254 / 81920 MB

========================================================================
Last 3 steps:
========================================================================
  Step  98: loss=1.2401 lr=7.52e-05 0:11:23
  Step  99: loss=1.2378 lr=7.51e-05 0:11:20
  Step 100: loss=1.2345 lr=7.50e-05 0:11:08

Press Ctrl+C to exit
```

自动每3秒刷新一次。

### 3. 手动监控方法

如果不想用脚本，也可以手动查看：

#### 查看实时训练进度
```bash
# 方法1: 实时跟踪 jsonl 日志
tail -f /path/to/checkpoints/xxx/trainer_log.jsonl

# 方法2: 每5秒显示最新状态
watch -n 5 "tail -1 /path/to/checkpoints/xxx/trainer_log.jsonl | python3 -m json.tool"

# 方法3: 只看关键信息
tail -f /path/to/checkpoints/xxx/trainer_log.jsonl | \
  python3 -c "import sys, json; [print(f\"Step {json.loads(l)['current_steps']}/{json.loads(l)['total_steps']} Loss={json.loads(l)['loss']:.4f} Remaining={json.loads(l)['remaining_time']}\") for l in sys.stdin]"
```

#### 查看 GPU 使用率
```bash
watch -n 1 nvidia-smi
```

#### 查看训练进程
```bash
ps aux | grep llamafactory
```

### 4. 日志文件位置

每次训练会生成以下日志文件：

```
checkpoints/d28cad738e2648368ac53046832b001e/
├── trainer_log.jsonl      # 训练进度 (实时更新)
├── train.log              # 完整训练输出 (训练结束后写入)
├── base_recipe.json       # 训练配置
└── train_result.json      # 训练结果摘要
```

## 未来改进建议

1. **实时流式输出**: 考虑将 `subprocess.run()` 改为 `subprocess.Popen()` 实时输出训练日志
2. **进度条**: 在主进程中显示训练进度条
3. **Webhook通知**: 训练完成时发送通知（邮件/Slack/微信）
4. **TensorBoard**: 集成 TensorBoard 可视化

## 测试

```bash
# 测试监控脚本
./playground/math_posttrain_datatree/monitor_training.sh \
  runs/math_posttrain_datatree_20260408_055528/workspaces/task_0/artifacts/checkpoints/d28cad738e2648368ac53046832b001e
```

应该能看到实时更新的训练进度界面。
