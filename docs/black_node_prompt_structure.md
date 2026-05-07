# Black Node 完整Prompt结构说明

## 概述

Black node收到的prompt由以下部分组成：

1. **[REDACTED]** (固定角色定义)
2. **User Prompt** (任务具体信息，包含变量替换)
3. **Benchmark Info** (benchmark特定的数据准备指南)

---

## 1. [REDACTED]

**文件位置**: `playground/math_posttrain_datatree_v2/prompts/black/system_prompt.md`

**内容摘要**:
- **角色**: 负责将候选数据转换为具体的post-training实验
- **目标**: 通过调整、过滤、清洗、混合、格式化数据来改进benchmark表现
- **工具**:
  - `execute_bash`: 运行shell命令和Python脚本
  - `validate_train_data`: 验证训练数据格式
  - `submit`: 提交训练数据并运行评估
  - memory_tree MCP工具: 浏览和注册数据集
- **成功标准**:
  - 生成有效的 `train.jsonl`
  - 通过 `validate_train_data` 验证
  - 调用 `submit` 并返回评估结果
- **数据清洗策略示例**:
  - 格式标准化
  - 去重
  - 质量过滤
  - 难度重平衡
  - 跨源合并
  - 答案格式对齐

**关键点**: 这部分是**通用的**，对所有benchmark都一样。

---

## 2. User Prompt

**文件位置**: `playground/math_posttrain_datatree_v2/prompts/black/user_prompt.md`

**变量替换** (运行时填充):
- `{task_description}`: 任务描述（从哪里来？见下文）
- `{benchmark_info}`: benchmark特定信息（自动生成）
- `{node_id}`: 当前节点ID
- `{task_workspace}`: 工作目录路径
- `{train_jsonl_path}`: 训练数据输出路径
- `{global_pool_manifest_path}`: 全局数据池清单路径
- `{dataset_manifest_summary}`: 数据集清单摘要
- `{parent_black_handoff_summary}`: 父节点传递的信息
- `{inspect_summary_text}`: 检查报告摘要
- `{memory_summary}`: 记忆摘要
- `{prep_feedback_summary}`: 之前的数据验证反馈

**内容结构**:
1. 任务描述
2. Benchmark信息（包含数据准备指南）
3. 节点信息
4. 可用上下文（数据池、全局清单）
5. 全局数据源摘要
6. 先前信号（父节点、检查、记忆、反馈）
7. 必需输出（train.jsonl格式要求）
8. 可选训练超参数
9. 完成标准
10. 禁止事项

**关键点**: 这部分是**通用框架**，但通过变量替换注入benchmark特定信息。

---

## 3. Benchmark Info (动态生成)

**生成位置**: `core/utils/benchmark_metadata.py` 的 `get_benchmark_info()` 函数

**以 aime_2025 为例，生成的内容**:

```markdown
## Target Benchmark: aime_2025

**Type**: Math Competition

**Description**: AIME (American Invitational Mathematics Examination) 2025 problems. High-difficulty competition math requiring multi-step reasoning.

**Input Format**: Problem statement (text)

**Output Format**: Step-by-step solution; last line must be "ANSWER: <integer 0-999>"

**Example**:
```
Problem: Find the number of positive integers n ≤ 1000 such that n² + n + 1 is divisible by 7.
Answer: ...
ANSWER: 143
```

**Answer Extraction from Source Data:**

Source datasets often contain answers in various formats. When preparing training data:

1. **Common answer formats in source data:**
   - `\boxed{143}` or `\boxed{n=143}` → extract 143
   - `n = 143` or `The answer is 143` → extract 143
   - `\frac{13}{6}` or other non-integers → SKIP (not valid AIME answer)
   - `\textbf{(A)} 26` → extract 26 (ignore multiple choice label)
   - Plain integers: `143`, `0`, `999` → use directly

2. **Filtering rules:**
   - Only keep examples where the final answer is an integer in range [0, 999]
   - Skip examples with fractional, negative, or out-of-range answers
   - Skip examples where answer cannot be reliably extracted

3. **Output format standardization:**
   - Preserve the step-by-step reasoning from source data
   - Replace the final answer line with: `ANSWER: <integer>`
   - Remove `\boxed{}` wrappers, LaTeX formatting from the answer line
   - Ensure the last non-empty line is exactly `ANSWER: <integer>`

4. **Quality checks:**
   - Verify each output ends with `ANSWER: <integer>` where integer is 0-999
   - Check that reasoning steps are present (not just answer-only)
   - Remove duplicates across sources
```

**关键点**: 这部分是**benchmark特定的**，每个benchmark有不同的数据准备指南。

---

## 4. Task Description (你可以控制的部分！)

**来源**: 在代码中调用Black node时传入

**示例** (从日志中提取):
```
Search public task-aligned data that improves weak benchmark domains without changing training code.
```

**在哪里设置**:
- 文件: `playground/math_posttrain_datatree_v2/core/playground.py`
- 或者: 通过配置文件/命令行参数传入

让我找一下具体位置...
