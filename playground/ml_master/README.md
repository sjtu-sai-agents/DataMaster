# ML-Master 智能体运行逻辑

## 概述

ML-Master 是一个基于 **蒙特卡洛树搜索 (MCTS)** 的自动化机器学习智能体，用于解决 Kaggle 风格的机器学习竞赛任务。系统通过生成代码、执行、评估和迭代改进的方式，构建一棵解决方案树，最终找到最优解。

## 目录结构

```
ml_master/
├── core/
│   ├── mcts_agent.py      # MCTS 智能体核心逻辑
│   ├── mcts_node.py       # MCTS 节点定义
│   ├── node.py            # 基础节点定义
│   ├── journal.py         # 解决方案树管理
│   ├── metric.py          # 评估指标定义
│   └── playground.py      # Playground 环境（EvoMaster 集成）
├── utils/
│   ├── llm_query.py       # LLM 查询工具
│   ├── response.py        # 响应解析工具
│   ├── data_preview.py    # 数据预览生成
│   └── mcts_utils.py      # MCTS 工具函数（探索衰减）
└── prompts/
    ├── draft.py           # Draft 阶段提示词
    ├── improve.py         # Improve 阶段提示词
    ├── debug.py           # Debug 阶段提示词
    └── review.py          # Review（评估）提示词
```

---

## 核心数据结构

### 1. Node (`core/node.py`)

基础节点类，存储解决方案的代码和执行结果。

**主要属性：**
- `code`: 代码字符串
- `plan`: 自然语言设计方案
- `parent`: 父节点
- `children`: 子节点集合
- `metric`: 评估指标值（`MetricValue` 类型）
- `is_buggy`: 是否有 bug
- `exec_time`: 执行时间
- `exc_type`: 异常类型

**主要属性方法：**
- `stage_name`: 返回节点阶段 ("draft", "improve", "debug")
- `debug_depth`: 当前调试路径的长度
- `absorb_exec_result()`: 吸收执行结果

### 2. MCTSNode (`core/mcts_node.py`)

扩展 `Node`，添加 MCTS 特有的属性。

**额外属性：**
- `visits`: 访问次数
- `total_reward`: 累计奖励
- `stage`: 节点阶段 ("root", "draft", "improve", "debug")
- `local_best_node`: 子树中的最佳节点
- `expected_child_count`: 预期子节点数量（用于并行搜索）

**主要方法：**
| 方法 | 功能 |
|------|------|
| `uct_value(exploration_constant)` | 计算 UCT 值（用于节点选择） |
| `is_fully_expanded(scfg)` | 检查节点是否完全扩展 |
| `has_no_bug_child()` | 检查是否有无 bug 的子节点 |
| `update(result, add)` | 更新节点统计信息 |
| `fetch_child_memory()` | 获取子节点的记忆（用于 prompt） |
| `fetch_parent_memory()` | 获取父节点的记忆（用于 prompt） |

### 3. Journal (`core/journal.py`)

存储整个解决方案树。

**主要属性：**
- `nodes`: 所有节点列表
- `draft_nodes`: 所有 draft 节点列表

**主要方法：**
| 方法 | 功能 |
|------|------|
| `append(node)` | 添加新节点 |
| `get_best_node()` | 获取最佳节点 |
| `get_metric_history()` | 获取所有指标历史 |
| `generate_summary()` | 生成树摘要 |

### 4. MetricValue (`core/metric.py`)

带比较语义的指标值。

**属性：**
- `value`: 数值
- `maximize`: 是否越大越好

---

## 核心类：MLMasterAgent (`core/mcts_agent.py`)

### 初始化 (`__init__`)

```python
MLMasterAgent.__init__(
    task_desc: str,      # 任务描述
    cfg: AgentConfig,    # 配置
    journal: Journal,    # 日志/树
    llm,                 # 代码生成 LLM
    feedback_llm,        # 反馈评估 LLM
    workspace_dir: str | Path,
)
```

### 全局状态维护

智能体维护以下全局状态：

| 状态 | 类型 | 说明 |
|------|------|------|
| `task_desc` | str | 任务描述 |
| `current_step` | int | 当前步数 |
| `current_node` | MCTSNode | 当前节点 |
| `best_metric` | float | 最佳指标值 |
| `best_node` | MCTSNode | 最佳节点 |
| `virtual_root` | MCTSNode | 虚拟根节点（stage="root"） |
| `all_root` | list | 所有 draft 节点 |
| `_locked_drafts` | set | 已锁定的 draft 节点 ID |
| `data_preview` | str | 数据预览缓存 |
| `search_start_time` | float | 搜索开始时间 |

### 核心方法详解

#### 1. `step(exec_callback)` - 执行一步 MCTS

**功能：** 执行单步 MCTS（选择 → 扩展 → 模拟 → 反向传播）

**流程：**
```
1. 选择阶段: select() → 选择要扩展的节点
2. 扩展阶段: _step_search() → 生成新节点
   - 若 parent = virtual_root: 调用 _draft()
   - 若 parent.is_buggy: 调用 _debug()
   - 否则: 调用 _improve()
3. 执行代码: exec_callback(code, node_id)
4. 评估结果: parse_exec_result()
5. 检查改进: check_improvement() → 决定是否回传
6. 更新最佳: 若找到更好的解，保存
7. 返回节点继续
```

#### 2. `select(node)` - 选择阶段

**功能：** 使用 UCT 算法选择要扩展的节点

**逻辑：**
```python
def select(node):
    current = node
    while current and not current.is_terminal:
        if not current.is_fully_expanded_with_expected(scfg):
            # 节点未完全扩展，返回当前节点进行扩展
            if current.is_buggy and current.is_debug_success:
                current = _uct_select(current)  # 调试成功，继续选择
            elif current.continue_improve and len(current.children) > 0:
                current = _uct_select(current)  # 继续改进
            else:
                return current  # 扩展此节点
        else:
            current = _uct_select(current)  # 选择 UCT 最高的子节点
    return current
```

#### 3. `_uct_select(node)` - UCT 选择

**功能：** 返回 UCT 值最高的子节点

**UCT 公式：**
```
UCT = Q + C * sqrt(ln(N) / n)
```
- `Q` = 平均奖励 (`total_reward / visits`)
- `C` = 探索常数
- `N` = 父节点访问次数
- `n` = 当前节点访问次数

#### 4. `_step_search(parent_node, exec_callback)` - 扩展与模拟

**功能：** 生成新节点、执行代码、评估结果

**返回：** `(should_return_to_root, result_node)`

**流程：**
```
1. 根据 parent_node 类型生成新节点:
   - draft: _draft()
   - buggy: _debug()
   - normal: _improve()

2. 执行代码: exec_callback(code, node_id)

3. 解析结果: parse_exec_result()

4. 检查改进: check_improvement()

5. 添加到 journal
```

#### 5. `_draft()` - 生成初始方案

**功能：** 生成新的 draft 节点（初始解决方案）

**流程：**
```
1. 获取实现指南和环境信息
2. 构建 draft prompt（包含任务描述、记忆、数据预览）
3. 调用 LLM 生成 plan + code
4. 创建新 MCTSNode（parent=virtual_root, stage="draft"）
5. 增加 virtual_root 的 expected_child_count
```

#### 6. `_improve(parent_node)` - 改进方案

**功能：** 基于父节点生成改进版本

**流程：**
```
1. 获取实现指南
2. 构建 improve prompt（包含任务、记忆、前代代码和输出）
3. 调用 LLM 生成改进的 plan + code
4. 创建新 MCTSNode（parent=parent_node, stage="improve"）
5. 继承 local_best_node
6. 增加 parent_node 的 expected_child_count
```

#### 7. `_debug(parent_node)` - 调试修复

**功能：** 修复有 bug 的节点

**流程：**
```
1. 获取实现指南
2. 构建 debug prompt（包含 bug 代码和执行输出）
3. 调用 LLM 生成修复的 plan + code
4. 创建新 MCTSNode（parent=parent_node, stage="debug"）
5. 继承 local_best_node
6. 增加 parent_node 的 expected_child_count
```

#### 8. `parse_exec_result(node, exec_result)` - 解析执行结果

**功能：** 评估代码执行结果，更新节点状态

**流程：**
```
1. 吸收执行结果到节点
2. 构建 review prompt（代码 + 输出）
3. 调用 feedback LLM 获取评估
4. 检查是否生成 submission.csv
5. 判断是否 buggy:
   - response["is_bug"] = True
   - 或 node.exc_type != None
   - 或 response["metric"] = None
   - 或没有 submission.csv
6. 更新 node.metric 和 node.analysis
```

#### 9. `check_improvement(cur_node, parent_node)` - 检查改进

**功能：** 判断改进是否足够，决定是否反向传播

**逻辑：**
```python
improvement = new_metric - local_best_metric

if improvement < threshold:
    if improve_failure_depth < max_improve_failure:
        # 改进不足，继续尝试改进
        improve_failure_depth += 1
        continue_improve = True
        return False
    else:
        # 达到最大尝试次数，停止改进
        continue_improve = False
        is_terminal = True
        should_backpropagate = True
else:
    # 改进足够，更新 local_best_node
    local_best_node = cur_node
    continue_improve = True

# 对于 debug 节点
if cur_node.debug_depth >= back_debug_depth:
    should_backpropagate = True
```

#### 10. `backpropagate(node, value)` - 反向传播

**功能：** 将奖励反向传播到根节点

**流程：**
```
while node is not None:
    # 更新调试成功状态
    if node.is_buggy == False and node.parent.is_buggy == True:
        node.parent.is_debug_success = True

    # 传播 continue_improve 标志
    if node.parent.stage != "root":
        node.parent.continue_improve = node.continue_improve

    # 解锁 draft 节点
    if node.stage == "draft" and node.lock:
        node.lock = False

    # 重置改进失败深度
    if node.improve_failure_depth > 0:
        node.improve_failure_depth = 0

    # 更新访问次数和奖励
    node.update(value)

    node = node.parent
```

#### 11. `_get_node_reward(node)` - 计算节点奖励

**功能：** 计算节点的奖励值

```python
reward = 0
if node.is_buggy or node.metric is None:
    return -1

# 相比全局最优有改进
if improvement > 0:
    reward += 1

# 基础奖励
reward += 1
```

---

## Pipeline 运行流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ML-Master Pipeline                          │
└─────────────────────────────────────────────────────────────────────┘

                                ┌──────────────┐
                                │   初始化      │
                                │ setup()      │
                                └──────┬───────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        主循环：MCTS 搜索                              │
│                  _run_mcts_search(exec_callback)                     │
└──────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────┐
                    │     并行执行 step 任务                │
                    │  (ThreadPoolExecutor,               │
                    │   max_workers=parallel_search_num)  │
                    └─────────────────────────────────────┘
                                       │
                        ┌──────────────┴───────────────┐
                        ▼                              ▼
              ┌──────────────────┐          ┌──────────────────┐
              │   step 1         │          │   step 2         │
              │   (线程 1)        │          │   (线程 2)        │
              └──────────────────┘          └──────────────────┘
                        │                              │
                        └──────────────┬───────────────┘
                                       ▼
                    ┌─────────────────────────────────────┐
                    │         单步：step()                 │
                    └─────────────────────────────────────┘
                                       │
                        ┌──────────────┴───────────────┐
                        ▼                              ▼
              ┌──────────────────┐          ┌──────────────────┐
              │    1. select()   │          │    2. _step_search│
              │   选择扩展节点     │          │    扩展与模拟     │
              └──────────────────┘          └────────┬─────────┘
                        │                              │
                        │                    ┌─────────┴─────────┐
                        │                    ▼                   ▼
                        │          ┌─────────────┐     ┌─────────────┐
                        │          │  _draft()   │     │ _improve()  │
                        │          │ _improve()  │     │  _debug()   │
                        │          │  _debug()   │     │             │
                        │          └──────┬──────┘     └─────────────┘
                        │                 │
                        │                 ▼
                        │    ┌────────────────────────┐
                        │    │  exec_callback(code)   │
                        │    │  执行代码生成 submission│
                        │    └───────────┬────────────┘
                        │                │
                        │                ▼
                        │    ┌────────────────────────┐
                        │    │ parse_exec_result()    │
                        │    │ LLM 评估执行结果        │
                        │    └───────────┬────────────┘
                        │                │
                        │                ▼
                        │    ┌────────────────────────┐
                        │    │ check_improvement()    │
                        │    │ 检查改进是否足够        │
                        │    └───────────┬────────────┘
                        │                │
                        └────────────────┼────────────────┐
                                         │                │
                        ┌────────────────┴────────────────┴────────┐
                        │                                          │
                        ▼                                          ▼
           ┌─────────────────────┐                   ┌─────────────────────┐
           │  改进足够/调试成功     │                   │  改进不足/继续调试     │
           │  should_backpropagate│                   │  continue_improve   │
           └──────────┬──────────┘                   └──────────┬──────────┘
                      │                                           │
                      ▼                                           ▼
           ┌─────────────────────┐                   ┌─────────────────────┐
           │   backpropagate()   │                   │  返回当前节点继续     │
           │   反向传播奖励        │                   │  改进/调试          │
           └──────────┬──────────┘                   └─────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ 更新 best_node      │
           │ 保存 best_solution  │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ current_step += 1   │
           │ 检查是否达到步数限制  │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │    搜索完成/返回     │
           └─────────────────────┘
```

---

## 全局状态维护详解

### 1. 虚拟根节点 (Virtual Root)

```python
self.virtual_root = MCTSNode(
    parent=None,
    plan="virtual root",
    code="# virtual root",
    metric=get_worst_metric(True),
    stage="root"
)
```

**作用：**
- 作为所有 draft 节点的共同父节点
- 统一管理多个初始方案分支
- 提供全局搜索起点

### 2. 节点锁机制 (Node Lock)

```python
# draft 节点锁定
selected_node.lock = True

# 解锁在 backpropagate 中进行
if node.stage == "draft" and node.lock:
    node.lock = False
```

**作用：** 防止并行搜索时多个线程同时扩展同一个 draft 节点

### 3. 预期子节点计数 (Expected Child Count)

```python
# 原子操作
with self._child_count_lock:
    self.expected_child_count += 1

# 用于并行搜索的扩展判断
def is_fully_expanded_with_expected(self, scfg):
    return self.expected_child_count >= scfg.num_drafts
```

**作用：** 在并行搜索中协调多个线程的扩展决策

### 4. 最佳节点追踪

```python
# 检查并更新最佳节点
if result_node.metric.value is not None:
    if self.best_node is None or self.best_node.metric < result_node.metric:
        self.best_node = result_node
        self.best_metric = result_node.metric.value
        self._save_best_solution(result_node)
```

### 5. 改进状态维护

```python
# continue_improve 标志
cur_node.continue_improve = True/False

# 在反向传播中向上传递
if node.parent.stage != "root":
    node.parent.continue_improve = node.continue_improve
```

**作用：** 控制是否继续在当前分支进行改进

### 6. 局部最佳节点 (Local Best Node)

```python
# 初始化
new_node = MCTSNode(
    ...
    local_best_node=parent_node.local_best_node
)

# 更新
if improvement > threshold:
    cur_node.local_best_node = cur_node
```

**作用：** 在子树中维护局部最优，用于改进决策

---

## 配置参数 (AgentConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `steps` | 500 | 最大搜索步数 |
| `time_limit` | 43200 | 时间限制（秒） |
| `num_drafts` | 5 | 每个 draft 节点的最大子节点数 |
| `num_bugs` | 3 | 每个 bug 节点的最大调试尝试 |
| `num_improves` | 3 | 每个 normal 节点的最大改进尝试 |
| `parallel_search_num` | 3 | 并行搜索线程数 |
| `exploration_constant` | 1.414 | UCT 探索常数 C |
| `metric_improvement_threshold` | 0.001 | 改进阈值 |
| `max_improve_failure` | 2 | 最大改进失败次数 |
| `max_debug_depth` | 3 | 最大调试深度 |
| `back_debug_depth` | 1 | 回退调试深度 |

---

## 探索常数衰减 (Decay)

支持多种衰减策略：

```python
# 线性衰减
C = max(initial_C - alpha * t, lower_bound)

# 指数衰减
C = max(initial_C * (gamma ** t), lower_bound)

# 分段衰减
if t < T1: C = initial_C
elif T1 <= t <= T2: C = max(initial_C - alpha * (t - T1), lower_bound)
else: C = lower_bound

# 动态分段衰减（基于时间估计）
progress = n_nodes / N_est
if progress < phase1_end: C = initial_C
elif progress < phase2_end: C = decayed...
else: C = lower_bound
```

---

## 工具函数

### `utils/llm_query.py`

| 函数 | 功能 |
|------|------|
| `plan_and_code_query()` | 生成 plan + code（用于 draft/improve/debug） |
| `query_with_feedback()` | 获取 LLM 反馈评估（用于 review） |
| `code_query()` | 单独生成代码 |

### `utils/response.py`

| 函数 | 功能 |
|------|------|
| `extract_code()` | 从 markdown 代码块提取代码 |
| `extract_text_up_to_code()` | 提取代码块前的文本（plan） |
| `wrap_code()` | 将代码包装为 markdown 格式 |
| `extract_review()` | 解析 JSON 格式的评估结果 |

### `utils/data_preview.py`

| 函数 | 功能 |
|------|------|
| `generate(input_dir)` | 生成数据预览字符串 |
| `generate_for_task(workspace)` | 为 Kaggle 任务生成数据预览 |

---

## Playground 集成 (`core/playground.py`)

`MLMasterPlayground` 类将 ML-Master 集成到 EvoMaster 框架。

**主要方法：**

| 方法 | 功能 |
|------|------|
| `setup()` | 初始化环境和智能体 |
| `_create_agent_config()` | 创建配置 |
| `_setup_agent()` | 设置智能体 |
| `_create_exec_callback()` | 创建代码执行回调 |
| `run(task_desc)` | 运行智能体 |
| `_run_mcts_search()` | 运行 MCTS 搜索循环 |

---

## 高级功能

### Steerable Reasoning（可引导推理）

**什么是 Steerable Reasoning？**

Steerable Reasoning 是一种让 LLM 在生成代码前先进行"思考"的技术，常见于 OpenAI 的 O1 系列模型。模型会在 `` 和 `` 标签之间输出推理过程，然后输出最终答案。

**工作原理：**

```
[模型输出示例]


# 我的解决方案

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
...
```
```

**系统支持：**

当 `steerable_reasoning=True` 时，系统会：

1. 调用 LLM 时启用推理模式
2. 接收完整响应（包括推理过程）
3. 自动提取 `` 之后的内容作为实际输出
4. 在 verbose 日志中记录完整推理过程

**配置：**

```python
# 在 AgentConfig 中启用
cfg = AgentConfig(
    steerable_reasoning=True,  # 启用推理模式
    code_model="gpt-5",        # 或其他支持推理的模型
)
```

**支持的模型：**
- OpenAI GPT-5 / O1 系列
- DeepSeek R1 系列
- 任何支持 `separate_reasoning` 参数的开源模型

**实现细节：**

```python
# utils/llm_query.py
def extract_after_think(text: str) -> str:
    """提取 `` 标签后的内容"""
    if "" in text:
        parts = text.split("", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return text

# plan_and_code_query() 函数会自动调用
plan, code = plan_and_code_query(
    llm, messages,
    steerable_reasoning=True  # 启用推理提取
)
```

### Check Format（格式验证）

**功能：** 自动验证生成的 `submission.csv` 格式是否符合要求。

**实现：** 调用外部格式验证服务器，检查提交文件的列名、数据类型、行数等。

**配置：**

```python
cfg = AgentConfig(
    check_format=True,  # 启用格式验证
)
```

### Save All Submission（保存所有提交）

**功能：** 所有提交文件自动保存在 `submission/` 目录下，文件名格式为 `submission_{node_id}.csv`。

**目录结构：**

```
workspace/
├── submission/            # 所有提交文件
│   ├── submission_{node_id1}.csv
│   ├── submission_{node_id2}.csv
│   └── ...
└── best_submission/       # 最佳提交
```

**说明：**
- 每个节点执行后，提交文件会自动保存为 `submission_{node_id}.csv`
- 所有历史提交都保留在 `submission/` 目录中
- 最佳提交会额外复制到 `best_submission/submission.csv`

### Data Preprocessing（数据预处理）

**功能：** 自动解压和处理原始数据文件。

**支持的格式：**
- `.tar.gz`
- `.zip`
- 其他压缩格式

**配置：**

```python
cfg = AgentConfig(
    preprocess_data=True,  # 自动预处理数据
)
```

### 并行代码执行

**功能：** 使用多进程并行执行代码，提高搜索效率。

**特性：**
- 多个 Python 进程同时运行
- CPU 亲和性设置（每个进程绑定特定 CPU 核心）
- 进程池管理和复用

**配置：**

```python
cfg = AgentConfig(
    parallel_search_num=3,  # 并行进程数
)
```

---

## 与原始 ML-Master 的差异对比

| 功能 | 原始 ML-Master | playground 版本 |
|------|---------------|----------------|
| **核心 MCTS 算法** | ✅ | ✅ |
| **Draft/Improve/Debug 阶段** | ✅ | ✅ |
| **Steerable Reasoning** | ✅ | ✅ |
| **Check Format** | ✅ | ✅ |
| **Save All Submission** | ✅ | ✅ |
| **数据预处理** | ✅ | ✅ |
| **并行代码执行** | ✅ (多进程) | ⚠️ (使用框架执行) |
| **Rich 日志** | ✅ | ⚠️ (使用框架日志) |
| **GPT-5/o1 API** | ✅ | ✅ |
| **EvoMaster 集成** | ❌ | ✅ |

**注：** playground 版本专为集成到 EvoMaster 框架而设计，在保持核心功能完整性的同时，利用框架提供的功能（如代码执行、日志管理）来简化实现。

---

## 新增功能详解

### 1. Steerable Reasoning 实现

**文件：** `utils/llm_query.py`

新增 `extract_after_think()` 函数，用于从支持推理模式的模型响应中提取实际内容：

```python
def extract_after_think(text: str) -> str:
    """提取 `` 标签后的内容"""
    if "" in text:
        parts = text.split("", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return text
```

在 `plan_and_code_query()` 中使用：

```python
def plan_and_code_query(
    llm, prompt, temperature=0.7, max_tokens=8192,
    steerable_reasoning: bool = False,  # 新增参数
    **kwargs
):
    # ... LLM 调用 ...

    # 提取推理后的内容
    if steerable_reasoning:
        completion_text = extract_after_think(completion_text)
```

### 2. Check Format 实现

**文件：** `utils/server_utils.py`

- `is_server_online()`: 检查验证服务器是否在线
- `call_validate()`: 调用服务器验证提交文件格式
- `validate_submission_format()`: 封装的验证接口

**集成位置：** `core/playground.py` 的 `exec_callback()` 中

```python
# 格式验证 if enabled
if self.agent and self.agent.cfg.check_format and submission_path.exists():
    is_valid, message = validate_submission_format(
        node_id=node_id,
        submission_path=submission_path,
        check_format=True
    )
    if not is_valid:
        term_out += f"\n\n[Format Validation Error]: {message}"
```

### 3. Save All Submission 实现

**说明：** 所有提交文件自动保存在 `submission/` 目录下。

每个节点执行后，提交文件会以 `submission_{node_id}.csv` 的格式自动保存。无需额外配置，所有历史提交都会保留。

**目录结构：**
```
workspace/
├── submission/           # 所有提交文件
│   ├── submission_{node_id1}.csv
│   ├── submission_{node_id2}.csv
│   └── ...
└── best_submission/      # 最佳提交
```

### 4. 数据预处理实现

**文件：** `utils/preproc_data.py`

提供以下函数：

| 函数 | 功能 |
|------|------|
| `extract_tar_file()` | 解压 tar/tar.gz/tgz 文件 |
| `extract_zip_file()` | 解压 zip 文件 |
| `preprocess_data()` | 批量预处理数据文件 |
| `create_directory_structure()` | 创建标准目录结构 |
| `verify_directory_structure()` | 验证目录结构 |

**集成位置：** `core/playground.py` 的 `setup()` 方法中

```python
# 创建目录结构
workspace_path = self.session.config.workspace_path
create_directory_structure(workspace_path)

# 数据预处理 if enabled
if self.agent and self.agent.cfg.preprocess_data:
    input_dir = workspace_path / "input"
    if input_dir.exists():
        stats = preprocess_data(input_dir, recursive=True)
        logger.info(f"Preprocessed {stats['extracted']} files")
```

---

## 使用示例

### 启用 Steerable Reasoning

```python
cfg = AgentConfig(
    steerable_reasoning=True,
    code_model="gpt-5",  # 或其他支持推理的模型
)
```

### 启用格式验证

```bash
# 设置验证服务器地址
export ML_MASTER_VALIDATE_SERVER="http://localhost:5001"

# 在配置中启用
cfg = AgentConfig(check_format=True)
```

**注意：** 保存所有提交文件是默认行为，所有提交会自动保存为 `submission_{node_id}.csv`，无需额外配置。

### 启用数据预处理

```python
cfg = AgentConfig(preprocess_data=True)
```

---

## 配置参数完整列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `steps` | int | 500 | 最大搜索步数 |
| `time_limit` | int | 43200 | 时间限制（秒） |
| `code_model` | str | "gpt-4" | 代码生成模型 |
| `code_temp` | float | 0.5 | 代码生成温度 |
| `feedback_model` | str | "gpt-4o" | 反馈评估模型 |
| `feedback_temp` | float | 0.0 | 反馈评估温度 |
| `num_drafts` | int | 5 | Draft 节点最大子节点数 |
| `num_bugs` | int | 3 | Bug 节点最大调试尝试 |
| `num_improves` | int | 3 | Normal 节点最大改进尝试 |
| `parallel_search_num` | int | 3 | 并行搜索线程数 |
| `exploration_constant` | float | 1.414 | UCT 探索常数 C |
| `metric_improvement_threshold` | float | 0.001 | 改进阈值 |
| `max_improve_failure` | int | 2 | 最大改进失败次数 |
| `max_debug_depth` | int | 3 | 最大调试深度 |
| `back_debug_depth` | int | 1 | 回退调试深度 |
| `obfuscate` | bool | False | 是否混淆任务描述 |
| **`steerable_reasoning`** | **bool** | **False** | **启用推理模式** |
| **`check_format`** | **bool** | **False** | **启用格式验证** |
| `save_all_submission` | bool | False | ~~已弃用~~（默认保存所有提交） |
| **`preprocess_data`** | **bool** | **False** | **数据预处理** |
| `convert_system_to_user` | bool | False | 转换系统消息为用户消息 |
| `expose_prediction` | bool | False | 暴露预测函数 |
| `k_fold_validation` | int | 1 | K 折验证数 |
