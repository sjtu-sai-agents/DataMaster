# DataTreePlayground 核心流程文档

## 目录
- [系统概述](#系统概述)
- [核心架构](#核心架构)
- [运行流程](#运行流程)
- [UCT 树管理](#uct-树管理)
- [节点类型与扩展策略](#节点类型与扩展策略)
- [并发执行机制](#并发执行机制)
- [关键数据结构](#关键数据结构)

---

## 系统概述

DataTreePlayground 是一个基于 UCT（Upper Confidence Bound）算法的并行搜索系统，用于自动化机器学习代码的迭代优化。系统通过树搜索策略，不断尝试新的数据增强和代码改进方案，最终找到最优解。

### 核心特性

- **两阶段执行**：串行初始化 → 并行搜索
- **UCT 优先级调度**：基于 UCT 值的 max-heap 优先执行高价值节点
- **事件驱动扩展**：节点完成立即触发扩展决策
- **并发安全**：线程锁保护共享状态
- **动态扩展策略**：父节点优先 → 当前节点次之

---

## 核心架构

### 主要组件

```
DataTreePlayground
├── UCTSearchManager      # UCT 树管理器
│   ├── UCTNode          # 树节点
│   ├── UCTMaxHeap       # 最大堆（优先队列）
│   └── UCTSearchConfig  # 搜索配置
├── Worker Loop          # 并行执行循环
├── InitialExp           # Initial 节点执行器
├── BlackExp             # Black 节点执行器
└── RedExp               # Red 节点执行器
```

### 执行流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    初始化阶段                               │
│  - 创建工作空间目录                                         │
│  - 初始化 UCT 搜索管理器                                    │
│  - 创建 Worker Agent 映射                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 1: 串行 Initial 阶段                     │
│  - 只有 Worker 0 工作                                       │
│  - 执行 root → initial                                      │
│  - Initial 成功后批量扩展 6 个子节点                        │
│    • 1 个 red 节点（数据搜索）                              │
│    • 5 个 black 节点（数据增强）                            │
│  - 所有节点加入 execution_heap（max-heap）                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: 并行执行阶段                          │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Worker Loop（每个 Worker 独立执行）               │     │
│  │                                                    │     │
│  │  1. 从 execution_heap 弹出 UCT 值最高的节点        │     │
│  │  2. 执行节点（运行代码、评测）                     │     │
│  │  3. 处理结果                                       │     │
│  │  4. 事件驱动扩展：                                 │     │
│  │     • 检查是否扩展父节点                           │     │
│  │     • 检查是否扩展当前节点                         │     │
│  │  5. 新节点加入 execution_heap                      │     │
│  │  6. 返回步骤 1                                     │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  直到：heap 为空 且 active_jobs = 0                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        返回最终结果
```

---

## 运行流程

### 主函数：`run()`

```python
def run(task_description: str, output_file: str | None = None) -> dict:
    try:
        # 1. 初始化
        setup_workspace_directories()
        search_mgr = create_search_manager()
        best_state = initialize_search_state()
        expand_node = create_expand_node_function()
        
        # 2. 启动并行 Workers
        worker_tasks = [partial(worker_loop, i) for i in range(max_workers)]
        worker_results = execute_parallel_tasks(worker_tasks)
        
        # 3. 返回结果
        return results
    finally:
        cleanup()
```

### Worker Loop 核心逻辑

```python
def worker_loop(worker_index: int) -> dict:
    while True:
        # 1. 选择节点
        should_wait, node = select_node_to_execute()
        if should_wait:
            sleep(0.1)
            continue
        if node is None:
            break
        
        # 2. 执行节点
        execute_and_process_node(node)
        
        # 3. 事件驱动扩展
        if initial_completed:
            expand_after_node_completion(node)
```

### 节点选择策略

#### Phase 1：串行 Initial

```
Worker 0:
  root → initial → [批量扩展 6 个子节点] → initial_completed = True
  
Worker 1-N:
  等待 initial_completed = True
```

#### Phase 2：并行执行

```
所有 Workers:
  从 execution_heap (max-heap) 弹出 UCT 值最高的节点
  
  UCT 值计算：
    uct_value = exploitation + exploration
    exploitation = total_reward / visits
    exploration = c * sqrt(log(parent_visits) / visits)
```

---

## UCT 树管理

### UCTSearchManager

管理整个搜索树的状态和操作。

#### 核心方法

```python
class UCTSearchManager:
    def select_next(self) -> UCTNode:
        """基于 UCT 算法选择下一个要扩展的节点"""
        
    def create_child(self, parent, stage, plan, code) -> UCTNode:
        """创建子节点"""
        
    def ingest_result(self, node, review) -> float:
        """处理节点执行结果，回传奖励"""
        
    def push_execution_node(self, node) -> None:
        """将节点加入执行堆"""
        
    def pop_execution_node(self) -> UCTNode:
        """从执行堆弹出优先级最高的节点"""
```

### UCTMaxHeap（最大堆）

基于 UCT 值的优先队列，用于并行执行时的节点调度。

```python
class UCTMaxHeap:
    def push(self, node: UCTNode) -> None:
        """将节点加入堆，按 UCT 值排序"""
        uct_value = self.uct_func(node)
        heapq.heappush(self.heap, (-uct_value, node.id))
        
    def pop_max(self) -> UCTNode:
        """弹出 UCT 值最大的节点"""
        neg_uct, node_id = heapq.heappop(self.heap)
        return self.nodes[node_id]
```

### UCT 树结构

```
root (虚拟)
  │
  └── initial (metric=0.85) ← 第一个执行的节点
       │
       ├── red_1 (metric=0.82) ← 数据搜索节点
       │    │
       │    ├── black_1 (metric=0.87)
       │    └── black_2 (metric=0.80)
       │
       ├── black_1 (metric=0.87) ← 数据增强节点
       │    │
       │    ├── red_1 (metric=0.85)
       │    └── black_1 (metric=0.89)
       │
       ├── black_2 (metric=0.80)
       ├── black_3 (metric=0.83)
       ├── black_4 (metric=0.81)
       └── black_5 (metric=0.84)
```

---

## 节点类型与扩展策略

### 节点类型

| 节点类型 | 英文 | 职责 | 上限 |
|---------|------|------|------|
| 根节点 | root | 虚拟起始节点，只用于扩展 initial | 1 个 |
| 初始节点 | initial | 生成初始版本的代码 | 1 个 |
| 红色节点 | red | 通过外部搜索接口进行数据搜索、下载数据集 | 每个父节点 1 个 |
| 黑色节点 | black | 对现有数据进行数据增强、数据整合 | 每个父节点 5 个 |

### 扩展规则

```python
def _select_stages_batch(target, search_cfg):
    if target.stage == "root":
        return [("initial", "", "", "")]
    
    if target.is_buggy:
        return []  # buggy 节点不扩展
    
    # 统计已有子节点
    num_red = count_children(target, "red")
    num_black = count_children(target, "black")
    
    stages_to_create = []
    
    # 先创建 red（最多 1 个）
    if num_red < search_cfg.num_red:
        stages_to_create.append(("red", ...))
    
    # 再创建 black（最多 num_black 个）
    remaining_black = search_cfg.num_black - num_black
    for _ in range(remaining_black):
        stages_to_create.append(("black", ...))
    
    return stages_to_create
```

### 事件驱动扩展

节点执行完成后，按以下优先级扩展：

```
优先级 1: 扩展父节点（创建兄弟节点）
  │
  ├─ 条件：should_expand_parent(node)
  │  - 父节点不是 root
  │  - 父节点不是 buggy
  │  - 父节点的子节点数未达上限（1 red + 5 black）
  │
  └─ 操作：创建 1 个 red 或 black 节点

优先级 2: 扩展当前节点（创建子节点）
  │
  ├─ 条件：should_expand_node(node)
  │  - 节点不是 buggy
  │  - 节点不是 terminal
  │  - 节点的子节点数未达上限
  │
  └─ 操作：创建 1 个 red 或 black 节点
```

---

## 并发执行机制

### 并发模型

- **并发度**：`max_workers`（默认 4）
- **执行模式**：
  - Phase 1：只有 Worker 0 工作（串行）
  - Phase 2：所有 Workers 并行工作

### 线程安全

#### 共享状态

```python
best_state = {
    "code": None,           # 最佳代码
    "metric": None,         # 最佳指标
    "node_id": None,        # 最佳节点 ID
    "dispatch_id": 0,       # 任务分发 ID
    "active_jobs": 0,       # 活跃任务数
    "initial_completed": False,  # initial 是否完成
}

state_lock = threading.Lock()  # 保护所有共享状态
```

#### 锁的使用

```python
# 读取共享状态
with state_lock:
    if search_mgr.current_step >= max_steps:
        break
    node = search_mgr.pop_execution_node()
    best_state["active_jobs"] += 1

# 更新共享状态
with state_lock:
    best_state["active_jobs"] -= 1
    node.code = res.get("code", "")
    review = build_review(res, has_submission)
    reward = search_mgr.ingest_result(node, review)
```

### 任务调度

```
┌─────────────────────────────────────────────────────────┐
│                  execution_heap (max-heap)              │
│                                                         │
│  节点按 UCT 值排序，每次弹出最高值的节点                │
│                                                         │
│  [node_1: UCT=2.5] ← 优先执行                           │
│  [node_2: UCT=2.1]                                      │
│  [node_3: UCT=1.8]                                      │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
         │                           │
         │ Worker 0                  │ Worker 1
         │ pop_max()                 │ pop_max()
         ▼                           ▼
    执行 node_1                  执行 node_2
         │                           │
         ▼                           ▼
    完成，扩展子节点              完成，扩展子节点
         │                           │
         └───────────┬───────────────┘
                     ▼
              新节点加入 heap
                     │
                     ▼
              继续循环...
```

---

## 关键数据结构

### UCTNode

```python
@dataclass
class UCTNode:
    # 基本信息
    stage: StageLiteral          # 节点类型
    plan: str                    # Agent 生成的计划
    code: str                    # Agent 生成的代码
    parent: Optional[UCTNode]    # 父节点
    id: str                      # 唯一标识（UUID）
    created_at: float            # 创建时间
    
    # 执行结果
    stdout: Optional[str]        # 代码执行输出
    exit_code: Optional[int]     # 退出码
    metric: MetricValue          # 指标值
    finish_time: Optional[float] # 完成时间
    
    # 状态标志
    is_buggy: Optional[bool]     # 是否有 bug
    is_valid: Optional[bool]     # 是否有效
    is_terminal: bool            # 是否终止节点
    continue_improve: bool       # 是否继续改进
    locked: bool                 # 是否被锁定
    
    # UCT 统计
    visits: int                  # 访问次数
    total_reward: float          # 累计奖励
    children: set                # 子节点集合
    expected_child_count: int    # 预期子节点数
```

### MetricReview

```python
@dataclass
class MetricReview:
    metric: Optional[float]      # 指标值
    lower_is_better: Optional[bool]  # 是否越小越好
    is_bug: bool                 # 是否有 bug
    has_submission: bool         # 是否有提交文件
    summary: str                 # 摘要
    raw_output: str              # 原始输出
```

### UCTSearchConfig

```python
@dataclass
class UCTSearchConfig:
    num_red: int = 1             # red 节点上限
    num_black: int = 5           # black 节点上限
    max_steps: int = 40          # 最大搜索步数
```

---

## 执行示例

### 简单执行流程

```
1. 初始化
   ├── 创建 workspace/best_solution
   ├── 创建 workspace/best_submission
   ├── 创建 workspace/submission
   └── 创建 UCTSearchManager

2. Phase 1: 串行 Initial
   Worker 0:
     ├── select_next() → root
     ├── create_child(root, "initial") → initial_node
     ├── 执行 initial_node
     ├── initial_node 完成 (metric=0.85)
     └── 批量扩展：
         ├── red_1 → execution_heap
         ├── black_1 → execution_heap
         ├── black_2 → execution_heap
         ├── black_3 → execution_heap
         ├── black_4 → execution_heap
         └── black_5 → execution_heap

3. Phase 2: 并行执行
   Worker 0                    Worker 1
     │                            │
     ├── pop_max() → black_1      ├── pop_max() → red_1
     │ (UCT=2.5)                  │ (UCT=2.3)
     │                            │
     ├── 执行 black_1             ├── 执行 red_1
     ├── metric=0.87              ├── metric=0.82
     │                            │
     ├── 扩展父节点               ├── 扩展父节点
     │ └── black_6                │ └── (已达上限)
     │                            │
     ├── 扩展当前节点             ├── 扩展当前节点
     │ └── black_1_red_1          │ └── red_1_black_1
     │                            │
     └── 新节点加入 heap          └── 新节点加入 heap

4. 继续循环直到 heap 为空

5. 返回结果
   ├── best_solution: 最佳代码
   ├── best_submission: 最佳提交文件
   └── trajectories: 执行轨迹
```

---

## 配置参数

### config.yaml

```yaml
session:
  local:
    parallel:
      enabled: true
      max_parallel: 4              # 最大并发数
      split_workspace_for_exp: false  # 是否为每个 experiment 分离 workspace

grading_servers: []               # 评测服务器列表
data_root: /path/to/data          # 数据根目录
exp_id: experiment_id             # 实验 ID
```

### UCT 参数

```python
max_steps = 40                    # 最大搜索步数
num_red = 1                       # 每个 parent 的 red 子节点上限
num_black = 5                     # 每个 parent 的 black 子节点上限
```

---

## 常见问题

### Q1: 为什么 Phase 1 是串行的？

A: Initial 节点是整个搜索的起点，必须成功后才能进行后续的并行搜索。串行执行确保了只有一个 initial 节点，避免重复计算。

### Q2: UCT 值如何计算？

A: 
```
UCT = exploitation + exploration
exploitation = total_reward / visits
exploration = c * sqrt(log(parent_visits) / visits)
```
其中 `c` 是探索系数，随搜索进度动态衰减。

### Q3: 如何保证并发安全？

A: 所有共享状态的访问都通过 `state_lock` 保护，确保同一时间只有一个线程修改状态。

### Q4: 什么时候停止搜索？

A: 满足以下任一条件：
1. `search_mgr.current_step >= max_steps`
2. `execution_heap` 为空 且 `active_jobs == 0`

### Q5: 如何选择最佳节点？

A: 节点执行完成后，通过 `search_mgr.ingest_result()` 更新 UCT 统计，系统自动跟踪最佳节点（`search_mgr.best_node`）。
