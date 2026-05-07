# Data Scientist PlayGround

专为 Kaggle 竞赛自动化设计的多智能体 playground，支持迭代改进工作流。

> Just Initial Version of Kaggle Competitions

## 概述

Data Scientist PlayGround 实现了完整的 Kaggle 竞赛解决工作流，包含多个专业化的 agent：

- **Data Collection Agent**: 自动收集数据
- **Draft Agent**: 生成初始解决方案代码
- **Debug Agent**: 调试和修复代码错误
- **Improve Agent**: 迭代改进解决方案
- **Research Agent**: 探索新的改进方向
- **Knowledge Promotion Agent**: 提取和维护知识
- **Metric Agent**: 评估解决方案性能

## Minimal Kaggle 工作流程

Data Scientist PlayGround 基于 Minimal Kaggle 的工作流，Minimal Kaggle 多智能体 Kaggle 竞赛自动化的工作流如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│                           初始化阶段                                  │
│  - 创建 6 个 agents（draft, debug, improve, research,               │
│    knowledge_promotion, metric）                                      │
│  - knowledge_promotion_agent 被创建但永不使用                         │
│  - self.knowledge = "There is no memory now." (永不更新)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DraftExp 阶段                               │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Draft Agent: 生成初始解决方案代码                                 │ │
│  │   - 输入: task_description, data_preview                         │ │
│  │   - 输出: Python 代码                                            │ │
│  │   - 执行: python run.py                                          │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                    │                                  │
│                          ┌─────────┴─────────┐                        │
│                          │ 执行成功?           │                        │
│                          └─────────┬─────────┘                        │
│                     ┌──────────────┴──────────────┐                   │
│                     │ YES                         │ NO                  │
│                     ▼                             ▼                     │
│          ┌──────────────────┐      ┌─────────────────────────────┐   │
│          │ Metric Agent     │      │ Debug Agent (最多3次)         │   │
│          │ 解析终端输出      │      │ - 输入: terminal_output      │   │
│          │ 提取 validation  │      │         buggy_code            │   │
│          │ score            │      │ - 修复并重新执行              │   │
│          └──────────────────┘      │ - 成功后调用 Metric Agent    │   │
│                                     └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        保存 best_solution 和 best_submission
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    迭代改进阶段 (固定10轮)                            │
│                  for reseach_round in range(10):                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │ ResearchExp                                                     ││
│  │   - 输入: task, data_preview, best_solution, knowledge(空)       ││
│  │   - 输出: research_plan (JSON格式)                               ││
│  │   示例结构:                                                      ││
│  │   {                                                              ││
│  │     "direction1": {"idea1": "...", "idea2": "..."},              ││
│  │     "direction2": {"idea1": "...", "idea2": "..."}               ││
│  │   }                                                              ││
│  └────────────────────────────────────────────────────────────────┘│
│                                    │                                  │
│                                    ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │ 遍历每个 direction                                               ││
│  │   direction_best_solution = 当前全局最佳                        ││
│  │   direction_best_score = 当前全局最佳                            ││
│  │                                                                  ││
│  │   遍历该 direction 下每个 idea:                                  ││
│  │                                                                  ││
│  │   ┌────────────────────────────────────────────────────────────┐││
│  │   │ ImproveExp                                                 │││
│  │   │   - 输入: task, data_preview, direction_best_solution, idea │││
│  │   │   - 执行: python run.py                                     │││
│  │   │                                                            │││
│  │   │   执行成功? → Metric Agent 解析分数                         │││
│  │   │   执行失败? → Debug Agent (最多3次) → Metric Agent          │││
│  │   │                                                            │││
│  │   │   如果新分数更好:                                           │││
│  │   │     direction_best_solution = 新方案                       │││
│  │   │     direction_best_score = 新分数                          │││
│  │   └────────────────────────────────────────────────────────────┘││
│  │                                                                  ││
│  │   该 direction 结束后:                                           ││
│  │     self.best_solution = direction_best_solution                ││
│  │     self.best_score = direction_best_score                      ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              返回完成状态
```

### 关键特点

| 特性 | 说明 |
|------|------|
| **实际使用的 Agents** | Draft, Debug, Improve, Research, Metric (5个) |
| **未使用的 Agent** | Knowledge Promotion (被创建但从不调用) |
| **外层循环** | 固定 10 轮 (`for reseach_round in range(10)`) |
| **Debug 机制** | 嵌套在 DraftExp/ImproveExp 内，最多3次重试 |
| **Metric 作用** | 解析终端输出，提取 `\boxed{分数}` 格式的验证分数 |
| **知识积累** | 不存在 (`knowledge` 永远是初始值) |
| **方向级联** | 每个 direction 内的 ideas 串行执行，基于该 direction 的最佳方案 |

## Data Scientist PlayGround 工作流程

Data Scientist PlayGround 在 Minimal Kaggle 的基础上增加了**数据发现**功能，允许系统自动搜索和整合外部数据集。

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Draft 阶段 (同 Minimal Kaggle)               │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    迭代改进阶段 (固定10轮)                        │
│                  for reseach_round in range(10):                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ ResearchExp (增强版)                                       │  │
│  │   - 输出固定结构: Model Architecture, Feature Engineering,      │  │
│  │     Training Strategy, Data Enhancement (可选)                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 遍历每个 direction                                        │  │
│  │                                                            │  │
│  │   ┌──────────────────────────────────────────────────────┐    │  │
│  │   │ direction == "Data Enhancement" ?                 │    │  │
│  │   │                                                   │    │  │
│  │   │ ┌─────────────┐      ┌──────────────────────┐   │    │  │
│  │   │ │  YES        │      │  NO                │   │    │  │
│  │   │ │             │      │                    │   │    │  │
│  │   │ │ ▼           │      │ ▼                  │   │    │  │
│  │   │ │ DataExp    │      │ ImproveExp (原流程)  │   │    │  │
│  │   │ │            │      │                    │   │    │  │
│  │   │ │ 1. 搜索数据集  │      │ - 直接执行改进        │   │    │  │
│  │   │ │ 2. 评估可行性  │      │ - 无需数据发现步骤    │   │    │  │
│  │   │ │ 3. 生成        │                       │   │    │  │
│  │   │ │   data_bridge.py│                       │   │    │  │
│  │   │ │             │      │                       │   │    │  │
│  │   │ │ ▼           │      ▼                       │   │    │  │
│  │   │ │ 可行?        │      Metric Agent            │   │    │  │
│  │   │ │             │                              │   │    │  │
│  │   │ │ ┌─────┴──────┐                           │   │    │  │
│  │   │ │ │ NO         │ YES                        │   │    │  │
│  │   │ │ │            │ ▼                         │   │    │  │
│  │   │ │ │ 跳过该idea  │ ImproveExp (带数据上下文)      │   │    │  │
│  │   │ │ │            │ - 使用 data_bridge.py      │   │    │  │
│  │   │ │ │            │   合并外部数据             │   │    │  │
│  │   │ │ └────────────┴───────────────────────────┘   │    │  │
│  │   └──────────────────────────────────────────────────────┘    │  │
│  │                                                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### DataExp 数据发现流程

当 Research Agent 输出 "Data Enhancement" 方向时，触发数据发现流程：

```
┌────────────────────────────────────────────────────────────────────┐
│                     DataExp 数据发现阶段                          │
├────────────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 解析 Idea                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 输入: "Search for weather dataset with date, temp,      │  │
│  │        humidity columns to merge on date"                  │  │
│  │                                                          │  │
│  │  输出:                                                   │  │
│  │  - search_query: "weather dataset"                      │  │
│  │  - expected_columns: ["date", "temp", "humidity"]           │  │
│  │  - merge_strategy: "left join on date"               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           │                                        │
│                           ▼                                        │
│  Step 2: 调用 Data Agent 搜索                                 │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 使用 search_datasets(), verify_dataset(), get_dataset_sample()   │  │
│  │ 从 HuggingFace 等源搜索候选数据集                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           │                                        │
│                           ▼                                        │
│  Step 3: 可行性评估                                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 对每个候选数据集打分 (0-1):                               │  │
│  │ - 是否 gated (0分)                                        │  │
│  │ - 下载量 (0-0.3分)                                      │  │
│  │ - likes 数 (0-0.2分)                                    │  │
│  │ - 列匹配度 (0-0.3分)                                   │  │
│  │ - 基础分 (0.2分)                                       │  │
│  │                                                          │  │
│  │ 分数 >= 0.5 → 继续处理                                 │  │
│  │ 分数 < 0.5  → 跳过该 idea                             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                           │                                        │
│                           ▼                                        │
│  Step 4: 生成 data_bridge.py                                   │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 自动生成数据合并脚本模板，包含:                             │  │
│  │ - load_external_dataset() 函数                              │  │
│  │ - merge_datasets() 函数                                    │  │
│  │ - main() 入口                                             │  │
│  │                                                          │  │
│  │ Improve Agent 会使用这个脚本来合并外部数据和原始数据              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键特性对比

| 特性 | Minimal Kaggle | Data Scientist |
|------|----------------|----------------|
| **Agents 数量** | 6 个 (draft, debug, improve, research, knowledge_promotion, metric) | 7 个 (+ data_agent) |
| **数据发现** | 无 | **有** - 自动搜索 HuggingFace 等数据源 |
| **外部数据支持** | 无 | **有** - 自动生成 data_bridge.py 合并脚本 |
| **Research 输出** | 自定义 major area | 固定结构 + 可选 Data Enhancement |
| **知识积累** | 无 | **有** - 记录数据搜索结果供后续参考 |
| **反馈机制** | 无 | **有** - 失败的数据方向会记录并避免重复 |



## 快速开始

### 1. 准备数据与环境

将 Kaggle 竞赛数据放入本地位置，后续在配置config.yaml后会自动将对应数据软链接到工作目录，示例任务已经提供了一个数据，位置在：

```
playground/minimal_kaggle/data/
├── private/
│   └── ...     # 测试数据
├── public/
│   └── ...    # 训练数据
```

如果需要运行示例机器学习任务，你需要根据`playground/minimal_kaggle/requirements.txt`在Evomaster基础环境上安装额外需要的环境

### 2. 配置

编辑 `configs/minimal_kaggle/deepseek-v3.2-example.yaml`：

```yaml
  local_sglang:
    provider: "deepseek"
    model: "deepseek-v3.2"
    api_key: "dummy"  # 本地部署可使用占位符
    base_url: "http://192.168.2.110:18889/v1"
    temperature: 0.7
    max_tokens: 16384
    timeout: 300  
    max_retries: 3
    retry_delay: 1.0

  # ... 如果使用openai api格式，需要同时修改每个agent的LLM配置，如
  agents:
    draft:
      llm: "local_sglang" #如修改成openai
```

### 3. 运行

```bash
# 示例已经在playground/minimal_kaggle/data下提供一个示例数据，如果使用自己的数据，需要修改config.yaml中的local下的软链接配置
python run.py --agent minimal_kaggle --config configs/minimal_kaggle/deepseek-v3.2-example.yaml --task playground/minimal_kaggle/data/public/description.md
```

### 4. 查看结果

结果保存在：

```
runs/minimal_kaggle/workspaces/
├── best_submission/
│   └── submission.csv      # 最佳提交文件
├── best_solution/
│   └── best_solution.py    # 最佳解决方案代码
├── submission/
│   └── submission_*.csv    # 所有提交
└── working/                # 工作文件
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `agents.draft.max_turns` | Draft agent 最大轮数 | `50` |
| `agents.debug.max_turns` | Debug agent 最大轮数 | `50` |
| `agents.improve.max_turns` | Improve agent 最大轮数 | `50` |
| `session.local.working_dir` | 工作目录路径 | `"./playground/minimal_kaggle/workspace"` |

## 使用示例

### 文本检测 (detecting-insults-in-social-commentary)
```bash
python run.py --agent minimal_kaggle --config configs/minimal_kaggle/deepseek-v3.2-example.yaml --task playground/minimal_kaggle/data/public/description.md
```


## 目录结构

```
playground/minimal_kaggle/
├── core/
│   ├── __init__.py
│   ├── playground.py       # 主 playground
│   ├── exp/
│   │   ├── draft_exp.py    # Draft 实验
│   │   ├── improve_exp.py  # Improve 实验
│   │   └── research_exp.py # Research 实验
│   └── utils/
│       ├── code.py         # 代码工具
│       └── data_preview.py # 数据预览
├── prompts/                # Agent 提示词
└── workspace/              # 工作目录
```

## 相关文档

- [EvoMaster 主 README](../../README.md)
- [配置示例](../../configs/)
