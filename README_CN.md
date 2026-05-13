<p align="center">
  <img src="docs/data_master_logo.png" alt="DataMaster" width="520">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <a href="https://arxiv.org/abs/2605.10906"><img src="https://img.shields.io/badge/arXiv-2605.10906-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  中文 | <a href="README.md">English</a>
</p>

<h1 align="center">DataMaster</h1>
<p align="center"><em>Towards Autonomous Data Engineering for Machine Learning</em></p>

<p align="center">
  <img src="docs/main.png" alt="DataMaster 框架总览" width="780">
  <br><sub><b>图 1.</b> DataMaster 自主数据工程框架总览。</sub>
</p>

---

## 📖 项目概览

DataMaster 聚焦于机器学习问题的**数据侧优化**。在固定建模算法或初始方案的前提下，自动搜索更优的数据流水线、外部数据源、特征变换、验证信号以及可复用的数据工件。框架同时面向 **MLE-Bench**（竞赛式机器学习任务）和 **PostTrainBench**（后训练增强：数学推理、领域微调等）。

框架通过 **DataTree（数据树）** 组织数据工程决策：**红节点**探索外部数据或数据变换，**黑节点**利用和优化候选对象，**Data Pool（候选数据池）** 存储可复用数据集，**Global Memory（全局记忆）** 跨搜索轮次保留结果。DataMaster 构建于 [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) 之上。

---

## 📦 发布范围

DataMaster 同时面向 **MLE-Bench** 和 **PostTrainBench** 工作流设计。当前开源版本包含 MLE-Bench 工作流：

| 组件 | 路径 |
|---|---|
| DataTree 核心工作流 | `playground/data_master` |
| MLE-Bench 基线工作流 | `playground/ml_master` |
| 数据集搜索与提交工具 | `playground/search_dataset_tools` |
| Benchmark 集成 | `mle-bench/` |
| 任务特定配置（75 个任务） | `configs/data_master/` |

> **PostTrainBench** 已纳入路线图，相关代码将在后续版本中发布。

---

## ✨ 核心特性

<table>
<tr>
<td width="50%">

- 🌲 **DataTree 搜索** — 基于树结构的可执行数据状态迭代搜索
- 🔴 **红节点** — 外部数据发现与候选数据源获取
- ⚫ **黑节点** — 数据清洗、精炼、适配与 DataLoader 构建

</td>
<td width="50%">

- 🗄️ **Data Pool（候选数据池）** — 跨搜索分支共享的候选数据集层
- 🧠 **Global Memory（全局记忆）** — 跨轮次存储节点结果、工件与可复用发现
- 🎯 **验证反馈** — MLE-Bench 与 PostTrainBench 任务执行与可配置评估

</td>
</tr>
</table>

<p align="center">
  <img src="docs/data_master_walkthrough.png" alt="DataMaster 流程总览" width="780">
  <br><sub><b>图 2.</b> DataTree 搜索流程总览：红节点探索，黑节点利用。</sub>
</p>

---

## 🏗️ 仓库结构

```text
DataMaster/
├── configs/
│   ├── ml_master/              # 基线 Agent 配置
│   └── data_master/            # DataTree 配置 + 75 个任务配置
├── docs/                       # 中英文文档
├── evomaster/                  # EvoMaster 核心组件
├── initial_code/               # 初始代码模板
├── mle-bench/                  # Benchmark 集成与参考工具
├── playground/
│   ├── ml_master/              # MLE-Bench 基线工作流
│   ├── data_master/            # DataTree 主工作流
│   └── search_dataset_tools/   # 数据集搜索与提交工具
├── scripts/                    # 辅助与可视化脚本
├── run.py                      # 命令行入口
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## 💿 安装

```bash
git clone https://github.com/zhifan-zhou/DataMaster.git
cd DataMaster
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **注意：** Python >= 3.10, < 3.13 版本要求。

---

## ⚙️ 配置说明

配置文件位于 `configs/ml_master/` 和 `configs/data_master/`。任务特定的 YAML 配置位于 `configs/data_master/yaml_configs/`，MCP 工具配置位于 `configs/data_master/json_configs/`。

本仓库**不**包含任何凭据。请通过环境变量提供密钥：

```bash
export DATA_ROOT=/path/to/mle-bench-lite
export LLM_MODEL=your-model-name
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://your-llm-endpoint/v1
export SERPER_API_KEY=optional-serper-key     # 网络搜索
export HF_TOKEN=optional-huggingface-token    # 数据集搜索
```

---

## 🚀 快速开始

```bash
export DATA_ROOT=/path/to/mle-bench-lite
export LLM_MODEL=your-model-name
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://your-llm-endpoint/v1

python run.py \
  --agent data_master \
  --config configs/data_master/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
  --task "$DATA_ROOT/detecting-insults-in-social-commentary/prepared/public/description.md"
```

使用初始方案代码：

```bash
python run.py \
  --agent data_master \
  --config configs/data_master/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
  --task "$DATA_ROOT/detecting-insults-in-social-commentary/prepared/public/description.md" \
  --initial-code initial_code/data_loader_format/detecting-insults-in-social-commentary/full_code.py
```

---

## 📜 主要脚本

| 脚本 | 用途 |
|---|---|
| `run.py` | 命令行入口 |
| `scripts/auto_config_exp.py` | 生成任务特定配置 |
| `scripts/build_full_initial_codes.py` | 组装初始代码清单 |
| `scripts/prefetch_models.py` | 本地模型预取辅助 |
| `scripts/check_port_conflicts.py` | 诊断评分服务器端口冲突 |
| `scripts/vis_node_by_tree_with_grade.py` | DataTree 交互式可视化 |

---

## 🧪 MLE-Bench 环境准备

本仓库**不**包含 MLE-Bench 数据集、Kaggle 竞赛数据、模型检查点或生成的实验产物。使用者需单独准备 MLE-Bench / MLE-Bench Lite 环境，并将 `DATA_ROOT` 指向本地 benchmark 目录。`mle-bench/` 目录提供了集成代码和参考工具。

---

## 🗺️ 路线图

| 项目 | 状态 |
|---|---|
| MLE-Bench / MLE-Bench Lite 工作流 | ✅ 已发布 |
| PostTrainBench 工作流 | 🔜 即将发布 |
| 补充文档与示例 | 🚧 进行中 |
| 可复现性脚本与基准测试 | 📋 计划中 |

---

## 🔒 安全说明

本仓库不会有意包含任何凭据。请勿提交 API 密钥、token、webhook、SSH 密钥、`.env` 文件、benchmark 数据、模型检查点、运行日志或私有服务配置。

---

## 📝 引用

如果您在研究中使用了 DataMaster，请引用：

```bibtex
@article{du2026datamaster,
  title   = {DataMaster: Towards Autonomous Data Engineering for Machine Learning},
  author  = {Yaxin Du and Xiyuan Yang and Zhifan Zhou and Wanxu Liu and
             Zixing Lei and Zimeng Chen and Fenyi Liu and Haotian Wu and
             Yuzhu Cai and Zexi Liu and Xinyu Zhu and WenHao Wang and
             Linfeng Zhang and Chen Qian and Siheng Chen},
  journal = {arXiv preprint arXiv:2605.10906},
  year    = {2026}
}
```

---

## 🙏 致谢

DataMaster 构建于 [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) 之上，复用了其核心 Agent 和运行时抽象。感谢 EvoMaster 项目提供的上游框架支持。

---

## 📄 License

本仓库基于 [Apache License 2.0](LICENSE) 发布。
