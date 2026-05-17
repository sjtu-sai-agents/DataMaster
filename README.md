<p align="center">
  <img src="docs/data_master_logo.png" alt="DataMaster" width="520">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <a href="https://arxiv.org/abs/2605.10906"><img src="https://img.shields.io/badge/arXiv-2605.10906-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <a href="README_CN.md">中文</a> | English
</p>

<h1 align="center">DataMaster</h1>
<p align="center"><em>Data-Centric Autonomous AI Research</em></p>

<p align="center">
  <img src="docs/main.png" alt="DataMaster Framework Overview" width="780">
  <br><sub><b>Figure 1.</b> Overview of the DataMaster autonomous data-engineering framework.</sub>
</p>

---

## 📖 Overview

DataMaster focuses on the **data side** of machine learning problem solving. Given a fixed modeling algorithm or starter solution, it searches for better data pipelines, external data sources, feature transformations, validation signals, and reusable data artifacts. It targets both **MLE-Bench** (competition-style ML tasks) and **PostTrainBench** (post-training enhancement: math, reasoning, domain-specific fine-tuning).

The framework organizes data-engineering decisions with a **DataTree**. **Red nodes** explore external data or transformations, **Black nodes** exploit and refine candidates, the **Data Pool** stores reusable datasets, and **Global Memory** retains outcomes across search rounds. DataMaster is built on top of [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster).

---

## 📦 Release Scope

DataMaster is designed to support both **MLE-Bench** and **PostTrainBench** workflows. The current open-source release includes the MLE-Bench workflow:

| Component | Path |
|---|---|
| DataTree core workflow | `playground/data_master` |
| Baseline MLE-Bench workflow | `playground/ml_master` |
| Dataset search & submission tools | `playground/search_dataset_tools` |
| Benchmark integration | `mle-bench/` |
| Task-specific configs (75 tasks) | `configs/data_master/` |

> **PostTrainBench** is on the DataMaster roadmap. Code will be released in a future update.

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

- 🌲 **DataTree Search** — tree-structured iterative search over executable data states
- 🔴 **Red Nodes** — external data discovery and candidate source acquisition
- ⚫ **Black Nodes** — data refinement, cleaning, adaptation, and DataLoader construction

</td>
<td width="50%">

- 🗄️ **Data Pool** — shared candidate dataset layer reused across search branches
- 🧠 **Global Memory** — stores node outcomes, artifacts, and reusable findings
- 🎯 **Validation Feedback** — MLE-Bench and PostTrainBench task execution with configurable metrics

</td>
</tr>
</table>

<p align="center">
  <img src="docs/data_master_walkthrough.png" alt="DataMaster Walkthrough" width="780">
  <br><sub><b>Figure 2.</b> Walkthrough of the DataTree search process: Red nodes explore, Black nodes exploit.</sub>
</p>

---

## 🏗️ Repository Structure

```text
DataMaster/
├── configs/
│   ├── ml_master/              # Baseline agent configs
│   └── data_master/            # DataTree configs + 75 task configs
├── docs/                       # English and Chinese documentation
├── evomaster/                  # EvoMaster core components
├── initial_code/               # Starter-code templates
├── mle-bench/                  # Benchmark integration and tooling
├── playground/
│   ├── ml_master/              # Baseline MLE-Bench workflow
│   ├── data_master/            # Main DataTree workflow
│   └── search_dataset_tools/   # Dataset search & submission tools
├── scripts/                    # Utility and visualization scripts
├── run.py                      # Main CLI entry point
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## 💿 Installation

```bash
git clone https://github.com/zhifan-zhou/DataMaster.git
cd DataMaster
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Note:** Python >= 3.10, < 3.13 required.

---

## ⚙️ Configuration

Configs live under `configs/ml_master/` and `configs/data_master/`. Task-specific YAML configs are under `configs/data_master/yaml_configs/`, with MCP tool configs under `configs/data_master/json_configs/`.

Credentials are **not** stored in this repository. Provide keys via environment variables:

```bash
export DATA_ROOT=/path/to/mle-bench-lite
export LLM_MODEL=your-model-name
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://your-llm-endpoint/v1
export SERPER_API_KEY=optional-serper-key     # web search
export HF_TOKEN=optional-huggingface-token    # dataset search
```

---

## 🚀 Quick Start

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

With a starter solution:

```bash
python run.py \
  --agent data_master \
  --config configs/data_master/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
  --task "$DATA_ROOT/detecting-insults-in-social-commentary/prepared/public/description.md" \
  --initial-code initial_code/data_loader_format/detecting-insults-in-social-commentary/full_code.py
```

---

## 📜 Main Scripts

| Script | Purpose |
|---|---|
| `run.py` | Main CLI entry point |
| `scripts/auto_config_exp.py` | Generate task-specific configs |
| `scripts/build_full_initial_codes.py` | Assemble starter-code manifests |
| `scripts/prefetch_models.py` | Local model prefetch helper |
| `scripts/check_port_conflicts.py` | Diagnose grading-server port conflicts |
| `scripts/vis_node_by_tree_with_grade.py` | Interactive DataTree visualization |

---

## 🧪 MLE-Bench Setup

This repository does **not** include MLE-Bench datasets, Kaggle data, model checkpoints, or generated artifacts. Prepare MLE-Bench / MLE-Bench Lite separately and point `DATA_ROOT` to the local benchmark directory. The vendored `mle-bench/` directory provides integration code and reference tooling.

---

## 🗺️ Roadmap

| Item | Status |
|---|---|
| MLE-Bench / MLE-Bench Lite workflow | ✅ Released |
| PostTrainBench workflow | 🔜 Coming soon |
| Additional documentation and examples | 🚧 In progress |
| Reproducibility scripts and benchmarks | 📋 Planned |

---

## 🔒 Security

No credentials are intentionally included. Do not commit API keys, tokens, webhooks, SSH keys, `.env` files, benchmark data, model checkpoints, run logs, or private service configuration.

---

## 📝 Citation

If you find DataMaster useful in your research, please cite:

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

## 🙏 Acknowledgements

DataMaster builds on [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) and reuses its core agent and runtime abstractions. We thank the EvoMaster project for the upstream framework.

---

## 📄 License

This repository is released under the [Apache License 2.0](LICENSE).
