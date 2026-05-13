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

# DataMaster: Towards Autonomous Data Engineering for Machine Learning

---

<p align="center">
  <img src="docs/main.png" alt="DataMaster Framework Overview" width="800">
</p>

---

## 📖 Overview

DataMaster focuses on the data side of machine learning problem solving. Given a fixed modeling algorithm or starter solution, it searches for better data pipelines, external data sources, feature transformations, validation signals, and reusable data artifacts. It targets both MLE-Bench (competition-style ML tasks) and PostTrainBench (post-training enhancement tasks such as math, reasoning, and domain-specific fine-tuning). 

The framework organizes data-engineering decisions with a DataTree. Red nodes explore potentially useful external data or data transformations, black nodes exploit and refine selected candidates, the Data Pool stores reusable candidate datasets, and Global Memory keeps outcomes that can inform later search rounds. Downstream validation feedback is used to decide which data-side changes should be expanded. DataMaster is built on top of EvoMaster: https://github.com/sjtu-sai-agents/EvoMaster

---

## 📦 Release Scope

DataMaster is designed to support both **MLE-Bench** and **PostTrainBench** workflows. The current open-source release includes the MLE-Bench workflow code:

- DataMaster core workflow with DataTree search (`playground/data_master`)
- Baseline MLE-Bench style workflow (`playground/ml_master`)
- Data-side search tools (`playground/search_dataset_tools`)
- MLE-Bench integration code (`mle-bench/`)
- Task-specific MLE-Bench Lite configs under `configs/`

**PostTrainBench** support is part of the DataMaster roadmap. PostTrainBench-related code will be released in a future update.

---

## ✨ Key Features

- **DataTree Search** — tree-structured iterative search over executable data states.
- **Red Nodes** — external data discovery and candidate source acquisition.
- **Black Nodes** — data refinement, cleaning, adaptation, and DataLoader construction.
- **Data Pool** — shared candidate dataset layer reused across search branches.
- **Global Memory** — stores node outcomes, artifacts, and reusable findings across rounds.
- **MLE-Bench & PostTrainBench** — validation-driven task execution with configurable feedback.

---

<p align="center">
  <img src="docs/data_master_walkthrough.png" alt="DataMaster Walkthrough" width="800">
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
python -m pip install --upgrade pip
python -m pip install -e .
```

If your environment does not need the full benchmark dependency stack, you can install the package metadata first and add task-specific dependencies as needed:

```bash
python -m pip install -e . --no-deps
python -m pip install -r requirements.txt
```

---

## ⚙️ Configuration

DataMaster configs live under `configs/ml_master/` and `configs/data_master/`. Task-specific MLE-Bench Lite configs are under `configs/data_master/yaml_configs/`, with matching MCP tool configs under `configs/data_master/json_configs/`.

Credentials are not included in this repository. Provide keys and private endpoints through environment variables or local untracked config files. Common variables include:

```bash
export DATA_ROOT=/path/to/mle-bench-lite
export MLE_BENCH_DATA_DIR="$DATA_ROOT"
export LLM_MODEL=your-model-name
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://your-llm-endpoint/v1
export SERPER_API_KEY=optional-serper-key
export HF_TOKEN=optional-huggingface-token
```

---

## 🚀 Quick Start

Prepare an MLE-Bench or MLE-Bench Lite task directory locally, then run a DataTree workflow with a task-specific config:

```bash
export DATA_ROOT=/path/to/mle-bench-lite
export MLE_BENCH_DATA_DIR="$DATA_ROOT"
export LLM_MODEL=your-model-name
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://your-llm-endpoint/v1

python run.py \
  --agent data_master \
  --config configs/data_master/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
  --task "$DATA_ROOT/detecting-insults-in-social-commentary/prepared/public/description.md"
```

Use `--initial-code` when you want to seed the initial node with a starter solution:

```bash
python run.py \
  --agent data_master \
  --config configs/data_master/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
  --task "$DATA_ROOT/detecting-insults-in-social-commentary/prepared/public/description.md" \
  --initial-code initial_code/data_loader_format/detecting-insults-in-social-commentary/full_code.py
```

---

## 📜 Main Scripts

- `run.py`: main command-line entry point for DataMaster and EvoMaster playgrounds.
- `scripts/auto_config_exp.py`: helper for generating task-specific DataMaster configs.
- `scripts/build_full_initial_codes.py`: helper for assembling starter-code manifests.
- `scripts/prefetch_models.py`: optional helper for local model preparation.
- `scripts/check_port_conflicts.py`: utility for diagnosing local grading-server ports.
- `scripts/vis_node_by_tree_with_grade.py`: interactive DataTree visualization with grading feedback.

---

## 🧪 MLE-Bench Setup

Benchmark datasets, Kaggle data, generated submissions, checkpoints, and run artifacts are not stored in this repository. Prepare MLE-Bench or MLE-Bench Lite separately and point `DATA_ROOT` or `MLE_BENCH_DATA_DIR` to the local benchmark directory. The vendored `mle-bench/` directory is kept for benchmark integration code and reference tooling.

---

## 🗺️ Roadmap

| Item | Status |
|---|---|
| MLE-Bench / MLE-Bench Lite workflow | Released |
| PostTrainBench workflow | Coming soon |
| Additional documentation and examples | In progress |
| Reproducibility scripts and benchmarks | Planned |

---

## 🔒 Security

No credentials are intentionally included. Do not commit API keys, tokens, webhooks, SSH keys, `.env` files, benchmark data, generated submissions, model checkpoints, run logs, or private service configuration. Use environment variables or local untracked config files for secrets and deployment-specific paths.

---

## 📝 Citation

If you find DataMaster useful in your research, please cite:

```bibtex
@article{zhou2025datamaster,
  title={DataMaster: Towards Autonomous Data Engineering for Machine Learning},
  author={Zhou, Zhifan and ...},
  journal={arXiv preprint arXiv:2605.10906},
  year={2025}
}
```

> **Note:** The full author list will be finalized upon paper publication. Please check the [arXiv page](https://arxiv.org/abs/2605.10906) for the latest version.

---

## 🙏 Acknowledgements

DataMaster builds on EvoMaster and reuses EvoMaster's core agent and runtime abstractions. We thank the EvoMaster project for the upstream framework: https://github.com/sjtu-sai-agents/EvoMaster

---

## 📄 License

This repository is released under the Apache License 2.0. See `LICENSE` for details.
