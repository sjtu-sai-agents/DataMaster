# DataMaster

DataMaster is an EvoMaster-based autonomous data-engineering framework for MLE-Bench-style machine learning workflows.

## Overview

DataMaster focuses on the data side of machine learning problem solving. Given a fixed modeling algorithm or starter solution, it searches for better data pipelines, external data sources, feature transformations, validation signals, and reusable data artifacts for MLE-Bench and MLE-Bench Lite style tasks.

The framework organizes data-engineering decisions with a DataTree. Red nodes explore potentially useful external data or data transformations, black nodes exploit and refine selected candidates, the Data Pool stores reusable candidate datasets, and Global Memory keeps outcomes that can inform later search rounds. Downstream validation feedback is used to decide which data-side changes should be expanded.

## Relationship to EvoMaster

DataMaster is built on top of EvoMaster. EvoMaster provides the core agent, runtime, tool, session, skill, and playground abstractions. This repository keeps the necessary EvoMaster core components and adds DataMaster-specific MLE-Bench playgrounds, data-side search tools, DataTree workflows, and MLE-Bench Lite experiment configuration.

Upstream EvoMaster: https://github.com/sjtu-sai-agents/EvoMaster

## Key Features

- DataTree search for iterative data-engineering workflows.
- Red nodes for data exploration and candidate acquisition.
- Black nodes for data exploitation, cleaning, refinement, and pipeline improvement.
- Data Pool for reusable candidate datasets and derived artifacts.
- Global Memory for reusable outcomes across search rounds.
- MLE-Bench-style task execution with validation feedback.
- Configurable local or Docker-backed execution through EvoMaster sessions.

## Repository Structure

```text
DataMaster/
|-- configs/
|   |-- ml_master/
|   `-- ml_master_datatree/
|-- docs/
|-- evomaster/
|-- initial_code/
|-- mle-bench/
|-- playground/
|   |-- ml_master/
|   |-- ml_master_datatree/
|   |-- ml_master_datatree_v2/
|   `-- search_dataset_tools/
|-- scripts/
|-- run.py
|-- pyproject.toml
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Installation

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

## Configuration

DataMaster configs live under `configs/ml_master/` and `configs/ml_master_datatree/`. Task-specific MLE-Bench Lite configs are under `configs/ml_master_datatree/yaml_configs/`, with matching MCP tool configs under `configs/ml_master_datatree/json_configs/`.

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

## Quick Start

Prepare an MLE-Bench or MLE-Bench Lite task directory locally, then run a DataTree workflow with a task-specific config:

```bash
export DATA_ROOT=/path/to/mle-bench-lite
export MLE_BENCH_DATA_DIR="$DATA_ROOT"
export LLM_MODEL=your-model-name
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://your-llm-endpoint/v1

python run.py \
  --agent ml_master_datatree \
  --config configs/ml_master_datatree/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
  --task "$DATA_ROOT/detecting-insults-in-social-commentary/prepared/public/description.md"
```

Use `--initial-code` when you want to seed the initial node with a starter solution:

```bash
python run.py \
  --agent ml_master_datatree \
  --config configs/ml_master_datatree/yaml_configs/detecting-insults-in-social-commentary/config_detecting-insults-in-social-commentary.yaml \
  --task "$DATA_ROOT/detecting-insults-in-social-commentary/prepared/public/description.md" \
  --initial-code initial_code/data_loader_format/detecting-insults-in-social-commentary/full_code.py
```

## Main Scripts

- `run.py`: main command-line entry point for DataMaster and EvoMaster playgrounds.
- `scripts/auto_config_exp.py`: helper for generating task-specific DataMaster configs.
- `scripts/build_full_initial_codes.py`: helper for assembling starter-code manifests.
- `scripts/prefetch_models.py`: optional helper for local model preparation.
- `scripts/check_port_conflicts.py`: utility for diagnosing local grading-server ports.
- `scripts/vis_node_by_tree_with_grade.py`: interactive DataTree visualization with grading feedback.

## MLE-Bench Setup

Benchmark datasets, Kaggle data, generated submissions, checkpoints, and run artifacts are not stored in this repository. Prepare MLE-Bench or MLE-Bench Lite separately and point `DATA_ROOT` or `MLE_BENCH_DATA_DIR` to the local benchmark directory. The vendored `mle-bench/` directory is kept for benchmark integration code and reference tooling.

## Security

No credentials are intentionally included. Do not commit API keys, tokens, webhooks, SSH keys, `.env` files, benchmark data, generated submissions, model checkpoints, run logs, or private service configuration. Use environment variables or local untracked config files for secrets and deployment-specific paths.

## Citation

Citation information will be added once the paper is publicly available.

## Acknowledgements

DataMaster builds on EvoMaster and reuses EvoMaster's core agent and runtime abstractions. We thank the EvoMaster project for the upstream framework: https://github.com/sjtu-sai-agents/EvoMaster

## License

This repository is released under the Apache License 2.0. See `LICENSE` for details.
