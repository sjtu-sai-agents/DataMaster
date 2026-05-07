# DataMaster

## Project overview

DataMaster is a research codebase for studying automated data-science agents. It includes agent orchestration code, dataset-search tools, MLE-Bench style playgrounds, post-training playgrounds, prompt templates, configuration examples, and analysis utilities.

The repository is prepared for anonymous review. Datasets, large generated artifacts, checkpoints, run outputs, local caches, and private credentials are intentionally omitted.

## Installation

Use Python 3.10 or newer. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e mle-bench
```

Some optional playgrounds may require additional packages or external services. Configure those services with environment variables rather than hard-coded credentials.

## Basic usage

Run an MLE-Bench style task with:

```bash
bash run_mle.sh <task_name>
```

Run the main entry point with:

```bash
python main.py --help
python run.py --help
```

Inspect a generated tree or run directory with:

```bash
python vis_node_by_tree_with_grade.py --run-dir <run_dir> --dataset <dataset_name>
python test/tool_inspection.py --run-dir <run_dir>
python test/tree_analysis.py --run-dir <run_dir>
```

## Main scripts

- `main.py`, `run.py`: top-level experiment entry points.
- `run_mle.sh` and related `run_mle_*.sh` scripts: launch MLE-Bench style experiments and ablations.
- `scripts/auto_config_exp.py`: generate or adapt experiment configurations.
- `search_dataset_tools/test_mcp_tool.py`: exercise the dataset-search MCP tools.
- `vis_node.py`, `vis_node_by_tree.py`, `vis_node_by_tree_with_grade.py`: visualize agent search trees and grading results.

## Configuration files

Configuration examples are stored under `configs/` and `agentcodebase/config/`. Replace placeholder values such as `${OPENAI_API_KEY}`, `${SERPER_API_KEY}`, `${IMAGE_MODEL_API_KEY}`, `${DATA_ROOT}`, `${MODEL_ROOT}`, and `${MCP_SERVER_URL}` with local environment variables or private deployment-specific settings before running experiments.

## Notes About Omitted Datasets And Large Artifacts

Large datasets, generated run directories, logs, model checkpoints, caches, and result exports are not included in this anonymous repository. Recreate them locally by following the configuration templates and pointing `DATA_ROOT`, `MODEL_ROOT`, and related environment variables to local resources.

## Anonymity Note

This repository has been scrubbed for anonymous review. It should not contain author names, institutional identifiers, personal paths, real repository URLs, API keys, tokens, webhooks, SSH keys, or private service credentials.
