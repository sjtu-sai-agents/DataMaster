#!/usr/bin/env bash
set -euo pipefail
BASE=${WORKSPACE_ROOT}
PROJECT=$BASE/DataScientistEvomaster2
for env in "$PROJECT/.venv" "$PROJECT/.venv_llamafactory" "$PROJECT/.venv_posttrainbench_eval"; do
  echo "== $env =="
  "$env/bin/python" --version
  "$env/bin/python" -c "import sys; print(sys.executable)"
done
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -c "import playground.math_posttrain_datatree.core.playground as p; print(p.__file__)"
"$PROJECT/.venv_llamafactory/bin/python" -c "import torch, torchaudio, torchvision, llamafactory; print(torch.__version__, torchaudio.__version__, torchvision.__version__); print(llamafactory.__file__)"
"$PROJECT/.venv_posttrainbench_eval/bin/python" -c "import importlib.metadata as m; import transformers, datasets, inspect_ai, inspect_evals.aime2025, inspect_ai.model._providers.vllm as vllm_provider; openai_version = m.version('openai'); major, minor, *_ = [int(part) for part in openai_version.split('.') if part.isdigit()]; assert (major, minor) >= (2, 26), f'openai version too old: {openai_version}'; vllm_version = m.version('vllm'); vllm_major, vllm_minor, *_ = [int(part) for part in vllm_version.split('.') if part.isdigit()]; assert (vllm_major, vllm_minor) >= (0, 8), f'vllm version too old: {vllm_version}'; print(transformers.__version__, datasets.__version__, inspect_ai.__version__, openai_version, 'vllm=' + vllm_version); print(inspect_evals.aime2025.__file__); print(vllm_provider.__file__)"
test -x "$PROJECT/.venv_posttrainbench_eval/bin/vllm" && echo "vllm binary: OK" || { echo "ERROR: vllm binary not found in eval venv"; exit 1; }
