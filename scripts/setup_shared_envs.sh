#!/usr/bin/env bash
set -euo pipefail

BASE=${WORKSPACE_ROOT}
PROJECT=$BASE/DataScientistEvomaster2
PLAYGROUND=$PROJECT/playground/math_posttrain_datatree
RUNTIME_BASE=$BASE/.runtime/python
SHARED_PY_DIR=$RUNTIME_BASE/cpython-3.12.13
SHARED_PY=$SHARED_PY_DIR/bin/python3.12
PY_URL=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/20260408/cpython-3.12.13%2B20260408-x86_64-unknown-linux-gnu-install_only.tar.gz
TMP_TGZ=/tmp/cpython-3.12.13-standalone.tar.gz
UV_CACHE_DIR=$BASE/.cache/uv
PIP_CACHE_DIR=$BASE/.cache/pip
HF_HOME=$BASE/.cache/huggingface
HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
HF_DATASETS_CACHE=$HF_HOME/datasets
INSPECT_EVALS_DIR=$PLAYGROUND/external/inspect_evals
export UV_CACHE_DIR PIP_CACHE_DIR HF_HOME HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE
export CC=gcc
export CXX=g++

mkdir -p "$RUNTIME_BASE" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME"

if [ ! -x "$SHARED_PY" ]; then
  rm -rf "$SHARED_PY_DIR"
  curl -L --fail --retry 3 "$PY_URL" -o "$TMP_TGZ"
  mkdir -p "$SHARED_PY_DIR"
  tar -xzf "$TMP_TGZ" -C "$SHARED_PY_DIR" --strip-components=1
fi

"$SHARED_PY" --version
"$SHARED_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$SHARED_PY" -m pip install --upgrade pip setuptools wheel uv

for stale in \
  "$PROJECT/.venv_gpu2" "$PROJECT/.venv_gpu3" \
  "$PROJECT/.venv_llamafactory_gpu2" "$PROJECT/.venv_llamafactory_gpu3" \
  "$PROJECT/.venv_posttrainbench_eval_gpu2" "$PROJECT/.venv_posttrainbench_eval_gpu3"; do
  rm -rf "$stale"
done

for env in "$PROJECT/.venv" "$PROJECT/.venv_llamafactory" "$PROJECT/.venv_posttrainbench_eval"; do
  rm -rf "$env"
  "$SHARED_PY" -m venv "$env"
  "$env/bin/python" -m pip install --upgrade pip setuptools wheel
done

"$SHARED_PY" -m uv sync --project "$PROJECT" --python "$PROJECT/.venv/bin/python"
"$PROJECT/.venv_llamafactory/bin/pip" install --upgrade -r "$PROJECT/requirements-llamafactory.txt"
"$PROJECT/.venv_posttrainbench_eval/bin/pip" install -r "$PLAYGROUND/external/PostTrainBench/containers/requirements-direct.txt"
"$PROJECT/.venv_posttrainbench_eval/bin/pip" install -r "$PLAYGROUND/external/PostTrainBench/src/eval/tasks/arenahardwriting/evaluation_code/requirements.txt"
"$PROJECT/.venv_posttrainbench_eval/bin/pip" install -r "$PLAYGROUND/external/PostTrainBench/src/eval/tasks/arenahardwriting/evaluation_code/requirements-optional.txt"
"$PROJECT/.venv_posttrainbench_eval/bin/pip" install "openai>=2.26.0"
"$PROJECT/.venv_posttrainbench_eval/bin/pip" install "vllm>=0.8.0"

if [ ! -d "$INSPECT_EVALS_DIR/.git" ]; then
  rm -rf "$INSPECT_EVALS_DIR"
  git clone --depth=1 https://github.com/UKGovernmentBEIS/inspect_evals.git "$INSPECT_EVALS_DIR"
fi
"$PROJECT/.venv_posttrainbench_eval/bin/pip" install -e "$INSPECT_EVALS_DIR" --no-deps

bash "$PROJECT/scripts/verify_shared_envs.sh"
