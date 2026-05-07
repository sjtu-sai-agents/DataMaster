#!/usr/bin/env bash

set -euo pipefail

ENV_DIR="${1:-./.venv_posttrainbench_eval}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POSTTRAINBENCH_DIR="${POSTTRAINBENCH_DIR:-./playground/math_posttrain_datatree/external/PostTrainBench}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.11.0}"
OPENAI_SPEC="${OPENAI_SPEC:-openai>=2.26.0}"
HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC:-huggingface_hub>=0.34.0,<1.0}"
INSPECT_EVALS_REPO="${INSPECT_EVALS_REPO:-https://github.com/UKGovernmentBEIS/inspect_evals.git}"
INSPECT_EVALS_MIRROR_REPO="${INSPECT_EVALS_MIRROR_REPO:-}"
INSPECT_EVALS_PIP_SPEC="${INSPECT_EVALS_PIP_SPEC:-inspect_evals}"
INSPECT_EVALS_ZIP_URL="${INSPECT_EVALS_ZIP_URL:-https://github.com/UKGovernmentBEIS/inspect_evals/archive/refs/heads/main.zip}"
TMPDIR="${TMPDIR:-/data/tmp}"
TEMP="${TEMP:-${TMPDIR}}"
TMP="${TMP:-${TMPDIR}}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/pip_cache}"
GIT_CLONE_RETRIES="${GIT_CLONE_RETRIES:-3}"

echo "Creating PostTrainBench eval environment at: ${ENV_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${POSTTRAINBENCH_DIR}" ]]; then
  echo "PostTrainBench repo not found: ${POSTTRAINBENCH_DIR}" >&2
  echo "Clone it first or set POSTTRAINBENCH_DIR." >&2
  exit 1
fi

REQ_FILE="${POSTTRAINBENCH_DIR}/containers/requirements-direct.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
export TMPDIR TEMP TMP PIP_CACHE_DIR

clone_repo() {
  local repo_url="$1"
  local target_dir="$2"
  local attempt=1
  while [[ "${attempt}" -le "${GIT_CLONE_RETRIES}" ]]; do
    if git -c http.version=HTTP/1.1 clone --depth 1 "${repo_url}" "${target_dir}"; then
      return 0
    fi
    echo "Clone failed for ${repo_url} (attempt ${attempt}/${GIT_CLONE_RETRIES}), retrying..." >&2
    rm -rf "${target_dir}"
    attempt=$((attempt + 1))
    sleep 2
  done
  return 1
}

"${PYTHON_BIN}" -m venv "${ENV_DIR}"

VENV_PYTHON="${ENV_DIR}/bin/python"
VENV_PIP="${ENV_DIR}/bin/pip"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_VLLM}" == "1" ]]; then
  "${VENV_PIP}" install "${VLLM_SPEC}"
fi

"${VENV_PIP}" install -r "${REQ_FILE}"
"${VENV_PIP}" install --upgrade "${OPENAI_SPEC}"

INSPECT_EVALS_SRC="${ENV_DIR}/src/inspect_evals"
rm -rf "${INSPECT_EVALS_SRC}"
if ! clone_repo "${INSPECT_EVALS_REPO}" "${INSPECT_EVALS_SRC}"; then
  if [[ -n "${INSPECT_EVALS_MIRROR_REPO}" ]]; then
    echo "Falling back to mirror repo: ${INSPECT_EVALS_MIRROR_REPO}" >&2
    if ! clone_repo "${INSPECT_EVALS_MIRROR_REPO}" "${INSPECT_EVALS_SRC}"; then
      echo "Mirror clone also failed. Falling back to pip package: ${INSPECT_EVALS_PIP_SPEC}" >&2
      "${VENV_PIP}" install "${INSPECT_EVALS_PIP_SPEC}" || "${VENV_PIP}" install "${INSPECT_EVALS_ZIP_URL}"
      INSPECT_EVALS_SRC=""
    fi
  else
    echo "Failed to clone inspect_evals from ${INSPECT_EVALS_REPO}. Falling back to pip package: ${INSPECT_EVALS_PIP_SPEC}" >&2
    "${VENV_PIP}" install "${INSPECT_EVALS_PIP_SPEC}" || "${VENV_PIP}" install "${INSPECT_EVALS_ZIP_URL}"
    INSPECT_EVALS_SRC=""
  fi
fi
if [[ -n "${INSPECT_EVALS_SRC}" ]]; then
  "${VENV_PIP}" install "${INSPECT_EVALS_SRC}"
fi

# Keep the PTB eval env compatible with transformers/vLLM.
"${VENV_PIP}" install --upgrade "${HUGGINGFACE_HUB_SPEC}"

cat <<EOF

PostTrainBench evaluation environment is ready.

Environment:
  ${ENV_DIR}

Quick checks:
  ${ENV_DIR}/bin/python ${POSTTRAINBENCH_DIR}/src/eval/tasks/aime2025/evaluate.py --help
  ./.venv/bin/python test/test_real_aime_eval_smoke.py

Temp/cache:
  TMPDIR=${TMPDIR}
  PIP_CACHE_DIR=${PIP_CACHE_DIR}
  GIT_CLONE_RETRIES=${GIT_CLONE_RETRIES}
  INSPECT_EVALS_PIP_SPEC=${INSPECT_EVALS_PIP_SPEC}
  OPENAI_SPEC=${OPENAI_SPEC}
  HUGGINGFACE_HUB_SPEC=${HUGGINGFACE_HUB_SPEC}

Suggested config:
  evaluation:
    backend: "posttrainbench"
    posttrainbench:
      repo_dir: "${POSTTRAINBENCH_DIR}"
      python_bin: "${ENV_DIR}/bin/python"
      templates_dir: "${POSTTRAINBENCH_DIR}/src/eval/templates"
EOF
