#!/usr/bin/env bash

set -euo pipefail

ENV_DIR="${1:-./.venv_llamafactory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LLAMAFACTORY_SPEC="${LLAMAFACTORY_SPEC:-llamafactory}"
HF_INDEX_URL="${HF_INDEX_URL:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0}"
EXTRA_COMPAT_PACKAGES="${EXTRA_COMPAT_PACKAGES:-pillow==11.3.0 fsspec==2025.3.0}"
TMPDIR="${TMPDIR:-/data/tmp}"
TEMP="${TEMP:-${TMPDIR}}"
TMP="${TMP:-${TMPDIR}}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/pip_cache}"

echo "Creating LLaMA Factory environment at: ${ENV_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
export TMPDIR TEMP TMP PIP_CACHE_DIR

"${PYTHON_BIN}" -m venv "${ENV_DIR}"

VENV_PYTHON="${ENV_DIR}/bin/python"
VENV_PIP="${ENV_DIR}/bin/pip"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ -n "${HF_INDEX_URL}" ]]; then
  "${VENV_PIP}" install --index-url "${HF_INDEX_URL}" "${LLAMAFACTORY_SPEC}"
else
  "${VENV_PIP}" install "${LLAMAFACTORY_SPEC}"
fi

echo "Pinning torch packages for the local GPU driver via ${TORCH_INDEX_URL}"
"${VENV_PIP}" install --force-reinstall --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES}
echo "Restoring package versions compatible with gradio/datasets"
"${VENV_PIP}" install --force-reinstall ${EXTRA_COMPAT_PACKAGES}

cat <<EOF

LLaMA Factory environment is ready.

Environment:
  ${ENV_DIR}

Quick checks:
  ${ENV_DIR}/bin/python -m llamafactory.cli --help
  ${ENV_DIR}/bin/python test/test_real_llamafactory_smoke.py --lf-env-dir ${ENV_DIR}

Temp/cache:
  TMPDIR=${TMPDIR}
  PIP_CACHE_DIR=${PIP_CACHE_DIR}

Torch pin:
  TORCH_INDEX_URL=${TORCH_INDEX_URL}
  TORCH_PACKAGES=${TORCH_PACKAGES}
  EXTRA_COMPAT_PACKAGES=${EXTRA_COMPAT_PACKAGES}

Suggested config:
  llama_factory_env:
    env_dir: "${ENV_DIR}"
    python_bin: null
EOF
