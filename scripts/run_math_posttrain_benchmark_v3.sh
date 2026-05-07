#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <benchmark> <gpu> [extra launch args...]"
    echo "Example: $0 aime_2025 0 --config-suffix manual --num-black 3 --max-rounds 100"
    exit 1
fi

BENCHMARK="$1"
GPU="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python "$REPO_ROOT/scripts/launch_math_posttrain_benchmarks_v3.py" \
  --base-config configs/math_posttrain_datatree_v3/config_gpu2.yaml \
  --benchmark "$BENCHMARK" \
  --gpu "$GPU" \
  "$@"
