#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <benchmark> <gpu> [extra launch args...]"
    echo "Example: $0 bfcl 3 --num-black 3 --max-rounds 18"
    exit 1
fi

BENCHMARK="$1"
GPU="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python "$REPO_ROOT/scripts/launch_math_posttrain_benchmarks_v2.py"   --base-config configs/math_posttrain_datatree_v2/config_gpu2.yaml   --benchmark "$BENCHMARK"   --gpu "$GPU"   "$@"
