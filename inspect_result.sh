#!/bin/bash

source .venv/bin/activate
which python

# 检查参数数量
if [ $# -lt 1 ]; then
    echo "Usage: $0 <RUN_DIR> [METRIC_DIRECTION]"
    echo ""
    echo "Arguments:"
    echo "  RUN_DIR            - Path to the run directory"
    echo "  METRIC_DIRECTION   - (Optional) Must be either 'lower' or 'higher'"
    echo "                      If not provided, will auto-detect from result.csv"
    echo ""
    echo "Examples:"
    echo "  $0 runs/ml_master_datatree_20260412_164926"
    echo "  $0 runs/ml_master_datatree_20260412_164926 lower"
    exit 1
fi

RUN_DIR=$1
METRIC_DIRECTION=$2

# 如果没有提供评测方向，自动从 result.csv 获取
if [ -z "$METRIC_DIRECTION" ]; then
    METRIC_DIRECTION=$(python get_metric_direction.py "$RUN_DIR")

    if [ $? -ne 0 ]; then
        echo "Error: Failed to auto-detect metric direction"
        echo "Please provide METRIC_DIRECTION manually (lower/higher)"
        exit 1
    fi
fi

# 标准化输入为小写
METRIC_DIRECTION=$(echo "$METRIC_DIRECTION" | tr '[:upper:]' '[:lower:]')

# 验证参数值
if [ "$METRIC_DIRECTION" != "lower" ] && [ "$METRIC_DIRECTION" != "higher" ]; then
    echo "Error: Invalid METRIC_DIRECTION '$METRIC_DIRECTION'"
    echo "Must be either 'lower' or 'higher'"
    exit 1
fi

# 根据 metric direction 设置 tree_analysis 参数
TREE_ANALYSIS_ARGS="--run-dir $RUN_DIR"
if [ "$METRIC_DIRECTION" = "lower" ]; then
    TREE_ANALYSIS_ARGS="$TREE_ANALYSIS_ARGS --force-minimize"
    echo "Info: Metric direction set to LOWER (smaller is better)"
else
    TREE_ANALYSIS_ARGS="$TREE_ANALYSIS_ARGS --force-maximize"
    echo "Info: Metric direction set to HIGHER (larger is better)"
fi

# run inspect
echo ""
echo "========================================"
echo "Running Grading Scripts"
echo "========================================"
python vis_node_by_tree_with_grade.py --run-dir $RUN_DIR

echo ""
echo "========================================"
echo "Running Analysis"
echo "========================================"
echo "Command: python test/tree_analysis.py $TREE_ANALYSIS_ARGS"
python test/tree_analysis.py $TREE_ANALYSIS_ARGS

echo ""
echo "Running tool inspection"
python test/tool_inspection.py --run-dir $RUN_DIR
python test/count_tool_visualize.py --run-dir $RUN_DIR

echo ""
echo "========================================"
echo "All analyses completed!"
echo "========================================"

