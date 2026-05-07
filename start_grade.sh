#!/bin/bash
# 启动 Grade HTTP 服务器（端口 7777）

# 设置默认值
DATA_ROOT=${ML_MASTER_DATA_ROOT:-"${DATA_ROOT}"}
HOST=${GRADE_SERVER_HOST:-"127.0.0.1"}
PORT=${GRADE_SERVER_PORT:-7777}
WORKERS=${GRADE_SERVER_WORKERS:-1}

echo "Starting Grade HTTP Server..."
echo "  Data root: $DATA_ROOT"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo ""

# 启动服务器
python initialize_grade_port.py \
    --data-root "$DATA_ROOT" \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS"