#!/bin/bash
set -uo pipefail

AGENT_TYPE=$1

if [ -z "${AGENT_TYPE:-}" ]; then
    echo "Error: please input agent type"
    exit 1
fi

TIME_LIMIT_SECS=43200   # 12h
KILL_AFTER="5m"

# 飞书机器人 webhook，从环境变量里读
FEISHU_WEBHOOK="${FEISHU_WEBHOOK:-}"

send_feishu() {
    local msg="$1"
    if [ -z "$FEISHU_WEBHOOK" ]; then
        return 0
    fi

    curl -s -X POST "$FEISHU_WEBHOOK" \
      -H 'Content-Type: application/json' \
      -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$msg\"}}" >/dev/null 2>&1 || true
}

echo "[start] task=${AGENT_TYPE}, limit=${TIME_LIMIT_SECS}s"

timeout -k "$KILL_AFTER" ${TIME_LIMIT_SECS}s bash run_mle_maintable.sh "$AGENT_TYPE"
status=$?

if [ "$status" -eq 124 ]; then
    echo "[timeout] task ${AGENT_TYPE} timed out after 12h"
    send_feishu "EvoMaster task ${AGENT_TYPE} timed out after 12h"
    exit 124
elif [ "$status" -eq 0 ]; then
    echo "[done] task ${AGENT_TYPE} finished successfully"
    send_feishu "EvoMaster task ${AGENT_TYPE} finished successfully"
    exit 0
else
    echo "[failed] task ${AGENT_TYPE} failed with exit code ${status}"
    send_feishu "EvoMaster task ${AGENT_TYPE} failed with exit code ${status}"
    exit "$status"
fi