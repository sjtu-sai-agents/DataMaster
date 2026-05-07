#!/bin/bash
set -uo pipefail

TIME_LIMIT_SECS=43200
KILL_AFTER="5m"

AGENT_TYPE=${1:-}

if [ -z "$AGENT_TYPE" ]; then
    echo "Error: please input agent type"
    exit 1
fi

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

echo "[start] rule-black task=${AGENT_TYPE}, limit=${TIME_LIMIT_SECS}s"

timeout -k "$KILL_AFTER" "${TIME_LIMIT_SECS}s" bash -c '
set -uo pipefail

python test/test_device.py

AGENT_TYPE="$1"
CONFIG="configs/ml_master_datatree/yaml_configs/${AGENT_TYPE}/config_${AGENT_TYPE}_rule_black.yaml"
TASK="${DATA_ROOT}/${AGENT_TYPE}/prepared/public/description.md"
INITIAL_CODE="${PROJECT_ROOT}/initial_code/data_loader_format/${AGENT_TYPE}/algo.py"

get_force_flag() {
    local task="$1"

    case "$task" in
        "aerial-cactus-identification" \
        | "aptos2019-blindness-detection" \
        | "mlsp-2013-birds" \
        | "plant-pathology-2020-fgvc7" \
        | "ranzcr-clip-catheter-line-classification" \
        | "detecting-insults-in-social-commentary" \
        | "histopathologic-cancer-detection" \
        | "jigsaw-toxic-comment-classification-challenge" \
        | "random-acts-of-pizza" \
        | "text-normalization-challenge-english-language" \
        | "siim-isic-melanoma-classification" \
        | "tabular-playground-series-dec-2021" \
        | "tabular-playground-series-may-2022" \
        | "tabular-playground-series-dec-2021-v2" \
        | "tabular-playground-series-may-2022-v2" \
        | "text-normalization-challenge-russian-language" \
        | "the-icml-2013-whale-challenge-right-whale-redux")
            echo "--force-maximize"
            ;;

        "new-york-city-taxi-fare-prediction" \
        | "spooky-author-identification" \
        | "dog-breed-identification" \
        | "dogs-vs-cats-redux-kernels-edition" \
        | "nomad2018-predict-transparent-conductors" \
        | "denoising-dirty-documents" \
        | "leaf-classification")
            echo "--force-minimize"
            ;;

        *)
            echo ""
            ;;
    esac
}

FORCE_FLAG="$(get_force_flag "$AGENT_TYPE")"

if [ -z "$FORCE_FLAG" ]; then
    echo "Error: unknown metric direction for task ${AGENT_TYPE}"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found: $CONFIG"
    exit 1
fi

if [ ! -f "$TASK" ]; then
    echo "Error: task description not found: $TASK"
    exit 1
fi

if [ ! -f "$INITIAL_CODE" ]; then
    echo "Error: initial code not found: $INITIAL_CODE"
    exit 1
fi

python run.py \
    --agent ml_master_datatree \
    --config "$CONFIG" \
    --task "$TASK" \
    --initial-code "$INITIAL_CODE" \
    --initial-instruction "Attention! You are allowed to finetune the hyperparameters of the given initial node, the core mission for you is to select a scalable algorithm that can gain better performance when getting more augmented data. You can proceed with the data preprocessing methods within the initial code which has been proved right" \
    --test-feedback \
    $FORCE_FLAG
' _ "$AGENT_TYPE"

status=$?

if [ "$status" -eq 124 ]; then
    echo "[timeout] rule-black task ${AGENT_TYPE} timed out after 12h"
    send_feishu "EvoMaster rule-black task ${AGENT_TYPE} timed out after 12h"
    exit 124
elif [ "$status" -eq 137 ]; then
    echo "[killed] rule-black task ${AGENT_TYPE} was killed with exit code 137"
    send_feishu "EvoMaster rule-black task ${AGENT_TYPE} was killed with exit code 137, likely OOM/system kill"
    exit 137
elif [ "$status" -eq 0 ]; then
    echo "[done] rule-black task ${AGENT_TYPE} finished successfully"
    send_feishu "EvoMaster rule-black task ${AGENT_TYPE} finished successfully"
    exit 0
else
    echo "[failed] rule-black task ${AGENT_TYPE} failed with exit code ${status}"
    send_feishu "EvoMaster rule-black task ${AGENT_TYPE} failed with exit code ${status}"
    exit "$status"
fi