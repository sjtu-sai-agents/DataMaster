#!/bin/bash

AGENT_TYPE=$1

if [ -z "$AGENT_TYPE" ]; then
    echo "Error: please input agent type"
    exit 1
fi


case "$AGENT_TYPE" in
    "minimal")
        python run.py \
            --agent minimal \
            --config configs/minimal/config.yaml \
            --task "今天是几号？"
        ;;

    "minimal_kaggle")
        python run.py \
            --agent minimal_kaggle \
            --config configs/minimal_kaggle/deepseek-v3.2-example.yaml \
            --task playground/minimal_kaggle/data/public/description.md
        ;;

    "data_sc1")
        python run.py \
            --agent data_scientist_ver1 \
            --config configs/data_scientist_ver1/deepseek-v3.2-example.yaml \
            --task playground/data_scientist_ver1/data/public/description.md
        ;;
    
    "ml_master")
        python run.py \
            --agent ml_master \
            --config configs/ml_master/config_simple.yaml \
            --task /data/public_data/exp_data/demo1bench/detecting-insults-in-social-commentary/prepared/public/description.md
        ;;

    "ml_master_multiturn")
        python run.py \
            --agent ml_master \
            --config configs/ml_master/config.yaml \
            --task /data/public_data/exp_data/demo1bench/detecting-insults-in-social-commentary/prepared/public/description.md
        ;;
    
    "ml_master_datatree")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/config.yaml \
            --task /data/public_data/exp_data/demo1bench/detecting-insults-in-social-commentary/prepared/public/description.md
        ;;

    "ml_master_datatree_dog")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/config_dogbreed.yaml \
            --task /data/public_data/exp_data/data_scientist_evomaster/dog-breed-identification/prepared/public/description.md
        ;;

    "ml_master_datatree_plant_pathology")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/config_plant_pathology.yaml \
            --task /data/public_data/exp_data/demo1bench/plant-pathology-2021-fgvc8/prepared/public/description.md
        ;;

    "ml_master_datatree_aptos2019")
        python run.py \
            --agent ml_master_datatree \
            --config configs/ml_master_datatree/config_aptos2019.yaml \
            --task /data/public_data/exp_data/demo1bench/aptos2019-blindness-detection/prepared/public/description.md
        ;;

    # ----------------------------------------------------------------
    # ml_master_datatree_v2: Red-Scout + Skilled-Black architecture
    # ----------------------------------------------------------------

    "ml_master_datatree_v2_plant_pathology")
        python run.py \
            --agent ml_master_datatree_v2 \
            --config configs/ml_master_datatree/config_plant_pathology_v2.yaml \
            --task /data/public_data/exp_data/data_scientist_evomaster/plant-pathology-2021-fgvc8/prepared/public/description.md
        ;;

    "ml_master_datatree_v2")
        python run.py \
            --agent ml_master_datatree_v2 \
                --config configs/ml_master_datatree/config_v2.yaml \
                --task /data/public_data/exp_data/demo1bench/detecting-insults-in-social-commentary/prepared/public/description.md
        ;;

    "math_posttrain_datatree")
        # One-time setup:
        #   bash scripts/create_llamafactory_env.sh
        #   bash scripts/create_posttrainbench_eval_env.sh
        MATH_PT_HTTP_PROXY="${http_proxy:-${HTTP_PROXY:-}}"
        MATH_PT_HTTPS_PROXY="${https_proxy:-${HTTPS_PROXY:-}}"
        case "${MATH_PT_HTTP_PROXY}" in
            127.0.0.1:*|localhost:*)
                MATH_PT_HTTP_PROXY="http://${MATH_PT_HTTP_PROXY}"
                ;;
        esac
        case "${MATH_PT_HTTPS_PROXY}" in
            127.0.0.1:*|localhost:*)
                MATH_PT_HTTPS_PROXY="http://${MATH_PT_HTTPS_PROXY}"
                ;;
        esac

        HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
        HF_HOME="${HF_HOME:-${HF_HOME}}" \
        HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}" \
        HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}" \
        HF_HUB_DISABLE_XET=1 \
        http_proxy="${MATH_PT_HTTP_PROXY}" \
        https_proxy="${MATH_PT_HTTPS_PROXY}" \
        HTTP_PROXY="${MATH_PT_HTTP_PROXY}" \
        HTTPS_PROXY="${MATH_PT_HTTPS_PROXY}" \
        python run.py \
            --agent posttrain_datatree \
            --config configs/posttrain_datatree/config_gpu3.yaml \
            --task "Collect, clean, post-train, and evaluate math data for AIME 2025."
        ;;

    *)
        echo "Error, not supported agent type '$AGENT_TYPE'"
        exit 1
        ;;
esac
