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
            --task "帮我 Verify 一下 ucberkeley-dlab/measuring-hate-speech 这个 dataset_id 的情况"
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
            --config configs/ml_master/deepseek_config.yaml \
            --task playground/ml_master/data/public/description.md
        ;;

    "ml_master_warmup")
        python run.py \
            --agent ml_master_warmup \
            --config configs/ml_master_warmup/config_simple.yaml \
            --task /data/public_data/exp_data/demo1bench/detecting-insults-in-social-commentary/prepared/public/description.md
        ;;

    "ml_master_warmup_multiturn")
        python run.py \
            --agent ml_master_warmup \
            --config configs/ml_master_warmup/config.yaml \
            --task /data/public_data/exp_data/demo1bench/detecting-insults-in-social-commentary/prepared/public/description.md
        ;;

    *)
        echo "Error, not supported agent type '$AGENT_TYPE'"
        exit 1
        ;;
esac
