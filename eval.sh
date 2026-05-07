#!/bin/bash

EXP_NAME="ml_master"
# SUBMISSION_DIR="runs/ml_master_20260226_095204"
SUBMISSION_DIR="runs/ml_master_20260226_095204/workspaces/task_0/submission"
FLAG=4

case $FLAG in
    1)
        python eval.py --exp_name $EXP_NAME
        ;;
    2)
        python eval.py --exp_name $EXP_NAME --submission_dir $SUBMISSION_DIR
        ;;
    3)
        python eval.py --exp_name $EXP_NAME --batch --save
        ;;
    4)
        python eval.py --exp_name $EXP_NAME --submission_dir $SUBMISSION_DIR --batch --save
        ;;
    *)
        echo "Error, no matching"
        exit 1
        ;;
esac