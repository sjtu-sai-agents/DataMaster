#!/bin/bash

RUN_DIR=$1

python vis_node.py \
    --run-dir $RUN_DIR \
    --host 127.0.0.1 --port 8787 --refresh-seconds 60