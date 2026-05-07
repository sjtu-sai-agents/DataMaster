#!/bin/bash

cd ${PROJECT_ROOT}/mle-bench/environment

export COMPETITION_ID="dog-breed-identification"
export PRIVATE_DATA_DIR="/data/public_data/exp_data/demo1bench"

${PROJECT_ROOT}/.venv/bin/python grading_server.py

cd -