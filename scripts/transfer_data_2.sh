#!/bin/bash
# rsync -avzP --info=progress2 -e "ssh" root@Voc_gpu_node2_ubuntu:/data/public_data/exp_data/data_scientist_evomaster_v2/ ${WORKSPACE_ROOT}/datamaster_mlebench/

rsync -avzP --info=progress2 -e "ssh" xiyuan@dp_gpu_4090_1:/data/exp_data/detecting-insults-in-social-commentary ${WORKSPACE_ROOT}/datamaster_mlebench/
${WORKSPACE_ROOT}/datamaster_mlebench/