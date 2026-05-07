# 环境设置

如果以后环境坏了，直接重跑：

bash ${WORKSPACE_ROOT}/DataScientistEvomaster2/scripts/setup_shared_envs.sh

只做验收则跑：

bash ${WORKSPACE_ROOT}/DataScientistEvomaster2/scripts/verify_shared_envs.sh

# 实验运行

## AIME
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2

bash scripts/run_math_posttrain_benchmark.sh aime_2025 5 \
  --config-suffix manual \
  --red-max-turns 50 --max-rounds 100 --num-black 3

bash scripts/run_math_posttrain_benchmark.sh aime_2025 0 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 4096 --train-template qwen3

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
bash scripts/run_math_posttrain_benchmark_v2.sh aime_2025 1 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 8192 --train-template qwen3

bash scripts/run_math_posttrain_benchmark_v3.sh aime_2025 2 \
    --config-suffix manual \
    --red-max-turns 50 \
    --max-rounds 100 \
    --num-black 3 \
    --cutoff-len 8192 \
    --train-template qwen3

## BFCL
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2

bash scripts/run_math_posttrain_benchmark.sh bfcl 7 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
bash scripts/run_math_posttrain_benchmark_v2.sh bfcl 3 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 4096 --train-template qwen3

## Arena Hard

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2

bash scripts/run_math_posttrain_benchmark.sh arena_hard_writing 6 \ --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
bash scripts/run_math_posttrain_benchmark_v2.sh arena_hard_writing 4 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 4096 --train-template qwen3

## HealthBench

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2

bash scripts/run_math_posttrain_benchmark.sh healthbench_easy 1 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3  --cutoff-len 4096 --train-template qwen3

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
bash scripts/run_math_posttrain_benchmark_v2.sh healthbench_easy 5 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 8192 --train-template qwen3

## HumanEval

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2

bash scripts/run_math_posttrain_benchmark.sh human_eval 2 \
  --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
bash scripts/run_math_posttrain_benchmark_v2.sh human_eval 6 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 2048 --train-template qwen3

## GSM8K

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2

bash scripts/run_math_posttrain_benchmark.sh gsm8k 3 \
  --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
bash scripts/run_math_posttrain_benchmark_v2.sh gsm8k 7 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 2048 --train-template qwen3

## GPQA
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2

bash scripts/run_math_posttrain_benchmark.sh gpqa_main 4 \
  --config-suffix manual --max-rounds 150 --num-black 3

cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
bash scripts/run_math_posttrain_benchmark_v2.sh gpqa_main 2 --config-suffix manual --red-max-turns 50 --max-rounds 100 --num-black 3 --cutoff-len 4096 --train-template qwen3

# 实验终止

pgrep -af "math_posttrain_datatree_aime_2025_gpu5_manual_20260417_172440"

ps aux | grep "run.py.*math_posttrain_datatree" | grep -v grep
kill -9 <PID>

# 实验评测

有一些benchmark不是全量的评测

## 取全树最高的节点分数
### gsm8k
```
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
source .venv_posttrainbench_eval/bin/activate

python scripts/rerun_best_posttrainbench_full_eval.py \
  ${WORKSPACE_ROOT}/DataScientistEvomaster2/runs/math_posttrain_datatree_v2_gsm8k_gpu7_manual_20260429_220033 \
  gsm8k \
  --gpu 1
```

### GPQA
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
source .venv_posttrainbench_eval/bin/activate
python scripts/rerun_best_posttrainbench_full_eval.py ${WORKSPACE_ROOT}/DataScientistEvomaster2/runs/math_posttrain_datatree_v2_gpqa_main_gpu3_manual_20260429_001415 gpqa_main --gpu 3

### human_eval
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
source .venv_posttrainbench_eval/bin/activate
python scripts/rerun_best_posttrainbench_full_eval.py \
 ${WORKSPACE_ROOT}/DataScientistEvomaster2/runs/math_posttrain_datatree_v2_human_eval_gpu6_manual_20260429_220031 \
  human_eval \
  --gpu 3

### healthbench_easy
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
source .venv_posttrainbench_eval/bin/activate
python scripts/rerun_best_posttrainbench_full_eval.py \
  ${WORKSPACE_ROOT}/DataScientistEvomaster2/runs/math_posttrain_datatree_v2_healthbench_easy_gpu5_manual_20260429_220028 \
  healthbench_easy \
  --gpu 7

### arena_hard
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
source .venv_posttrainbench_eval/bin/activate
python scripts/rerun_best_posttrainbench_full_eval.py  ${WORKSPACE_ROOT}/DataScientistEvomaster2/runs/math_posttrain_datatree_v2_arena_hard_writing_gpu4_manual_20260429_220026 arena_hard_writing --gpu 1

### bfcl
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2
source .venv_posttrainbench_eval/bin/activate
python scripts/rerun_best_posttrainbench_full_eval.py ${WORKSPACE_ROOT}/DataScientistEvomaster2/runs/math_posttrain_datatree_v2_bfcl_gpu1_manual_20260429_220053 bfcl --gpu 4

现在行为是：

如果你不传 --gpu，就沿用原 run 的 config.yaml 里的设备
如果你传了 --gpu，就强制用你指定的卡
输出的 rerun_summary.json 里也会记录 selected_device

## 指定某节点的分数
如果你想指定节点，不让它自动选最高分：

python scripts/rerun_best_posttrainbench_full_eval.py \
  ${WORKSPACE_ROOT}/DataScientistEvomaster2/runs/math_posttrain_datatree_gsm8k_20260413_020122 \
  gsm8k \
  --node-id 41ccb09e98374be996c69623d5c5f472


# 提前保存cache
./.venv_gpu2/bin/python scripts/extract_red_node_dataset_cache.py \
  runs/math_posttrain_datatree_aime_2025_gpu4_manual_20260415_175801 \
  --all-configs


# 设置梯子
启动梯子
 cd ${WORKSPACE_ROOT}/clash
 ./start_mihomo.sh

启动端口
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891

# 查看运行情况
apt update
apt install -y nvtop

nvtop
