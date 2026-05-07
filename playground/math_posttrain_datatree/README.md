# Math Post-Train DataTree

`math_posttrain_datatree` is a math-focused post-training playground built on EvoMaster.

It keeps the search space on data strategy:

- `seed` initializes the run context and fixed training recipe
- `red` searches and records public math datasets
- `black` cleans data, exports LLaMA Factory Alpaca data, and triggers fixed train/eval/inspect runners

Training, evaluation, and inspection are deterministic runners rather than free-form agent nodes.

Useful commands:

- Initialize the bundled `PostTrainBench` submodule after clone:
  - `git submodule update --init --recursive`

- Create a dedicated LLaMA Factory env:
  - `bash scripts/create_llamafactory_env.sh`
- Create a dedicated PostTrainBench eval env:
  - `bash scripts/create_posttrainbench_eval_env.sh`
- Run the real LLaMA Factory smoke test against that env:
  - `./.venv/bin/python test/test_real_llamafactory_smoke.py --lf-env-dir ./.venv_llamafactory`
- Run the AIME evaluation smoke test:
  - `./.venv/bin/python test/test_real_aime_eval_smoke.py`
- Run the AIME evaluation smoke test on your own files:
  - `./.venv/bin/python test/test_real_aime_eval_smoke.py --workspace playground/math_posttrain_datatree/workspace/aime_eval --benchmark-file /path/to/aime.jsonl --prediction-file /path/to/preds.jsonl`

Config snippet for real benchmark evaluation:

```yaml
evaluation:
  backend: "builtin"
  benchmark_files:
    aime_2025: "/abs/path/to/aime_2025.jsonl"
    arena_hard_writing: null
    bfcl: null
    gpqa_main: null
    gsm8k: null
    healthbench_easy: null
    human_eval: null
  prediction_files:
    aime_2025: "/abs/path/to/aime_2025_predictions.jsonl"
    arena_hard_writing: null
    bfcl: null
    gpqa_main: null
    gsm8k: null
    healthbench_easy: null
    human_eval: null
```

When these paths are set, the fixed eval runner in `black` will use them automatically instead of proxy scoring.

Config snippet for PostTrainBench-backed AIME evaluation:

```yaml
evaluation:
  backend: "posttrainbench"
  benchmark_files:
    aime_2025: null
    arena_hard_writing: null
    bfcl: null
    gpqa_main: null
    gsm8k: null
    healthbench_easy: null
    human_eval: null
  prediction_files:
    aime_2025: null
    arena_hard_writing: null
    bfcl: null
    gpqa_main: null
    gsm8k: null
    healthbench_easy: null
    human_eval: null
  posttrainbench:
    repo_dir: "/abs/path/to/PostTrainBench"
    python_bin: null
    templates_dir: null
    limit: 32
    max_tokens: 16000
    max_connections: 6
    gpu_memory_utilization: 0.8
```

This mode supports `aime_2025`, `arena_hard_writing`, `bfcl`, `gpqa_main`, `gsm8k`, `healthbench_easy`, and `human_eval` by invoking the corresponding `src/eval/tasks/<task>/evaluate.py` entrypoints from the PostTrainBench repo and scoring the trained checkpoint directly.

Recommended setup for this backend:

1. `bash scripts/create_posttrainbench_eval_env.sh`
2. Keep `evaluation.posttrainbench.python_bin` pointed at `./.venv_posttrainbench_eval/bin/python`
3. Keep `evaluation.posttrainbench.templates_dir` pointed at `./playground/math_posttrain_datatree/external/PostTrainBench/src/eval/templates`
