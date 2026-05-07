from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playground.math_posttrain_datatree.core.utils.io import write_jsonl
from playground.math_posttrain_datatree.core.utils.llama_factory import (
    ensure_llamafactory_available,
    run_llama_factory_sft,
    validate_alpaca_dataset,
)


def _build_dataset(dataset_path: Path) -> Path:
    rows = [
        {
            "instruction": "Solve the following math problem carefully and end with the final answer explicitly.",
            "input": "What is 1+1?",
            "output": "We compute 1+1=2.\n\nFinal answer: 2",
        },
        {
            "instruction": "Solve the following math problem carefully and end with the final answer explicitly.",
            "input": "What is 3+4?",
            "output": "3+4=7.\n\nFinal answer: 7",
        },
        {
            "instruction": "Solve the following math problem carefully and end with the final answer explicitly.",
            "input": "What is 6-2?",
            "output": "6-2=4.\n\nFinal answer: 4",
        },
        {
            "instruction": "Solve the following math problem carefully and end with the final answer explicitly.",
            "input": "What is 3*3?",
            "output": "3*3=9.\n\nFinal answer: 9",
        },
    ]
    return write_jsonl(dataset_path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real LLaMA Factory smoke test.")
    parser.add_argument("--output-dir", default="playground/math_posttrain_datatree/workspace/smoke_test")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lf-env-dir", default=None)
    parser.add_argument("--lf-python-bin", default=None)
    args = parser.parse_args()

    ok, detail = ensure_llamafactory_available(
        python_bin=args.lf_python_bin,
        env_dir=args.lf_env_dir,
    )
    if not ok:
        print(detail, file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _build_dataset(output_dir / "alpaca_smoke.jsonl")
    valid, msg = validate_alpaca_dataset(dataset_path)
    if not valid:
        print(msg, file=sys.stderr)
        return 3

    result = run_llama_factory_sft(
        dataset_path=dataset_path,
        recipe_path=output_dir / "recipe.json",
        output_dir=output_dir / "run",
        base_model=args.base_model,
        overrides={
            "num_train_epochs": args.epochs,
            "learning_rate": args.lr,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "cutoff_len": 256,
            "max_samples": 4,
        },
        dry_run=False,
        env_overrides={
            "CUDA_VISIBLE_DEVICES": "0",
            "NPROC_PER_NODE": "1",
            "OMP_NUM_THREADS": "1",
        },
        python_bin=args.lf_python_bin,
        env_dir=args.lf_env_dir,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
