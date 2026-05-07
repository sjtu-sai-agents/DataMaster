#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_v2_eval_runner():
    package_paths = {
        "playground": REPO_ROOT / "playground",
        "playground.math_posttrain_datatree_v2": REPO_ROOT / "playground" / "math_posttrain_datatree_v2",
        "playground.math_posttrain_datatree_v2.core": REPO_ROOT / "playground" / "math_posttrain_datatree_v2" / "core",
        "playground.math_posttrain_datatree_v2.core.utils": REPO_ROOT / "playground" / "math_posttrain_datatree_v2" / "core" / "utils",
    }
    for name, path in package_paths.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    module_name = "playground.math_posttrain_datatree_v2.core.utils.eval"
    eval_path = REPO_ROOT / "playground" / "math_posttrain_datatree_v2" / "core" / "utils" / "eval.py"
    spec = importlib.util.spec_from_file_location(module_name, eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load eval runner from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.run_posttrainbench_eval


run_posttrainbench_eval = _load_v2_eval_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PostTrainBench eval for the base model.")
    parser.add_argument("benchmark", help="Benchmark id, e.g. gsm8k, human_eval, bfcl.")
    parser.add_argument(
        "--config",
        default="configs/math_posttrain_datatree_v2/config.yaml",
        help="Config file to read base_model and evaluation.posttrainbench from.",
    )
    parser.add_argument("--model-path", default=None, help="Override base model path.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id for vLLM.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override sample limit. Use -1 for full eval. Defaults to config benchmark_limits or global limit.",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Override generation token limit.")
    parser.add_argument("--max-connections", type=int, default=None, help="Override eval concurrency.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Override vLLM max_num_seqs.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="Override vLLM GPU memory fraction.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to runs/base_posttrainbench_eval/<benchmark>_<timestamp>.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _per_benchmark_option(options: dict, key: str, benchmark: str, default=None):
    values = options.get(key)
    if isinstance(values, dict) and benchmark in values:
        return values[benchmark]
    return default


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    posttrainbench_cfg = (((cfg.get("evaluation") or {}).get("posttrainbench")) or {}).copy()
    if not posttrainbench_cfg:
        raise ValueError(f"Missing evaluation.posttrainbench configuration in {config_path}")

    model_path = args.model_path or cfg.get("base_model")
    if not model_path:
        raise ValueError(f"Missing base_model in {config_path}; pass --model-path")

    selected_device = args.gpu
    if selected_device is None:
        selected_device = (posttrainbench_cfg.get("benchmark_devices") or {}).get(
            args.benchmark,
            posttrainbench_cfg.get("device"),
        )

    limit = args.limit
    if limit is None:
        limit = _per_benchmark_option(
            posttrainbench_cfg,
            "benchmark_limits",
            args.benchmark,
            posttrainbench_cfg.get("limit"),
        )
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = _per_benchmark_option(
            posttrainbench_cfg,
            "benchmark_max_tokens",
            args.benchmark,
            posttrainbench_cfg.get("max_tokens"),
        )
    max_connections = args.max_connections
    if max_connections is None:
        max_connections = _per_benchmark_option(
            posttrainbench_cfg,
            "benchmark_max_connections",
            args.benchmark,
            posttrainbench_cfg.get("max_connections"),
        )
    max_num_seqs = args.max_num_seqs
    if max_num_seqs is None:
        max_num_seqs = _per_benchmark_option(
            posttrainbench_cfg,
            "benchmark_max_num_seqs",
            args.benchmark,
            posttrainbench_cfg.get("max_num_seqs", 32),
        )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "runs" / "base_posttrainbench_eval" / f"{args.benchmark}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    score, detail = run_posttrainbench_eval(
        eval_dir=output_dir,
        benchmark_id=args.benchmark,
        model_path=model_path,
        repo_dir=posttrainbench_cfg.get("repo_dir"),
        python_bin=posttrainbench_cfg.get("python_bin"),
        limit=limit,
        max_tokens=max_tokens,
        max_connections=max_connections,
        gpu_memory_utilization=(
            args.gpu_memory_utilization
            if args.gpu_memory_utilization is not None
            else posttrainbench_cfg.get("gpu_memory_utilization")
        ),
        templates_dir=posttrainbench_cfg.get("templates_dir"),
        env_overrides=posttrainbench_cfg.get("env_overrides"),
        device=selected_device,
        auto_select_device=bool(posttrainbench_cfg.get("auto_select_device", True)) and args.gpu is None,
        max_num_seqs=max_num_seqs,
    )

    summary = {
        "benchmark": args.benchmark,
        "config_path": str(config_path),
        "model_path": str(model_path),
        "selected_device": selected_device,
        "limit": limit,
        "max_tokens": max_tokens,
        "max_connections": max_connections,
        "max_num_seqs": max_num_seqs,
        "output_dir": str(output_dir),
        "score": score,
        "backend_detail": detail,
    }
    summary_path = output_dir / "base_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if score is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
