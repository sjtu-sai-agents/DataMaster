#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
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
    parser = argparse.ArgumentParser(
        description=(
            "Find the best subset-eval node for a benchmark inside a run directory and rerun "
            "that checkpoint on the full PostTrainBench test set."
        )
    )
    parser.add_argument("run_dir", help="Path to the run directory.")
    parser.add_argument("benchmark", help="Benchmark id, e.g. gpqa_main, gsm8k, human_eval.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional config.yaml to use for PostTrainBench settings. If omitted, the script "
            "uses run_dir/config.yaml or a compatible sibling run config."
        ),
    )
    parser.add_argument(
        "--node-id",
        default=None,
        help="Optional explicit node id. If omitted, the script picks the highest subset score.",
    )
    parser.add_argument(
        "--output-suffix",
        default="full_eval_best",
        help="Suffix for the output directory under artifacts/evals_full/.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Optional GPU override for the full evaluation run.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _extract_score(metrics: dict) -> float | None:
    for key in ("accuracy", "acc", "exact_match", "score"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _per_benchmark_option(options: dict, key: str, benchmark: str, default=None):
    values = options.get(key)
    if isinstance(values, dict) and benchmark in values:
        return values[benchmark]
    return default


def _run_name_matches_benchmark(path: Path, benchmark: str) -> bool:
    name = path.parent.name
    return (
        name.startswith("math_posttrain_datatree_v2_")
        and f"_{benchmark}_" in f"_{name}_"
    )


def _resolve_config_path(run_dir: Path, benchmark: str, explicit_config: str | None) -> Path:
    if explicit_config:
        config_path = Path(explicit_config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Explicit config not found: {config_path}")
        return config_path

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        return config_path

    runs_root = run_dir.parent
    sibling_configs = [
        candidate
        for candidate in runs_root.glob("math_posttrain_datatree_v2_*/config.yaml")
        if _run_name_matches_benchmark(candidate, benchmark)
    ]
    if len(sibling_configs) == 1:
        return sibling_configs[0].resolve()
    if len(sibling_configs) > 1:
        return sorted(sibling_configs, key=lambda path: path.parent.name)[-1].resolve()

    raise FileNotFoundError(
        f"Run config not found: {config_path}. Pass --config /path/to/config.yaml for archived runs."
    )


def _find_best_node(run_dir: Path, benchmark: str) -> tuple[str, float, Path, dict]:
    submits_root = run_dir / "workspaces" / "task_0" / "artifacts" / "submits"
    best: tuple[str, float, Path, dict] | None = None
    for submit_path in sorted(submits_root.glob("*/best_success.json")):
        payload = json.loads(submit_path.read_text(encoding="utf-8"))
        if payload.get("benchmark") != benchmark:
            continue
        score = payload.get("score")
        if not isinstance(score, (int, float)):
            continue
        candidate = (str(payload.get("node_id") or submit_path.parent.name), float(score), submit_path, payload)
        if best is None or candidate[1] > best[1]:
            best = candidate
    if best is not None:
        return best

    eval_root = run_dir / "workspaces" / "task_0" / "artifacts" / "evals"
    metric_name = f"{benchmark}_posttrainbench_metrics.json"
    for metrics_path in sorted(eval_root.glob(f"*/{metric_name}")):
        if metrics_path.parent.name.startswith("seed_"):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = _extract_score(metrics)
        if score is None:
            continue
        candidate = (metrics_path.parent.name, score, metrics_path, {})
        if best is None or score > best[1]:
            best = candidate
    if best is None:
        raise FileNotFoundError(
            f"No trained-node subset metrics found for benchmark={benchmark} under {eval_root} or {submits_root}"
        )
    return best


def _is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    model_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).exists() for name in model_files) or any(path.glob("model-*.safetensors"))


def _is_lora_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists() and (
        (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    )


def _checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _candidate_model_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates = [root, root / "merged_model"]
    candidates.extend(sorted(root.glob("checkpoint-*"), key=_checkpoint_number, reverse=True))
    for checkpoint_dir in sorted(root.glob("checkpoint-*"), key=_checkpoint_number, reverse=True):
        candidates.append(checkpoint_dir / "merged_model")
    return candidates


def _select_eval_model_path(path: Path) -> Path | None:
    for candidate in _candidate_model_dirs(path):
        if _is_hf_model_dir(candidate):
            return candidate
    for candidate in _candidate_model_dirs(path):
        if _is_lora_adapter_dir(candidate):
            merged_model = candidate / "merged_model"
            if _is_hf_model_dir(merged_model):
                return merged_model
    return None


def _resolve_checkpoint(run_dir: Path, node_id: str, submit_payload: dict | None = None) -> Path:
    if isinstance(submit_payload, dict):
        for key in ("checkpoint_path", "train_path"):
            raw_path = submit_payload.get(key)
            if not raw_path:
                continue
            candidate = Path(raw_path)
            model_path = _select_eval_model_path(candidate)
            if model_path is not None:
                return model_path

    ckpt_dir = run_dir / "workspaces" / "task_0" / "artifacts" / "checkpoints" / node_id
    model_path = _select_eval_model_path(ckpt_dir)
    if model_path is not None:
        return model_path
    raise FileNotFoundError(f"Checkpoint directory not found for node_id={node_id}: {ckpt_dir}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = _resolve_config_path(run_dir, args.benchmark, args.config)
    if config_path.parent != run_dir:
        print(f"Using config: {config_path}")

    cfg = _load_yaml(config_path)
    posttrainbench_cfg = (((cfg.get("evaluation") or {}).get("posttrainbench")) or {}).copy()
    if not posttrainbench_cfg:
        raise ValueError(f"Missing evaluation.posttrainbench configuration in {config_path}")

    eval_root = run_dir / "workspaces" / "task_0" / "artifacts" / "evals"
    subset_score: float | None = None
    submit_payload: dict = {}
    if args.node_id is None:
        node_id, subset_score, metrics_path, submit_payload = _find_best_node(run_dir, args.benchmark)
        print(f"Selected best node: {node_id} (subset score={subset_score:.6f}) from {metrics_path}")
    else:
        node_id = args.node_id
        submit_path = run_dir / "workspaces" / "task_0" / "artifacts" / "submits" / node_id / "best_success.json"
        if submit_path.exists():
            submit_payload = json.loads(submit_path.read_text(encoding="utf-8"))
            score = submit_payload.get("score")
            subset_score = float(score) if isinstance(score, (int, float)) else None
        metric_name = f"{args.benchmark}_posttrainbench_metrics.json"
        metrics_path = eval_root / node_id / metric_name
        if subset_score is None and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            subset_score = _extract_score(metrics)
        print(f"Using explicit node: {node_id} (subset score={subset_score})")

    model_path = _resolve_checkpoint(run_dir, node_id, submit_payload)
    output_dir = run_dir / "workspaces" / "task_0" / "artifacts" / "evals_full" / f"{node_id}_{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_device = args.gpu
    if selected_device is None:
        selected_device = (posttrainbench_cfg.get("benchmark_devices") or {}).get(
            args.benchmark,
            posttrainbench_cfg.get("device"),
        )

    full_eval_score, detail = run_posttrainbench_eval(
        eval_dir=output_dir,
        benchmark_id=args.benchmark,
        model_path=model_path,
        repo_dir=posttrainbench_cfg.get("repo_dir"),
        python_bin=posttrainbench_cfg.get("python_bin"),
        limit=-1,
        max_tokens=_per_benchmark_option(
            posttrainbench_cfg,
            "benchmark_max_tokens",
            args.benchmark,
            posttrainbench_cfg.get("max_tokens"),
        ),
        max_connections=_per_benchmark_option(
            posttrainbench_cfg,
            "benchmark_max_connections",
            args.benchmark,
            posttrainbench_cfg.get("max_connections"),
        ),
        gpu_memory_utilization=posttrainbench_cfg.get("gpu_memory_utilization"),
        templates_dir=posttrainbench_cfg.get("templates_dir"),
        env_overrides=posttrainbench_cfg.get("env_overrides"),
        device=selected_device,
        auto_select_device=bool(posttrainbench_cfg.get("auto_select_device", True)) and args.gpu is None,
        max_num_seqs=_per_benchmark_option(
            posttrainbench_cfg,
            "benchmark_max_num_seqs",
            args.benchmark,
            posttrainbench_cfg.get("max_num_seqs", 32),
        ),
    )

    summary = {
        "run_dir": str(run_dir),
        "benchmark": args.benchmark,
        "selected_node_id": node_id,
        "subset_score": subset_score,
        "model_path": str(model_path),
        "selected_device": selected_device,
        "full_eval_dir": str(output_dir),
        "full_eval_score": full_eval_score,
        "backend_detail": detail,
    }
    summary_path = output_dir / "rerun_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if full_eval_score is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
