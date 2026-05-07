#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

DEFAULT_BENCHMARK_DEVICE_MAP = {
    "aime_2025": 0,
    "arena_hard_writing": 1,
    "healthbench_easy": 2,
    "bfcl": 3,
    "gpqa_main": 5,
    "human_eval": 6,
    "gsm8k": 7,
}

VALID_BENCHMARKS = tuple(DEFAULT_BENCHMARK_DEVICE_MAP.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch math_posttrain_datatree runs. Use single-run mode with --benchmark/--gpu, "
            "or batch mode with the default benchmark->GPU map."
        )
    )
    parser.add_argument(
        "--base-config",
        default="configs/math_posttrain_datatree/config_gpu2.yaml",
        help="Base config path relative to repo root.",
    )
    parser.add_argument(
        "--benchmark",
        choices=VALID_BENCHMARKS,
        help="Launch a single benchmark run.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="GPU index for single-benchmark mode.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Task text to pass to run.py. Defaults to a benchmark-specific prompt in single-run mode, "
            "or the historical AIME prompt in batch mode."
        ),
    )
    parser.add_argument(
        "--config-suffix",
        default="",
        help="Optional human-readable suffix appended to generated config/run names.",
    )
    parser.add_argument(
        "--num-black",
        type=int,
        default=None,
        help="Optional override for num_black.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional override for max_rounds.",
    )
    parser.add_argument(
        "--red-max-turns",
        type=int,
        default=None,
        help="Optional override for agents.red.max_turns.",
    )
    parser.add_argument(
        "--black-max-turns",
        type=int,
        default=None,
        help="Optional override for agents.black.max_turns.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override evaluation.posttrainbench.max_tokens (generation token limit during eval).",
    )
    parser.add_argument(
        "--cutoff-len",
        type=int,
        default=None,
        help="Override training cutoff_len (max training sequence length).",
    )
    parser.add_argument(
        "--train-template",
        type=str,
        default=None,
        help="Override LLaMA Factory training template (e.g., 'qwen3', 'llama3').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands and generated config paths.",
    )
    return parser.parse_args()


def resolve_single_run_request(args: argparse.Namespace) -> tuple[str | None, int | None]:
    benchmark = args.benchmark or os.environ.get("MATH_PT_BENCHMARK")
    gpu_value = args.gpu
    if gpu_value is None and os.environ.get("MATH_PT_GPU"):
        gpu_value = int(os.environ["MATH_PT_GPU"])

    if (benchmark is None) != (gpu_value is None):
        raise SystemExit("Provide both --benchmark and --gpu together for single-run mode.")

    return benchmark, gpu_value


def build_task_text(benchmark: str, explicit_task: str | None) -> str:
    if explicit_task:
        return explicit_task
    return f"Collect, clean, post-train, and evaluate task-aligned data for {benchmark}."


def build_config(
    base_cfg: dict,
    benchmark: str,
    device: int,
    repo_root: Path,
    *,
    num_black: int | None = None,
    max_rounds: int | None = None,
    red_max_turns: int | None = None,
    black_max_turns: int | None = None,
    max_tokens: int | None = None,
    cutoff_len: int | None = None,
    train_template: str | None = None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["benchmark_suite"] = [benchmark]

    if num_black is not None:
        cfg["num_black"] = num_black
    if max_rounds is not None:
        cfg["max_rounds"] = max_rounds

    def abs_repo(path: str | None) -> str | None:
        if not path:
            return path
        candidate = Path(path)
        if candidate.is_absolute():
            return str(candidate)
        raw = str(path)
        if raw.startswith("prompts/"):
            return str(repo_root / "playground" / "math_posttrain_datatree" / raw)
        if raw.startswith("mcp_config"):
            return str(repo_root / "configs" / "math_posttrain_datatree" / raw)
        if raw.startswith("./"):
            return str(repo_root / raw[2:])
        return str(repo_root / candidate)

    cfg["system_prompt_file"] = abs_repo(cfg.get("system_prompt_file"))
    mcp_cfg = cfg.setdefault("mcp", {})
    mcp_cfg["config_file"] = abs_repo(mcp_cfg.get("config_file"))

    for agent_name in ("red", "black"):
        agent_cfg = cfg.setdefault("agents", {}).setdefault(agent_name, {})
        agent_cfg["system_prompt_file"] = abs_repo(agent_cfg.get("system_prompt_file"))
        agent_cfg["user_prompt_file"] = abs_repo(agent_cfg.get("user_prompt_file"))
        if agent_name == "red" and red_max_turns is not None:
            agent_cfg["max_turns"] = red_max_turns
        if agent_name == "black" and black_max_turns is not None:
            agent_cfg["max_turns"] = black_max_turns
        tools = agent_cfg.setdefault("tools", {})
        tools["mcp"] = abs_repo(tools.get("mcp"))

    lf_env = cfg.setdefault("llama_factory_env", {})
    lf_env["cuda_visible_devices"] = str(device)
    lf_env["nproc_per_node"] = 1
    lf_env["env_dir"] = abs_repo(lf_env.get("env_dir"))
    lf_env["python_bin"] = abs_repo(lf_env.get("python_bin")) if lf_env.get("python_bin") else None

    evaluation = cfg.setdefault("evaluation", {})
    posttrainbench = evaluation.setdefault("posttrainbench", {})
    posttrainbench["repo_dir"] = abs_repo(posttrainbench.get("repo_dir"))
    posttrainbench["python_bin"] = abs_repo(posttrainbench.get("python_bin"))
    posttrainbench["templates_dir"] = abs_repo(posttrainbench.get("templates_dir"))
    posttrainbench["auto_select_device"] = False
    posttrainbench["device"] = device
    posttrainbench["parallel_benchmarks"] = False
    posttrainbench["max_parallel_benchmarks"] = 1
    posttrainbench["benchmark_devices"] = {benchmark: device}

    if max_tokens is not None:
        posttrainbench["max_tokens"] = max_tokens

    training_defaults = dict(cfg.get("training_defaults") or {})
    if cutoff_len is not None:
        training_defaults["cutoff_len"] = cutoff_len
    if train_template is not None:
        training_defaults["template"] = train_template
    if training_defaults:
        cfg["training_defaults"] = training_defaults

    return cfg


def sanitize_suffix(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return cleaned.strip("_")


def main() -> int:
    args = parse_args()
    single_benchmark, single_gpu = resolve_single_run_request(args)

    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (repo_root / args.base_config).resolve()
    base_cfg = yaml.safe_load(base_config_path.read_text())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = sanitize_suffix(args.config_suffix)
    suffix_part = f"_{suffix}" if suffix else ""
    processes = []

    if single_benchmark is not None:
        benchmark_device_pairs = [(single_benchmark, int(single_gpu))]
    else:
        benchmark_device_pairs = list(DEFAULT_BENCHMARK_DEVICE_MAP.items())

    for benchmark, device in benchmark_device_pairs:
        run_dir = repo_root / "runs" / f"math_posttrain_datatree_{benchmark}_gpu{device}{suffix_part}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = build_config(
            base_cfg,
            benchmark,
            device,
            repo_root,
            num_black=args.num_black,
            max_rounds=args.max_rounds,
            red_max_turns=args.red_max_turns,
            black_max_turns=args.black_max_turns,
            max_tokens=args.max_tokens,
            cutoff_len=args.cutoff_len,
            train_template=args.train_template,
        )
        cfg_path = run_dir / "launch_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

        task_text = build_task_text(benchmark, args.task)
        cmd = [
            str(repo_root / ".venv" / "bin" / "python"),
            str(repo_root / "run.py"),
            "--agent",
            "math_posttrain_datatree",
            "--config",
            str(cfg_path),
            "--task",
            task_text,
            "--run-dir",
            str(run_dir),
        ]
        print(f"[{benchmark}] device={device} config={cfg_path} run_dir={run_dir}")
        print(" ".join(cmd))
        if not args.dry_run:
            proc = subprocess.Popen(cmd, cwd=repo_root)
            processes.append((benchmark, device, proc.pid, cfg_path, run_dir))

    if not args.dry_run:
        print("\nLaunched processes:")
        for benchmark, device, pid, cfg_path, run_dir in processes:
            print(f"- {benchmark}: pid={pid}, device={device}, config={cfg_path}, run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
