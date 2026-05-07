from __future__ import annotations

import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .data import normalize_final_answer
from .io import read_jsonl, write_json, write_jsonl
from .types import EvalReport

DEFAULT_BENCHMARK_SUITE = [
    "aime_2025",
    "arena_hard_writing",
    "bfcl",
    "gpqa_main",
    "gsm8k",
    "healthbench_easy",
    "human_eval",
]
POSTTRAINBENCH_TASK_MAP = {
    "aime_2025": "aime2025",
    "arena_hard_writing": "arenahardwriting",
    "bfcl": "bfcl",
    "gpqa_main": "gpqamain",
    "gsm8k": "gsm8k",
    "healthbench_easy": "healthbench",
    "human_eval": "humaneval",
}
logger = logging.getLogger(__name__)


POSTTRAINBENCH_UNSUPPORTED_CLI_ARGS: dict[str, set[str]] = {
    "arena_hard_writing": {"max_tokens", "max_connections", "gpu_memory_utilization"},
    "healthbench_easy": {"max_tokens", "max_connections", "gpu_memory_utilization"},
}

BENCHMARK_FEEDBACK_SCHEMA_VERSION = 1
BENCHMARK_FEEDBACK_PROFILES = {
    "aime_2025": "sparse_exact_match",
    "arena_hard_writing": "llm_judge_writing",
    "bfcl": "tool_use",
    "gpqa_main": "multiple_choice_qa",
    "gsm8k": "exact_match_reasoning",
    "healthbench_easy": "llm_judge_health",
    "human_eval": "code_execution",
}
KNOWN_LAST_LINE_PREFIXES = ("ANSWER:", "Final answer:", "Final Answer:")


def _absolute_path_preserve_symlink(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def load_benchmark_samples(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    if file_path.suffix.lower() == ".jsonl":
        return read_jsonl(file_path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def score_answer_predictions(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    pred_map = {
        str(item.get("id", idx)): normalize_final_answer(
            str(
                item.get("prediction")
                or item.get("final_answer")
                or item.get("answer")
                or item.get("response")
                or item.get("output")
                or ""
            )
        )
        for idx, item in enumerate(predictions)
    }
    results: list[dict[str, Any]] = []
    correct = 0
    for idx, sample in enumerate(samples):
        sample_id = str(sample.get("id", idx))
        gold = normalize_final_answer(str(sample.get("answer") or sample.get("final_answer") or ""))
        pred = pred_map.get(sample_id, "")
        is_correct = bool(gold and pred and gold == pred)
        correct += int(is_correct)
        results.append(
            {
                "id": sample_id,
                "gold": gold,
                "prediction": pred,
                "correct": is_correct,
            }
        )
    accuracy = correct / len(samples) if samples else 0.0
    return accuracy, results


def resolve_posttrainbench_task_dir(repo_dir: str | Path | None, benchmark_id: str) -> Path | None:
    if repo_dir is None:
        return None
    task_name = POSTTRAINBENCH_TASK_MAP.get(benchmark_id)
    if task_name is None:
        return None
    candidate = Path(repo_dir).resolve() / "src" / "eval" / "tasks" / task_name
    if candidate.exists():
        return candidate
    return None


def _extract_metric_value(metrics: dict[str, Any]) -> float | None:
    preferred_keys = ("accuracy", "acc", "exact_match", "score")
    for key in preferred_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _pick_best_cuda_device() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None

    best_device: int | None = None
    best_free_mem = -1
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            device = int(parts[0])
            free_mem = int(parts[1])
        except ValueError:
            continue
        if free_mem > best_free_mem:
            best_device = device
            best_free_mem = free_mem
    return best_device


def _benchmark_profile(benchmark_id: str) -> str:
    return BENCHMARK_FEEDBACK_PROFILES.get(benchmark_id, "generic")


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _classify_last_line_prefix(last_line: str) -> str:
    if not last_line:
        return "<empty>"
    for prefix in KNOWN_LAST_LINE_PREFIXES:
        if last_line.startswith(prefix):
            return prefix
    if ":" in last_line:
        head = last_line.split(":", 1)[0].strip()
        if head:
            return f"{head[:40]}:"
    return last_line[:40]


def _extract_prompt_variant(inspect_payload: dict[str, Any] | None) -> str:
    if not isinstance(inspect_payload, dict):
        return "official"
    candidates = [
        inspect_payload.get("metadata"),
        (inspect_payload.get("eval") or {}).get("metadata"),
        (inspect_payload.get("eval") or {}).get("task_attribs"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            value = candidate.get("prompt_variant")
            if value:
                return str(value)
    return "official"


def _find_matching_posttrainbench_log(
    task_dir: Path,
    model_path: str | Path,
    created_log_names: set[str] | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    logs_dir = task_dir / "logs"
    if not logs_dir.exists():
        return None, None

    candidates = sorted(logs_dir.glob("*.json"))
    if created_log_names:
        filtered = [path for path in candidates if path.name in created_log_names]
        if filtered:
            candidates = filtered

    resolved_model_path = str(Path(model_path).resolve())
    matches: list[tuple[float, Path, dict[str, Any]]] = []
    for path in candidates:
        payload = _read_json_file(path)
        if payload is None:
            continue
        logged_model = str(((payload.get("eval") or {}).get("model") or ""))
        if resolved_model_path not in logged_model:
            continue
        matches.append((path.stat().st_mtime, path, payload))

    if not matches and created_log_names:
        return _find_matching_posttrainbench_log(task_dir, model_path, None)
    if not matches:
        return None, None

    _, best_path, best_payload = max(matches, key=lambda item: item[0])
    return best_path, best_payload


def _capture_posttrainbench_raw_artifacts(
    *,
    eval_dir: Path,
    benchmark_id: str,
    task_dir: Path,
    model_path: str | Path,
    created_log_names: set[str] | None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    inspect_log_path, inspect_payload = _find_matching_posttrainbench_log(
        task_dir,
        model_path,
        created_log_names,
    )
    raw_artifacts: dict[str, str] = {}
    if inspect_log_path is None or inspect_payload is None:
        return raw_artifacts, None

    copied_log_path = eval_dir / f"{benchmark_id}_inspect_eval_log.json"
    try:
        shutil.copy2(inspect_log_path, copied_log_path)
        raw_artifacts["inspect_eval_log_path"] = str(copied_log_path)
    except Exception:
        raw_artifacts["inspect_eval_log_path"] = str(inspect_log_path)
    return raw_artifacts, inspect_payload


def _build_generic_posttrainbench_feedback(
    *,
    benchmark_id: str,
    score: float | None,
    metrics: dict[str, Any],
    raw_artifacts: dict[str, str],
    inspect_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    num_samples = None
    if isinstance(inspect_payload, dict):
        samples = inspect_payload.get("samples")
        if isinstance(samples, list):
            num_samples = len(samples)
        elif isinstance(inspect_payload.get("results"), dict):
            total_samples = inspect_payload["results"].get("total_samples")
            if isinstance(total_samples, int):
                num_samples = total_samples

    return {
        "schema_version": BENCHMARK_FEEDBACK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "benchmark_profile": _benchmark_profile(benchmark_id),
        "score": score,
        "num_samples": num_samples,
        "num_correct": None,
        "format_adherence_rate": None,
        "parseable_answer_rate": None,
        "numeric_match_rate": None,
        "raw_artifacts": raw_artifacts,
        "extras": {
            "prompt_variant": _extract_prompt_variant(inspect_payload),
            "raw_metrics": metrics,
        },
    }


def _build_aime_posttrainbench_feedback(
    *,
    eval_dir: Path,
    benchmark_id: str,
    score: float | None,
    metrics: dict[str, Any],
    raw_artifacts: dict[str, str],
    inspect_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(inspect_payload, dict):
        return _build_generic_posttrainbench_feedback(
            benchmark_id=benchmark_id,
            score=score,
            metrics=metrics,
            raw_artifacts=raw_artifacts,
            inspect_payload=inspect_payload,
        )

    prompt_variant = _extract_prompt_variant(inspect_payload)
    expected_prefix = "Final answer:" if prompt_variant == "aligned_final_answer" else "ANSWER:"
    samples = inspect_payload.get("samples") or []
    rows: list[dict[str, Any]] = []
    prefix_counter: Counter[str] = Counter()
    num_correct = 0
    num_parseable = 0
    num_expected_prefix = 0
    num_numeric_match = 0

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        completion = str(((sample.get("output") or {}).get("completion") or ""))
        last_line = _extract_last_nonempty_line(completion)
        prefix_counter[_classify_last_line_prefix(last_line)] += 1
        numeric_tokens = re.findall(r"-?\d+", last_line)
        parsed_value = numeric_tokens[-1] if numeric_tokens else None
        score_payload = ((sample.get("scores") or {}).get("aime_scorer") or {})
        score_code = score_payload.get("value")
        target = str(sample.get("target") or "")
        is_correct = score_code == "C"
        expected_prefix_match = last_line.startswith(expected_prefix)
        parseable_answer = parsed_value is not None
        numeric_match = parsed_value == target if parsed_value is not None else False

        num_correct += int(is_correct)
        num_parseable += int(parseable_answer)
        num_expected_prefix += int(expected_prefix_match)
        num_numeric_match += int(numeric_match)
        rows.append(
            {
                "id": sample.get("id"),
                "target": target,
                "last_line": last_line,
                "parsed_numeric_tail": parsed_value,
                "score": score_code,
                "is_correct": is_correct,
                "expected_prefix_match": expected_prefix_match,
                "parseable_answer": parseable_answer,
                "numeric_match": numeric_match,
            }
        )

    if rows:
        sample_summary_path = write_jsonl(eval_dir / f"{benchmark_id}_sample_summary.jsonl", rows)
        raw_artifacts = {**raw_artifacts, "sample_summary_path": str(sample_summary_path)}

    num_samples = len(rows)
    generic_feedback = _build_generic_posttrainbench_feedback(
        benchmark_id=benchmark_id,
        score=score,
        metrics=metrics,
        raw_artifacts=raw_artifacts,
        inspect_payload=inspect_payload,
    )
    generic_feedback.update(
        {
            "num_samples": num_samples,
            "num_correct": num_correct,
            "format_adherence_rate": round(num_expected_prefix / num_samples, 6) if num_samples else 0.0,
            "parseable_answer_rate": round(num_parseable / num_samples, 6) if num_samples else 0.0,
            "numeric_match_rate": round(num_numeric_match / num_samples, 6) if num_samples else 0.0,
        }
    )
    extras = dict(generic_feedback.get("extras") or {})
    extras.update(
        {
            "expected_last_line_prefix": expected_prefix,
            "common_last_line_prefixes": dict(prefix_counter.most_common(5)),
        }
    )
    generic_feedback["extras"] = extras
    return generic_feedback


def _build_builtin_exact_match_feedback(
    *,
    benchmark_id: str,
    accuracy: float,
    sample_results: list[dict[str, Any]],
) -> dict[str, Any]:
    num_samples = len(sample_results)
    num_correct = sum(1 for item in sample_results if item.get("correct"))
    return {
        "schema_version": BENCHMARK_FEEDBACK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "benchmark_profile": _benchmark_profile(benchmark_id),
        "score": accuracy,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "format_adherence_rate": None,
        "parseable_answer_rate": None,
        "numeric_match_rate": round(num_correct / num_samples, 6) if num_samples else 0.0,
        "raw_artifacts": {},
        "extras": {"source": "builtin_exact_match"},
    }


def _build_posttrainbench_benchmark_feedback(
    *,
    eval_dir: Path,
    benchmark_id: str,
    score: float | None,
    metrics: dict[str, Any],
    raw_artifacts: dict[str, str],
    inspect_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if benchmark_id == "aime_2025":
        return _build_aime_posttrainbench_feedback(
            eval_dir=eval_dir,
            benchmark_id=benchmark_id,
            score=score,
            metrics=metrics,
            raw_artifacts=raw_artifacts,
            inspect_payload=inspect_payload,
        )
    return _build_generic_posttrainbench_feedback(
        benchmark_id=benchmark_id,
        score=score,
        metrics=metrics,
        raw_artifacts=raw_artifacts,
        inspect_payload=inspect_payload,
    )


def _extend_cli_args(command: list[str], extra_cli_args: dict[str, Any] | None) -> None:
    if not extra_cli_args:
        return
    for key, value in extra_cli_args.items():
        if value is None or value is False:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if value is True:
            command.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                command.extend([flag, str(item)])
            continue
        command.extend([flag, str(value)])


def run_posttrainbench_eval(
    *,
    eval_dir: str | Path,
    benchmark_id: str,
    model_path: str | Path,
    repo_dir: str | Path,
    python_bin: str | Path | None = None,
    limit: int | None = None,
    max_tokens: int | None = None,
    max_connections: int | None = None,
    gpu_memory_utilization: float | None = None,
    templates_dir: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
    device: int | None = None,
    auto_select_device: bool = True,
    max_num_seqs: int | None = 32,
    extra_cli_args: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    task_dir = resolve_posttrainbench_task_dir(repo_dir, benchmark_id)
    if task_dir is None:
        return None, {"status": "unavailable", "reason": f"missing PostTrainBench task for {benchmark_id}"}
    script_path = task_dir / "evaluate.py"
    if not script_path.exists():
        return None, {"status": "unavailable", "reason": f"missing evaluate.py for {benchmark_id}"}

    eval_dir = _absolute_path_preserve_symlink(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / f"{benchmark_id}_posttrainbench_metrics.json"
    if python_bin:
        python_path = _absolute_path_preserve_symlink(python_bin)
        if not python_path.exists():
            return None, {
                "status": "failed",
                "reason": f"configured posttrainbench python not found: {python_path}",
                "script_path": str(script_path),
            }
        python_exe = str(python_path)
    else:
        python_exe = sys.executable
    command = [
        python_exe,
        str(script_path),
        "--model-path",
        str(Path(model_path).resolve()),
        "--json-output-file",
        str(metrics_path),
    ]
    unsupported_cli_args = POSTTRAINBENCH_UNSUPPORTED_CLI_ARGS.get(benchmark_id, set())
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if max_tokens is not None and "max_tokens" not in unsupported_cli_args:
        command.extend(["--max-tokens", str(max_tokens)])
    if max_connections is not None and "max_connections" not in unsupported_cli_args:
        command.extend(["--max-connections", str(max_connections)])

    if gpu_memory_utilization is None:
        try:
            import torch
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                free_gb = free_mem / (1024**3)
                if free_gb < 60:
                    gpu_memory_utilization = 0.4
                    logger.info(f"Auto-set gpu_memory_utilization=0.4 (free memory: {free_gb:.1f} GB)")
                else:
                    gpu_memory_utilization = 0.7
                    logger.info(f"Auto-set gpu_memory_utilization=0.7 (free memory: {free_gb:.1f} GB)")
        except Exception:
            gpu_memory_utilization = 0.5
            logger.info("Auto-set gpu_memory_utilization=0.5 (default)")

    if gpu_memory_utilization is not None and "gpu_memory_utilization" not in unsupported_cli_args:
        command.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if templates_dir is not None:
        command.extend(["--templates-dir", str(Path(templates_dir).resolve())])
    _extend_cli_args(command, extra_cli_args)

    env = os.environ.copy()
    if python_bin:
        env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(python_path.parent.parent)
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://huggingface.co")
    env["FORCE_ONLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "0"
    env["HF_HUB_OFFLINE"] = "0"
    env.pop("DATASETS_OFFLINE", None)
    env["HF_HOME"] = env.get("HF_HOME", "${HF_HOME}")
    env["HUGGINGFACE_HUB_CACHE"] = env.get("HUGGINGFACE_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
    env["HF_DATASETS_CACHE"] = env.get("HF_DATASETS_CACHE", os.path.join(env["HF_HOME"], "datasets"))
    server_args: dict[str, Any] = {}
    selected_device = device
    if selected_device is None and auto_select_device:
        selected_device = _pick_best_cuda_device()
    if selected_device is not None:
        server_args["device"] = selected_device

    if max_num_seqs is None:
        try:
            import torch
            if torch.cuda.is_available():
                free_mem, _ = torch.cuda.mem_get_info(selected_device if selected_device is not None else 0)
                free_gb = free_mem / (1024**3)
                if free_gb < 40:
                    max_num_seqs = 8
                    logger.info(f"Auto-set max_num_seqs=8 (free memory: {free_gb:.1f} GB)")
                elif free_gb < 60:
                    max_num_seqs = 16
                    logger.info(f"Auto-set max_num_seqs=16 (free memory: {free_gb:.1f} GB)")
                else:
                    max_num_seqs = 32
                    logger.info(f"Auto-set max_num_seqs=32 (free memory: {free_gb:.1f} GB)")
        except Exception:
            max_num_seqs = 16
            logger.info("Auto-set max_num_seqs=16 (default)")

    if max_num_seqs is not None:
        server_args["max_num_seqs"] = max_num_seqs
    if server_args:
        env["VLLM_DEFAULT_SERVER_ARGS"] = json.dumps(server_args)
    if env_overrides:
        env.update(env_overrides)
    log_path = eval_dir / f"{benchmark_id}_posttrainbench_eval.log"
    logs_dir = task_dir / "logs"
    existing_inspect_logs = {path.name for path in logs_dir.glob("*.json")} if logs_dir.exists() else set()
    logger.info(
        "Starting PostTrainBench eval: benchmark=%s backend=posttrainbench model=%s device=%s max_num_seqs=%s log=%s",
        benchmark_id,
        Path(model_path).resolve(),
        selected_device if selected_device is not None else "auto",
        max_num_seqs if max_num_seqs is not None else "default",
        log_path,
    )
    logger.info("PostTrainBench command: %s", " ".join(command))

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"COMMAND: {' '.join(command)}\n")
        log_file.write(
            "SERVER_ARGS: "
            + json.dumps(server_args, ensure_ascii=False, sort_keys=True)
            + "\n\n"
        )
        log_file.flush()

        proc = subprocess.Popen(
            command,
            cwd=task_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        start_time = time.monotonic()
        last_heartbeat = start_time
        heartbeat_interval = 30.0
        while proc.poll() is None:
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                logger.info(
                    "PostTrainBench eval still running: benchmark=%s elapsed=%ss pid=%s log=%s",
                    benchmark_id,
                    int(now - start_time),
                    proc.pid,
                    log_path,
                )
                last_heartbeat = now
            time.sleep(2.0)
        returncode = proc.wait()

    created_inspect_logs = None
    if logs_dir.exists():
        created_inspect_logs = {path.name for path in logs_dir.glob("*.json")} - existing_inspect_logs
    raw_artifacts, inspect_payload = _capture_posttrainbench_raw_artifacts(
        eval_dir=eval_dir,
        benchmark_id=benchmark_id,
        task_dir=task_dir,
        model_path=model_path,
        created_log_names=created_inspect_logs,
    )

    if returncode != 0 or not metrics_path.exists():
        logger.warning(
            "PostTrainBench eval failed: benchmark=%s returncode=%s log=%s",
            benchmark_id,
            returncode,
            log_path,
        )
        detail = {
            "status": "failed",
            "reason": f"evaluate.py exited with code {returncode}",
            "command": command,
            "log_path": str(log_path),
            "script_path": str(script_path),
        }
        if raw_artifacts:
            detail["raw_artifacts"] = raw_artifacts
        return None, detail

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = _extract_metric_value(metrics)
    if score is None:
        logger.warning(
            "PostTrainBench eval finished without numeric metric: benchmark=%s metrics=%s log=%s",
            benchmark_id,
            metrics_path,
            log_path,
        )
        detail = {
            "status": "failed",
            "reason": "no numeric metric found in PostTrainBench metrics output",
            "command": command,
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "script_path": str(script_path),
        }
        if raw_artifacts:
            detail["raw_artifacts"] = raw_artifacts
        return None, detail
    logger.info(
        "PostTrainBench eval completed: benchmark=%s score=%s metrics=%s log=%s",
        benchmark_id,
        score,
        metrics_path,
        log_path,
    )
    benchmark_feedback = _build_posttrainbench_benchmark_feedback(
        eval_dir=eval_dir,
        benchmark_id=benchmark_id,
        score=score,
        metrics=metrics,
        raw_artifacts=raw_artifacts,
        inspect_payload=inspect_payload,
    )
    return score, {
        "status": "completed",
        "command": command,
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "script_path": str(script_path),
        "raw_metrics": metrics,
        "raw_artifacts": raw_artifacts,
        "benchmark_feedback": benchmark_feedback,
    }


def proxy_accuracy(pack_manifest: dict[str, Any], pack_stats: dict[str, Any], benchmark_id: str) -> float:
    sample_count = int(pack_manifest.get("sample_count", 0))
    coverage = set(pack_manifest.get("coverage_tags", []))
    style_dist = pack_stats.get("style_distribution", {}) or {}
    duplicate_rate = float(pack_stats.get("duplicate_rate", 1.0) or 1.0)
    short_count = int(style_dist.get("short_answer", 0))
    long_count = int(style_dist.get("long_reasoning", 0))
    total = max(short_count + long_count, 1)
    style_balance = 1.0 - abs(short_count / total - 0.5) * 2.0

    score = 0.10
    score += min(sample_count / 2000.0, 0.25)
    score += max(style_balance, 0.0) * 0.20
    score += max(0.0, 0.20 - duplicate_rate * 0.30)
    if benchmark_id == "aime_2025" and {"aime", "competition_math"} & coverage:
        score += 0.20
    elif benchmark_id == "gsm8k" and "gsm8k" in coverage:
        score += 0.15
    elif benchmark_id == "math500" and "competition_math" in coverage:
        score += 0.15
    return round(min(max(score, 0.0), 0.95), 6)


def run_eval(
    *,
    eval_dir: str | Path,
    benchmark_suite: list[str],
    pack_manifest: dict[str, Any],
    pack_stats: dict[str, Any],
    benchmark_files: dict[str, str] | None = None,
    prediction_files: dict[str, str] | None = None,
    eval_backend: str = "builtin",
    evaluation_options: dict[str, Any] | None = None,
    model_path: str | Path | None = None,
) -> EvalReport:
    eval_dir = _absolute_path_preserve_symlink(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    benchmark_files = benchmark_files or {}
    prediction_files = prediction_files or {}
    evaluation_options = evaluation_options or {}

    benchmark_scores: dict[str, float] = {}
    benchmark_feedback: dict[str, Any] = {}
    all_sample_results: list[dict[str, Any]] = []
    normalized_predictions: list[dict[str, Any]] = []
    status = "proxy"
    backend_details: dict[str, Any] = {}
    backend_results: dict[str, tuple[float | None, dict[str, Any]]] = {}

    if eval_backend == "posttrainbench" and model_path is not None:
        benchmark_devices = evaluation_options.get("benchmark_devices") or {}
        benchmark_cli_args = evaluation_options.get("benchmark_cli_args") or {}
        parallel_benchmarks = bool(evaluation_options.get("parallel_benchmarks", False))
        max_parallel_benchmarks = int(
            evaluation_options.get("max_parallel_benchmarks") or len(benchmark_suite) or 1
        )

        def _run_backend_eval(benchmark_id: str) -> tuple[str, float | None, dict[str, Any]]:
            benchmark_device = benchmark_devices.get(benchmark_id, evaluation_options.get("device"))
            per_benchmark_cli_args = benchmark_cli_args.get(benchmark_id) or {}
            score, backend_detail = run_posttrainbench_eval(
                eval_dir=eval_dir,
                benchmark_id=benchmark_id,
                model_path=model_path,
                repo_dir=evaluation_options.get("repo_dir"),
                python_bin=evaluation_options.get("python_bin"),
                limit=evaluation_options.get("limit"),
                max_tokens=evaluation_options.get("max_tokens"),
                max_connections=evaluation_options.get("max_connections"),
                gpu_memory_utilization=evaluation_options.get("gpu_memory_utilization"),
                templates_dir=evaluation_options.get("templates_dir"),
                env_overrides=evaluation_options.get("env_overrides"),
                device=benchmark_device,
                auto_select_device=bool(evaluation_options.get("auto_select_device", True)),
                max_num_seqs=evaluation_options.get("max_num_seqs", 32),
                extra_cli_args=per_benchmark_cli_args if isinstance(per_benchmark_cli_args, dict) else None,
            )
            return benchmark_id, score, backend_detail

        if parallel_benchmarks and len(benchmark_suite) > 1:
            worker_count = max(1, min(max_parallel_benchmarks, len(benchmark_suite)))
            logger.info(
                "Running PostTrainBench benchmarks in parallel: workers=%s suite=%s",
                worker_count,
                ",".join(benchmark_suite),
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_run_backend_eval, benchmark_id): benchmark_id
                    for benchmark_id in benchmark_suite
                }
                for future in as_completed(futures):
                    benchmark_id, score, backend_detail = future.result()
                    backend_results[benchmark_id] = (score, backend_detail)
        else:
            for benchmark_id in benchmark_suite:
                _, score, backend_detail = _run_backend_eval(benchmark_id)
                backend_results[benchmark_id] = (score, backend_detail)

    for benchmark_id in benchmark_suite:
        if benchmark_id in backend_results:
            score, backend_detail = backend_results[benchmark_id]
            backend_details[benchmark_id] = backend_detail
            feedback = backend_detail.get("benchmark_feedback") if isinstance(backend_detail, dict) else None
            if isinstance(feedback, dict):
                benchmark_feedback[benchmark_id] = feedback
            if score is not None:
                benchmark_scores[benchmark_id] = score
                status = "completed"
                continue

        samples = load_benchmark_samples(benchmark_files.get(benchmark_id))
        predictions = load_benchmark_samples(prediction_files.get(benchmark_id))
        if samples and predictions:
            accuracy, sample_results = score_answer_predictions(samples, predictions)
            benchmark_scores[benchmark_id] = accuracy
            benchmark_feedback[benchmark_id] = _build_builtin_exact_match_feedback(
                benchmark_id=benchmark_id,
                accuracy=accuracy,
                sample_results=sample_results,
            )
            status = "completed"
            all_sample_results.extend(
                [{**item, "benchmark_id": benchmark_id} for item in sample_results]
            )
            normalized_predictions.extend(
                [
                    {
                        "benchmark_id": benchmark_id,
                        "id": item["id"],
                        "prediction": item["prediction"],
                    }
                    for item in sample_results
                ]
            )
        else:
            benchmark_scores[benchmark_id] = proxy_accuracy(pack_manifest, pack_stats, benchmark_id)

    overall = (
        sum(benchmark_scores.values()) / len(benchmark_scores)
        if benchmark_scores
        else 0.0
    )
    sample_results_path = write_jsonl(eval_dir / "sample_results.jsonl", all_sample_results)
    normalized_predictions_path = write_jsonl(
        eval_dir / "normalized_predictions.jsonl",
        normalized_predictions,
    )
    report = EvalReport(
        status=status,
        overall_accuracy=round(overall, 6),
        benchmark_scores=benchmark_scores,
        sample_results_path=str(sample_results_path),
        normalized_predictions_path=str(normalized_predictions_path),
        metadata={
            "benchmark_suite": benchmark_suite,
            "eval_backend": eval_backend,
            "backend_details": backend_details,
            "benchmark_feedback": benchmark_feedback,
        },
    )
    write_json(eval_dir / "eval_report.json", report.to_dict())
    return report
