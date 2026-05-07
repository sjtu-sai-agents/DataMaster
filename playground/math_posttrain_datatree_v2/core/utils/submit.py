from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
from typing import Any

from .data import validate_train_config
from .eval import run_eval
from .inspect import run_inspect
from .io import read_json, write_json
from .llama_factory import run_llama_factory_sft


DEFAULT_NPROC_PER_NODE = 1


class SubmitError(RuntimeError):
    def __init__(self, message: str, *, result: dict[str, Any] | None = None):
        self.result = result or {}
        super().__init__(message)


@dataclass
class SubmitResult:
    status: str
    score: float | None
    node_id: str
    benchmark: str
    trial_id: str
    trial_path: str
    train_path: str
    eval_path: str
    checkpoint_path: str
    recipe_path: str
    train_result_path: str
    eval_report_path: str
    inspect_report_path: str
    train_config_path: str
    train_data_path: str
    summary: str
    recommended_next_action: str = ""
    inspect_summary: str = ""
    benchmark_feedback_summary: str = ""
    benchmark_feedback: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "node_id": self.node_id,
            "benchmark": self.benchmark,
            "trial_id": self.trial_id,
            "trial_path": self.trial_path,
            "train_path": self.train_path,
            "eval_path": self.eval_path,
            "checkpoint_path": self.checkpoint_path,
            "recipe_path": self.recipe_path,
            "train_result_path": self.train_result_path,
            "eval_report_path": self.eval_report_path,
            "inspect_report_path": self.inspect_report_path,
            "train_config_path": self.train_config_path,
            "train_data_path": self.train_data_path,
            "summary": self.summary,
            "recommended_next_action": self.recommended_next_action,
            "inspect_summary": self.inspect_summary,
            "benchmark_feedback_summary": self.benchmark_feedback_summary,
            "benchmark_feedback": self.benchmark_feedback or {},
        }


def config_section_to_dict(section: Any) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if hasattr(value, "model_dump"):
            return {key: _convert(val) for key, val in value.model_dump().items()}
        if hasattr(value, "dict"):
            return {key: _convert(val) for key, val in value.dict().items()}
        if hasattr(value, "__dict__"):
            return {key: _convert(val) for key, val in vars(value).items() if not key.startswith("_")}
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    if section is None:
        return {}
    if isinstance(section, dict):
        return _convert(section)
    converted = _convert(section)
    return converted if isinstance(converted, dict) else {}


def resolve_llama_factory_runtime(config_section: Any) -> tuple[str | None, str | None, dict[str, str]]:
    raw = config_section_to_dict(config_section)
    python_bin = raw.get("python_bin")
    env_dir = raw.get("env_dir")

    env_overrides: dict[str, str] = {}
    nproc_per_node = raw.get("nproc_per_node", DEFAULT_NPROC_PER_NODE)
    if nproc_per_node is not None:
        env_overrides["NPROC_PER_NODE"] = str(int(nproc_per_node))

    cuda_visible_devices = raw.get("cuda_visible_devices")
    if cuda_visible_devices not in (None, ""):
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    return python_bin, env_dir, env_overrides


def format_submit_observation(result: dict[str, Any]) -> str:
    lines = [
        f"submit_status={result.get('status', 'unknown')}",
        f"benchmark={result.get('benchmark', '')}",
        f"score={result.get('score')}",
        f"trial_path={result.get('trial_path', '')}",
        f"eval_report_path={result.get('eval_report_path', '')}",
        f"recipe_path={result.get('recipe_path', '')}",
    ]
    reason = result.get("reason")
    if reason:
        lines.append(f"reason={reason}")
    train_log_path = result.get("train_log_path")
    if train_log_path:
        lines.append(f"train_log_path={train_log_path}")
    train_result_path = result.get("train_result_path")
    if train_result_path:
        lines.append(f"train_result_path={train_result_path}")
    train_config_feedback_path = result.get("train_config_feedback_path")
    if train_config_feedback_path:
        lines.append(f"train_config_feedback_path={train_config_feedback_path}")
    feedback = result.get("benchmark_feedback_summary")
    if feedback:
        lines.append(f"feedback={feedback}")
    inspect = result.get("inspect_summary")
    if inspect:
        lines.append(f"inspect={inspect}")
    return "\n".join(lines)


def compact_submit_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "status": result.get("status"),
        "score": result.get("score"),
        "node_id": result.get("node_id"),
        "benchmark": result.get("benchmark"),
        "trial_path": result.get("trial_path"),
        "eval_report_path": result.get("eval_report_path"),
        "recipe_path": result.get("recipe_path"),
        "checkpoint_path": result.get("checkpoint_path"),
        "benchmark_feedback_summary": result.get("benchmark_feedback_summary"),
        "inspect_summary": result.get("inspect_summary"),
    }
    for key in ("reason", "train_log_path", "train_result_path", "train_config_feedback_path"):
        if result.get(key):
            compact[key] = result.get(key)
    return compact


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clip_text(value: str | None, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "out of memory",
)


def _read_log_tail(path: str | Path, *, max_bytes: int = 12000) -> str:
    log_path = Path(path)
    if not log_path.exists() or not log_path.is_file():
        return ""
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _matching_log_line(text: str, markers: tuple[str, ...]) -> str:
    for line in reversed(text.splitlines()):
        normalized = line.lower()
        if any(marker in normalized for marker in markers):
            return _clip_text(line, 320)
    return ""


def _summarize_training_failure(train_result: Any) -> str:
    metrics = _safe_dict(getattr(train_result, "metrics", None))
    status = str(getattr(train_result, "status", "") or "failed")
    log_path = str(getattr(train_result, "train_log_path", "") or "")
    log_tail = _read_log_tail(log_path) if log_path else ""
    oom_line = _matching_log_line(log_tail, _OOM_MARKERS)
    log_hint = f"; see train_log_path={log_path}" if log_path else ""
    if oom_line:
        return f"training failed: CUDA out of memory ({oom_line}){log_hint}"
    reason = str(metrics.get("reason") or metrics.get("merge_error") or "").strip()
    if reason:
        return f"training failed: {_clip_text(reason, 320)}{log_hint}"
    returncode = metrics.get("returncode")
    if returncode is not None:
        return f"training failed with status={status}, returncode={returncode}{log_hint}"
    return f"training failed with status={status}{log_hint}"


def _extract_benchmark_feedback(eval_report: Any) -> dict[str, Any]:
    if hasattr(eval_report, "to_dict"):
        payload = eval_report.to_dict()
    elif isinstance(eval_report, dict):
        payload = eval_report
    else:
        payload = {}
    metadata = _safe_dict(payload.get("metadata"))
    feedback = metadata.get("benchmark_feedback")
    return feedback if isinstance(feedback, dict) else {}


def _format_benchmark_feedback_summary(feedback: dict[str, Any]) -> str:
    rows: list[str] = []
    for benchmark_id, entry in feedback.items():
        if not isinstance(entry, dict):
            continue
        parts = [str(benchmark_id)]
        score = entry.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"score={float(score):.4f}")
        num_correct = entry.get("num_correct")
        num_samples = entry.get("num_samples")
        if isinstance(num_correct, int) and isinstance(num_samples, int) and num_samples > 0:
            parts.append(f"correct={num_correct}/{num_samples}")
        rows.append(", ".join(parts))
    return "; ".join(rows)


def _summarize_inspect_report(inspect_report: dict[str, Any]) -> str:
    if not isinstance(inspect_report, dict) or not inspect_report:
        return ""
    parts: list[str] = []
    next_action = inspect_report.get("recommended_next_action")
    if next_action:
        parts.append(f"next={next_action}")
    failure_clusters = inspect_report.get("failure_clusters")
    if isinstance(failure_clusters, list) and failure_clusters:
        parts.append("clusters=" + ",".join(str(item) for item in failure_clusters[:4]))
    rationale = _clip_text(str(inspect_report.get("rationale") or ""), 220)
    if rationale:
        parts.append(f"rationale={rationale}")
    return " | ".join(parts)


def _make_trial_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _append_submission_index(task_workspace: Path, node_id: str, result: dict[str, Any]) -> None:
    node_submit_dir = task_workspace / "artifacts" / "submits" / node_id
    index_path = node_submit_dir / "submissions.json"
    index_payload = read_json(index_path, default={}) or {}
    submissions = index_payload.get("submissions")
    if not isinstance(submissions, list):
        submissions = []
    submissions.append(result)
    successful = [
        item for item in submissions
        if isinstance(item, dict) and item.get("status") == "completed" and isinstance(item.get("score"), (int, float))
    ]
    best = max(successful, key=lambda item: float(item["score"])) if successful else None
    write_json(
        index_path,
        {
            "node_id": node_id,
            "submissions": submissions,
            "best_success": best,
            "successful_count": len(successful),
        },
    )
    if best:
        write_json(node_submit_dir / "best_success.json", best)


def load_best_submit(task_workspace: str | Path, node_id: str) -> dict[str, Any] | None:
    node_submit_dir = Path(task_workspace) / "artifacts" / "submits" / node_id
    best = read_json(node_submit_dir / "best_success.json", default=None)
    return best if isinstance(best, dict) else None


def submit_training_eval(
    *,
    task_workspace: str | Path,
    config: Any,
    node_id: str,
    benchmark: str,
    train_config: str | Path | dict[str, Any] | None,
    train_data_path: str | Path,
    pack_manifest: dict[str, Any] | None = None,
    pack_stats: dict[str, Any] | None = None,
    trial_id: str | None = None,
) -> SubmitResult:
    workspace = Path(task_workspace)
    train_data = Path(train_data_path)
    if not train_data.exists():
        raise SubmitError(f"train_data_path does not exist: {train_data}")
    if not benchmark:
        raise SubmitError("benchmark is required")
    if not node_id:
        raise SubmitError("node_id is required")

    trial_id = trial_id or _make_trial_id()
    trial_dir = workspace / "artifacts" / "submits" / node_id / trial_id
    train_dir = trial_dir / "checkpoint"
    eval_dir = trial_dir / "eval"
    inspect_dir = trial_dir / "inspect"
    trial_dir.mkdir(parents=True, exist_ok=True)
    copied_train_data = trial_dir / train_data.name
    if train_data.resolve() != copied_train_data.resolve():
        shutil.copy2(train_data, copied_train_data)

    # Resolve train_config from file or dict
    if isinstance(train_config, (str, Path)):
        source_path = Path(train_config)
        if not source_path.exists():
            raise SubmitError(f"train_config path does not exist: {source_path}")
        agent_train_config = read_json(source_path, default={}) or {}
    elif isinstance(train_config, dict):
        agent_train_config = dict(train_config)
    else:
        agent_train_config = {}

    # Save agent's train_config for audit
    train_config_path = write_json(trial_dir / "train_config.json", agent_train_config)

    # Get training_defaults from config.yaml
    training_defaults = config_section_to_dict(getattr(config, "training_defaults", None))

    # Validate and merge with correct priority
    train_config_validation = validate_train_config(
        train_config_path,
        output_path=trial_dir / "train_config_validation_report.json",
        effective_output_path=trial_dir / "effective_train_config.json",
        defaults=training_defaults,
    )
    if train_config_validation.get("status") != "passed":
        failure = {
            "status": "failed",
            "node_id": node_id,
            "benchmark": benchmark,
            "trial_id": trial_id,
            "trial_path": str(trial_dir),
            "train_path": str(train_dir),
            "train_config_path": str(train_config_path),
            "train_config_feedback_path": str(trial_dir / "train_config_validation_report.json"),
            "reason": f"train config validation failed: {train_config_validation.get('reason') or 'unknown reason'}",
        }
        write_json(trial_dir / "submit_result.json", failure)
        _append_submission_index(workspace, node_id, failure)
        raise SubmitError(failure["reason"], result=failure)

    effective_train_config = train_config_validation.get("effective_config", {})

    base_model = getattr(
        config,
        "base_model",
        "${BASE_MODEL_PATH}",
    )
    evaluation_cfg = config_section_to_dict(getattr(config, "evaluation", None))
    eval_backend = str(evaluation_cfg.get("backend", "builtin"))
    benchmark_files = config_section_to_dict(evaluation_cfg.get("benchmark_files"))
    prediction_files = config_section_to_dict(evaluation_cfg.get("prediction_files"))
    posttrainbench_cfg = config_section_to_dict(evaluation_cfg.get("posttrainbench"))
    training_mode = str(getattr(config, "training_mode", "lora_sft"))
    template_override = training_defaults.get("template") if training_defaults else None
    lf_python_bin, lf_env_dir, lf_env_overrides = resolve_llama_factory_runtime(
        getattr(config, "llama_factory_env", None)
    )

    train_result = run_llama_factory_sft(
        dataset_path=copied_train_data,
        recipe_path=train_dir / "base_recipe.json",
        output_dir=train_dir,
        base_model=base_model,
        overrides=effective_train_config,
        template_override=template_override,
        dry_run=bool(getattr(config, "dry_run_training", True)),
        env_overrides=lf_env_overrides,
        python_bin=lf_python_bin,
        env_dir=lf_env_dir,
        merge_for_evaluation=eval_backend == "posttrainbench",
        training_mode=training_mode,
    )
    if train_result.status != "completed":
        failure = {
            "status": "failed",
            "node_id": node_id,
            "benchmark": benchmark,
            "trial_id": trial_id,
            "trial_path": str(trial_dir),
            "train_path": str(train_dir),
            "train_result_path": str(train_dir / "train_result.json"),
            "train_log_path": train_result.train_log_path,
            "reason": _summarize_training_failure(train_result),
        }
        write_json(trial_dir / "submit_result.json", failure)
        _append_submission_index(workspace, node_id, failure)
        raise SubmitError(failure["reason"], result=failure)

    eval_report = run_eval(
        eval_dir=eval_dir,
        benchmark_suite=[benchmark],
        pack_manifest=pack_manifest or {},
        pack_stats=pack_stats or {},
        benchmark_files=benchmark_files,
        prediction_files=prediction_files,
        eval_backend=eval_backend,
        evaluation_options=posttrainbench_cfg,
        model_path=train_result.checkpoint_path,
    )
    backend_details = (eval_report.metadata or {}).get("backend_details", {}) or {}
    failed_eval_backends = [
        benchmark_id
        for benchmark_id, detail in backend_details.items()
        if isinstance(detail, dict) and detail.get("status") == "failed"
    ]
    if failed_eval_backends:
        failure = {
            "status": "failed",
            "node_id": node_id,
            "benchmark": benchmark,
            "trial_id": trial_id,
            "trial_path": str(trial_dir),
            "train_path": str(train_dir),
            "checkpoint_path": train_result.checkpoint_path,
            "recipe_path": train_result.recipe_path,
            "eval_path": str(eval_dir),
            "eval_report_path": str(eval_dir / "eval_report.json"),
            "reason": f"evaluation backend failed for: {', '.join(sorted(failed_eval_backends))}",
        }
        write_json(trial_dir / "submit_result.json", failure)
        _append_submission_index(workspace, node_id, failure)
        raise SubmitError(failure["reason"], result=failure)

    inspect_report = run_inspect(
        eval_report=eval_report.to_dict(),
        pack_manifest=pack_manifest or {},
        pack_stats=pack_stats or {},
        output_path=inspect_dir / "inspect_report.json",
    )
    benchmark_feedback = _extract_benchmark_feedback(eval_report)
    benchmark_feedback_summary = _format_benchmark_feedback_summary(benchmark_feedback)
    inspect_summary = _summarize_inspect_report(inspect_report.to_dict())
    result = SubmitResult(
        status="completed",
        score=eval_report.overall_accuracy,
        node_id=node_id,
        benchmark=benchmark,
        trial_id=trial_id,
        trial_path=str(trial_dir),
        train_path=str(train_dir),
        eval_path=str(eval_dir),
        checkpoint_path=train_result.checkpoint_path,
        recipe_path=train_result.recipe_path,
        train_result_path=str(train_dir / "train_result.json"),
        eval_report_path=str(eval_dir / "eval_report.json"),
        inspect_report_path=str(inspect_dir / "inspect_report.json"),
        train_config_path=str(train_config_path),
        train_data_path=str(copied_train_data),
        summary=f"score={eval_report.overall_accuracy} eval_path={eval_dir}",
        recommended_next_action=inspect_report.recommended_next_action,
        inspect_summary=inspect_summary,
        benchmark_feedback_summary=benchmark_feedback_summary,
        benchmark_feedback=benchmark_feedback,
    )
    payload = result.to_dict()
    write_json(trial_dir / "submit_result.json", payload)
    _append_submission_index(workspace, node_id, payload)
    return result
