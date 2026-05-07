from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, write_json


def get_memory_dir(task_workspace: Path) -> Path:
    return ensure_dir(task_workspace / "artifacts" / "memory")


def get_node_memory_path(task_workspace: Path, node_id: str) -> Path:
    return get_memory_dir(task_workspace) / f"{node_id}.json"


def _clip_text(value: str | None, limit: int = 400) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_manifest_sources(metric_detail: dict[str, Any]) -> list[str]:
    manifest_path = metric_detail.get("manifest_path") or metric_detail.get("pack_manifest_path")
    if not manifest_path:
        return []
    payload = read_json(manifest_path, default={}) or {}
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, list):
        return []
    sources: list[str] = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or item.get("name") or "").strip()
        if source_id:
            sources.append(source_id)
    return sources


def _extract_benchmark_feedback(metric_detail: dict[str, Any]) -> dict[str, Any]:
    eval_report_path = metric_detail.get("eval_report_path")
    if not eval_report_path:
        return {}
    payload = read_json(eval_report_path, default={}) or {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    feedback = metadata.get("benchmark_feedback") if isinstance(metadata, dict) else None
    return feedback if isinstance(feedback, dict) else {}


def _format_benchmark_feedback_summary(feedback: dict[str, Any]) -> str:
    rows: list[str] = []
    for benchmark_id, entry in feedback.items():
        if not isinstance(entry, dict):
            continue
        parts = [str(benchmark_id)]
        score = entry.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"score={score:.4f}")
        num_correct = entry.get("num_correct")
        num_samples = entry.get("num_samples")
        if isinstance(num_correct, int) and isinstance(num_samples, int) and num_samples > 0:
            parts.append(f"correct={num_correct}/{num_samples}")
        for key, label in (
            ("format_adherence_rate", "format"),
            ("parseable_answer_rate", "parseable"),
            ("numeric_match_rate", "match"),
        ):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={value:.2f}")
        rows.append(", ".join(parts))
    return "; ".join(rows)


def write_node_memory(task_workspace: Path, node: Any, res: dict[str, Any], review: Any) -> Path:
    metric_detail = _safe_dict(res.get("metric_detail"))
    benchmark_feedback = _extract_benchmark_feedback(metric_detail)
    payload = {
        "node_id": node.id,
        "stage": node.stage,
        "parent_id": getattr(node.parent, "id", None),
        "metric": getattr(getattr(node, "metric", None), "value", None),
        "is_buggy": node.is_buggy,
        "recommended_next_action": getattr(node, "recommended_next_action", None),
        "plan": str(res.get("plan") or "").strip(),
        "summary": _clip_text(review.summary or res.get("raw_response") or "", 700),
        "stdout": _clip_text(_safe_dict(res.get("exec")).get("stdout", ""), 700),
        "raw_response": _clip_text(str(res.get("raw_response") or ""), 1200),
        "selected_sources": _extract_manifest_sources(metric_detail),
        "benchmark_feedback": benchmark_feedback,
        "artifacts": {
            "manifest_path": metric_detail.get("manifest_path") or getattr(node, "output_manifest_path", None),
            "global_pool_manifest_path": metric_detail.get("global_pool_manifest_path") or getattr(node, "global_pool_manifest_path", None),
            "pack_manifest_path": metric_detail.get("pack_manifest_path") or getattr(node, "pack_manifest_path", None),
            "pack_stats_path": metric_detail.get("pack_stats_path") or getattr(node, "pack_stats_path", None),
            "eval_report_path": metric_detail.get("eval_report_path") or getattr(node, "eval_report_path", None),
            "inspect_report_path": metric_detail.get("inspect_report_path") or getattr(node, "inspect_report_path", None),
            "black_handoff_path": metric_detail.get("black_handoff_path") or getattr(node, "black_handoff_path", None),
            "train_config_path": metric_detail.get("train_config_path") or getattr(node, "train_config_path", None),
            "effective_train_config_path": metric_detail.get("effective_train_config_path") or getattr(node, "effective_train_config_path", None),
            "submit_trial_path": metric_detail.get("submit_trial_path"),
            "submit_recipe_path": metric_detail.get("submit_recipe_path"),
            "submit_train_config_path": metric_detail.get("submit_train_config_path"),
            "submit_train_data_path": metric_detail.get("submit_train_data_path"),
            "checkpoint_path": metric_detail.get("checkpoint_path"),
        },
    }
    return write_json(get_node_memory_path(task_workspace, node.id), payload)


def load_node_memory(task_workspace: Path, node_id: str) -> dict[str, Any]:
    payload = read_json(get_node_memory_path(task_workspace, node_id), default={}) or {}
    return payload if isinstance(payload, dict) else {}


def summarize_memory_entry(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    parts = [
        f"stage={payload.get('stage', 'unknown')}",
        f"id={str(payload.get('node_id', ''))[:8]}",
        f"metric={payload.get('metric')}",
        f"buggy={payload.get('is_buggy')}",
    ]
    selected_sources = payload.get("selected_sources")
    if isinstance(selected_sources, list) and selected_sources:
        parts.append("sources=" + ", ".join(str(item) for item in selected_sources[:4]))
    if payload.get("plan"):
        parts.append(f"plan={_clip_text(str(payload['plan']), 220)}")
    if payload.get("summary"):
        parts.append(f"summary={_clip_text(str(payload['summary']), 260)}")
    feedback = payload.get("benchmark_feedback")
    if isinstance(feedback, dict) and feedback:
        feedback_summary = _format_benchmark_feedback_summary(feedback)
        if feedback_summary:
            parts.append(f"feedback={_clip_text(feedback_summary, 260)}")
    if payload.get("recommended_next_action"):
        parts.append(f"next={payload['recommended_next_action']}")
    return " | ".join(parts)


def summarize_parent_memory(task_workspace: Path, node: Any) -> str:
    parent = getattr(node, "parent", None)
    if parent is None:
        return "No parent memory available."
    payload = load_node_memory(task_workspace, parent.id)
    summary = summarize_memory_entry(payload)
    return summary or "No parent memory available."


def summarize_sibling_memory(task_workspace: Path, node: Any, max_items: int = 6) -> str:
    parent = getattr(node, "parent", None)
    if parent is None:
        return "There is no previous memory"
    summaries: list[str] = []
    siblings = sorted(parent.children, key=lambda item: item.created_at)
    for sibling in siblings:
        if sibling.id == node.id or sibling.is_buggy is None:
            continue
        payload = load_node_memory(task_workspace, sibling.id)
        if not payload:
            payload = {
                "node_id": sibling.id,
                "stage": sibling.stage,
                "metric": getattr(getattr(sibling, "metric", None), "value", None),
                "is_buggy": sibling.is_buggy,
                "summary": getattr(sibling, "analysis", "") or getattr(sibling, "stdout", ""),
                "plan": getattr(sibling, "plan", ""),
                "recommended_next_action": getattr(sibling, "recommended_next_action", None),
            }
        summary = summarize_memory_entry(payload)
        if summary:
            summaries.append(summary)
        if len(summaries) >= max_items:
            break
    return "\n".join(summaries) if summaries else "There is no previous memory"


def build_prompt_memory(task_workspace: Path, node: Any) -> str:
    parent_summary = summarize_parent_memory(task_workspace, node)
    sibling_summary = summarize_sibling_memory(task_workspace, node)
    return (
        "Parent memory:\n"
        f"{parent_summary}\n\n"
        "Sibling history:\n"
        f"{sibling_summary}"
    )


def build_node_memory_index(task_workspace: Path) -> Path:
    memory_dir = get_memory_dir(task_workspace)
    rows: list[dict[str, Any]] = []
    for file_path in sorted(memory_dir.glob("*.json")):
        payload = read_json(file_path, default={}) or {}
        if isinstance(payload, dict) and payload:
            rows.append(payload)
    return write_json(memory_dir / "index.json", rows)
