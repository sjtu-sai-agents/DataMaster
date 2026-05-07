from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json


def get_global_pool_manifest_path(task_workspace: Path) -> Path:
    manifests_dir = ensure_dir(task_workspace / "artifacts" / "manifests")
    return manifests_dir / "global_pool_manifest.json"


def get_black_handoff_path(task_workspace: Path, node_id: str) -> Path:
    handoff_dir = ensure_dir(task_workspace / "artifacts" / "handoffs")
    return handoff_dir / f"black_handoff_{node_id}.json"


def load_json_payload(path_value: str | Path | None) -> dict[str, Any]:
    if not path_value:
        return {}
    payload = read_json(path_value, default={}) or {}
    return payload if isinstance(payload, dict) else {}


def _entry_key(entry: dict[str, Any]) -> str:
    return str(
        entry.get("source_id")
        or entry.get("name")
        or entry.get("full_local_path")
        or entry.get("local_path")
        or entry.get("url")
        or ""
    ).strip()


def _normalize_entry(entry: Any) -> dict[str, Any]:
    if hasattr(entry, "to_dict"):
        payload = entry.to_dict()
    elif isinstance(entry, dict):
        payload = dict(entry)
    else:
        payload = {}
    coverage_tags = payload.get("coverage_tags")
    if not isinstance(coverage_tags, list):
        payload["coverage_tags"] = []
    quality_signals = payload.get("quality_signals")
    if not isinstance(quality_signals, dict):
        payload["quality_signals"] = {}
    full_local_path = str(payload.get("full_local_path") or "").strip()
    local_path = str(payload.get("local_path") or "").strip()
    if full_local_path and not local_path:
        payload["local_path"] = full_local_path
    probe_sample_rows_path = payload.get("probe_sample_rows_path")
    if probe_sample_rows_path in (None,):
        payload["probe_sample_rows_path"] = ""
    return payload


def merge_global_pool_manifest(
    existing_payload: dict[str, Any] | None,
    new_entries: list[Any],
    *,
    node_id: str,
    search_goal: str,
    agent_summary: str = "",
) -> dict[str, Any]:
    payload = dict(existing_payload or {})
    existing_raw = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    merged: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}

    for raw in existing_raw:
        entry = _normalize_entry(raw)
        key = _entry_key(entry)
        if not key:
            continue
        index_by_key[key] = len(merged)
        merged.append(entry)

    latest_added: list[str] = []
    for raw in new_entries:
        entry = _normalize_entry(raw)
        key = _entry_key(entry)
        if not key:
            continue
        if key in index_by_key:
            current = merged[index_by_key[key]]
            updated = dict(current)
            for field, value in entry.items():
                if field == "coverage_tags":
                    tags = list(dict.fromkeys([*(current.get("coverage_tags") or []), *(value or [])]))
                    updated[field] = tags
                elif field == "quality_signals":
                    merged_quality = dict(current.get("quality_signals") or {})
                    if isinstance(value, dict):
                        merged_quality.update(value)
                    updated[field] = merged_quality
                elif value not in (None, "", [], {}):
                    updated[field] = value
            merged[index_by_key[key]] = updated
        else:
            index_by_key[key] = len(merged)
            merged.append(entry)
            latest_added.append(str(entry.get("source_id") or entry.get("name") or key))

    coverage_tags = sorted(
        {
            str(tag)
            for entry in merged
            for tag in (entry.get("coverage_tags") or [])
            if str(tag).strip()
        }
    )
    payload.update(
        {
            "manifest_id": "global_pool_manifest",
            "created_from_node": node_id,
            "search_goal": search_goal,
            "datasets": merged,
            "coverage_tags": coverage_tags,
            "source_summary": {
                "source_count": len(merged),
                "latest_added_source_ids": latest_added,
                "new_source_count": len(latest_added),
                "last_updated_by": node_id,
                "last_search_goal": search_goal,
                "last_agent_summary": agent_summary[:500],
            },
        }
    )
    return payload


def summarize_global_pool_manifest(payload: dict[str, Any], max_sources: int = 12) -> dict[str, Any]:
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    source_summary = payload.get("source_summary") if isinstance(payload.get("source_summary"), dict) else {}
    sample_sources: list[str] = []
    for item in datasets[:max_sources]:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or item.get("name") or "").strip()
        if source_id:
            sample_sources.append(source_id)
    return {
        "manifest_id": payload.get("manifest_id") or "global_pool_manifest",
        "source_count": len(datasets),
        "coverage_tags": payload.get("coverage_tags") or [],
        "latest_added_source_ids": source_summary.get("latest_added_source_ids") or [],
        "sample_sources": sample_sources,
    }


def summarize_black_handoff(payload: dict[str, Any], max_sources: int = 8) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    selected_sources = payload.get("selected_sources") if isinstance(payload.get("selected_sources"), list) else []
    preferred = [str(item) for item in selected_sources[:max_sources]]
    summary: dict[str, Any] = {
        "node_id": payload.get("node_id"),
        "metric": payload.get("metric"),
        "selected_sources": preferred,
        "recommended_next_action": payload.get("recommended_next_action"),
        "inspect_summary": payload.get("inspect_summary") or "",
        "benchmark_feedback_summary": payload.get("benchmark_feedback_summary") or "",
        "train_config_path": payload.get("train_config_path") or "",
        "effective_train_config_path": payload.get("effective_train_config_path") or "",
    }
    return summary
