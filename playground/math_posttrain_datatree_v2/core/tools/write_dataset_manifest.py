from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class WriteDatasetManifestParams(BaseToolParams):
    """Merge red-node dataset choices into the global source pool.

    Use this instead of manually writing JSON. The tool validates that `datasets`
    is a non-empty list of objects, normalizes common fields, and updates the
    shared global pool manifest visible to later red and black nodes.
    """

    name: ClassVar[str] = "write_dataset_manifest"
    manifest_path: str = Field(default="", description="Deprecated; ignored. The global pool is the canonical manifest.")
    global_pool_manifest_path: str = Field(description="Absolute path to global_pool_manifest.json")
    node_id: str = Field(description="Current red node id")
    search_goal: str = Field(default="", description="Current red search goal")
    datasets: list[dict[str, Any]] = Field(description="Dataset entries to register")
    agent_summary: str = Field(default="", description="Short summary of selected sources")


class WriteDatasetManifestTool(BaseTool):
    name: ClassVar[str] = "write_dataset_manifest"
    params_class: ClassVar[type[BaseToolParams]] = WriteDatasetManifestParams

    def execute(self, session: Any, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)
        try:
            from playground.math_posttrain_datatree_v2.core.utils.handoff import (
                load_json_payload,
                merge_global_pool_manifest,
                summarize_global_pool_manifest,
            )
            from playground.math_posttrain_datatree_v2.core.utils.io import write_json

            datasets, issues = _normalize_datasets(params.datasets)
            if not datasets:
                result = {
                    "status": "failed",
                    "reason": "datasets must be a non-empty list of valid dataset objects",
                    "issues": issues,
                }
                return _format_manifest_observation(result), {"manifest_result": result}

            global_path = Path(params.global_pool_manifest_path)
            global_path.parent.mkdir(parents=True, exist_ok=True)
            global_payload = load_json_payload(global_path)
            global_payload = merge_global_pool_manifest(
                global_payload,
                datasets,
                node_id=params.node_id,
                search_goal=params.search_goal,
                agent_summary=params.agent_summary,
            )
            write_json(global_path, global_payload)
            global_summary = summarize_global_pool_manifest(global_payload)

            result = {
                "status": "passed",
                "global_pool_manifest_path": str(global_path),
                "dataset_count": len(datasets),
                "source_ids": [str(item.get("source_id") or item.get("name") or "") for item in datasets],
                "global_pool_source_count": int(global_summary.get("source_count") or 0),
                "global_pool_latest_added_source_ids": global_summary.get("latest_added_source_ids") or [],
                "issues": issues,
            }
        except Exception as exc:
            result = {"status": "failed", "reason": f"manifest write error: {exc}"}
        return _format_manifest_observation(result), {"manifest_result": result}


def _normalize_datasets(raw_datasets: Any) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    if not isinstance(raw_datasets, list):
        return [], ["datasets is not a list"]

    datasets: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_datasets):
        if not isinstance(raw, dict):
            issues.append(f"datasets[{idx}] is not an object")
            continue
        source_id = str(raw.get("source_id") or raw.get("dataset_id") or raw.get("name") or "").strip()
        if not source_id:
            issues.append(f"datasets[{idx}] missing source_id/name")
            continue
        name = str(raw.get("name") or source_id).strip()
        url = str(raw.get("url") or raw.get("huggingface_url") or "").strip()
        if not url and "/" in source_id:
            url = f"https://huggingface.co/datasets/{source_id}"

        quality_signals = raw.get("quality_signals")
        if not isinstance(quality_signals, dict):
            quality_signals = {}
        coverage_tags = raw.get("coverage_tags")
        if not isinstance(coverage_tags, list):
            coverage_tags = []

        entry = dict(raw)
        entry.update(
            {
                "source_id": source_id,
                "name": name,
                "url": url or "agent_search",
                "license": str(raw.get("license") or "unknown"),
                "local_path": str(raw.get("local_path") or raw.get("full_local_path") or ""),
                "full_local_path": str(raw.get("full_local_path") or raw.get("local_path") or ""),
                "probe_sample_rows_path": str(raw.get("probe_sample_rows_path") or ""),
                "split": str(raw.get("split") or ""),
                "config": str(raw.get("config") or raw.get("config_name") or raw.get("subset") or ""),
                "task_type": str(raw.get("task_type") or "math_reasoning"),
                "answer_style": str(raw.get("answer_style") or "mixed"),
                "difficulty": str(raw.get("difficulty") or "unknown"),
                "language": str(raw.get("language") or "en"),
                "quality_signals": quality_signals,
                "coverage_tags": [str(tag) for tag in coverage_tags if str(tag).strip()],
            }
        )
        datasets.append(entry)
    return datasets, issues


def _format_manifest_observation(result: dict[str, Any]) -> str:
    lines = [
        f"Status: {result.get('status', 'unknown')}",
    ]
    if result.get("reason"):
        lines.append(f"Reason: {result.get('reason')}")
    if result.get("global_pool_manifest_path"):
        lines.append(f"Global pool manifest path: {result.get('global_pool_manifest_path')}")
    if result.get("dataset_count") is not None:
        lines.append(f"Dataset count: {result.get('dataset_count')}")
    if result.get("source_ids"):
        lines.append(f"Source IDs: {', '.join(str(item) for item in result.get('source_ids') or [])}")
    if result.get("global_pool_source_count") is not None:
        lines.append(f"Global pool source count: {result.get('global_pool_source_count')}")
    issues = result.get("issues") or []
    if issues:
        lines.append("Issues:")
        for issue in issues[:10]:
            lines.append(f"  - {issue}")
    return "\n".join(lines)
