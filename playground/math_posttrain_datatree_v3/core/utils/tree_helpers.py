from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .io import ensure_dir
from .uct import MetricReview, UCTSearchManager


def append_trajectory(playground: Any, record: dict[str, Any]) -> None:
    if not hasattr(playground, "trajectories") or playground.trajectories is None:
        playground.trajectories = []
    playground.trajectories.append(record)


def write_uct_trajectory(workspace_path: Path, records: list[dict[str, Any]]) -> Path:
    report_dir = ensure_dir(workspace_path / "artifacts" / "reports")
    output_path = report_dir / "uct_trajectory.json"
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _serialize_node(node: Any, search_mgr: UCTSearchManager) -> dict[str, Any]:
    parent_visits = node.parent.visits if node.parent else 1
    try:
        uct_val = node.uct_value(search_mgr._exploration_constant(), parent_visits)
    except Exception:
        uct_val = None
    return {
        "id": node.id,
        "stage": node.stage,
        "parent": getattr(node.parent, "id", None),
        "metric": getattr(getattr(node, "metric", None), "value", None),
        "is_buggy": node.is_buggy,
        "visits": node.visits,
        "total_reward": node.total_reward,
        "uct_value": uct_val,
        "bound_manifest_path": getattr(node, "bound_manifest_path", None),
        "output_manifest_path": getattr(node, "output_manifest_path", None),
        "global_pool_manifest_path": getattr(node, "global_pool_manifest_path", None),
        "input_black_handoff_path": getattr(node, "input_black_handoff_path", None),
        "black_handoff_path": getattr(node, "black_handoff_path", None),
        "global_advice_path": getattr(node, "global_advice_path", None),
        "inspect_report_path": getattr(node, "inspect_report_path", None),
        "pack_manifest_path": getattr(node, "pack_manifest_path", None),
        "eval_report_path": getattr(node, "eval_report_path", None),
        "recommended_next_action": getattr(node, "recommended_next_action", None),
        "children": [],
    }


def _walk(node: Any, search_mgr: UCTSearchManager) -> dict[str, Any]:
    payload = _serialize_node(node, search_mgr)
    payload["children"] = [
        _walk(child, search_mgr)
        for child in sorted(node.children, key=lambda item: item.created_at)
    ]
    return payload


def save_node_snapshot(
    run_dir: str | Path | None,
    workspace_path: Path,
    node: Any,
    review: MetricReview,
    reward: float,
    search_mgr: UCTSearchManager,
    snapshot_event: str,
    task_description: str | None = None,
) -> None:
    base = Path(run_dir) / "logs" / "uct_nodes" if run_dir else workspace_path / "logs" / "uct_nodes"
    ensure_dir(base)
    payload = {
        "snapshot_event": snapshot_event,
        "snapshot_ts": time.time(),
        "task": task_description,
        "id": node.id,
        "stage": node.stage,
        "metric": getattr(getattr(node, "metric", None), "value", None),
        "is_buggy": node.is_buggy,
        "reward": reward,
        "summary": review.summary,
        "bound_manifest_path": getattr(node, "bound_manifest_path", None),
        "output_manifest_path": getattr(node, "output_manifest_path", None),
        "global_pool_manifest_path": getattr(node, "global_pool_manifest_path", None),
        "input_black_handoff_path": getattr(node, "input_black_handoff_path", None),
        "black_handoff_path": getattr(node, "black_handoff_path", None),
        "global_advice_path": getattr(node, "global_advice_path", None),
        "pack_manifest_path": getattr(node, "pack_manifest_path", None),
        "pack_stats_path": getattr(node, "pack_stats_path", None),
        "eval_report_path": getattr(node, "eval_report_path", None),
        "inspect_report_path": getattr(node, "inspect_report_path", None),
        "recommended_next_action": getattr(node, "recommended_next_action", None),
        "stdout": getattr(node, "stdout", "") or "",
    }
    (base / f"{node.id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tree_payload = {
        "task": task_description,
        "generated_at": time.time(),
        "root": _walk(search_mgr.root, search_mgr),
    }
    (base / "node.json").write_text(
        json.dumps(tree_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
