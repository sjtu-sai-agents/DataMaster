"""Shared helpers for MLMasterPlayground to keep playground.py small."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .uct import MetricReview, UCTSearchManager


def copy_submission(
    submission_dir: Path,
    node_id: str,
    source_submission_dir: Path | None = None,
) -> Path | None:
    """Copy submission.csv to a node-specific file if it exists."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    src_dir = source_submission_dir or submission_dir
    src = src_dir / "submission.csv"
    if not src.exists():
        return None
    dst = submission_dir / f"submission_{node_id}.csv"
    shutil.copy(src, dst)
    # Remove the shared submission to avoid accidental reuse by later nodes.
    try:
        src.unlink()
    except Exception:
        pass
    return dst


def build_review(res: dict[str, Any], has_submission: bool) -> MetricReview:
    """Construct MetricReview from experiment result."""
    detail = res.get("metric_detail", {}) or {}
    metric = detail.get("metric") if detail else res.get("metric")
    return MetricReview(
        metric=metric,
        lower_is_better=detail.get("lower_is_better"),
        is_bug=detail.get("is_bug", False) or metric is None,
        has_submission=has_submission,
        summary=(res.get("exec", {}).get("stdout", "") or "")[-500:],
        raw_output=res.get("raw_response"),
    )


def append_trajectory(playground: Any, record: dict[str, Any], logger: logging.Logger | None = None) -> None:
    """Append trajectory record to memory (no longer persists to disk)."""
    logger = logger or logging.getLogger(__name__)
    if not hasattr(playground, "trajectories") or playground.trajectories is None:
        playground.trajectories = []
    playground.trajectories.append(record)

    # Removed trajectory.jsonl persistence - trajectory.jsonl is now handled by BaseAgent
    # with a more efficient jsonl format (append-only, no read-rewrite)


def save_node_snapshot(
    run_dir: str | Path | None,
    workspace_path: Path,
    node: Any,
    submission_path: Path | None,
    review: MetricReview,
    reward: float,
    search_mgr: UCTSearchManager,
    task_description: str | None = None,
    snapshot_event: str = "updated",
) -> None:
    """Persist key state for a search node for later inspection."""
    base_dir = (
        Path(run_dir) / "logs" / "uct_nodes"
        if run_dir
        else Path(workspace_path) / "logs" / "uct_nodes"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    parent_visits = node.parent.visits if node.parent else 1
    try:
        uct_val = node.uct_value(search_mgr._exploration_constant(), parent_visits)
    except Exception:
        uct_val = None
    snapshot = {
        "snapshot_event": snapshot_event,
        "snapshot_ts": time.time(),
        "id": node.id,
        "stage": node.stage,
        "parent": getattr(node.parent, "id", None),
        "metric": getattr(node.metric, "value", None),
        "submission_score": getattr(node, "submission_score", None),
        "submission_valid": getattr(node, "submission_valid", None),
        "submission_detail": getattr(node, "submission_detail", None),
        "maximize": getattr(node.metric, "maximize", True) if getattr(node, "metric", None) else None,
        "is_buggy": node.is_buggy,
        "has_submission": review.has_submission,
        "reward": reward,
        "visits": node.visits,
        "total_reward": node.total_reward,
        "uct_value": uct_val,
        "submission_file": str(submission_path) if submission_path else None,
        "code": getattr(node, "code", "") or "",
        "stdout": getattr(node, "stdout", ""),
        "initial_reward": getattr(node, "initial_reward", None),
        "initial_total_reward": getattr(node, "initial_total_reward", None),
        "initial_visits": getattr(node, "initial_visits", None),
        "initial_uct": getattr(node, "initial_uct", None),
    }
    snap_path = base_dir / f"{node.id}.json"
    snap_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also persist a tree-style summary for all nodes.
    try:
        tree_path = base_dir / "node.json"
        tree_snapshot = _build_tree_snapshot(search_mgr, task_description=task_description)
        _atomic_write_json(tree_path, tree_snapshot)
    except Exception as exc:  # pragma: no cover - best-effort tree export
        logging.getLogger(__name__).warning("Failed to write node tree snapshot: %s", exc)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _node_status(node: Any) -> str:
    if getattr(node, "is_buggy", None) is True:
        return "buggy"
    metric_val = getattr(getattr(node, "metric", None), "value", None)
    if metric_val is None and getattr(node, "is_buggy", None) is None:
        return "pending"
    if getattr(node, "is_terminal", False):
        return "terminal"
    return "ok"


def _serialize_node(node: Any, search_mgr: UCTSearchManager) -> dict[str, Any]:
    parent_visits = node.parent.visits if node.parent else 1
    try:
        uct_val = node.uct_value(search_mgr._exploration_constant(), parent_visits)
    except Exception:
        uct_val = None
    metric_val = getattr(getattr(node, "metric", None), "value", None)
    submission_score = getattr(node, "submission_score", None)
    return {
        "id": node.id,
        "stage": node.stage,
        "status": _node_status(node),
        "parent": getattr(node.parent, "id", None),
        "plan": getattr(node, "plan", "") or "",
        "code": getattr(node, "code", "") or "",
        "stdout": getattr(node, "stdout", "") or "",
        "exit_code": getattr(node, "exit_code", None),
        "is_buggy": getattr(node, "original_is_buggy", getattr(node, "is_buggy", None)),  # Use original state for snapshot tracing
        "is_valid": getattr(node, "is_valid", None),
        "is_terminal": getattr(node, "is_terminal", False),
        "visits": getattr(node, "visits", 0),
        "total_reward": getattr(node, "total_reward", 0.0),
        "reward": getattr(node, "initial_reward", None),
        "uct_value": uct_val,
        "metric": metric_val,  # Add for compatibility with vis_node.py
        "submission_score": submission_score,
        "submission_valid": getattr(node, "submission_valid", None),
        "submission_detail": getattr(node, "submission_detail", None),
        "maximize": getattr(getattr(node, "metric", None), "maximize", True)
        if getattr(node, "metric", None)
        else None,
        "children": [],
    }


def _build_tree_snapshot(
    search_mgr: UCTSearchManager,
    task_description: str | None = None,
) -> dict[str, Any]:
    root = search_mgr.root
    visited: set[str] = set()

    def walk(node: Any) -> dict[str, Any]:
        visited.add(node.id)
        payload = _serialize_node(node, search_mgr)
        children = sorted(list(getattr(node, "children", [])), key=lambda c: c.created_at)
        payload["children"] = [walk(child) for child in children if child.id not in visited]
        return payload

    return {
        "task": task_description,
        "generated_at": time.time(),
        "root": walk(root),
    }


def save_best(logger: logging.Logger, workspace: Path, best_code: str, submission_csv: Path | None) -> None:
    """Persist best solution code and submission csv if present."""
    best_solution_path = workspace / "best_solution" / "best_solution.py"
    best_solution_path.write_text(best_code, encoding="utf-8")

    best_submission_path = workspace / "best_submission" / "best_submission.csv"
    if submission_csv is not None and submission_csv.exists():
        shutil.copy(submission_csv, best_submission_path)
    else:
        logger.debug("No submission csv to save as best (None or missing).")

    logger.info("Saved best_solution: %s", str(best_solution_path))
    logger.info("Saved best_submission: %s", str(best_submission_path))
