#!/usr/bin/env python3
"""Paper-style agent tool usage analysis.

This script reads trajectory files from a run folder and analyzes target tool calls.

Main outputs:
    - paper_tool_counts.pdf
    - paper_tool_proportions_donut.pdf
    - paper_tool_by_stage_grouped.pdf
    - paper_tool_stage_heatmap_counts.pdf
    - paper_tool_stage_heatmap_normalized.pdf
    - paper_tool_per_conversation_average.pdf
    - paper_tool_sequence_heatmap.pdf
    - paper_tool_timeline_stacked.pdf
    - tool_usage_summary.json

Usage:
    python count_tool_visualize.py --run-dir /path/to/runs/ml_master_*
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Paper-style plotting
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.linewidth": 0.9,

    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#374151",
    "ytick.color": "#374151",

    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.85,

    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.facecolor": "white",
    "legend.edgecolor": "#E5E7EB",
    "legend.fontsize": 9,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


PAPER_COLORS = {
    "blue": "#2F5D8C",
    "orange": "#D9822B",
    "green": "#3A7D44",
    "red": "#B23A48",
    "purple": "#6C5B7B",
    "teal": "#287C7C",
    "brown": "#8B5E34",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}

STAGE_COLORS = {
    "initial": "#111827",
    "black": "#2F5D8C",
    "red": "#B23A48",
    "unknown": "#6B7280",
}

TARGET_TOOLS = {
    "operate_submission_run_code",
    "operate_submission_grade_code",
    "operate_submission_read_code",
    "operate_submission_validate_submission",
    "operate_submission_write_code",
    "operate_submission_fix_code",
}

TOOL_ORDER = [
    "operate_submission_read_code",
    "operate_submission_write_code",
    "operate_submission_run_code",
    "operate_submission_fix_code",
    "operate_submission_validate_submission",
    "operate_submission_grade_code",
]

TOOL_COLORS = {
    "operate_submission_read_code": "#2F5D8C",
    "operate_submission_write_code": "#287C7C",
    "operate_submission_run_code": "#D9822B",
    "operate_submission_fix_code": "#B23A48",
    "operate_submission_validate_submission": "#6C5B7B",
    "operate_submission_grade_code": "#3A7D44",
}

TOOL_SHORT_NAMES = {
    "operate_submission_read_code": "read",
    "operate_submission_write_code": "write",
    "operate_submission_run_code": "run",
    "operate_submission_fix_code": "fix",
    "operate_submission_validate_submission": "validate",
    "operate_submission_grade_code": "grade",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


# =============================================================================
# Helpers
# =============================================================================

def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_viz_dir(run_dir: Path) -> Path:
    viz_dir = run_dir / "visualize"
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _tool_label(tool: str) -> str:
    return TOOL_SHORT_NAMES.get(tool, tool.replace("operate_submission_", ""))


def _clean_stage(stage: Any) -> str:
    if stage is None:
        return "unknown"
    stage = str(stage).strip()
    return stage if stage else "unknown"


def _ordered_tools(tools: Sequence[str]) -> List[str]:
    tools_set = set(tools)
    ordered = [t for t in TOOL_ORDER if t in tools_set]
    ordered += sorted(t for t in tools_set if t not in TOOL_ORDER)
    return ordered


def _ordered_stages(stages: Sequence[str]) -> List[str]:
    stage_set = set(stages)
    preferred = ["initial", "black", "red"]
    ordered = [s for s in preferred if s in stage_set]
    ordered += sorted(s for s in stage_set if s not in preferred)
    return ordered


def _short_id(node_id: Optional[str], n: int = 8) -> str:
    if not node_id:
        return "None"
    return str(node_id)[:n]


# =============================================================================
# Loading
# =============================================================================

def discover_run_dir(explicit: Optional[str]) -> Path:
    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
        if not run_dir.exists():
            raise SystemExit(f"run_dir not found: {run_dir}")
        return run_dir

    if not RUNS_DIR.exists():
        raise SystemExit(f"runs dir not found: {RUNS_DIR}")

    candidates = []
    for d in RUNS_DIR.glob("ml_master_*"):
        if d.is_dir() and (d / "logs" / "uct_nodes").exists():
            candidates.append(d)

    if not candidates:
        raise SystemExit("no run found under runs/")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_node_tree(run_dir: Path) -> Dict[str, Any]:
    nodes_dir = run_dir / "logs" / "uct_nodes"
    node_file = nodes_dir / "node.json"

    if node_file.exists():
        obj = _read_json(node_file)
        if isinstance(obj, dict):
            if isinstance(obj.get("root"), dict):
                return obj["root"]
            return obj

    # Fallback: build a flat root-like object from individual node JSON files.
    nodes = []
    for fp in sorted(nodes_dir.glob("*.json")):
        if fp.name == "node.json":
            continue
        obj = _read_json(fp)
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            nodes.append(obj)

    if nodes:
        root_candidates = [n for n in nodes if not isinstance(n.get("parent"), str)]
        root = root_candidates[0] if root_candidates else nodes[0]
        return root

    raise SystemExit(f"cannot load node tree from: {nodes_dir}")


def collect_all_nodes(
    node: Dict[str, Any],
    nodes_by_stage: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if nodes_by_stage is None:
        nodes_by_stage = defaultdict(list)

    stage = _clean_stage(node.get("stage", "unknown"))
    nodes_by_stage[stage].append(node)

    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            collect_all_nodes(child, nodes_by_stage)

    return dict(nodes_by_stage)


def flatten_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = [node]
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            out.extend(flatten_nodes(child))
    return out


def match_node_to_stage(node_id: str, nodes_by_stage: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    if not node_id:
        return None

    for stage, nodes in nodes_by_stage.items():
        for node in nodes:
            full_id = node.get("id")
            if not isinstance(full_id, str):
                continue

            if full_id == node_id:
                return stage

            if full_id[:8] == node_id[:8]:
                return stage

            if node_id.startswith(full_id[:8]) or full_id.startswith(node_id[:8]):
                return stage

    return None


def load_trajectories(run_dir: Path) -> List[Dict[str, Any]]:
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        return []

    all_trajectories: List[Dict[str, Any]] = []

    for task_dir in sorted(traj_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue

        traj_file = task_dir / "trajectory.json"
        if not traj_file.exists():
            continue

        data = _read_json(traj_file)
        if isinstance(data, list):
            for idx, entry in enumerate(data):
                if isinstance(entry, dict):
                    enriched = dict(entry)
                    enriched["_task_dir"] = task_dir.name
                    enriched["_local_index"] = idx
                    all_trajectories.append(enriched)

    return all_trajectories


# =============================================================================
# Tool-call parsing
# =============================================================================

def parse_tool_call(tc: Any) -> Tuple[str, str]:
    if tc is None:
        return ("unknown", "")

    if isinstance(tc, dict):
        fn = tc.get("function", {}) or {}

        if isinstance(fn, dict):
            name = fn.get("name", "") or tc.get("name", "unknown")
            args = fn.get("arguments", "") or tc.get("arguments", "")
        else:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", "")

        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)

        return (str(name), str(args) if args else "")

    if isinstance(tc, str):
        name_match = re.search(r"FunctionCall\(name='([^']+)'", tc)
        if name_match:
            name = name_match.group(1)
        else:
            fallback = re.search(r"name='([^']+)'", tc)
            name = fallback.group(1) if fallback else "unknown"

        args_match = re.search(r"arguments='([^']*)'", tc, re.DOTALL)
        args = args_match.group(1) if args_match else ""

        return (name, args)

    return ("unknown", "")


def extract_tool_calls_from_message(msg: Dict[str, Any]) -> List[Tuple[str, str]]:
    raw_calls = msg.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return []

    out = []
    for tc in raw_calls:
        name, args = parse_tool_call(tc)
        if name and name != "unknown":
            out.append((name, args))

    return out


def extract_node_id_prefix(agent_id: str) -> str:
    if not agent_id:
        return ""

    # Common form: nodeid_xxx or nodeid-agentname.
    parts = re.split(r"[_\s/]+", agent_id)
    return parts[0] if parts else agent_id


# =============================================================================
# Analysis
# =============================================================================

def analyze_tool_usage(
    trajectories: List[Dict[str, Any]],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    by_tool: Counter[str] = Counter()
    by_stage: Dict[str, Counter[str]] = defaultdict(Counter)
    by_conversation: List[Dict[str, Any]] = []

    for conv_idx, traj_entry in enumerate(trajectories):
        agent_id = str(traj_entry.get("agent_id", ""))
        node_id_prefix = extract_node_id_prefix(agent_id)
        stage = match_node_to_stage(node_id_prefix, nodes_by_stage) or "unknown"

        traj = traj_entry.get("trajectory", {})
        if not isinstance(traj, dict):
            traj = {}

        messages = traj.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        conversation_tools = []
        assistant_turn_count = 0

        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            if msg.get("role") != "assistant":
                continue

            assistant_turn_count += 1

            tool_calls = extract_tool_calls_from_message(msg)
            for call_idx, (tool_name, args) in enumerate(tool_calls):
                if tool_name not in TARGET_TOOLS:
                    continue

                by_tool[tool_name] += 1
                by_stage[stage][tool_name] += 1

                conversation_tools.append({
                    "tool": tool_name,
                    "args": args,
                    "message_index": msg_idx,
                    "assistant_turn_index": assistant_turn_count - 1,
                    "call_index": call_idx,
                    "global_call_index": len(conversation_tools),
                })

        by_conversation.append({
            "conversation_index": conv_idx,
            "stage": stage,
            "node_id": node_id_prefix,
            "agent_id": agent_id,
            "task_dir": traj_entry.get("_task_dir", ""),
            "local_index": traj_entry.get("_local_index", None),
            "assistant_turn_count": assistant_turn_count,
            "tool_calls": conversation_tools,
        })

    total_calls = sum(by_tool.values())
    proportions = {
        tool: count / total_calls if total_calls > 0 else 0.0
        for tool, count in by_tool.items()
    }

    return {
        "by_tool": by_tool,
        "by_stage": dict(by_stage),
        "by_conversation": by_conversation,
        "proportions": proportions,
        "total_calls": total_calls,
    }


def build_stage_tool_matrix(
    by_stage: Dict[str, Counter[str]],
    normalize: bool = False,
) -> Tuple[List[str], List[str], np.ndarray]:
    stages = _ordered_stages(list(by_stage.keys()))
    all_tools = set()

    for counter in by_stage.values():
        all_tools.update(counter.keys())

    tools = _ordered_tools(list(all_tools))
    matrix = np.zeros((len(stages), len(tools)), dtype=float)

    for i, stage in enumerate(stages):
        total = sum(by_stage.get(stage, Counter()).values())
        for j, tool in enumerate(tools):
            val = by_stage.get(stage, Counter()).get(tool, 0)
            if normalize:
                matrix[i, j] = val / total if total > 0 else 0.0
            else:
                matrix[i, j] = float(val)

    return stages, tools, matrix


def save_summary_json(run_dir: Path, analysis: Dict[str, Any], nodes_by_stage: Dict[str, List[Dict[str, Any]]]) -> Path:
    viz_dir = _ensure_viz_dir(run_dir)

    by_tool = analysis["by_tool"]
    by_stage = analysis["by_stage"]
    by_conversation = analysis["by_conversation"]

    output = {
        "total_tool_calls": int(sum(by_tool.values())),
        "tool_counts": {k: int(v) for k, v in by_tool.items()},
        "tool_proportions": {k: float(v) for k, v in analysis["proportions"].items()},
        "stage_tool_counts": {
            stage: {tool: int(count) for tool, count in counter.items()}
            for stage, counter in by_stage.items()
        },
        "stage_node_counts": {
            stage: len(nodes)
            for stage, nodes in nodes_by_stage.items()
        },
        "conversation_count": len(by_conversation),
        "conversation_summaries": [
            {
                "conversation_index": c["conversation_index"],
                "stage": c["stage"],
                "node_id": c["node_id"],
                "tool_call_count": len(c["tool_calls"]),
                "assistant_turn_count": c["assistant_turn_count"],
                "tool_counts": dict(Counter(call["tool"] for call in c["tool_calls"])),
            }
            for c in by_conversation
        ],
    }

    path = viz_dir / "tool_usage_summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =============================================================================
# Visualizations
# =============================================================================

def create_tool_counts_plot(run_dir: Path, analysis: Dict[str, Any]) -> Optional[Path]:
    by_tool = analysis["by_tool"]
    if not by_tool:
        return None

    tools = _ordered_tools(list(by_tool.keys()))
    counts = [by_tool[t] for t in tools]
    colors = [TOOL_COLORS.get(t, PAPER_COLORS["gray"]) for t in tools]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    bars = ax.bar(
        np.arange(len(tools)),
        counts,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=1.1,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=9,
            color=PAPER_COLORS["dark"],
        )

    ax.set_title("Tool call frequency", loc="left")
    ax.set_ylabel("Call count")
    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels([_tool_label(t) for t in tools], rotation=25, ha="right")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    total = sum(counts)
    ax.text(
        0.985,
        0.93,
        f"Total calls: {total}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#E5E7EB",
            alpha=0.95,
        ),
    )

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_counts.pdf"
    return _save_figure(fig, out)


def create_tool_proportion_donut(run_dir: Path, analysis: Dict[str, Any]) -> Optional[Path]:
    by_tool = analysis["by_tool"]
    if not by_tool:
        return None

    tools = _ordered_tools(list(by_tool.keys()))
    counts = np.asarray([by_tool[t] for t in tools], dtype=float)
    colors = [TOOL_COLORS.get(t, PAPER_COLORS["gray"]) for t in tools]

    if counts.sum() <= 0:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[_tool_label(t) for t in tools],
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        counterclock=False,
        pctdistance=0.78,
        labeldistance=1.08,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.6),
        textprops=dict(fontsize=9, color=PAPER_COLORS["dark"]),
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    ax.text(
        0,
        0,
        f"{int(counts.sum())}\nTotal calls",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=PAPER_COLORS["dark"],
    )

    ax.set_title("Tool call composition", fontsize=14, fontweight="bold", pad=18)
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_proportions_donut.pdf"
    return _save_figure(fig, out)


def create_tool_by_stage_grouped(run_dir: Path, analysis: Dict[str, Any]) -> Optional[Path]:
    by_stage = analysis["by_stage"]
    if not by_stage:
        return None

    stages, tools, matrix = build_stage_tool_matrix(by_stage, normalize=False)
    if matrix.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(12.4, 6.4))

    x = np.arange(len(tools))
    width = 0.78 / max(len(stages), 1)

    for i, stage in enumerate(stages):
        values = matrix[i]
        offset = (i - len(stages) / 2 + 0.5) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=stage.title(),
            color=STAGE_COLORS.get(stage, PAPER_COLORS["gray"]),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(int(value)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=PAPER_COLORS["dark"],
                )

    ax.set_title("Tool usage by node stage", loc="left")
    ax.set_ylabel("Call count")
    ax.set_xticks(x)
    ax.set_xticklabels([_tool_label(t) for t in tools], rotation=25, ha="right")
    ax.grid(axis="y")
    ax.legend(loc="best", ncol=min(len(stages), 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_by_stage_grouped.pdf"
    return _save_figure(fig, out)


def create_stage_tool_heatmap(
    run_dir: Path,
    analysis: Dict[str, Any],
    normalize: bool,
) -> Optional[Path]:
    by_stage = analysis["by_stage"]
    if not by_stage:
        return None

    stages, tools, matrix = build_stage_tool_matrix(by_stage, normalize=normalize)
    if matrix.size == 0 or len(stages) == 0 or len(tools) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10.8, 5.4))

    cmap = "magma" if not normalize else "viridis"
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalize else str(int(value))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > matrix.max() * 0.35 else PAPER_COLORS["dark"],
            )

    ax.set_title(
        "Stage-tool usage heatmap" + (" (row-normalized)" if normalize else " (counts)"),
        loc="left",
    )
    ax.set_xlabel("Tool")
    ax.set_ylabel("Stage")
    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels([_tool_label(t) for t in tools], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels([s.title() for s in stages])

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Proportion" if normalize else "Count", fontsize=9)

    fig.tight_layout()

    suffix = "normalized" if normalize else "counts"
    out = _ensure_viz_dir(run_dir) / f"paper_tool_stage_heatmap_{suffix}.pdf"
    return _save_figure(fig, out)


def create_per_conversation_average_plot(run_dir: Path, analysis: Dict[str, Any]) -> Optional[Path]:
    conversations = analysis["by_conversation"]
    if not conversations:
        return None

    stage_stats: Dict[str, Dict[str, Any]] = {}

    for conv in conversations:
        stage = conv["stage"]
        if stage not in stage_stats:
            stage_stats[stage] = {
                "count": 0,
                "tool_counts": Counter(),
                "total_calls": 0,
            }

        calls = [call["tool"] for call in conv["tool_calls"]]
        stage_stats[stage]["count"] += 1
        stage_stats[stage]["tool_counts"].update(calls)
        stage_stats[stage]["total_calls"] += len(calls)

    stages = _ordered_stages(stage_stats.keys())
    all_tools = set()

    for stats in stage_stats.values():
        all_tools.update(stats["tool_counts"].keys())

    tools = _ordered_tools(list(all_tools))
    if not tools:
        return None

    fig, ax = plt.subplots(figsize=(12.4, 6.3))

    x = np.arange(len(tools))
    width = 0.78 / max(len(stages), 1)

    for i, stage in enumerate(stages):
        stats = stage_stats[stage]
        conv_count = max(stats["count"], 1)

        values = [
            stats["tool_counts"].get(tool, 0) / conv_count
            for tool in tools
        ]

        offset = (i - len(stages) / 2 + 0.5) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=stage.title(),
            color=STAGE_COLORS.get(stage, PAPER_COLORS["gray"]),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, value in zip(bars, values):
            if value > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=PAPER_COLORS["dark"],
                )

    ax.set_title("Average tool calls per conversation", loc="left")
    ax.set_ylabel("Average calls / conversation")
    ax.set_xticks(x)
    ax.set_xticklabels([_tool_label(t) for t in tools], rotation=25, ha="right")
    ax.grid(axis="y")
    ax.legend(loc="best", ncol=min(len(stages), 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_per_conversation_average.pdf"
    return _save_figure(fig, out)


def create_tool_sequence_heatmap(run_dir: Path, analysis: Dict[str, Any], max_conversations: int = 80) -> Optional[Path]:
    conversations = [
        c for c in analysis["by_conversation"]
        if len(c["tool_calls"]) > 0
    ]

    if not conversations:
        return None

    # Sort by stage first, then by original conversation order.
    stage_rank = {s: i for i, s in enumerate(_ordered_stages([c["stage"] for c in conversations]))}
    conversations.sort(key=lambda c: (stage_rank.get(c["stage"], 99), c["conversation_index"]))

    if len(conversations) > max_conversations:
        conversations = conversations[:max_conversations]

    all_tools = _ordered_tools(
        sorted({call["tool"] for c in conversations for call in c["tool_calls"]})
    )

    if not all_tools:
        return None

    tool_to_idx = {tool: i + 1 for i, tool in enumerate(all_tools)}
    max_len = max(len(c["tool_calls"]) for c in conversations)

    matrix = np.zeros((len(conversations), max_len), dtype=float)

    for i, conv in enumerate(conversations):
        for j, call in enumerate(conv["tool_calls"]):
            matrix[i, j] = tool_to_idx.get(call["tool"], 0)

    fig_height = max(5.8, min(15.0, 0.18 * len(conversations) + 2.5))
    fig, ax = plt.subplots(figsize=(13.2, fig_height))

    cmap = plt.cm.get_cmap("tab10", len(all_tools) + 1)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(all_tools))

    y_labels = [
        f"{i:02d} | {c['stage']} | {_short_id(c['node_id'])}"
        for i, c in enumerate(conversations)
    ]

    ax.set_title("Tool-call sequence heatmap by conversation", loc="left")
    ax.set_xlabel("Tool-call position within conversation")
    ax.set_ylabel("Conversation | stage | node")
    ax.set_yticks(np.arange(len(conversations)))
    ax.set_yticklabels(y_labels, fontsize=7)

    if max_len <= 40:
        ax.set_xticks(np.arange(max_len))
        ax.set_xticklabels([str(i + 1) for i in range(max_len)])
    else:
        step = max(1, max_len // 12)
        ticks = np.arange(0, max_len, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(i + 1) for i in ticks])

    # Manual legend.
    handles = []
    labels = []

    for tool, idx in tool_to_idx.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=cmap(idx),
                markeredgecolor="none",
                markersize=8,
            )
        )
        labels.append(_tool_label(tool))

    ax.legend(
        handles,
        labels,
        title="Tool",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_sequence_heatmap.pdf"
    return _save_figure(fig, out)


def create_tool_timeline_stacked(run_dir: Path, analysis: Dict[str, Any], bins: int = 12) -> Optional[Path]:
    conversations = analysis["by_conversation"]
    if not conversations:
        return None

    all_tools = _ordered_tools(
        sorted({call["tool"] for c in conversations for call in c["tool_calls"]})
    )
    if not all_tools:
        return None

    n = len(conversations)
    bins = max(1, min(bins, n))
    bin_edges = np.linspace(0, n, bins + 1, dtype=int)

    # Avoid duplicate edges when n is small.
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 1:
        return None

    bin_labels = []
    data = {tool: [] for tool in all_tools}

    for b in range(len(bin_edges) - 1):
        start = bin_edges[b]
        end = bin_edges[b + 1]
        chunk = conversations[start:end]

        counter = Counter()
        for conv in chunk:
            counter.update(call["tool"] for call in conv["tool_calls"])

        bin_labels.append(f"{start + 1}-{end}")

        for tool in all_tools:
            data[tool].append(counter.get(tool, 0))

    fig, ax = plt.subplots(figsize=(12.8, 6.2))

    x = np.arange(len(bin_labels))
    bottom = np.zeros(len(bin_labels), dtype=float)

    for tool in all_tools:
        values = np.asarray(data[tool], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=_tool_label(tool),
            color=TOOL_COLORS.get(tool, PAPER_COLORS["gray"]),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values

    ax.set_title("Tool-call timeline across conversations", loc="left")
    ax.set_ylabel("Call count")
    ax.set_xlabel("Conversation index range")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=25, ha="right")
    ax.grid(axis="y")
    ax.legend(loc="best", ncol=min(len(all_tools), 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_timeline_stacked.pdf"
    return _save_figure(fig, out)


def create_stage_total_efficiency_plot(
    run_dir: Path,
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> Optional[Path]:
    by_stage = analysis["by_stage"]
    conversations = analysis["by_conversation"]

    if not by_stage:
        return None

    stages = _ordered_stages(by_stage.keys())
    if not stages:
        return None

    stage_conv_counts = Counter(c["stage"] for c in conversations)

    total_calls = []
    calls_per_node = []
    calls_per_conv = []

    for stage in stages:
        calls = sum(by_stage.get(stage, Counter()).values())
        node_count = len(nodes_by_stage.get(stage, []))
        conv_count = stage_conv_counts.get(stage, 0)

        total_calls.append(calls)
        calls_per_node.append(calls / node_count if node_count > 0 else 0)
        calls_per_conv.append(calls / conv_count if conv_count > 0 else 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False)

    configs = [
        (axes[0], total_calls, "Total calls", "Count"),
        (axes[1], calls_per_node, "Calls per node", "Calls / node"),
        (axes[2], calls_per_conv, "Calls per conversation", "Calls / conversation"),
    ]

    x = np.arange(len(stages))
    colors = [STAGE_COLORS.get(s, PAPER_COLORS["gray"]) for s in stages]

    for ax, values, title, ylabel in configs:
        bars = ax.bar(
            x,
            values,
            color=colors,
            alpha=0.86,
            edgecolor="white",
            linewidth=1.0,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}" if isinstance(value, float) and not float(value).is_integer() else str(int(value)),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([s.title() for s in stages])
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Stage-level tool-call efficiency", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_stage_tool_efficiency.pdf"
    return _save_figure(fig, out)


def create_visualizations(
    run_dir: Path,
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> List[Path]:
    output_files: List[Path] = []

    plot_calls = [
        create_tool_counts_plot(run_dir, analysis),
        create_tool_proportion_donut(run_dir, analysis),
        create_tool_by_stage_grouped(run_dir, analysis),
        create_stage_tool_heatmap(run_dir, analysis, normalize=False),
        create_stage_tool_heatmap(run_dir, analysis, normalize=True),
        create_per_conversation_average_plot(run_dir, analysis),
        create_tool_sequence_heatmap(run_dir, analysis),
        create_tool_timeline_stacked(run_dir, analysis),
        create_stage_total_efficiency_plot(run_dir, analysis, nodes_by_stage),
    ]

    for out in plot_calls:
        if out is not None:
            output_files.append(out)
            print(f"[INFO] saved: {out}")

    summary_path = save_summary_json(run_dir, analysis, nodes_by_stage)
    output_files.append(summary_path)
    print(f"[INFO] saved: {summary_path}")

    return output_files


# =============================================================================
# Console report
# =============================================================================

def print_statistics(analysis: Dict[str, Any], nodes_by_stage: Dict[str, List[Dict[str, Any]]]) -> None:
    by_tool: Counter[str] = analysis["by_tool"]
    by_stage: Dict[str, Counter[str]] = analysis["by_stage"]
    by_conversation: List[Dict[str, Any]] = analysis["by_conversation"]

    total_calls = sum(by_tool.values())

    print("\n" + "=" * 80)
    print("AGENT TOOL USAGE ANALYSIS REPORT")
    print("=" * 80)

    print("\n[Overall]")
    print(f"  Total target tool calls: {total_calls}")
    print(f"  Unique target tools used: {len(by_tool)}")
    print(f"  Conversations: {len(by_conversation)}")

    if by_conversation:
        non_empty = sum(1 for c in by_conversation if c["tool_calls"])
        print(f"  Conversations with target tool calls: {non_empty}")

    print("\n[Tool counts]")
    for tool in _ordered_tools(list(by_tool.keys())):
        count = by_tool[tool]
        pct = count / total_calls * 100 if total_calls > 0 else 0
        print(f"  {_tool_label(tool):12s} {count:6d}  ({pct:5.1f}%)  [{tool}]")

    print("\n[Stage summary]")
    stages = _ordered_stages(by_stage.keys())
    stage_conv_counts = Counter(c["stage"] for c in by_conversation)

    for stage in stages:
        stage_calls = sum(by_stage[stage].values())
        node_count = len(nodes_by_stage.get(stage, []))
        conv_count = stage_conv_counts.get(stage, 0)

        print(f"\n  {stage.upper()}")
        print(f"    node_count:       {node_count}")
        print(f"    conversation_count:{conv_count}")
        print(f"    total_tool_calls: {stage_calls}")

        if node_count > 0:
            print(f"    calls_per_node:   {stage_calls / node_count:.2f}")
        if conv_count > 0:
            print(f"    calls_per_conv:   {stage_calls / conv_count:.2f}")

        for tool in _ordered_tools(list(by_stage[stage].keys())):
            count = by_stage[stage][tool]
            pct = count / stage_calls * 100 if stage_calls > 0 else 0
            avg = count / conv_count if conv_count > 0 else 0
            print(f"      {_tool_label(tool):12s} {count:6d} ({pct:5.1f}%) avg={avg:.2f}/conv")

    print("\n[Per-conversation totals by stage]")
    for stage in _ordered_stages([c["stage"] for c in by_conversation]):
        convs = [c for c in by_conversation if c["stage"] == stage]
        totals = [len(c["tool_calls"]) for c in convs]

        if not totals:
            continue

        print(
            f"  {stage:12s} "
            f"mean={np.mean(totals):.2f}, "
            f"median={np.median(totals):.2f}, "
            f"max={np.max(totals):.0f}, "
            f"n={len(totals)}"
        )

    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze and visualize agent tool usage patterns")
    parser.add_argument("--run-dir", type=str, default=None, help="Target run directory")
    args = parser.parse_args()

    run_dir = discover_run_dir(args.run_dir)
    print(f"[INFO] analyzing run: {run_dir}")

    print("[INFO] loading node tree...")
    tree = load_node_tree(run_dir)

    print("[INFO] collecting nodes by stage...")
    nodes_by_stage = collect_all_nodes(tree)

    print("[INFO] loading trajectories...")
    trajectories = load_trajectories(run_dir)
    print(f"[INFO] loaded {len(trajectories)} trajectory entries")

    print("[INFO] analyzing tool usage...")
    analysis = analyze_tool_usage(trajectories, nodes_by_stage)

    print_statistics(analysis, nodes_by_stage)

    print("[INFO] creating visualizations...")
    output_files = create_visualizations(run_dir, analysis, nodes_by_stage)

    print(f"\n[INFO] generated {len(output_files)} files:")
    for f in output_files:
        print(f"  - {f}")

    print(f"\n[INFO] visualizations saved to: {run_dir / 'visualize'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())