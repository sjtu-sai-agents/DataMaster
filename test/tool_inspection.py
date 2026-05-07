#!/usr/bin/env python3
"""Paper-style Tool Usage Analysis for UCT Nodes.

This script reads trajectory files from a run folder and analyzes tool-call patterns
across UCT node stages.

Compared with the original version, this version:
- Uses a cleaner paper-style plotting theme.
- Fixes stacked bar bottom accumulation.
- Fixes create_visualizations sometimes returning None.
- Makes sampled sequence visualization deterministic.
- Adds category-stage heatmaps.
- Adds red-vs-black differential plots.
- Adds tool transition matrix.
- Adds sequence-position category heatmap.
- Adds stage-level efficiency summary.
- Saves a machine-readable JSON summary.

Usage:
    python tool_inspection.py --run-dir /path/to/runs/ml_master_*

Output:
    Visualizations saved to:
        run_dir/visualize/
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
# Repository paths
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


# =============================================================================
# Tool categorization
# =============================================================================

BASIC_TOOLS = {"finish", "think"}

# Tools to exclude from statistics.
# The original script treats run_code as a bad tool.
BAD_TOOLS = {"run_code"}


def get_tool_category(tool_name: str) -> str:
    """Return coarse category for a tool name."""
    if tool_name in BASIC_TOOLS:
        return "basic_tool"

    if tool_name == "execute_bash" or tool_name.startswith("operate_submission"):
        return "operate"

    if tool_name.startswith("memory_tree"):
        return "memory"

    if tool_name.startswith(
        (
            "search_huggingface",
            "search_web",
            "search_github",
            "search_scholar",
        )
    ):
        return "search"

    return "other"


CATEGORY_ORDER = ["operate", "memory", "search", "basic_tool", "other"]

CATEGORY_LABELS = {
    "operate": "Operate",
    "memory": "Memory",
    "search": "Search",
    "basic_tool": "Basic",
    "other": "Other",
}

STAGE_ORDER = ["initial", "black", "red", "unknown"]


# =============================================================================
# Plot style
# =============================================================================

PAPER_COLORS = {
    "dark": "#111827",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "blue": "#2F5D8C",
    "green": "#3A7D44",
    "red": "#B23A48",
    "orange": "#D9822B",
    "purple": "#6C5B7B",
    "teal": "#287C7C",
    "brown": "#8B5E34",
}

CATEGORY_COLORS = {
    "operate": "#B23A48",
    "memory": "#287C7C",
    "search": "#2F5D8C",
    "basic_tool": "#D9822B",
    "other": "#6B7280",
}

STAGE_COLORS = {
    "initial": "#111827",
    "black": "#2F5D8C",
    "red": "#B23A48",
    "unknown": "#6B7280",
}


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


# =============================================================================
# Small helpers
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


def _clean_stage(stage: Any) -> str:
    if stage is None:
        return "unknown"
    stage = str(stage).strip()
    return stage if stage else "unknown"


def _ordered_stages(stages: Sequence[str]) -> List[str]:
    stage_set = set(stages)
    ordered = [s for s in STAGE_ORDER if s in stage_set]
    ordered += sorted(s for s in stage_set if s not in STAGE_ORDER)
    return ordered


def _ordered_categories(categories: Sequence[str]) -> List[str]:
    category_set = set(categories)
    ordered = [c for c in CATEGORY_ORDER if c in category_set]
    ordered += sorted(c for c in category_set if c not in CATEGORY_ORDER)
    return ordered


def _short_tool_name(tool: str, max_len: int = 24) -> str:
    name = (
        tool.replace("operate_submission_", "op_")
        .replace("search_huggingface_", "hf_")
        .replace("search_github_", "gh_")
        .replace("search_scholar_", "scholar_")
        .replace("search_web_", "web_")
        .replace("memory_tree_", "mem_")
    )
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _short_node_id(node_id: Optional[str], n: int = 8) -> str:
    if not node_id:
        return "None"
    return str(node_id)[:n]


def _as_counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


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

    raise SystemExit(f"cannot load node tree from {nodes_dir}")


def collect_all_nodes(
    node: Dict[str, Any],
    nodes_by_stage: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect all nodes grouped by stage."""
    if nodes_by_stage is None:
        nodes_by_stage = defaultdict(list)

    stage = _clean_stage(node.get("stage", "unknown"))
    nodes_by_stage[stage].append(node)

    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            collect_all_nodes(child, nodes_by_stage)

    return dict(nodes_by_stage)


def collect_node_ids(node: Dict[str, Any], ids: Optional[List[str]] = None) -> List[str]:
    """Recursively collect all node IDs."""
    if ids is None:
        ids = []

    node_id = node.get("id", "")
    if node_id and node_id != "__virtual_root__":
        ids.append(node_id)

    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            collect_node_ids(child, ids)

    return ids


def match_node_to_stage(node_id: str, nodes_by_stage: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """Find stage for a given node id or id prefix."""
    if not node_id:
        return None

    for stage, nodes in nodes_by_stage.items():
        for node in nodes:
            full_id = node.get("id", "")
            if not isinstance(full_id, str) or not full_id:
                continue

            if full_id == node_id:
                return stage

            if full_id[:8] == node_id[:8]:
                return stage

            if node_id.startswith(full_id[:8]) or full_id.startswith(node_id[:8]):
                return stage

    return None


def load_trajectories(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all trajectory entries from trajectories/task_*/trajectory.json."""
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
    """Parse tool call and return (tool_name, args_string)."""
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


def extract_tool_calls_from_message(
    msg: Dict[str, Any],
    filter_bad: bool = True,
) -> List[Tuple[str, str]]:
    """Extract tool calls from an assistant message."""
    raw_calls = msg.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return []

    tool_calls: List[Tuple[str, str]] = []

    for tc in raw_calls:
        name, args = parse_tool_call(tc)
        if not name or name == "unknown":
            continue

        if filter_bad and name in BAD_TOOLS:
            continue

        tool_calls.append((name, args))

    return tool_calls


def extract_node_id_prefix(agent_id: str) -> str:
    if not agent_id:
        return ""

    parts = re.split(r"[_\s/]+", agent_id)
    return parts[0] if parts else agent_id


# =============================================================================
# Analysis
# =============================================================================

def analyze_tool_usage(
    trajectories: List[Dict[str, Any]],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Analyze tool usage patterns."""
    by_tool: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_stage: Dict[str, Counter[str]] = defaultdict(Counter)
    by_category_stage: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    by_node: Dict[str, Counter[str]] = defaultdict(Counter)
    by_conversation: List[Dict[str, Any]] = []
    sequences_by_stage: Dict[str, List[List[Tuple[str, str]]]] = defaultdict(list)
    transition_counts: Counter[Tuple[str, str]] = Counter()

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

        conversation_tools: List[Tuple[str, str]] = []
        sequence: List[Tuple[str, str]] = []
        assistant_turn_count = 0

        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            if msg.get("role") != "assistant":
                continue

            assistant_turn_count += 1
            tool_calls = extract_tool_calls_from_message(msg, filter_bad=True)

            for call_idx, (tool_name, _args) in enumerate(tool_calls):
                category = get_tool_category(tool_name)

                by_tool[tool_name] += 1
                by_category[category] += 1
                by_stage[stage][tool_name] += 1
                by_node[node_id_prefix][tool_name] += 1
                by_category_stage[stage][category][tool_name] += 1

                conversation_tools.append((tool_name, category))
                sequence.append((tool_name, category))

        for i in range(len(sequence) - 1):
            transition_counts[(sequence[i][1], sequence[i + 1][1])] += 1

        by_conversation.append({
            "conversation_index": conv_idx,
            "stage": stage,
            "node_id": node_id_prefix,
            "agent_id": agent_id,
            "task_dir": traj_entry.get("_task_dir", ""),
            "local_index": traj_entry.get("_local_index", None),
            "assistant_turn_count": assistant_turn_count,
            "tool_calls": conversation_tools,
            "sequence": sequence,
        })

        if sequence:
            sequences_by_stage[stage].append(sequence)

    return {
        "by_tool": by_tool,
        "by_category": by_category,
        "by_stage": dict(by_stage),
        "by_category_stage": {
            stage: dict(cat_map)
            for stage, cat_map in by_category_stage.items()
        },
        "by_node": dict(by_node),
        "by_conversation": by_conversation,
        "sequences_by_stage": dict(sequences_by_stage),
        "transition_counts": transition_counts,
    }


# =============================================================================
# Matrix builders
# =============================================================================

def build_stage_category_matrix(
    analysis: Dict[str, Any],
    normalize: str = "none",
) -> Tuple[List[str], List[str], np.ndarray]:
    """Build stage x category matrix.

    normalize:
        none: raw counts
        row: row-normalized by stage
        node: average per node is handled elsewhere, not here
    """
    by_category_stage = analysis["by_category_stage"]

    stages = _ordered_stages(by_category_stage.keys())

    categories_seen = set()
    for stage_map in by_category_stage.values():
        categories_seen.update(stage_map.keys())

    categories = _ordered_categories(categories_seen)

    matrix = np.zeros((len(stages), len(categories)), dtype=float)

    for i, stage in enumerate(stages):
        for j, category in enumerate(categories):
            counter = by_category_stage.get(stage, {}).get(category, Counter())
            matrix[i, j] = float(sum(counter.values()))

    if normalize == "row":
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums > 0,
        )

    return stages, categories, matrix


def build_stage_tool_average_matrix(
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
    top_k: int = 15,
) -> Tuple[List[str], List[str], np.ndarray]:
    by_stage = analysis["by_stage"]

    stages = _ordered_stages(by_stage.keys())

    total_by_tool = Counter()
    for counter in by_stage.values():
        total_by_tool.update(counter)

    top_tools = [
        tool
        for tool, _ in total_by_tool.most_common(top_k)
        if tool not in BAD_TOOLS
    ]

    matrix = np.zeros((len(stages), len(top_tools)), dtype=float)

    for i, stage in enumerate(stages):
        node_count = len(nodes_by_stage.get(stage, []))
        denom = node_count if node_count > 0 else 1

        for j, tool in enumerate(top_tools):
            matrix[i, j] = by_stage.get(stage, Counter()).get(tool, 0) / denom

    return stages, top_tools, matrix


# =============================================================================
# Visualizations
# =============================================================================

def create_category_overall_plot(
    run_dir: Path,
    analysis: Dict[str, Any],
) -> Optional[Path]:
    by_category = analysis["by_category"]
    if not by_category:
        return None

    categories = _ordered_categories(by_category.keys())
    values = [by_category[c] for c in categories]
    colors = [CATEGORY_COLORS.get(c, PAPER_COLORS["gray"]) for c in categories]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    bars = ax.bar(
        np.arange(len(categories)),
        values,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=1.0,
    )

    total = sum(values)

    for bar, value in zip(bars, values):
        pct = value / total * 100 if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value)}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PAPER_COLORS["dark"],
        )

    ax.set_title("Overall tool usage by category", loc="left")
    ax.set_ylabel("Tool call count")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c.title()) for c in categories])
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.985,
        0.94,
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

    out = _ensure_viz_dir(run_dir) / "paper_tool_category_overall.pdf"
    return _save_figure(fig, out)


def create_category_donut_by_stage(
    run_dir: Path,
    analysis: Dict[str, Any],
) -> Optional[Path]:
    by_category_stage = analysis["by_category_stage"]
    if not by_category_stage:
        return None

    stages = [s for s in ["red", "black", "initial"] if s in by_category_stage]
    if not stages:
        stages = _ordered_stages(by_category_stage.keys())[:3]

    if not stages:
        return None

    fig, axes = plt.subplots(1, len(stages), figsize=(5.2 * len(stages), 5.4))

    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        stage_map = by_category_stage.get(stage, {})

        categories = []
        values = []

        for category in CATEGORY_ORDER:
            counter = stage_map.get(category, Counter())
            total = sum(counter.values())
            if total > 0:
                categories.append(category)
                values.append(total)

        if not values:
            ax.text(
                0.5,
                0.5,
                f"No data for {stage}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            ax.axis("off")
            continue

        colors = [CATEGORY_COLORS.get(c, PAPER_COLORS["gray"]) for c in categories]

        wedges, _texts, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
            startangle=90,
            counterclock=False,
            pctdistance=0.78,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
        )

        for text in autotexts:
            text.set_color("white")
            text.set_fontsize(9)
            text.set_fontweight("bold")

        total_sum = sum(values)

        ax.text(
            0,
            0.07,
            stage.title(),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=PAPER_COLORS["dark"],
        )
        ax.text(
            0,
            -0.09,
            f"n={total_sum}",
            ha="center",
            va="center",
            fontsize=10,
            color=PAPER_COLORS["gray"],
        )

        legend_labels = [
            f"{CATEGORY_LABELS.get(c, c)}: {v / total_sum * 100:.1f}%"
            for c, v in zip(categories, values)
        ]

        ax.legend(
            wedges,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
            fontsize=8,
            frameon=False,
        )

    fig.suptitle("Tool category composition by stage", y=1.03, fontsize=15, fontweight="bold")
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_category_donut_by_stage.pdf"
    return _save_figure(fig, out)


def create_stage_category_heatmap(
    run_dir: Path,
    analysis: Dict[str, Any],
    normalize: bool = False,
) -> Optional[Path]:
    stages, categories, matrix = build_stage_category_matrix(
        analysis,
        normalize="row" if normalize else "none",
    )

    if matrix.size == 0 or not stages or not categories:
        return None

    fig, ax = plt.subplots(figsize=(9.4, 5.3))

    cmap = "viridis" if normalize else "magma"
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    max_value = matrix.max() if matrix.size else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = f"{value:.2f}" if normalize else str(int(value))
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if max_value > 0 and value > max_value * 0.35 else PAPER_COLORS["dark"],
            )

    ax.set_title(
        "Stage-category tool usage heatmap" + (" (row-normalized)" if normalize else " (counts)"),
        loc="left",
    )
    ax.set_xlabel("Tool category")
    ax.set_ylabel("Node stage")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c.title()) for c in categories])
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels([s.title() for s in stages])

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Proportion" if normalize else "Count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    suffix = "normalized" if normalize else "counts"
    out = _ensure_viz_dir(run_dir) / f"paper_stage_category_heatmap_{suffix}.pdf"
    return _save_figure(fig, out)


def create_stage_tool_average_heatmap(
    run_dir: Path,
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> Optional[Path]:
    stages, tools, matrix = build_stage_tool_average_matrix(
        analysis,
        nodes_by_stage,
        top_k=15,
    )

    if matrix.size == 0 or not stages or not tools:
        return None

    fig, ax = plt.subplots(figsize=(12.8, 5.5))

    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")

    max_value = matrix.max() if matrix.size else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if max_value > 0 and value > max_value * 0.45 else PAPER_COLORS["dark"],
            )

    ax.set_title("Top tools by stage: average calls per node", loc="left")
    ax.set_xlabel("Tool")
    ax.set_ylabel("Node stage")
    ax.set_xticks(np.arange(len(tools)))
    ax.set_xticklabels([_short_tool_name(t, 18) for t in tools], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels([s.title() for s in stages])

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Avg calls / node", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_stage_tool_average_heatmap.pdf"
    return _save_figure(fig, out)


def create_red_black_difference_plot(
    run_dir: Path,
    analysis: Dict[str, Any],
) -> Optional[Path]:
    by_conversation = analysis["by_conversation"]

    red_conv = [c for c in by_conversation if c["stage"] == "red"]
    black_conv = [c for c in by_conversation if c["stage"] == "black"]

    if not red_conv or not black_conv:
        return None

    all_tools = set()
    for conv in red_conv + black_conv:
        for tool, _category in conv["tool_calls"]:
            if tool not in BAD_TOOLS:
                all_tools.add(tool)

    if not all_tools:
        return None

    red_avg = {}
    black_avg = {}

    for tool in all_tools:
        red_total = sum(1 for conv in red_conv for t, _ in conv["tool_calls"] if t == tool)
        black_total = sum(1 for conv in black_conv for t, _ in conv["tool_calls"] if t == tool)

        red_avg[tool] = red_total / len(red_conv)
        black_avg[tool] = black_total / len(black_conv)

    tools = sorted(
        all_tools,
        key=lambda t: abs(red_avg[t] - black_avg[t]),
        reverse=True,
    )[:15]

    diffs = [red_avg[t] - black_avg[t] for t in tools]
    y = np.arange(len(tools))

    colors = [
        STAGE_COLORS["red"] if d > 0 else STAGE_COLORS["black"]
        for d in diffs
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    ax.barh(y, diffs, color=colors, alpha=0.86, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=PAPER_COLORS["dark"], linewidth=1.1)

    ax.set_yticks(y)
    ax.set_yticklabels([_short_tool_name(t, 28) for t in tools])
    ax.invert_yaxis()

    ax.set_title("Red vs black tool usage difference", loc="left")
    ax.set_xlabel("Average calls per conversation: red minus black")
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, diff in zip(y, diffs):
        ha = "left" if diff >= 0 else "right"
        offset = 0.02 if diff >= 0 else -0.02
        ax.text(
            diff + offset,
            yi,
            f"{diff:+.2f}",
            ha=ha,
            va="center",
            fontsize=8.5,
            color=PAPER_COLORS["dark"],
        )

    ax.text(
        0.985,
        0.04,
        f"red conversations: {len(red_conv)}\nblack conversations: {len(black_conv)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#E5E7EB",
            alpha=0.95,
        ),
    )

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_red_black_tool_difference.pdf"
    return _save_figure(fig, out)


def create_stage_efficiency_plot(
    run_dir: Path,
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> Optional[Path]:
    by_stage = analysis["by_stage"]
    by_conversation = analysis["by_conversation"]

    if not by_stage:
        return None

    stages = _ordered_stages(by_stage.keys())
    conv_counts = Counter(c["stage"] for c in by_conversation)

    total_calls = []
    calls_per_node = []
    calls_per_conversation = []
    mean_sequence_length = []

    for stage in stages:
        calls = sum(by_stage.get(stage, Counter()).values())
        node_count = len(nodes_by_stage.get(stage, []))
        conv_count = conv_counts.get(stage, 0)

        total_calls.append(calls)
        calls_per_node.append(calls / node_count if node_count > 0 else 0)
        calls_per_conversation.append(calls / conv_count if conv_count > 0 else 0)

        seq_lengths = [
            len(c["sequence"])
            for c in by_conversation
            if c["stage"] == stage
        ]
        mean_sequence_length.append(float(np.mean(seq_lengths)) if seq_lengths else 0.0)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.8))
    x = np.arange(len(stages))
    colors = [STAGE_COLORS.get(s, PAPER_COLORS["gray"]) for s in stages]

    configs = [
        (axes[0], total_calls, "Total calls", "Count"),
        (axes[1], calls_per_node, "Calls per node", "Calls / node"),
        (axes[2], calls_per_conversation, "Calls per conversation", "Calls / conv"),
        (axes[3], mean_sequence_length, "Mean sequence length", "Calls / sequence"),
    ]

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
            label = f"{value:.2f}" if abs(value - round(value)) > 1e-9 else str(int(value))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([s.title() for s in stages], rotation=20, ha="right")
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Stage-level tool-call efficiency", y=1.03, fontsize=15, fontweight="bold")
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_stage_tool_efficiency.pdf"
    return _save_figure(fig, out)


def create_sequence_category_heatmap(
    run_dir: Path,
    analysis: Dict[str, Any],
    max_position: int = 30,
) -> Optional[Path]:
    sequences_by_stage = analysis["sequences_by_stage"]

    if not sequences_by_stage:
        return None

    stages = _ordered_stages(sequences_by_stage.keys())
    categories = CATEGORY_ORDER

    # One heatmap per stage: category count by sequence position.
    fig, axes = plt.subplots(
        len(stages),
        1,
        figsize=(13.5, max(3.2, 2.6 * len(stages))),
        sharex=True,
    )

    if len(stages) == 1:
        axes = [axes]

    has_any = False

    for ax, stage in zip(axes, stages):
        sequences = sequences_by_stage.get(stage, [])
        matrix = np.zeros((len(categories), max_position), dtype=float)

        for seq in sequences:
            for pos, (_tool, category) in enumerate(seq[:max_position]):
                if category in categories:
                    matrix[categories.index(category), pos] += 1

        if matrix.sum() <= 0:
            ax.text(
                0.5,
                0.5,
                f"No sequence data for {stage}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True

        # Normalize by number of sequences for comparability.
        matrix_norm = matrix / max(len(sequences), 1)

        im = ax.imshow(matrix_norm, aspect="auto", cmap="YlGnBu")

        ax.set_title(f"{stage.title()} nodes", loc="left")
        ax.set_ylabel("Category")
        ax.set_yticks(np.arange(len(categories)))
        ax.set_yticklabels([CATEGORY_LABELS.get(c, c.title()) for c in categories])

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    if not has_any:
        plt.close(fig)
        return None

    axes[-1].set_xlabel("Tool-call position in conversation")
    axes[-1].set_xticks(np.arange(0, max_position, 2))
    axes[-1].set_xticklabels([str(i + 1) for i in range(0, max_position, 2)])

    fig.suptitle("Category usage by sequence position", y=1.01, fontsize=15, fontweight="bold")
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_sequence_category_position_heatmap.pdf"
    return _save_figure(fig, out)


def create_tool_sequence_samples(
    run_dir: Path,
    analysis: Dict[str, Any],
    samples_per_stage: int = 5,
    max_steps: int = 40,
) -> Optional[Path]:
    sequences_by_stage = analysis["sequences_by_stage"]
    if not sequences_by_stage:
        return None

    stages = [s for s in ["red", "black", "initial"] if s in sequences_by_stage]
    if not stages:
        stages = _ordered_stages(sequences_by_stage.keys())[:3]

    if not stages:
        return None

    fig, axes = plt.subplots(
        len(stages),
        1,
        figsize=(14.5, max(4.0, 2.8 * len(stages))),
        sharex=True,
    )

    if len(stages) == 1:
        axes = [axes]

    has_any = False

    for ax, stage in zip(axes, stages):
        sequences = sequences_by_stage.get(stage, [])
        if not sequences:
            ax.text(
                0.5,
                0.5,
                f"No sequences for {stage}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True

        # Deterministic sample: choose earliest non-empty sequences.
        sampled = sequences[:samples_per_stage]

        for seq_idx, sequence in enumerate(sampled):
            clipped = sequence[:max_steps]

            for x_pos, (tool, category) in enumerate(clipped):
                ax.barh(
                    seq_idx,
                    1,
                    left=x_pos,
                    height=0.64,
                    color=CATEGORY_COLORS.get(category, PAPER_COLORS["gray"]),
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.88,
                )

                if len(clipped) <= 18:
                    ax.text(
                        x_pos + 0.5,
                        seq_idx,
                        _short_tool_name(tool, 10),
                        ha="center",
                        va="center",
                        fontsize=6,
                        rotation=45 if len(tool) > 12 else 0,
                        color=PAPER_COLORS["dark"],
                    )

        ax.set_title(f"{stage.title()} node tool-call samples", loc="left")
        ax.set_ylabel("Sample")
        ax.set_yticks(np.arange(len(sampled)))
        ax.set_yticklabels([f"Seq {i + 1}" for i in range(len(sampled))])
        ax.grid(axis="x", alpha=0.3)

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=CATEGORY_COLORS.get(cat, PAPER_COLORS["gray"]),
                markeredgecolor="none",
                markersize=8,
                label=CATEGORY_LABELS.get(cat, cat.title()),
            )
            for cat in CATEGORY_ORDER
        ]
        ax.legend(handles=handles, loc="upper right", ncol=min(5, len(handles)))

    if not has_any:
        plt.close(fig)
        return None

    axes[-1].set_xlabel("Tool-call step")
    axes[-1].set_xlim(0, max_steps)

    fig.suptitle("Representative tool-call sequences", y=1.01, fontsize=15, fontweight="bold")
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_tool_sequence_samples.pdf"
    return _save_figure(fig, out)


def create_category_transition_matrix(
    run_dir: Path,
    analysis: Dict[str, Any],
    normalize: bool = True,
) -> Optional[Path]:
    transition_counts: Counter[Tuple[str, str]] = analysis["transition_counts"]
    if not transition_counts:
        return None

    categories_seen = set()
    for src, dst in transition_counts:
        categories_seen.add(src)
        categories_seen.add(dst)

    categories = _ordered_categories(categories_seen)
    if not categories:
        return None

    matrix = np.zeros((len(categories), len(categories)), dtype=float)

    for (src, dst), count in transition_counts.items():
        if src in categories and dst in categories:
            i = categories.index(src)
            j = categories.index(dst)
            matrix[i, j] += count

    if matrix.sum() <= 0:
        return None

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums > 0,
        )
    else:
        display_matrix = matrix

    fig, ax = plt.subplots(figsize=(7.4, 6.2))

    im = ax.imshow(display_matrix, aspect="auto", cmap="viridis")

    max_value = display_matrix.max() if display_matrix.size else 0

    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            value = display_matrix[i, j]
            raw = matrix[i, j]
            if raw <= 0:
                label = ""
            elif normalize:
                label = f"{value:.2f}\n({int(raw)})"
            else:
                label = str(int(raw))

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if max_value > 0 and value > max_value * 0.38 else PAPER_COLORS["dark"],
            )

    ax.set_title(
        "Category transition matrix" + (" (row-normalized)" if normalize else " (counts)"),
        loc="left",
    )
    ax.set_xlabel("Next category")
    ax.set_ylabel("Current category")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c.title()) for c in categories], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels([CATEGORY_LABELS.get(c, c.title()) for c in categories])

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Transition probability" if normalize else "Count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    suffix = "normalized" if normalize else "counts"
    out = _ensure_viz_dir(run_dir) / f"paper_category_transition_matrix_{suffix}.pdf"
    return _save_figure(fig, out)


def create_top_tool_bar_by_category(
    run_dir: Path,
    analysis: Dict[str, Any],
    top_k_per_category: int = 8,
) -> Optional[Path]:
    by_tool = analysis["by_tool"]
    if not by_tool:
        return None

    categories = _ordered_categories({get_tool_category(t) for t in by_tool.keys()})
    categories = [c for c in categories if c != "other"] + ([c for c in categories if c == "other"])

    if not categories:
        return None

    fig, axes = plt.subplots(
        len(categories),
        1,
        figsize=(11.5, max(4.0, 3.0 * len(categories))),
    )

    if len(categories) == 1:
        axes = [axes]

    has_any = False

    for ax, category in zip(axes, categories):
        tools = [
            (tool, count)
            for tool, count in by_tool.items()
            if get_tool_category(tool) == category
        ]

        tools.sort(key=lambda x: x[1], reverse=True)
        tools = tools[:top_k_per_category]

        if not tools:
            ax.text(
                0.5,
                0.5,
                f"No data for {category}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True

        labels = [_short_tool_name(t, 35) for t, _ in tools]
        values = [count for _, count in tools]
        y = np.arange(len(labels))

        ax.barh(
            y,
            values,
            color=CATEGORY_COLORS.get(category, PAPER_COLORS["gray"]),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.8,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(f"{CATEGORY_LABELS.get(category, category.title())} tools", loc="left")
        ax.set_xlabel("Call count")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for yi, value in zip(y, values):
            ax.text(
                value,
                yi,
                f" {value}",
                ha="left",
                va="center",
                fontsize=8.5,
            )

    if not has_any:
        plt.close(fig)
        return None

    fig.suptitle("Top tools within each category", y=1.01, fontsize=15, fontweight="bold")
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_top_tools_by_category.pdf"
    return _save_figure(fig, out)


def create_visualizations(
    run_dir: Path,
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> List[Path]:
    """Create all visualization PDFs and return generated paths."""
    outputs: List[Path] = []

    plot_outputs = [
        create_category_overall_plot(run_dir, analysis),
        create_category_donut_by_stage(run_dir, analysis),
        create_stage_category_heatmap(run_dir, analysis, normalize=False),
        create_stage_category_heatmap(run_dir, analysis, normalize=True),
        create_stage_tool_average_heatmap(run_dir, analysis, nodes_by_stage),
        create_red_black_difference_plot(run_dir, analysis),
        create_stage_efficiency_plot(run_dir, analysis, nodes_by_stage),
        create_sequence_category_heatmap(run_dir, analysis),
        create_tool_sequence_samples(run_dir, analysis),
        create_category_transition_matrix(run_dir, analysis, normalize=True),
        create_category_transition_matrix(run_dir, analysis, normalize=False),
        create_top_tool_bar_by_category(run_dir, analysis),
    ]

    for path in plot_outputs:
        if path is not None:
            outputs.append(path)
            print(f"[INFO] saved: {path}")

    summary_path = save_summary_json(run_dir, analysis, nodes_by_stage)
    outputs.append(summary_path)
    print(f"[INFO] saved: {summary_path}")

    return outputs


# =============================================================================
# JSON summary
# =============================================================================

def save_summary_json(
    run_dir: Path,
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> Path:
    viz_dir = _ensure_viz_dir(run_dir)

    by_tool: Counter[str] = analysis["by_tool"]
    by_category: Counter[str] = analysis["by_category"]
    by_stage: Dict[str, Counter[str]] = analysis["by_stage"]
    by_category_stage = analysis["by_category_stage"]
    by_conversation = analysis["by_conversation"]
    transition_counts: Counter[Tuple[str, str]] = analysis["transition_counts"]

    output = {
        "bad_tools_excluded": sorted(BAD_TOOLS),
        "total_tool_calls": int(sum(by_tool.values())),
        "unique_tools": int(len(by_tool)),
        "tool_counts": _as_counter_dict(by_tool),
        "category_counts": _as_counter_dict(by_category),
        "stage_tool_counts": {
            stage: _as_counter_dict(counter)
            for stage, counter in by_stage.items()
        },
        "stage_category_counts": {
            stage: {
                category: int(sum(counter.values()))
                for category, counter in cat_map.items()
            }
            for stage, cat_map in by_category_stage.items()
        },
        "stage_node_counts": {
            stage: int(len(nodes))
            for stage, nodes in nodes_by_stage.items()
        },
        "conversation_count": int(len(by_conversation)),
        "transition_counts": {
            f"{src}->{dst}": int(count)
            for (src, dst), count in transition_counts.items()
        },
        "conversation_summaries": [
            {
                "conversation_index": int(c["conversation_index"]),
                "stage": c["stage"],
                "node_id": c["node_id"],
                "agent_id": c["agent_id"],
                "assistant_turn_count": int(c["assistant_turn_count"]),
                "tool_call_count": int(len(c["tool_calls"])),
                "category_counts": _as_counter_dict(
                    Counter(category for _tool, category in c["tool_calls"])
                ),
                "tool_counts": _as_counter_dict(
                    Counter(tool for tool, _category in c["tool_calls"])
                ),
            }
            for c in by_conversation
        ],
    }

    path = viz_dir / "tool_inspection_summary.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =============================================================================
# Console report
# =============================================================================

def print_statistics(
    analysis: Dict[str, Any],
    nodes_by_stage: Dict[str, List[Dict[str, Any]]],
) -> None:
    by_tool: Counter[str] = analysis["by_tool"]
    by_category: Counter[str] = analysis["by_category"]
    by_stage: Dict[str, Counter[str]] = analysis["by_stage"]
    by_category_stage = analysis["by_category_stage"]
    by_conversation = analysis["by_conversation"]
    sequences_by_stage = analysis["sequences_by_stage"]

    total_calls = sum(by_tool.values())

    print("\n" + "=" * 80)
    print("TOOL USAGE ANALYSIS REPORT")
    print("=" * 80)

    print("\n[Overall]")
    print(f"  Total tool calls:       {total_calls}")
    print(f"  Unique tools:           {len(by_tool)}")
    print(f"  Conversations:          {len(by_conversation)}")
    print(f"  Bad tools excluded:     {', '.join(sorted(BAD_TOOLS)) if BAD_TOOLS else 'None'}")

    non_empty = sum(1 for c in by_conversation if c["tool_calls"])
    print(f"  Non-empty conversations:{non_empty}")

    print("\n[Top tools]")
    for tool, count in by_tool.most_common(20):
        pct = count / total_calls * 100 if total_calls > 0 else 0
        print(f"  {tool:42s}: {count:6d} ({pct:5.1f}%)")

    print("\n[Tool categories]")
    for category in _ordered_categories(by_category.keys()):
        count = by_category[category]
        pct = count / total_calls * 100 if total_calls > 0 else 0
        print(f"  {category:15s}: {count:6d} ({pct:5.1f}%)")

    print("\n[Stage summaries]")
    conv_counts = Counter(c["stage"] for c in by_conversation)

    for stage in _ordered_stages(by_stage.keys()):
        stage_calls = sum(by_stage[stage].values())
        node_count = len(nodes_by_stage.get(stage, []))
        conv_count = conv_counts.get(stage, 0)

        print(f"\n  {stage.upper()}")
        print(f"    Node count:          {node_count}")
        print(f"    Conversation count:  {conv_count}")
        print(f"    Total tool calls:    {stage_calls}")
        print(f"    Avg per node:        {stage_calls / node_count:.2f}" if node_count else "    Avg per node:        N/A")
        print(f"    Avg per conversation:{stage_calls / conv_count:.2f}" if conv_count else "    Avg per conversation:N/A")

        print("\n    Category breakdown:")
        stage_cat_map = by_category_stage.get(stage, {})
        for category in _ordered_categories(stage_cat_map.keys()):
            cat_count = sum(stage_cat_map[category].values())
            cat_pct = cat_count / stage_calls * 100 if stage_calls > 0 else 0
            avg_conv = cat_count / conv_count if conv_count > 0 else 0
            print(f"      {category:15s}: {cat_count:6d} ({cat_pct:5.1f}%) avg={avg_conv:.2f}/conv")

        print("\n    Top tools:")
        for tool, count in by_stage[stage].most_common(12):
            pct = count / stage_calls * 100 if stage_calls > 0 else 0
            avg_conv = count / conv_count if conv_count > 0 else 0
            print(f"      {tool:38s}: {count:6d} ({pct:5.1f}%) avg={avg_conv:.2f}/conv")

    if "red" in by_stage and "black" in by_stage:
        print("\n[Red vs Black]")
        red_tools = set(by_stage["red"].keys())
        black_tools = set(by_stage["black"].keys())

        only_red = sorted(red_tools - black_tools)
        only_black = sorted(black_tools - red_tools)
        common = sorted(red_tools & black_tools)

        print(f"  Tools only in red:    {len(only_red)}")
        for tool in only_red[:20]:
            print(f"    - {tool}: {by_stage['red'][tool]}")

        print(f"  Tools only in black:  {len(only_black)}")
        for tool in only_black[:20]:
            print(f"    - {tool}: {by_stage['black'][tool]}")

        print(f"  Shared tools:         {len(common)}")

    print("\n[Sequence statistics]")
    for stage in _ordered_stages(sequences_by_stage.keys()):
        seqs = sequences_by_stage.get(stage, [])
        if not seqs:
            continue

        lengths = [len(seq) for seq in seqs]
        print(f"  {stage:12s}: n={len(seqs)}, mean={np.mean(lengths):.2f}, median={np.median(lengths):.2f}, min={min(lengths)}, max={max(lengths)}")

    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze tool usage patterns from trajectory data")
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