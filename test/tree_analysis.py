#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in {"test", "scripts", "analysis"} else SCRIPT_DIR
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


# =============================================================================
# Paper-style plotting
# =============================================================================

PAPER_COLORS = {
    "blue": "#2F5D8C",
    "orange": "#D9822B",
    "green": "#3A7D44",
    "red": "#B23A48",
    "purple": "#6C5B7B",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
    "background": "#FAFAFA",
}

STAGE_COLORS = {
    "initial": "#111827",
    "black": "#2F5D8C",
    "red": "#B23A48",
    "unknown": "#6B7280",
}


def apply_paper_style() -> None:
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


apply_paper_style()


# =============================================================================
# General helpers
# =============================================================================

def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        val = float(value)
        return val if math.isfinite(val) else None
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _clean_stage(stage: Any) -> str:
    if stage is None:
        return "unknown"
    stage = str(stage).strip()
    return stage if stage else "unknown"


def _short_id(node_id: Optional[str], n: int = 8) -> str:
    if not node_id:
        return "None"
    return str(node_id)[:n]


def _ensure_viz_dir(run_dir: Path) -> Path:
    viz_dir = run_dir / "visualize"
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _best_record(records: Sequence["NodeRecord"], attr: str, maximize: bool) -> Optional["NodeRecord"]:
    scored = [r for r in records if getattr(r, attr) is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: getattr(r, attr)) if maximize else min(scored, key=lambda r: getattr(r, attr))


def _polyfit_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    degree: int,
    n_points: int = 200,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(xs) < degree + 1:
        return None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < degree + 1:
        return None

    if len(np.unique(x)) < degree + 1:
        return None

    try:
        coeff = np.polyfit(x, y, degree)
        poly = np.poly1d(coeff)
        x_line = np.linspace(float(x.min()), float(x.max()), n_points)
        y_line = poly(x_line)
        return x_line, y_line
    except Exception:
        return None


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


# =============================================================================
# Data model
# =============================================================================

@dataclass
class NodeRecord:
    run_name: str
    node_id: str
    parent: Optional[str]
    stage: str
    depth: int
    time_key: Optional[float]
    metric: Optional[float]
    test_score: Optional[float]
    maximize: bool
    visits: Optional[int]
    reward: Optional[float]
    total_reward: Optional[float]
    is_buggy: Optional[bool]
    has_submission: Optional[bool]


# =============================================================================
# Loading
# =============================================================================

def discover_run_dir(run_dir: Optional[str], runs_root: Optional[str]) -> Path:
    if run_dir:
        p = Path(run_dir).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"run_dir not found: {p}")
        return p

    root = Path(runs_root).expanduser().resolve() if runs_root else DEFAULT_RUNS_DIR
    if not root.exists():
        raise SystemExit(f"runs root not found: {root}")

    candidates = [
        d for d in root.glob("ml_master*")
        if d.is_dir() and (d / "logs" / "uct_nodes").exists()
    ]
    if not candidates:
        raise SystemExit(f"no run dirs found under: {root}")

    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def load_tree_or_nodes(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    nodes_dir = run_dir / "logs" / "uct_nodes"
    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    if not nodes_dir.exists():
        return nodes_by_id

    for fp in sorted(nodes_dir.glob("*.json")):
        if fp.name == "node.json":
            continue

        obj = _read_json(fp)
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            data = dict(obj)
            data["_mtime"] = fp.stat().st_mtime
            nodes_by_id[obj["id"]] = data

    snapshot = _read_json(nodes_dir / "node.json")
    if isinstance(snapshot, dict):
        root = snapshot.get("root") if isinstance(snapshot.get("root"), dict) else snapshot

        def walk(node: Dict[str, Any]) -> None:
            nid = node.get("id")
            if isinstance(nid, str) and nid:
                merged = dict(node)
                if nid in nodes_by_id:
                    merged.update(nodes_by_id[nid])
                nodes_by_id[nid] = merged

            for child in node.get("children", []) or []:
                if isinstance(child, dict):
                    walk(child)

        if isinstance(root, dict):
            walk(root)

    return nodes_by_id


def load_grade_results(run_dir: Path) -> Dict[str, float]:
    candidates = [
        run_dir / "trajectories" / "task_0" / "grade_results.json",
        run_dir / "test" / "grade_results.json",
    ]

    for path in candidates:
        obj = _read_json(path)
        if not isinstance(obj, dict):
            continue

        out: Dict[str, float] = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                score = _safe_float(v.get("score"))
            else:
                score = _safe_float(v)

            if score is not None:
                out[str(k)] = score

        if out:
            return out

    return {}


def attach_test_scores(nodes_by_id: Dict[str, Dict[str, Any]], grade_results: Dict[str, float]) -> None:
    if not grade_results:
        return

    for node_id, node in nodes_by_id.items():
        score = grade_results.get(node_id)

        if score is None:
            short = node_id[:8]
            for sid, sval in grade_results.items():
                if sid.startswith(short) or short == sid[:8]:
                    score = sval
                    break

        if score is not None:
            node["test_score"] = score


def infer_maximize(
    nodes_by_id: Dict[str, Dict[str, Any]],
    force_minimize: bool,
    force_maximize: bool,
) -> bool:
    if force_minimize and force_maximize:
        raise SystemExit("cannot set both --force-minimize and --force-maximize")

    if force_minimize:
        return False
    if force_maximize:
        return True

    flags = [
        n.get("maximize")
        for n in nodes_by_id.values()
        if isinstance(n.get("maximize"), bool)
    ]
    if flags:
        return sum(1 for x in flags if x) >= len(flags) / 2

    return True


def compute_depths(nodes_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    parent_map = {nid: node.get("parent") for nid, node in nodes_by_id.items()}
    cache: Dict[str, int] = {}

    def depth(nid: str) -> int:
        if nid in cache:
            return cache[nid]

        d = 0
        cur = nid
        seen = set()

        while True:
            if cur in seen:
                cache[nid] = d
                return d

            seen.add(cur)
            parent = parent_map.get(cur)

            if not isinstance(parent, str) or parent not in nodes_by_id:
                cache[nid] = d
                return d

            cur = parent
            d += 1

    for nid in nodes_by_id:
        depth(nid)

    return cache


def build_node_records(run_dir: Path, force_minimize: bool, force_maximize: bool) -> List[NodeRecord]:
    nodes_by_id = load_tree_or_nodes(run_dir)
    if not nodes_by_id:
        return []

    attach_test_scores(nodes_by_id, load_grade_results(run_dir))
    maximize = infer_maximize(nodes_by_id, force_minimize, force_maximize)
    depths = compute_depths(nodes_by_id)

    records: List[NodeRecord] = []

    for node_id, node in nodes_by_id.items():
        created_at = _safe_float(node.get("created_at"))
        finish_time = _safe_float(node.get("finish_time"))
        snapshot_ts = _safe_float(node.get("snapshot_ts"))
        fallback_mtime = _safe_float(node.get("_mtime"))

        time_key = (
            created_at
            if created_at is not None
            else finish_time
            if finish_time is not None
            else snapshot_ts
            if snapshot_ts is not None
            else fallback_mtime
        )

        metric = _safe_float(node.get("metric"))
        if metric is None:
            metric = _safe_float(node.get("submission_score"))

        test_score = _safe_float(node.get("test_score"))

        records.append(
            NodeRecord(
                run_name=run_dir.name,
                node_id=node_id,
                parent=node.get("parent") if isinstance(node.get("parent"), str) else None,
                stage=_clean_stage(node.get("stage", "unknown")),
                depth=depths.get(node_id, 0),
                time_key=time_key,
                metric=metric,
                test_score=test_score,
                maximize=maximize,
                visits=_safe_int(node.get("visits")),
                reward=_safe_float(node.get("reward")),
                total_reward=_safe_float(node.get("total_reward")),
                is_buggy=node.get("is_buggy") if isinstance(node.get("is_buggy"), bool) else None,
                has_submission=node.get("has_submission") if isinstance(node.get("has_submission"), bool) else None,
            )
        )

    records.sort(key=lambda r: (float("inf") if r.time_key is None else r.time_key, r.node_id))
    return records


# =============================================================================
# Summaries
# =============================================================================

def get_initial_record(records: Sequence[NodeRecord]) -> Optional[NodeRecord]:
    if not records:
        return None

    initial_candidates = [r for r in records if r.stage == "initial"]
    if initial_candidates:
        return sorted(
            initial_candidates,
            key=lambda r: (float("inf") if r.time_key is None else r.time_key, r.depth),
        )[0]

    return sorted(
        records,
        key=lambda r: (float("inf") if r.time_key is None else r.time_key, r.depth),
    )[0]


def summarize_against_initial(records: Sequence[NodeRecord], attr: str, maximize: bool) -> Dict[str, Any]:
    initial = get_initial_record(records)
    initial_score = getattr(initial, attr) if initial else None
    scored = [r for r in records if getattr(r, attr) is not None]

    non_initial_scored = [
        r for r in scored
        if initial is None or r.node_id != initial.node_id
    ]

    if non_initial_scored:
        best = (
            max(non_initial_scored, key=lambda r: getattr(r, attr))
            if maximize
            else min(non_initial_scored, key=lambda r: getattr(r, attr))
        )
    else:
        best = None

    improvements: List[float] = []
    better_count = 0

    for r in scored:
        score = getattr(r, attr)
        if initial_score is None or score is None:
            continue

        delta = score - initial_score if maximize else initial_score - score
        improvements.append(delta)

        if delta > 0:
            better_count += 1

    return {
        "initial_score": initial_score,
        "best_score": getattr(best, attr) if best else None,
        "best_node_id": best.node_id if best else None,
        "best_stage": best.stage if best else None,
        "best_depth": best.depth if best else None,
        "nodes_with_score": len(scored),
        "better_than_initial": better_count,
        "better_ratio": (better_count / len(improvements)) if improvements else None,
        "mean_improvement": float(np.mean(improvements)) if improvements else None,
        "median_improvement": float(np.median(improvements)) if improvements else None,
    }


def make_stage_values(
    records: Sequence[NodeRecord],
    attr: str,
    stages: Sequence[str],
) -> Tuple[List[float], List[int], List[str]]:
    means: List[float] = []
    counts: List[int] = []
    labels: List[str] = []

    for stage in stages:
        vals = [
            float(getattr(r, attr))
            for r in records
            if r.stage == stage and getattr(r, attr) is not None
        ]

        if vals:
            means.append(float(np.mean(vals)))
            counts.append(len(vals))
        else:
            means.append(np.nan)
            counts.append(0)

        labels.append(stage.title())

    return means, counts, labels


def ordered_stages(records: Sequence[NodeRecord]) -> List[str]:
    preferred = ["initial", "black", "red"]
    seen = {r.stage for r in records}
    stages = [s for s in preferred if s in seen]
    stages += sorted(s for s in seen if s not in preferred)
    return stages


# =============================================================================
# Plots
# =============================================================================

def create_time_trend_plot(run_dir: Path, records: Sequence[NodeRecord]) -> Optional[Path]:
    timed = [r for r in records if r.time_key is not None]
    if not timed:
        return None

    initial = get_initial_record(records)
    non_initial_timed = [
        r for r in timed
        if initial is None or r.node_id != initial.node_id
    ]

    x_non_initial = np.arange(len(non_initial_timed), dtype=float) + 1.0
    x_initial = 0.0

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True)

    configs = [
        (
            axes[0],
            "metric",
            "Validation score trajectory",
            "Validation score",
            PAPER_COLORS["blue"],
        ),
        (
            axes[1],
            "test_score",
            "Test score trajectory",
            "Test score",
            PAPER_COLORS["red"],
        ),
    ]

    has_any = False

    for ax, attr, title, ylabel, color in configs:
        y_raw = [getattr(r, attr) for r in non_initial_timed]
        initial_y = getattr(initial, attr) if initial is not None else None

        mask_idx = [i for i, y in enumerate(y_raw) if y is not None]

        if not mask_idx and initial_y is None:
            ax.text(
                0.5,
                0.5,
                f"No data for {ylabel.lower()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True

        for stage in ordered_stages(records):
            idx = [
                i for i, r in enumerate(non_initial_timed)
                if r.stage == stage and y_raw[i] is not None
            ]
            if not idx:
                continue

            xs = x_non_initial[idx]
            ys = np.asarray([y_raw[i] for i in idx], dtype=float)

            ax.scatter(
                xs,
                ys,
                s=36,
                alpha=0.78,
                color=STAGE_COLORS.get(stage, PAPER_COLORS["gray"]),
                edgecolor="white",
                linewidth=0.55,
                label=stage.title(),
                zorder=3,
            )

        if mask_idx:
            xs_fit = np.asarray(mask_idx, dtype=float)
            ys = np.asarray([y_raw[i] for i in mask_idx], dtype=float)

            linear = _polyfit_curve(xs_fit, ys, degree=1)
            if linear is not None:
                ax.plot(
                    linear[0] + 1.0,
                    linear[1],
                    color=PAPER_COLORS["dark"],
                    linewidth=2.1,
                    label="Linear trend",
                    zorder=4,
                )

            rolling_window = max(3, min(10, len(mask_idx) // 5 if len(mask_idx) >= 15 else 3))
            if len(mask_idx) >= rolling_window:
                xs_sorted = np.asarray([i + 1.0 for i in mask_idx], dtype=float)
                ys_sorted = np.asarray([y_raw[i] for i in mask_idx], dtype=float)
                kernel = np.ones(rolling_window) / rolling_window
                rolling = np.convolve(ys_sorted, kernel, mode="valid")
                rolling_x = xs_sorted[rolling_window - 1:]

                ax.plot(
                    rolling_x,
                    rolling,
                    color=PAPER_COLORS["orange"],
                    linewidth=2.2,
                    linestyle="-",
                    label=f"Rolling mean ({rolling_window})",
                    zorder=5,
                )

            best = _best_record([non_initial_timed[i] for i in mask_idx], attr, records[0].maximize)
            if best is not None:
                best_idx = non_initial_timed.index(best) + 1
                best_y = getattr(best, attr)

                ax.scatter(
                    [best_idx],
                    [best_y],
                    s=140,
                    marker="*",
                    color=PAPER_COLORS["orange"],
                    edgecolor=PAPER_COLORS["dark"],
                    linewidth=0.7,
                    zorder=7,
                    label="Best node",
                )
                ax.annotate(
                    f"Best\n{_short_id(best.node_id)}",
                    xy=(best_idx, best_y),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=PAPER_COLORS["dark"],
                    arrowprops=dict(
                        arrowstyle="->",
                        color=PAPER_COLORS["gray"],
                        lw=0.8,
                    ),
                )

        if initial_y is not None:
            ax.scatter(
                [x_initial],
                [initial_y],
                s=150,
                marker="*",
                color=PAPER_COLORS["dark"],
                edgecolor="white",
                linewidth=0.8,
                zorder=8,
                label="Initial baseline",
            )
            ax.annotate(
                "Initial",
                xy=(x_initial, initial_y),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=PAPER_COLORS["dark"],
            )

        ax.axvline(x=0.5, color="#9CA3AF", linewidth=1.0, linestyle=":", alpha=0.9)
        ax.axvspan(-0.35, 0.5, color="#F3F4F6", alpha=0.9, zorder=0)

        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(ncol=4, loc="best")

    if not has_any:
        plt.close(fig)
        return None

    total_non_initial = len(non_initial_timed)
    if total_non_initial > 0:
        step = max(1, total_non_initial // 8)
        ticks = [0] + list(range(1, total_non_initial + 1, step))
        if ticks[-1] != total_non_initial:
            ticks.append(total_non_initial)

        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels(["Initial" if t == 0 else str(t) for t in ticks])

    axes[-1].set_xlabel("Temporal order of non-initial nodes")
    fig.suptitle("Search score progression", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = _ensure_viz_dir(run_dir) / "paper_score_over_time.pdf"
    return _save_figure(fig, out)


def create_depth_distribution_plot(run_dir: Path, records: Sequence[NodeRecord]) -> Optional[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.8), sharex=True)

    configs = [
        (axes[0], "metric", "Validation score distribution by tree depth", "Validation score", PAPER_COLORS["blue"]),
        (axes[1], "test_score", "Test score distribution by tree depth", "Test score", PAPER_COLORS["red"]),
    ]

    rng = np.random.default_rng(42)
    has_any = False

    for ax, attr, title, ylabel, color in configs:
        groups: Dict[int, List[float]] = {}

        for r in records:
            val = getattr(r, attr)
            if val is not None:
                groups.setdefault(r.depth, []).append(float(val))

        if not groups:
            ax.text(
                0.5,
                0.5,
                f"No data for {ylabel.lower()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True
        depths = sorted(groups)
        values = [groups[d] for d in depths]

        bp = ax.boxplot(
            values,
            positions=depths,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=PAPER_COLORS["dark"], linewidth=1.6),
            boxprops=dict(facecolor="#F3F4F6", color="#9CA3AF", linewidth=1.0),
            whiskerprops=dict(color="#9CA3AF", linewidth=1.0),
            capprops=dict(color="#9CA3AF", linewidth=1.0),
        )

        for patch in bp["boxes"]:
            patch.set_alpha(0.95)

        means = []
        for d in depths:
            vals = np.asarray(groups[d], dtype=float)
            means.append(float(vals.mean()))

            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), d) + jitter,
                vals,
                s=20,
                alpha=0.33,
                color=color,
                edgecolor="none",
                zorder=3,
            )

            ax.text(
                d,
                float(np.max(vals)),
                f"n={len(vals)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PAPER_COLORS["gray"],
            )

        ax.plot(
            depths,
            means,
            color=PAPER_COLORS["dark"],
            linewidth=2.0,
            marker="D",
            markersize=5,
            label="Mean",
            zorder=4,
        )

        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.legend(loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if not has_any:
        plt.close(fig)
        return None

    axes[-1].set_xlabel("Tree depth")
    fig.suptitle("Depth-wise score distribution", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = _ensure_viz_dir(run_dir) / "paper_score_by_depth_distribution.pdf"
    return _save_figure(fig, out)


def create_improvement_plot(run_dir: Path, records: Sequence[NodeRecord]) -> Optional[Path]:
    initial = get_initial_record(records)
    if initial is None:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.8), sharex=False)

    configs = [
        (axes[0], "metric", "Validation improvement relative to initial node", "Validation improvement"),
        (axes[1], "test_score", "Test improvement relative to initial node", "Test improvement"),
    ]

    has_any = False

    for ax, attr, title, xlabel in configs:
        initial_score = getattr(initial, attr)
        if initial_score is None:
            ax.text(
                0.5,
                0.5,
                f"Initial {attr} missing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        items = []
        for r in records:
            score = getattr(r, attr)
            if score is None:
                continue

            delta = score - initial_score if r.maximize else initial_score - score
            items.append((r.node_id, delta, r.stage, r.depth))

        if not items:
            ax.text(
                0.5,
                0.5,
                f"No improvement data for {attr}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True
        items.sort(key=lambda x: x[1])

        vals = [x[1] for x in items]
        y = np.arange(len(items))

        colors = [
            PAPER_COLORS["green"] if v > 0 else PAPER_COLORS["red"] if v < 0 else PAPER_COLORS["gray"]
            for v in vals
        ]

        ax.barh(y, vals, color=colors, alpha=0.88, height=0.72)
        ax.axvline(0, color=PAPER_COLORS["dark"], linewidth=1.2)

        better = sum(1 for v in vals if v > 0)
        worse = sum(1 for v in vals if v < 0)

        ax.text(
            0.985,
            0.05,
            f"better: {better}/{len(vals)}\nworse: {worse}/{len(vals)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.92,
                edgecolor="#E5E7EB",
            ),
        )

        if len(vals) <= 40:
            labels = [f"{_short_id(nid)} | {stage} | d={depth}" for nid, _, stage, depth in items]
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.set_yticks([])

        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nodes sorted by improvement")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if not has_any:
        plt.close(fig)
        return None

    fig.suptitle("Improvement over initial baseline", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = _ensure_viz_dir(run_dir) / "paper_improvement_vs_initial.pdf"
    return _save_figure(fig, out)


def create_stage_summary_bar(run_dir: Path, records: Sequence[NodeRecord]) -> Optional[Path]:
    stages = ordered_stages(records)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.3), sharex=True)

    configs = [
        (axes[0], "metric", "Validation score by node stage", "Validation score"),
        (axes[1], "test_score", "Test score by node stage", "Test score"),
    ]

    rng = np.random.default_rng(123)
    has_any = False

    for ax, attr, title, ylabel in configs:
        means, counts, labels = make_stage_values(records, attr, stages)
        means_arr = np.asarray(means, dtype=float)

        if not np.isfinite(means_arr).any():
            ax.text(
                0.5,
                0.5,
                f"No data for {ylabel.lower()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True
        x = np.arange(len(labels))
        colors = [STAGE_COLORS.get(s, PAPER_COLORS["gray"]) for s in stages]

        bars = ax.bar(
            x,
            means,
            color=colors,
            alpha=0.84,
            width=0.58,
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
        )

        for bar, n, mean_v in zip(bars, counts, means):
            if np.isfinite(mean_v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_v,
                    f"{mean_v:.4f}\n(n={n})",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=PAPER_COLORS["dark"],
                )

        for i, stage in enumerate(stages):
            vals = [
                float(getattr(r, attr))
                for r in records
                if r.stage == stage and getattr(r, attr) is not None
            ]
            if not vals:
                continue

            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=20,
                color=PAPER_COLORS["dark"],
                alpha=0.25,
                zorder=3,
            )

        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if not has_any:
        plt.close(fig)
        return None

    fig.suptitle("Stage-level score summary", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = _ensure_viz_dir(run_dir) / "paper_score_by_stage_summary.pdf"
    return _save_figure(fig, out)


def create_stage_depth_heatmap(run_dir: Path, records: Sequence[NodeRecord]) -> Optional[Path]:
    stages = ordered_stages(records)
    depths = sorted({r.depth for r in records})

    if not stages or not depths:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    configs = [
        (axes[0], "metric", "Mean validation score"),
        (axes[1], "test_score", "Mean test score"),
    ]

    has_any = False

    for ax, attr, title in configs:
        matrix = np.full((len(stages), len(depths)), np.nan)
        counts = np.zeros((len(stages), len(depths)), dtype=int)

        for i, stage in enumerate(stages):
            for j, depth in enumerate(depths):
                vals = [
                    float(getattr(r, attr))
                    for r in records
                    if r.stage == stage and r.depth == depth and getattr(r, attr) is not None
                ]
                if vals:
                    matrix[i, j] = float(np.mean(vals))
                    counts[i, j] = len(vals)

        if not np.isfinite(matrix).any():
            ax.text(
                0.5,
                0.5,
                f"No data for {title.lower()}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=PAPER_COLORS["gray"],
            )
            continue

        has_any = True
        masked = np.ma.masked_invalid(matrix)

        im = ax.imshow(masked, aspect="auto", cmap="viridis")

        for i in range(len(stages)):
            for j in range(len(depths)):
                if np.isfinite(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.3f}\n(n={counts[i, j]})",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )

        ax.set_title(title, loc="left")
        ax.set_xticks(np.arange(len(depths)))
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_yticks(np.arange(len(stages)))
        ax.set_yticklabels([s.title() for s in stages])
        ax.set_xlabel("Tree depth")
        ax.set_ylabel("Stage")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    if not has_any:
        plt.close(fig)
        return None

    fig.suptitle("Stage-depth score heatmap", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_stage_depth_score_heatmap.pdf"
    return _save_figure(fig, out)


def create_search_quality_scatter(run_dir: Path, records: Sequence[NodeRecord]) -> Optional[Path]:
    usable = [
        r for r in records
        if r.metric is not None and (r.visits is not None or r.reward is not None or r.total_reward is not None)
    ]

    if not usable:
        return None

    x_attr = "visits" if any(r.visits is not None for r in usable) else "total_reward"
    x_label = "UCT visits" if x_attr == "visits" else "Total reward"

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    has_any = False

    for stage in ordered_stages(records):
        items = [r for r in usable if r.stage == stage and getattr(r, x_attr) is not None]
        if not items:
            continue

        has_any = True

        xs = np.asarray([float(getattr(r, x_attr)) for r in items], dtype=float)
        ys = np.asarray([float(r.metric) for r in items], dtype=float)

        sizes = []
        for r in items:
            if r.reward is not None:
                sizes.append(40 + 120 * min(abs(r.reward), 1.0))
            else:
                sizes.append(52)

        ax.scatter(
            xs,
            ys,
            s=sizes,
            alpha=0.72,
            color=STAGE_COLORS.get(stage, PAPER_COLORS["gray"]),
            edgecolor="white",
            linewidth=0.7,
            label=stage.title(),
        )

    if not has_any:
        plt.close(fig)
        return None

    ax.set_title("UCT search signal vs validation score", loc="left")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation score")
    ax.grid(axis="both")
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_uct_signal_vs_score.pdf"
    return _save_figure(fig, out)


def create_bug_submission_status_plot(run_dir: Path, records: Sequence[NodeRecord]) -> Optional[Path]:
    has_bug = any(r.is_buggy is not None for r in records)
    has_submission = any(r.has_submission is not None for r in records)

    if not has_bug and not has_submission:
        return None

    stages = ordered_stages(records)
    labels = []
    buggy_rates = []
    submission_rates = []
    counts = []

    for stage in stages:
        items = [r for r in records if r.stage == stage]
        if not items:
            continue

        labels.append(stage.title())
        counts.append(len(items))

        bug_known = [r for r in items if r.is_buggy is not None]
        sub_known = [r for r in items if r.has_submission is not None]

        buggy_rates.append(
            sum(1 for r in bug_known if r.is_buggy) / len(bug_known)
            if bug_known else np.nan
        )
        submission_rates.append(
            sum(1 for r in sub_known if r.has_submission) / len(sub_known)
            if sub_known else np.nan
        )

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(10.2, 5.7))

    x = np.arange(len(labels))
    width = 0.36

    if has_bug:
        ax.bar(
            x - width / 2,
            buggy_rates,
            width,
            label="Buggy rate",
            color=PAPER_COLORS["red"],
            alpha=0.82,
            edgecolor="white",
            linewidth=1.0,
        )

    if has_submission:
        ax.bar(
            x + width / 2,
            submission_rates,
            width,
            label="Submission rate",
            color=PAPER_COLORS["green"],
            alpha=0.82,
            edgecolor="white",
            linewidth=1.0,
        )

    for i, n in enumerate(counts):
        ax.text(
            i,
            1.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=PAPER_COLORS["gray"],
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate")
    ax.set_title("Bug and submission status by stage", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = _ensure_viz_dir(run_dir) / "paper_bug_submission_status_by_stage.pdf"
    return _save_figure(fig, out)


# =============================================================================
# Reporting
# =============================================================================

def print_run_brief(run_dir: Path, records: Sequence[NodeRecord]) -> None:
    maximize = records[0].maximize if records else True
    stages = ordered_stages(records)

    print("\n" + "=" * 88)
    print(f"RUN: {run_dir.name}")
    print("=" * 88)
    print(f"Nodes: {len(records)}")
    print(f"Scoring direction: {'higher is better' if maximize else 'lower is better'}")
    print(f"Stages: {', '.join(stages)}")

    print("\n[Stage counts]")
    for stage in stages:
        print(f"  {stage:12s}: {sum(1 for r in records if r.stage == stage)}")

    for attr, label in [("metric", "VAL"), ("test_score", "TEST")]:
        summary = summarize_against_initial(records, attr, maximize)

        print(f"\n[{label}]")
        print(f"  initial_score:       {summary['initial_score']}")
        print(f"  best_score:          {summary['best_score']}")
        print(f"  best_node_id:        {summary['best_node_id']}")
        print(f"  best_stage/depth:    {summary['best_stage']} / {summary['best_depth']}")
        print(f"  nodes_with_score:    {summary['nodes_with_score']}")
        print(f"  better_than_initial: {summary['better_than_initial']} (ratio={summary['better_ratio']})")
        print(f"  mean_improvement:    {summary['mean_improvement']}")
        print(f"  median_improvement:  {summary['median_improvement']}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paper-style visualization for EvoMaster UCT run analysis"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Target run directory. If omitted, use latest ml_master* run under --runs-root.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help="Directory containing ml_master* runs. Used only when --run-dir is omitted.",
    )
    parser.add_argument("--force-minimize", action="store_true", help="Treat lower score as better")
    parser.add_argument("--force-maximize", action="store_true", help="Treat higher score as better")
    args = parser.parse_args()

    run_dir = discover_run_dir(args.run_dir, args.runs_root)
    print(f"[INFO] analyzing run: {run_dir}")

    records = build_node_records(run_dir, args.force_minimize, args.force_maximize)
    if not records:
        raise SystemExit(f"no node records found under: {run_dir}")

    plot_fns = [
        create_time_trend_plot,
        create_depth_distribution_plot,
        create_improvement_plot,
        create_stage_summary_bar,
        create_stage_depth_heatmap,
        create_search_quality_scatter,
        create_bug_submission_status_plot,
    ]

    outputs: List[Path] = []

    for fn in plot_fns:
        out = fn(run_dir, records)
        if out is not None:
            outputs.append(out)
            print(f"[INFO] saved: {out}")

    print_run_brief(run_dir, records)

    print("\n[INFO] generated files:")
    for out in outputs:
        print(f"  - {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())