#!/usr/bin/env python3
"""
Revised visualization for MLE-Lite best-node leakage analysis.

This script reads:
    <run_dir>/mle_lite_best_node_leakage_report.json

and generates:
    <run_dir>/visualize/best_node_leakage_revised.png
    <run_dir>/visualize/best_node_leakage_revised.pdf

Usage:
python scripts/plot_best_node_leakage.py \
    --run-dir /path/to/run_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def load_report(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "mle_lite_best_node_leakage_report.json"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Cannot find leakage report: {report_path}\n"
            "Please run check_mle_lite_leakage.py first."
        )

    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_list(report: dict[str, Any], key: str) -> list[Any]:
    value = report.get("hash_overlap_curve", {}).get(key, [])
    if value is None:
        return []
    return value


def to_float_array(values: list[Any]) -> np.ndarray:
    result = []

    for value in values:
        try:
            result.append(float(value))
        except Exception:
            result.append(np.nan)

    return np.array(result, dtype=float)


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f7",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,

            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,

            "axes.grid": True,
            "grid.color": "#cfcfcf",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,

            "lines.linewidth": 2.2,
            "lines.markersize": 5.0,

            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def style_inset_box(box_ax) -> None:
    """
    Make an inset axes look like a fixed-size text box.
    """
    box_ax.set_facecolor("white")
    box_ax.set_xticks([])
    box_ax.set_yticks([])
    box_ax.set_xlim(0, 1)
    box_ax.set_ylim(0, 1)

    for spine in box_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#dddddd")
        spine.set_linewidth(0.9)

    box_ax.patch.set_alpha(0.96)


def plot_leakage_curve(run_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    curve = report.get("hash_overlap_curve", {})

    sizes = to_float_array(safe_list(report, "dataset_sizes"))
    exact_rates = to_float_array(safe_list(report, "hash_overlap_rates"))
    nn_rates = to_float_array(safe_list(report, "nn_overlap_rates"))

    if sizes.size == 0 or exact_rates.size == 0:
        raise ValueError(
            "The leakage report does not contain a valid hash_overlap_curve. "
            "Expected keys: dataset_sizes, hash_overlap_rates, nn_overlap_rates."
        )

    nn_threshold = curve.get("nn_threshold", report.get("nn_threshold", None))

    exact_final = float(exact_rates[-1]) if exact_rates.size else 0.0
    nn_final = float(nn_rates[-1]) if nn_rates.size else 0.0

    visualize_dir = run_dir / "visualize"
    visualize_dir.mkdir(parents=True, exist_ok=True)

    png_path = visualize_dir / "best_node_leakage_revised.png"
    pdf_path = visualize_dir / "best_node_leakage_revised.pdf"

    set_paper_style()

    fig, ax = plt.subplots(figsize=(5.2, 4.8))

    # Highlight exact hash overlap in red.
    # Use blue for nearest-neighbor overlap.
    exact_color = "#D62728"
    nn_color = "#4C78A8"

    ax.plot(
        sizes,
        exact_rates,
        marker="o",
        color=exact_color,
        label="Exact hash overlap",
        zorder=3,
    )

    has_nn_curve = nn_rates.size > 0 and not np.all(np.isnan(nn_rates))

    if has_nn_curve:
        nn_label = "Nearest-neighbor overlap"
        if nn_threshold is not None:
            nn_label += f" ($\\tau={nn_threshold}$)"

        ax.plot(
            sizes,
            nn_rates,
            marker="^",
            linestyle="--",
            color=nn_color,
            label=nn_label,
            zorder=3,
        )
    else:
        nn_label = "Nearest-neighbor overlap"

    ax.axhline(
        exact_final,
        color=exact_color,
        linestyle=":",
        linewidth=1.8,
        alpha=0.8,
    )

    if has_nn_curve and not np.isnan(nn_final):
        ax.axhline(
            nn_final,
            color=nn_color,
            linestyle=":",
            linewidth=1.8,
            alpha=0.8,
        )

    x_min = float(np.nanmin(sizes))
    x_max = float(np.nanmax(sizes))
    x_text = x_max * 1.005

    ax.text(
        x_text,
        exact_final,
        f"{exact_final * 100:.1f}%",
        color=exact_color,
        fontsize=9.5,
        fontweight="bold",
        va="center",
        ha="left",
    )

    if has_nn_curve and not np.isnan(nn_final):
        ax.text(
            x_text,
            nn_final,
            f"{nn_final * 100:.1f}%",
            color=nn_color,
            fontsize=9.5,
            fontweight="bold",
            va="center",
            ha="left",
        )

    ax.set_xlabel("Test Dataset Size", fontweight="bold")
    ax.set_ylabel("Overlap Rate (%)", fontweight="bold")

    y_candidates = [exact_rates]
    if has_nn_curve:
        y_candidates.append(nn_rates)

    y_all = np.concatenate(y_candidates)
    y_all = y_all[~np.isnan(y_all)]

    if y_all.size:
        y_upper = max(0.05, float(np.max(y_all)) * 1.12)
        y_upper = min(1.05, max(0.12, y_upper))
    else:
        y_upper = 1.0

    ax.set_ylim(0.0, y_upper)
    ax.set_xlim(x_min, x_max * 1.06)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ============================================================
    # Stacked in-plot boxes.
    # Summary box on top, legend box below.
    # Both are centered, larger, and non-overlapping.
    # ============================================================

    summary_x = 0.055
    summary_y = 0.255
    summary_w = 0.490
    summary_h = 0.125

    legend_x = 0.055
    legend_y = 0.075
    legend_w = 0.690
    legend_h = 0.145

    summary_fontsize = 8.2
    legend_fontsize = 8.2

    summary_ax = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(summary_x, summary_y, summary_w, summary_h),
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0,
    )
    style_inset_box(summary_ax)

    summary_ax.text(
        0.050,
        0.66,
        f"Exact hash final: {exact_final * 100:.2f}%",
        ha="left",
        va="center",
        fontsize=summary_fontsize,
        fontweight="bold",
        color="black",
    )
    summary_ax.text(
        0.050,
        0.32,
        f"NN final: {nn_final * 100:.2f}%",
        ha="left",
        va="center",
        fontsize=summary_fontsize,
        fontweight="bold",
        color="black",
    )

    legend_ax = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(legend_x, legend_y, legend_w, legend_h),
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0,
    )
    style_inset_box(legend_ax)

    # Manual legend row 1: exact hash overlap
    legend_ax.plot(
        [0.050, 0.145],
        [0.68, 0.68],
        color=exact_color,
        linewidth=2.8,
        solid_capstyle="round",
        clip_on=False,
    )
    legend_ax.plot(
        [0.097],
        [0.68],
        marker="o",
        markersize=7.0,
        color=exact_color,
        clip_on=False,
    )
    legend_ax.text(
        0.185,
        0.68,
        "Exact hash overlap",
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        fontweight="bold",
        color="black",
    )

    # Manual legend row 2: nearest-neighbor overlap
    legend_ax.plot(
        [0.050, 0.145],
        [0.32, 0.32],
        color=nn_color,
        linewidth=2.8,
        linestyle="--",
        solid_capstyle="round",
        clip_on=False,
    )
    legend_ax.plot(
        [0.097],
        [0.32],
        marker="^",
        markersize=7.5,
        color=nn_color,
        clip_on=False,
    )
    legend_ax.text(
        0.185,
        0.32,
        nn_label,
        ha="left",
        va="center",
        fontsize=legend_fontsize,
        fontweight="bold",
        color="black",
    )

    fig.subplots_adjust(left=0.16, right=0.95, top=0.96, bottom=0.15)

    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help=(
            "Path to the MLE-Lite run directory containing "
            "mle_lite_best_node_leakage_report.json."
        ),
    )

    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    report = load_report(run_dir)
    png_path, pdf_path = plot_leakage_curve(run_dir, report)

    print("Saved revised leakage visualization:")
    print(f"- {png_path}")
    print(f"- {pdf_path}")


if __name__ == "__main__":
    main()