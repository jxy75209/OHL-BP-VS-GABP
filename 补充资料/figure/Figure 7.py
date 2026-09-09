#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare underestimation risk for BP and asymmetric GA-BP."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Underestimation risk figure"
OUTPUT_STEM = OUTPUT_DIR / "Figure_underestimation_risk_dual_axis"

# Results across the 44 held-out LOOCV predictions.
MODELS = ["BP", "GA-BP"]
UR_VALUES = np.array([52.27, 40.91], dtype=float)
CUMULATIVE_UNDERESTIMATION = np.array([12308.29, 9833.43], dtype=float)
UR_BLUE = "#4C86B9"
CUMULATIVE_RED = "#C74355"

FIGURE_WIDTH_IN = 5.40  # approximately 137 mm for a more compact layout
FIGURE_HEIGHT_IN = 3.20
PNG_DPI = 600


def configure_style() -> None:
    """Apply a compact publication style with editable vector text."""
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_figure() -> plt.Figure:
    """Create a grouped bar chart with separate axes for the two risk metrics."""
    bar_width = 0.08
    small_gap = 0.02
    large_gap = 0.04

    # Four consecutive bars: BP-UR, BP-cumulative, GA-BP-UR, GA-BP-cumulative.
    # The between-model gap is exactly twice the within-model gap.
    left_edges = np.array(
        [
            0.0,
            bar_width + small_gap,
            2.0 * bar_width + small_gap + large_gap,
            3.0 * bar_width + 2.0 * small_gap + large_gap,
        ]
    )
    bar_centres = left_edges + bar_width / 2.0
    bar_centres -= np.mean(bar_centres)
    ur_x = bar_centres[[0, 2]]
    cumulative_x = bar_centres[[1, 3]]
    model_x = np.array(
        [
            np.mean(bar_centres[:2]),
            np.mean(bar_centres[2:]),
        ]
    )
    difference_pp = UR_VALUES[0] - UR_VALUES[1]
    cumulative_reduction_pct = (
        1.0 - CUMULATIVE_UNDERESTIMATION[1] / CUMULATIVE_UNDERESTIMATION[0]
    ) * 100.0

    fig, ax_left = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        constrained_layout=True,
    )
    ax_right = ax_left.twinx()

    bars_ur = ax_left.bar(
        ur_x,
        UR_VALUES,
        width=bar_width,
        color=UR_BLUE,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label="Underestimation rate",
    )
    bars_cumulative = ax_right.bar(
        cumulative_x,
        CUMULATIVE_UNDERESTIMATION,
        width=bar_width,
        color=CUMULATIVE_RED,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label="Cumulative underestimation magnitude",
    )

    for bar, value in zip(bars_ur, UR_VALUES):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
        )
    for bar, value in zip(bars_cumulative, CUMULATIVE_UNDERESTIMATION):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 250,
            f"{value:,.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

    ax_left.set_ylabel("Underestimation rate (%)", color="black")
    ax_right.set_ylabel(
        "Cumulative underestimation magnitude (10⁴ CNY)",
        color="black",
    )

    ax_left.set_xticks(model_x, MODELS)
    ax_left.set_ylim(0, 65)
    ax_left.set_yticks(np.arange(0, 61, 10))
    # The 0–16,250 range aligns each 2,500-unit right-axis interval with
    # each 10-percentage-point left-axis interval.
    ax_right.set_ylim(0, 16250)
    ax_right.set_yticks(np.arange(0, 15001, 2500))
    ax_right.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

    ax_left.yaxis.grid(True, color="#D9DDE3", linewidth=0.6, linestyle=(0, (2, 3)))
    ax_left.set_axisbelow(True)
    ax_left.tick_params(axis="x", length=0, pad=4)
    ax_left.tick_params(axis="y", direction="in", length=3, colors="black")
    ax_right.tick_params(axis="y", direction="in", length=3, colors="black")
    ax_left.spines["left"].set_color("black")
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["right"].set_color("black")
    ax_right.spines["top"].set_visible(False)

    ax_left.legend(
        [bars_ur[0], bars_cumulative[0]],
        ["Underestimation rate", "Cumulative underestimation magnitude"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        columnspacing=1.8,
    )
    ax_left.text(
        0.25,
        0.93,
        f"UR reduction: {difference_pp:.2f} percentage points",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="black",
    )
    ax_left.text(
        0.75,
        0.93,
        f"Cumulative reduction: {cumulative_reduction_pct:.2f}%",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="black",
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    """Export a high-resolution preview and editable vector versions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=PNG_DPI)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))


def main() -> None:
    configure_style()
    fig = make_figure()
    save_figure(fig)
    plt.close(fig)
    print(f"Saved figure files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
