#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate Figure 6 from the 44 held-out LOOCV predictions."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
INPUT_PATH = ROOT_DIR / "消融实验_predictions.xlsx"
OUTPUT_DIR = BASE_DIR / "Figure6_output"
OUTPUT_PREFIX = OUTPUT_DIR / "Figure6_agreement"

SHEET_NAME = "LOO44_predictions"
METRICS_SHEET_NAME = "Metrics"
TRUE_COLUMN = "y_true"
BP_COLUMN = "BP_NN_pred"
GABP_COLUMN = "GA_BP_Pinball_tau0.2_pred"
BP_MODEL_NAME = "BP_NN"
GABP_MODEL_NAME = "GA_BP_Pinball_tau0.2"

MM_TO_INCH = 1 / 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 82

COLORS = {
    "actual": "#1F5F99",
    "predicted": "#D94B3D",
    "over": "#F6D98F",
    "under": "#B9DCE6",
    "grid": "#D8E1E8",
    "text": "#20252A",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate all 44 held-out predictions."""
    data = pd.read_excel(path, sheet_name=SHEET_NAME)
    required = [TRUE_COLUMN, BP_COLUMN, GABP_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {SHEET_NAME}: {missing}")

    data = data.loc[:, required].apply(pd.to_numeric, errors="raise")
    if len(data) != 44:
        raise ValueError(f"Expected 44 LOOCV predictions, found {len(data)}")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Prediction data contain non-finite values")

    return data.sort_values(TRUE_COLUMN, kind="stable").reset_index(drop=True)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the metrics reported by the model output workbook."""
    error = predicted - actual
    absolute_error = np.abs(error)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MedAE": float(median_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
        "EVS": float(explained_variance_score(actual, predicted)),
        "MAPE_%": float(np.mean(absolute_error / np.abs(actual)) * 100.0),
        "UR_%": float(np.mean(error < 0.0) * 100.0),
    }


def validate_reported_metrics(
    path: Path,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Verify that plotted predictions reproduce the workbook metrics."""
    reported = pd.read_excel(path, sheet_name=METRICS_SHEET_NAME)
    required_columns = [
        "Model",
        "RMSE",
        "MAE",
        "MedAE",
        "R2",
        "EVS",
        "MAPE_%",
        "UR_%",
    ]
    missing = [column for column in required_columns if column not in reported.columns]
    if missing:
        raise ValueError(f"Missing columns in {METRICS_SHEET_NAME}: {missing}")

    reported = reported.loc[:, required_columns].set_index("Model")
    recalculated_rows = []
    for model_name, predicted in predictions.items():
        if model_name not in reported.index:
            raise ValueError(f"Missing model in {METRICS_SHEET_NAME}: {model_name}")
        values = calculate_metrics(actual, predicted)
        for metric_name, calculated_value in values.items():
            reported_value = float(reported.loc[model_name, metric_name])
            if not np.isclose(calculated_value, reported_value, rtol=1e-8, atol=1e-6):
                raise ValueError(
                    f"{model_name} {metric_name} mismatch: "
                    f"predictions={calculated_value:.9f}, "
                    f"Metrics sheet={reported_value:.9f}"
                )
        recalculated_rows.append({"Model": model_name, **values})

    return pd.DataFrame(recalculated_rows)


def draw_panel(
    ax: plt.Axes,
    rank: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    panel_label: str,
) -> None:
    """Draw one model panel with explicit over- and underestimation regions."""
    overestimated = predicted >= actual
    underestimated = predicted < actual

    ax.fill_between(
        rank,
        actual,
        predicted,
        where=overestimated,
        interpolate=True,
        color=COLORS["over"],
        alpha=0.62,
        linewidth=0,
        zorder=1,
    )
    ax.fill_between(
        rank,
        actual,
        predicted,
        where=underestimated,
        interpolate=True,
        color=COLORS["under"],
        alpha=0.72,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        rank,
        actual,
        color=COLORS["actual"],
        linewidth=1.25,
        marker="o",
        markersize=2.4,
        markeredgewidth=0,
        zorder=3,
    )
    ax.plot(
        rank,
        predicted,
        color=COLORS["predicted"],
        linewidth=1.15,
        linestyle=(0, (3.2, 2.0)),
        marker="o",
        markersize=2.1,
        markeredgewidth=0,
        zorder=4,
    )

    score = r2_score(actual, predicted)
    ax.text(
        0.025,
        0.91,
        f"R² = {score:.3f}",
        transform=ax.transAxes,
        fontsize=7,
        color=COLORS["text"],
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#B8C0C8",
            "linewidth": 0.55,
        },
        zorder=5,
    )
    ax.set_title(title, pad=7, fontweight="bold", color=COLORS["text"])
    ax.text(
        -0.105,
        1.045,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        color=COLORS["text"],
    )
    ax.set_xlabel("Project rank (sorted by actual cost)")
    ax.set_xticks([1, 10, 20, 30, 40, 44])
    ax.set_xlim(1, 44)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, linestyle=(0, (2, 2)))
    ax.tick_params(axis="both", width=0.65, length=3)


def main() -> None:
    data = load_predictions(INPUT_PATH)
    rank = np.arange(1, len(data) + 1)
    actual = data[TRUE_COLUMN].to_numpy(dtype=float)
    bp_pred = data[BP_COLUMN].to_numpy(dtype=float)
    gabp_pred = data[GABP_COLUMN].to_numpy(dtype=float)
    verified_metrics = validate_reported_metrics(
        INPUT_PATH,
        actual,
        {
            BP_MODEL_NAME: bp_pred,
            GABP_MODEL_NAME: gabp_pred,
        },
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, FIGURE_HEIGHT_MM * MM_TO_INCH),
        sharey=True,
    )
    draw_panel(axes[0], rank, actual, bp_pred, "BP neural network", "a")
    draw_panel(
        axes[1],
        rank,
        actual,
        gabp_pred,
        "Asymmetric GA–BP (τ = 0.20)",
        "b",
    )
    axes[0].set_ylabel("Cost (10⁴ CNY)")

    minimum = min(actual.min(), bp_pred.min(), gabp_pred.min())
    maximum = max(actual.max(), bp_pred.max(), gabp_pred.max())
    lower_limit = min(0.0, np.floor(minimum / 500.0) * 500.0)
    upper_limit = np.ceil(maximum / 1000.0) * 1000.0
    axes[0].set_ylim(lower_limit, upper_limit)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["actual"],
            linewidth=1.25,
            marker="o",
            markersize=3,
            label="Actual cost",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["predicted"],
            linewidth=1.15,
            linestyle=(0, (3.2, 2.0)),
            marker="o",
            markersize=3,
            label="Predicted cost",
        ),
        Patch(
            facecolor=COLORS["over"],
            edgecolor="none",
            alpha=0.62,
            label="Overestimation",
        ),
        Patch(
            facecolor=COLORS["under"],
            edgecolor="none",
            alpha=0.72,
            label="Underestimation",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        handlelength=2.4,
        columnspacing=1.35,
        handletextpad=0.5,
    )
    fig.text(
        0.995,
        0.995,
        "n = 44 held-out projects",
        ha="right",
        va="top",
        fontsize=6.8,
        color="#68727C",
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.20, top=0.78, wspace=0.16)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    bp_abs_error = np.abs(bp_pred - actual)
    gabp_abs_error = np.abs(gabp_pred - actual)
    improved = int(np.sum(gabp_abs_error < bp_abs_error))
    print(f"Loaded {len(data)} held-out LOOCV projects")
    print("Verified metrics:")
    print(verified_metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"GA-BP has a smaller absolute error for {improved}/{len(data)} projects")
    print(f"Saved figure files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
