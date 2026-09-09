#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解释7特征BP与GA-BP模型。

模型参数来自“BP VS GABP.py”。代码遍历128个特征组合，
主图使用标准化SHAP值，CSV同时保留万元贡献值。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning


# 配置

BASE_DIR = Path(__file__).resolve().parent
MODEL_SCRIPT = BASE_DIR / "BP VS GABP.py"
DATA_PATH = BASE_DIR / "bp1-data.csv"
OUTPUT_DIR = BASE_DIR / "SHAP 7 features PNG output"

RANDOM_STATE = 53
PREDICTION_BATCH_SIZE = 8192
LOCAL_PROJECT_ID: int | None = None

FEATURE_COLUMNS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
FEATURE_NAMES = [
    "Steel price",
    "OPGW erection length",
    "Conductor unit price",
    "Conductor erection length",
    "Conductor cross-sectional area",
    "Number of crossings",
    "Voltage level",
]
CATEGORICAL_FEATURE_INDEXES = {4, 6}

MODEL_SPECS = (
    ("BP", "BP_NN", "BP"),
    ("GA-BP", "GA_BP_tau0.2", "GA_BP"),
)

# 论文配色
DEEP_RED = "#B2182B"
VERMILION = "#E34A33"
ORANGE = "#FDAE61"
PALE_YELLOW = "#FEE391"
NEUTRAL_GREY = "#F0F2F5"
LIGHT_BLUE = "#ABD9E9"
MID_BLUE = "#4393C3"
DEEP_BLUE = "#2166AC"

MODEL_COLORS = {"BP": DEEP_BLUE, "GA-BP": DEEP_RED}

# 低特征值为深蓝，高特征值为深红
FEATURE_VALUE_CMAP = LinearSegmentedColormap.from_list(
    "feature_value_low_blue_high_red",
    [DEEP_BLUE, MID_BLUE, LIGHT_BLUE, NEUTRAL_GREY, PALE_YELLOW, ORANGE, VERMILION, DEEP_RED],
)

FONT_FAMILY = "Times New Roman"
DOUBLE_COLUMN_WIDTH_IN = 7.09
PNG_DPI = 600

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [FONT_FAMILY],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


@dataclass
class SHAPResult:
    """单个模型的SHAP结果。"""

    label: str
    key: str
    slug: str
    predicted_cost: np.ndarray
    predicted_standardized: np.ndarray
    base_standardized: np.ndarray
    shap_standardized: np.ndarray
    importance_standardized: np.ndarray
    target_mean: float
    target_scale: float
    max_additivity_error: float


def load_model_module(path: Path):
    """导入模型脚本。"""
    if not path.exists():
        raise FileNotFoundError(f"Model script not found: {path}")
    specification = importlib.util.spec_from_file_location("bp_gabp_model_module", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import model script: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def predict_in_batches(predict_function, values: np.ndarray) -> np.ndarray:
    """分批预测。"""
    values = np.asarray(values, dtype=float)
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), PREDICTION_BATCH_SIZE):
        stop = min(start + PREDICTION_BATCH_SIZE, len(values))
        outputs.append(
            np.asarray(predict_function(values[start:stop]), dtype=float).reshape(-1)
        )
    return np.concatenate(outputs)


def exact_interventional_shap(
    predict_function,
    explained_values: np.ndarray,
    background_values: np.ndarray,
    model_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """精确计算经验干预SHAP。"""
    explained_values = np.asarray(explained_values, dtype=float)
    background_values = np.asarray(background_values, dtype=float)
    if explained_values.ndim != 2 or background_values.ndim != 2:
        raise ValueError("Explained values and background values must be two-dimensional.")
    if explained_values.shape[1] != background_values.shape[1]:
        raise ValueError("Explained and background data must contain the same features.")

    n_projects, n_features = explained_values.shape
    if n_features != 7:
        raise ValueError(f"Expected 7 features, received {n_features}.")
    if len(background_values) == 0:
        raise ValueError("The SHAP background cannot be empty.")

    n_coalitions = 1 << n_features
    coalition_ids = np.arange(n_coalitions, dtype=np.int64)
    feature_bits = 1 << np.arange(n_features, dtype=np.int64)
    membership = (coalition_ids[:, None] & feature_bits[None, :]) != 0
    coalition_sizes = membership.sum(axis=1)
    weights_by_size = np.asarray(
        [
            math.factorial(size)
            * math.factorial(n_features - size - 1)
            / math.factorial(n_features)
            for size in range(n_features)
        ],
        dtype=float,
    )

    shap_values = np.zeros((n_projects, n_features), dtype=float)
    base_values = np.zeros(n_projects, dtype=float)

    for project_index, project in enumerate(explained_values):
        # 维度：组合×背景项目×特征
        coalition_rows = np.where(
            membership[:, None, :],
            project[None, None, :],
            background_values[None, :, :],
        )
        predictions = predict_in_batches(
            predict_function,
            coalition_rows.reshape(-1, n_features),
        )
        coalition_values = predictions.reshape(
            n_coalitions,
            len(background_values),
        ).mean(axis=1)
        base_values[project_index] = coalition_values[0]

        for feature_index, feature_bit in enumerate(feature_bits):
            eligible = (coalition_ids & feature_bit) == 0
            subsets = coalition_ids[eligible]
            marginal = (
                coalition_values[subsets | feature_bit]
                - coalition_values[subsets]
            )
            shap_values[project_index, feature_index] = np.sum(
                weights_by_size[coalition_sizes[subsets]] * marginal
            )

        if (project_index + 1) % 10 == 0 or project_index + 1 == n_projects:
            print(
                f"  {model_label}: completed {project_index + 1}/{n_projects} projects"
            )

    return shap_values, base_values


def fit_and_explain(
    model,
    label: str,
    key: str,
    slug: str,
    X: pd.DataFrame,
    y: np.ndarray,
    background: np.ndarray,
) -> SHAPResult:
    """拟合模型并计算标准化SHAP。"""
    print(f"Fitting {label} ({key})...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y)

    predicted_cost = np.asarray(model.predict(X), dtype=float).reshape(-1)

    if not hasattr(model, "named_steps") or "model" not in model.named_steps:
        raise TypeError("Expected a Pipeline containing a 'model' step.")
    target_regressor = model.named_steps["model"]
    if not hasattr(target_regressor, "transformer_"):
        raise TypeError("The fitted model does not expose its target transformer.")
    target_transformer = target_regressor.transformer_
    target_mean = float(np.asarray(target_transformer.mean_).reshape(-1)[0])
    target_scale = float(np.asarray(target_transformer.scale_).reshape(-1)[0])
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("The fitted target standard deviation must be positive.")

    predicted_standardized = np.asarray(
        target_transformer.transform(predicted_cost.reshape(-1, 1)),
        dtype=float,
    ).reshape(-1)

    def predict_standardized(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(values, dtype=float), columns=FEATURE_COLUMNS)
        cost_predictions = np.asarray(model.predict(frame), dtype=float).reshape(-1, 1)
        return np.asarray(
            target_transformer.transform(cost_predictions),
            dtype=float,
        ).reshape(-1)

    print(f"Calculating exact SHAP values for {label}...")
    shap_values, base_values = exact_interventional_shap(
        predict_standardized,
        X.to_numpy(dtype=float),
        background,
        label,
    )

    reconstructed = base_values + shap_values.sum(axis=1)
    max_error = float(np.max(np.abs(reconstructed - predicted_standardized)))
    tolerance = max(1e-8, 1e-7 * float(np.max(np.abs(predicted_standardized))))
    if max_error > tolerance:
        raise RuntimeError(
            f"{label} SHAP additivity failed: error={max_error:.6g}, "
            f"tolerance={tolerance:.6g}."
        )

    return SHAPResult(
        label=label,
        key=key,
        slug=slug,
        predicted_cost=predicted_cost,
        predicted_standardized=predicted_standardized,
        base_standardized=base_values,
        shap_standardized=shap_values,
        importance_standardized=np.mean(np.abs(shap_values), axis=0),
        target_mean=target_mean,
        target_scale=target_scale,
        max_additivity_error=max_error,
    )


def normalized_feature_values(values: np.ndarray) -> np.ndarray:
    """归一化点颜色。"""
    values = np.asarray(values, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Feature values must be finite for plotting.")
    if high <= low:
        return np.full_like(values, 0.5, dtype=float)
    return (values - low) / (high - low)


def apply_times_new_roman(fig: plt.Figure) -> None:
    """统一图片字体。"""
    for artist in fig.findobj(match=mpl.text.Text):
        artist.set_fontfamily(FONT_FAMILY)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    """保存PNG图片。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_times_new_roman(fig)
    with warnings.catch_warnings():
        # 忽略色条布局提示
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout()
    path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    return [path]


def plot_model_summary(result: SHAPResult, X: pd.DataFrame) -> list[Path]:
    """绘制汇总图。"""
    order = np.argsort(result.importance_standardized)[::-1]
    fig, (ax_swarm, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.85),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.72},
    )
    y_positions = np.arange(len(FEATURE_NAMES))

    for rank, feature_index in enumerate(order):
        rng = np.random.default_rng(RANDOM_STATE + feature_index)
        jitter = rng.uniform(-0.13, 0.13, len(X))
        colours = normalized_feature_values(X.iloc[:, feature_index].to_numpy(dtype=float))
        ax_swarm.scatter(
            result.shap_standardized[:, feature_index],
            rank + jitter,
            c=colours,
            cmap=FEATURE_VALUE_CMAP,
            vmin=0,
            vmax=1,
            s=19,
            linewidths=0.2,
            edgecolors="white",
            alpha=0.95,
            zorder=3,
        )

    ax_swarm.axvline(0, color="#666666", linewidth=0.8, zorder=1)
    ax_swarm.grid(axis="y", color="#D9D9D9", linewidth=0.6, linestyle=(0, (1, 3)))
    ax_swarm.set_yticks(y_positions)
    ax_swarm.set_yticklabels([FEATURE_NAMES[index] for index in order])
    ax_swarm.invert_yaxis()
    ax_swarm.set_xlabel("SHAP value (standardized model output)")
    ax_swarm.set_title(f"{result.label}: contribution direction", fontweight="bold")
    ax_swarm.tick_params(axis="y", length=0)

    colourbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=FEATURE_VALUE_CMAP),
        ax=ax_swarm,
        fraction=0.045,
        pad=0.025,
    )
    colourbar.set_ticks([0, 1], labels=["Low", "High"])
    colourbar.set_label("Feature value")

    importance = result.importance_standardized[order]
    ax_bar.barh(
        y_positions,
        importance,
        color=MODEL_COLORS[result.label],
        edgecolor="none",
        alpha=0.78,
        height=0.68,
    )
    ax_bar.set_yticks(y_positions)
    # 右图沿用左图顺序
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean |SHAP value|")
    ax_bar.set_title(f"{result.label}: global importance", fontweight="bold")
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.grid(axis="x", color="#E1E1E1", linewidth=0.6, linestyle=(0, (1, 3)))
    ax_bar.set_axisbelow(True)
    offset = max(float(np.max(importance)) * 0.025, 0.006)
    ax_bar.set_xlim(0, float(np.max(importance)) * 1.20)
    for position, value in zip(y_positions, importance):
        ax_bar.text(value + offset, position, f"{value:.3f}", va="center", fontsize=6.5)

    fig.text(0.01, 0.985, "a", fontweight="bold", fontsize=9, va="top")
    fig.text(0.655, 0.985, "b", fontweight="bold", fontsize=9, va="top")
    return save_figure(fig, f"Figure_SHAP_{result.slug}_summary")


def plot_importance_comparison(results: list[SHAPResult]) -> list[Path]:
    """绘制模型重要性对比图。"""
    if len(results) != 2:
        raise ValueError("The importance comparison requires exactly two models.")
    importance = np.vstack([item.importance_standardized for item in results])
    order = np.argsort(np.mean(importance, axis=0))[::-1]
    y_positions = np.arange(len(FEATURE_NAMES))
    bar_height = 0.34

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.55))
    for model_index, result in enumerate(results):
        offset = (model_index - 0.5) * bar_height
        ax.barh(
            y_positions + offset,
            result.importance_standardized[order],
            height=bar_height * 0.88,
            color=MODEL_COLORS[result.label],
            label=result.label,
            alpha=0.82,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_NAMES[index] for index in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (standardized model output)")
    ax.set_title("BP and GA-BP global feature importance", fontweight="bold")
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.6, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")
    return save_figure(fig, "Figure_SHAP_BP_vs_GA_BP_importance")


def plot_dependence(result: SHAPResult, X: pd.DataFrame) -> list[Path]:
    """绘制SHAP依赖图。"""
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.05),
        sharey=False,
    )
    axes_flat = axes.ravel()
    for feature_index, (axis, feature_name) in enumerate(zip(axes_flat, FEATURE_NAMES)):
        x_values = X.iloc[:, feature_index].to_numpy(dtype=float)
        y_values = result.shap_standardized[:, feature_index]
        if feature_index in CATEGORICAL_FEATURE_INDEXES:
            categories = np.unique(x_values)
            category_positions = np.searchsorted(categories, x_values).astype(float)
            rng = np.random.default_rng(RANDOM_STATE + 100 + feature_index)
            plot_x = category_positions + rng.uniform(-0.055, 0.055, len(x_values))
        else:
            categories = np.asarray([], dtype=float)
            plot_x = x_values
        colours = normalized_feature_values(x_values)
        axis.scatter(
            plot_x,
            y_values,
            s=17,
            c=colours,
            cmap=FEATURE_VALUE_CMAP,
            vmin=0,
            vmax=1,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.90,
        )
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.grid(color="#E5E5E5", linewidth=0.5, linestyle=(0, (1, 3)))
        axis.set_axisbelow(True)
        axis.set_title(feature_name, fontsize=7.5)
        axis.set_xlabel(
            "Feature category"
            if feature_index in CATEGORICAL_FEATURE_INDEXES
            else "Feature value"
        )
        axis.set_ylabel("SHAP value")
        if feature_index in CATEGORICAL_FEATURE_INDEXES:
            axis.set_xticks(np.arange(len(categories)))
            axis.set_xticklabels(
                [f"{value:g}" for value in categories],
                rotation=30,
                rotation_mode="anchor",
                ha="right",
            )

    axes_flat[-1].axis("off")
    colourbar_axis = axes_flat[-1].inset_axes([0.42, 0.18, 0.12, 0.64])
    colourbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=FEATURE_VALUE_CMAP),
        cax=colourbar_axis,
    )
    colourbar.set_ticks([0, 1], labels=["Low", "High"])
    colourbar.set_label("Feature value")
    fig.suptitle(
        f"{result.label}: feature values and SHAP contributions",
        y=1.01,
        fontsize=9,
        fontweight="bold",
    )
    return save_figure(fig, f"Figure_SHAP_{result.slug}_dependence")


def plot_local_contribution(
    result: SHAPResult,
    X: pd.DataFrame,
    project_ids: np.ndarray,
    project_id: int,
) -> list[Path]:
    """绘制项目局部贡献图。"""
    matches = np.flatnonzero(project_ids.astype(str) == str(project_id))
    if len(matches) != 1:
        raise ValueError(f"Project ID {project_id!r} was not found uniquely.")
    index = int(matches[0])
    contributions = result.shap_standardized[index]
    order = np.argsort(np.abs(contributions))[::-1]
    colours = np.where(contributions[order] >= 0, VERMILION, MID_BLUE)

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.45))
    y_positions = np.arange(len(FEATURE_NAMES))
    ax.barh(y_positions, contributions[order], color=colours, alpha=0.85)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_NAMES[item] for item in order])
    ax.invert_yaxis()
    ax.set_xlabel("Local SHAP contribution (standardized model output)")
    ax.set_title(
        f"{result.label}, project {project_id}: predicted cost "
        f"{result.predicted_cost[index]:.1f} × 10⁴ CNY",
        fontweight="bold",
    )
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.6, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    return save_figure(fig, f"Figure_SHAP_{result.slug}_project_{project_id}")


def safe_feature_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def save_model_tables(
    result: SHAPResult,
    X: pd.DataFrame,
    y: np.ndarray,
    project_ids: np.ndarray,
) -> list[Path]:
    """保存模型SHAP表。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    project_table = pd.DataFrame(
        {
            "project_id": project_ids,
            "observed_cost_10k_CNY": y,
            "predicted_cost_10k_CNY": result.predicted_cost,
            "observed_cost_standardized": (y - result.target_mean) / result.target_scale,
            "predicted_output_standardized": result.predicted_standardized,
            "base_value_standardized": result.base_standardized,
            "base_value_10k_CNY": (
                result.target_mean + result.base_standardized * result.target_scale
            ),
        }
    )
    for feature_index, (column, feature_name) in enumerate(
        zip(FEATURE_COLUMNS, FEATURE_NAMES)
    ):
        safe_name = safe_feature_name(feature_name)
        project_table[f"value__{safe_name}"] = X[column].to_numpy(dtype=float)
        project_table[f"shap_standardized__{safe_name}"] = (
            result.shap_standardized[:, feature_index]
        )
        project_table[f"shap_10k_CNY__{safe_name}"] = (
            result.shap_standardized[:, feature_index] * result.target_scale
        )

    project_path = OUTPUT_DIR / f"SHAP_values_{result.slug}.csv"
    project_table.to_csv(project_path, index=False, encoding="utf-8-sig")

    total = max(float(np.sum(result.importance_standardized)), 1e-12)
    importance_table = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "mean_abs_SHAP_standardized": result.importance_standardized,
            "mean_abs_SHAP_10k_CNY": (
                result.importance_standardized * result.target_scale
            ),
            "importance_share_pct": result.importance_standardized / total * 100,
        }
    ).sort_values("mean_abs_SHAP_standardized", ascending=False)
    importance_table["importance_rank"] = np.arange(1, len(importance_table) + 1)
    importance_path = OUTPUT_DIR / f"SHAP_importance_{result.slug}.csv"
    importance_table.to_csv(importance_path, index=False, encoding="utf-8-sig")
    return [project_path, importance_path]


def save_comparison_table(results: list[SHAPResult]) -> Path:
    """保存模型对比表。"""
    table = pd.DataFrame({"feature": FEATURE_NAMES})
    for result in results:
        total = max(float(np.sum(result.importance_standardized)), 1e-12)
        table[f"{result.slug}__mean_abs_SHAP_standardized"] = (
            result.importance_standardized
        )
        table[f"{result.slug}__mean_abs_SHAP_10k_CNY"] = (
            result.importance_standardized * result.target_scale
        )
        table[f"{result.slug}__importance_share_pct"] = (
            result.importance_standardized / total * 100
        )
        table[f"{result.slug}__importance_rank"] = (
            pd.Series(result.importance_standardized)
            .rank(method="min", ascending=False)
            .astype(int)
            .to_numpy()
        )
    table = table.sort_values("GA_BP__importance_rank")
    path = OUTPUT_DIR / "SHAP_importance_BP_vs_GA_BP.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_settings(X: pd.DataFrame, results: list[SHAPResult]) -> Path:
    """保存计算设置。"""
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.label,
                "model_key": result.key,
                "analysis_scope": "final model fitted on all available projects",
                "SHAP_method": "exact empirical interventional Shapley enumeration",
                "explained_projects": len(X),
                "background_projects": len(X),
                "features": X.shape[1],
                "coalitions_per_project": 1 << X.shape[1],
                "figure_output_scale": "standardized target prediction",
                "target_mean_10k_CNY": result.target_mean,
                "target_scale_10k_CNY": result.target_scale,
                "cost_conversion": (
                    "SHAP_10k_CNY = SHAP_standardized * target_scale_10k_CNY"
                ),
                "global_importance": "mean(abs(project-level SHAP))",
                "random_state": RANDOM_STATE,
                "maximum_additivity_error": result.max_additivity_error,
                "interpretation_boundary": "model association; not a causal effect",
            }
        )
    path = OUTPUT_DIR / "SHAP_calculation_settings.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    if len(FEATURE_COLUMNS) != 7 or len(FEATURE_NAMES) != 7:
        raise ValueError("Exactly seven feature columns and names are required.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    model_module = load_model_module(MODEL_SCRIPT)
    X, y_series, project_id_series = model_module.load_data(DATA_PATH)
    X = X.loc[:, FEATURE_COLUMNS].copy()
    y = y_series.to_numpy(dtype=float)
    project_ids = project_id_series.to_numpy()

    if len(X) != len(y):
        raise ValueError("Feature and target sample counts do not match.")
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y).all():
        raise ValueError("The SHAP analysis requires finite feature and target values.")

    # 两个模型使用相同背景数据
    background = X.to_numpy(dtype=float)
    available_models = model_module.make_models(random_state=RANDOM_STATE)
    results: list[SHAPResult] = []
    for label, key, slug in MODEL_SPECS:
        if key not in available_models:
            raise KeyError(f"Model key {key!r} is unavailable in the model script.")
        results.append(
            fit_and_explain(
                available_models[key],
                label,
                key,
                slug,
                X,
                y,
                background,
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for result in results:
        output_paths.extend(plot_model_summary(result, X))
        output_paths.extend(plot_dependence(result, X))
        output_paths.extend(save_model_tables(result, X, y, project_ids))
        if LOCAL_PROJECT_ID is not None:
            output_paths.extend(
                plot_local_contribution(
                    result,
                    X,
                    project_ids,
                    LOCAL_PROJECT_ID,
                )
            )

    output_paths.extend(plot_importance_comparison(results))
    output_paths.append(save_comparison_table(results))
    output_paths.append(save_settings(X, results))

    print("\nSHAP analysis completed successfully.")
    print(f"Projects explained per model: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Coalitions per project: {1 << X.shape[1]}")
    for result in results:
        minimum = float(np.min(result.shap_standardized))
        maximum = float(np.max(result.shap_standardized))
        print(
            f"{result.label}: SHAP range [{minimum:.3f}, {maximum:.3f}], "
            f"maximum additivity error {result.max_additivity_error:.3e}"
        )
    print(f"Output directory: {OUTPUT_DIR}")
    for path in output_paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
