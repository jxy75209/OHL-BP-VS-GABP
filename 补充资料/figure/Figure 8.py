"""基于多随机种子结果绘制 BP 与 GA-BP 稳健性四联图。

默认读取“多种子稳健性_结果.xlsx”的 raw_40runs 工作表，生成：
1. R²；2. RMSE；3. 低估率（UR）；4. 累计低估幅度（CUM）的箱线图和种子散点。

用法示例：
    python 多随机种子稳健性检验.py
    python 多随机种子稳健性检验.py --input "D:\\data\\多种子稳健性_结果.xlsx"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


DEFAULT_INPUT = Path(
    r"D:\Users\milip\Documents\xwechat_files\wxid_1azbxx69yfny22_2ceb"
    r"\msg\file\2026-09\多种子稳健性_结果.xlsx"
)
DEFAULT_SHEET = "raw_40runs"
OUTPUT_DIR_NAME = "多随机种子稳健性检验输出"

MODEL_ORDER = ["BP", "GABP_010", "GABP_015", "GABP_020"]
MODEL_LABELS = {
    "BP": "BP",
    "GABP_010": "GA-BP\n(τ=0.10)",
    "GABP_015": "GA-BP\n(τ=0.15)",
    "GABP_020": "GA-BP\n(τ=0.20)",
}
# 箱体和散点分别设置颜色；使用不透明填充，直接呈现指定 HEX 色值。
BOX_COLORS = {
    "BP": "#D7E1E8",
    "GABP_010": "#E7F3F2",
    "GABP_015": "#F2DADA",
    "GABP_020": "#D89090",
}
POINT_COLORS = {
    "BP": "#104E8B",
    "GABP_010": "#90D1D5",
    "GABP_015": "#E5B5B5",
    "GABP_020": "#B22222",
}

METRICS = [
    ("R2", "(a) R² across 10 random seeds", "R²"),
    ("RMSE", "(b) RMSE across 10 random seeds", "RMSE"),
    ("UR_%", "(c) Underestimation rate (UR) across 10 random seeds", "UR (%)"),
    (
        "CUM",
        "(d) Cumulative underestimation magnitude across 10 random seeds",
        "Cumulative underestimation magnitude (10⁴ CNY)",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制多随机种子稳健性 2×2 四联图。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入 Excel 文件（默认：{DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help=f"原始结果工作表（默认：{DEFAULT_SHEET}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_DIR_NAME,
        help="图片输出目录（默认：脚本同级的“多随机种子稳健性检验输出”）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG 分辨率（默认：600 dpi）",
    )
    return parser.parse_args()


def _model_key(row: pd.Series) -> str:
    """优先利用 tau 数值识别 GA-BP，避免依赖 Model 列中的特殊字符编码。"""
    model = str(row["Model"]).strip().upper().replace(" ", "")
    if model == "BP":
        return "BP"

    tau = pd.to_numeric(pd.Series([row["tau"]]), errors="coerce").iloc[0]
    if pd.isna(tau):
        raise ValueError(f"无法识别模型：Model={row['Model']!r}, tau={row['tau']!r}")
    if np.isclose(tau, 0.10):
        return "GABP_010"
    if np.isclose(tau, 0.15):
        return "GABP_015"
    if np.isclose(tau, 0.20):
        return "GABP_020"
    raise ValueError(f"发现未配置的 GA-BP τ 值：{tau}")


def load_and_validate(input_path: Path, sheet_name: str) -> pd.DataFrame:
    """读取原始 40 次运行结果，并检查 4 模型 × 10 种子的完整性。"""
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    data = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")
    required = {"Model", "tau", "seed", "R2", "RMSE", "UR_%", "CUM"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"工作表 {sheet_name!r} 缺少字段：{', '.join(missing)}")

    data = data.copy()
    data["model_key"] = data.apply(_model_key, axis=1)
    data["seed"] = pd.to_numeric(data["seed"], errors="raise").astype(int)
    for metric, _, _ in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="raise")

    if data[["seed", "model_key"]].duplicated().any():
        duplicate_rows = data.loc[
            data[["seed", "model_key"]].duplicated(keep=False),
            ["seed", "model_key"],
        ]
        raise ValueError(
            "存在重复的模型—种子记录：\n" + duplicate_rows.to_string(index=False)
        )

    unexpected_models = sorted(set(data["model_key"]) - set(MODEL_ORDER))
    if unexpected_models:
        raise ValueError(f"存在未配置模型：{unexpected_models}")

    seed_sets = {
        key: set(data.loc[data["model_key"] == key, "seed"]) for key in MODEL_ORDER
    }
    counts = {key: len(seeds) for key, seeds in seed_sets.items()}
    if any(count != 10 for count in counts.values()):
        raise ValueError(f"每个模型应有 10 个随机种子，当前数量为：{counts}")

    reference_seeds = seed_sets[MODEL_ORDER[0]]
    if any(seeds != reference_seeds for seeds in seed_sets.values()):
        details = {key: sorted(seeds) for key, seeds in seed_sets.items()}
        raise ValueError(f"四个模型的随机种子集合不一致：{details}")

    if len(data) != 40:
        raise ValueError(f"应有 40 条模型—种子记录，实际为 {len(data)} 条。")
    if not data["UR_%"].between(0, 100, inclusive="both").all():
        raise ValueError("UR_% 应采用 0–100 的百分数单位。")

    return data


def _set_y_limits(ax: plt.Axes, values: np.ndarray, metric: str) -> None:
    """为箱线图留出适度边距，并确保真实极值（含散点）均可见。"""
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    span = upper - lower
    padding = 0.08 * span if span > 0 else max(abs(upper) * 0.08, 1.0)

    if metric == "UR_%":
        ax.set_ylim(max(0.0, lower - padding), min(100.0, upper + padding))
    else:
        ax.set_ylim(lower - padding, upper + padding)


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    rng: np.random.Generator,
) -> None:
    grouped = [
        data.loc[data["model_key"] == key, metric].to_numpy(dtype=float)
        for key in MODEL_ORDER
    ]
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.5},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
        boxprops={"linewidth": 1.1},
    )

    for key, patch in zip(MODEL_ORDER, box["boxes"]):
        patch.set_facecolor(BOX_COLORS[key])
        patch.set_edgecolor(POINT_COLORS[key])
        patch.set_alpha(1.0)

    for key, median in zip(MODEL_ORDER, box["medians"]):
        median.set_color(POINT_COLORS[key])

    for position, key, values in zip(positions, MODEL_ORDER, grouped):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=30,
            color=POINT_COLORS[key],
            edgecolor="white",
            linewidth=0.55,
            alpha=1.0,
            zorder=3,
        )

    ax.set_title(title, loc="left", fontsize=11.2, fontweight="semibold", pad=9)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS[key] for key in MODEL_ORDER])
    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    ax.tick_params(axis="x", pad=7)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_values = np.concatenate(grouped)
    _set_y_limits(ax, all_values, metric)

    if metric == "R2" and np.nanmin(all_values) < 0:
        ax.axhline(0, color="#7A7A7A", linewidth=0.8, linestyle="--", zorder=0)
    elif metric == "UR_%":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    elif metric in {"RMSE", "CUM"}:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def print_summary(data: pd.DataFrame) -> None:
    """在终端输出可复核的均值和样本标准差。"""
    summary = (
        data.groupby("model_key", sort=False)[["R2", "RMSE", "UR_%", "CUM"]]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
    )
    summary.index = [MODEL_LABELS[key].replace("\n", " ") for key in MODEL_ORDER]
    print("\n4 个模型 × 10 个随机种子的汇总（均值、样本标准差）：")
    print(summary.round(4).to_string())


def make_figure(data: pd.DataFrame, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2))
    rng = np.random.default_rng(20260905)  # 仅用于固定散点横向抖动，不改变数据。
    for ax, (metric, title, ylabel) in zip(axes.flat, METRICS):
        draw_panel(ax, data, metric, title, ylabel, rng)

    legend_handles = [
        Patch(
            facecolor=BOX_COLORS[key],
            edgecolor=POINT_COLORS[key],
            linewidth=1.3,
            label=MODEL_LABELS[key].replace("\n", " "),
        )
        for key in MODEL_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.08, top=0.905, wspace=0.22, hspace=0.30)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "多随机种子稳健性检验_四联图.png"
    pdf_path = output_dir / "多随机种子稳健性检验_四联图.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    data = load_and_validate(args.input.resolve(), args.sheet)
    print_summary(data)
    png_path, pdf_path = make_figure(data, args.output_dir.resolve(), args.dpi)
    seeds = sorted(data["seed"].unique().tolist())
    print(f"\n完整性校验通过：4 个模型 × 10 个种子；种子为 {seeds}")
    print(f"PNG：{png_path}")
    print(f"PDF：{pdf_path}")


if __name__ == "__main__":
    main()
