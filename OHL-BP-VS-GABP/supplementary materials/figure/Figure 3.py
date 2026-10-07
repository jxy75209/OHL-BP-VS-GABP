from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import gaussian_kde


DATA_FILE = Path(r"C:\python code\BP\bp1-data.csv")
OUTPUT_FILE = Path(__file__).resolve().parent / "Figure_3_variable_distributions.png"

FONT_FILES = [
    Path(r"C:\Windows\Fonts\times.ttf"),
    Path(r"C:\Windows\Fonts\timesbd.ttf"),
    Path(r"C:\Windows\Fonts\timesi.ttf"),
    Path(r"C:\Windows\Fonts\timesbi.ttf"),
]
for font_file in FONT_FILES:
    if not font_file.exists():
        raise FileNotFoundError(f"Times New Roman font file not found: {font_file}")
    font_manager.fontManager.addfont(str(font_file))

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

COLORS = {
    "ink": "#20252B",
    "muted": "#66707A",
    "grid": "#D8E0E6",
    "light_blue": "#ABD9E9",
    "mid_blue": "#4393C3",
    "deep_blue": "#2166AC",
    "orange": "#FDAE61",
    "vermilion": "#E34A33",
    "deep_red": "#B2182B",
}

PANEL_COLORS = [
    ("#B2182B", "#7A1020"),
    ("#E34A33", "#A72D23"),
    ("#FDAE61", "#D66D22"),
    ("#FEE391", "#B58516"),
    ("#B8BFC5", "#66707A"),
    ("#ABD9E9", "#4393C3"),
    ("#4393C3", "#2166AC"),
    ("#2166AC", "#16477A"),
]

VARIABLES = [
    ("Steel price", "Steel price", "CNY t⁻¹", "continuous"),
    ("OPGW erection length", "OPGW erection length", "km", "continuous"),
    ("Conductor unit price", "Conductor unit price", "CNY t⁻¹", "continuous"),
    ("Conductor erection length", "Conductor erection length", "km", "continuous"),
    (
        "Conductor cross sectional area",
        "Conductor cross-sectional area",
        "mm²",
        "categorical",
    ),
    ("Number of crossings", "Number of crossings", "Crossings per project", "continuous"),
    ("Voltage level", "Voltage level", "kV", "categorical"),
    ("Cost", "Cost", "10⁴ CNY", "continuous"),
]


def load_data():
    raw = pd.read_csv(DATA_FILE)
    id_col = raw.columns[0]
    metadata_row = raw[id_col].astype(str).str.strip().str.lower().eq("index")
    if metadata_row.sum() != 1:
        raise ValueError("Expected one variable-code row labelled 'index'.")

    columns = [item[0] for item in VARIABLES]
    data = raw.loc[~metadata_row, columns].copy()
    data[columns] = data[columns].apply(pd.to_numeric, errors="raise")

    if len(data) != 44:
        raise ValueError(f"Expected 44 project observations, found {len(data)}.")
    if data.isna().any().any():
        raise ValueError("Missing values were found in the plotting variables.")

    print(f"Rows read: {len(raw)}; metadata rows removed: {int(metadata_row.sum())}; projects plotted: {len(data)}")
    return data


def fd_edges(values):
    edges = np.histogram_bin_edges(values, bins="fd")
    n_bins = len(edges) - 1
    if n_bins < 5:
        edges = np.linspace(values.min(), values.max(), 6)
    elif n_bins > 9:
        edges = np.linspace(values.min(), values.max(), 10)
    return edges


def format_axis(ax, *, show_ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(axis="both", labelsize=6.2, width=0.7, length=3, colors=COLORS["ink"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, linestyle=(0, (2, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Number of projects" if show_ylabel else "", fontsize=6.7)


def draw_continuous(ax, values, *, fill, line):
    values = np.asarray(values, dtype=float)
    edges = fd_edges(values)

    ax.hist(
        values,
        bins=edges,
        color=fill,
        edgecolor=line,
        linewidth=0.7,
        alpha=0.78,
        zorder=2,
    )

    x_grid = np.linspace(values.min(), values.max(), 300)
    density = gaussian_kde(values)(x_grid)
    bin_width = float(np.mean(np.diff(edges)))
    ax.plot(x_grid, density * len(values) * bin_width, color=line, linewidth=1.25, zorder=4)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if np.nanmax(values) >= 1000:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))


def draw_categorical(ax, values, *, fill, line):
    counts = pd.Series(values).value_counts().sort_index()
    bars = ax.bar(
        np.arange(len(counts)),
        counts.values,
        width=0.68,
        color=fill,
        edgecolor=line,
        linewidth=0.65,
        alpha=0.82,
        zorder=3,
    )
    ax.set_xticks(np.arange(len(counts)), [f"{int(v)}" for v in counts.index])
    ax.set_ylim(0, counts.max() * 1.18)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.035,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=COLORS["ink"],
        )


def draw_figure():
    data = load_data()
    width_mm, height_mm = 180, 96
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        facecolor="white",
    )
    letters = "abcdefgh"

    for i, (column, title, xlabel, variable_type) in enumerate(VARIABLES):
        ax = axes.flat[i]
        fill, line = PANEL_COLORS[i]
        if variable_type == "categorical":
            draw_categorical(ax, data[column], fill=fill, line=line)
        else:
            draw_continuous(ax, data[column], fill=fill, line=line)

        format_axis(ax, show_ylabel=(i % 4 == 0))
        ax.set_xlabel(xlabel, fontsize=6.7, labelpad=3)
        ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=6)
        ax.text(
            -0.145,
            1.075,
            letters[i],
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=COLORS["ink"],
        )

    legend_handles = [
        Patch(facecolor="#D9DEE2", edgecolor=COLORS["muted"], linewidth=0.7, label="Project frequency"),
        Line2D([0], [0], color=COLORS["ink"], linewidth=1.25, label="Kernel density estimate"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.992),
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handlelength=1.8,
        columnspacing=1.6,
    )
    fig.text(
        0.988,
        0.986,
        "n = 44 projects",
        ha="right",
        va="top",
        fontsize=6.4,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.070, right=0.990, top=0.895, bottom=0.135, wspace=0.36, hspace=0.62)
    fig.savefig(OUTPUT_FILE, dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
