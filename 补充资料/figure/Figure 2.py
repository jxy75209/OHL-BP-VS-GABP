from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox


# 字体
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
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 7,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


COLORS = {
    "ink": "#20252B",
    "muted": "#66707A",
    "line": "#88939C",
    "panel": "#F0F2F5",
    "white": "#FFFFFF",
    "deep_red": "#B2182B",
    "vermilion": "#E34A33",
    "orange": "#FDAE61",
    "light_yellow": "#FEE391",
    "light_blue": "#ABD9E9",
    "mid_blue": "#4393C3",
    "deep_blue": "#2166AC",
    "pale_red": "#FAE4E1",
    "pale_blue": "#E4F1F6",
}


def rounded_box(ax, x, y, w, h, face, edge, *, lw=0.8, radius=0.012, z=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def text(
    ax,
    x,
    y,
    value,
    *,
    size=7,
    weight="normal",
    color=None,
    ha="center",
    va="center",
    style="normal",
    linespacing=1.05,
    z=5,
):
    return ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        color=color or COLORS["ink"],
        linespacing=linespacing,
        zorder=z,
    )


def arrow(ax, x0, y0, x1, y1, *, color=None, lw=0.9, scale=7, z=4):
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color or COLORS["ink"],
        transform=ax.transAxes,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def bullet_list(ax, x, y_top, items, *, color, size, step, detail_color=None):
    for i, item in enumerate(items):
        y = y_top - i * step
        ax.scatter(
            [x],
            [y],
            s=7,
            color=color,
            edgecolors="none",
            transform=ax.transAxes,
            zorder=6,
        )
        text(
            ax,
            x + 0.013,
            y,
            item,
            size=size,
            color=detail_color or COLORS["ink"],
            ha="left",
        )


def count_box(ax, y, label, face, edge):
    x, w, h = 0.4075, 0.185, 0.068
    rounded_box(ax, x, y, w, h, face, edge, lw=0.9, radius=0.012, z=3)
    text(ax, x + w / 2, y + h / 2, label, size=7.8, weight="bold", color=COLORS["white"])
    return x, y, w, h


def stage_box(ax, y, label, face, edge):
    x, w, h = 0.130, 0.225, 0.064
    rounded_box(ax, x, y, w, h, face, edge, lw=0.8, radius=0.011)
    text(ax, x + w / 2, y + h / 2, label, size=7.0, weight="bold", linespacing=0.96)
    return x, y, w, h


def draw_figure():
    width_mm, height_mm = 180, 112
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column headings
    text(ax, 0.2425, 0.958, "Feature-screening stage", size=7.0, weight="bold")
    text(ax, 0.500, 0.958, "Retained feature space", size=7.0, weight="bold")
    text(ax, 0.7525, 0.958, "Excluded features and basis", size=7.0, weight="bold")
    ax.plot([0.110, 0.890], [0.934, 0.934], color=COLORS["line"], lw=0.7, transform=ax.transAxes)

    # Retained-feature trajectory
    top = count_box(ax, 0.820, "20 candidate\nengineering features", COLORS["deep_red"], COLORS["deep_red"])
    retained_16 = count_box(ax, 0.646, "16 engineering\nfeatures retained", COLORS["orange"], COLORS["vermilion"])
    retained_14 = count_box(ax, 0.472, "14 engineering\nfeatures retained", COLORS["mid_blue"], COLORS["mid_blue"])
    retained_7 = count_box(ax, 0.298, "7 early-stage\ninput features", COLORS["deep_blue"], COLORS["deep_blue"])

    # Stage 1
    s1 = stage_box(ax, 0.735, "Expert appraisal", COLORS["light_yellow"], COLORS["orange"])
    r1 = (0.640, 0.735, 0.225, 0.064)
    rounded_box(ax, *r1, COLORS["pale_red"], COLORS["deep_red"], lw=0.75, radius=0.010)
    text(
        ax,
        r1[0] + r1[2] / 2,
        r1[1] + r1[3] / 2,
        "Excluded: lower suitability (4)",
        size=5.9,
        weight="bold",
        color=COLORS["deep_red"],
    )

    # Stage 2
    s2 = stage_box(ax, 0.561, "Exploratory factor\nanalysis", COLORS["panel"], COLORS["line"])
    r2 = (0.640, 0.561, 0.225, 0.064)
    rounded_box(ax, *r2, COLORS["panel"], COLORS["mid_blue"], lw=0.75, radius=0.010)
    text(
        ax,
        r2[0] + r2[2] / 2,
        r2[1] + r2[3] / 2,
        "Excluded by EFA diagnosis (2)",
        size=5.9,
        weight="bold",
        color=COLORS["mid_blue"],
    )

    # Stage 3
    s3 = stage_box(ax, 0.387, "Early-stage information\navailability", COLORS["light_blue"], COLORS["mid_blue"])
    r3 = (0.640, 0.387, 0.225, 0.064)
    rounded_box(ax, *r3, COLORS["pale_blue"], COLORS["deep_blue"], lw=0.75, radius=0.010)
    text(
        ax,
        r3[0] + r3[2] / 2,
        r3[1] + r3[3] / 2,
        "Excluded by early availability (7)",
        size=5.8,
        weight="bold",
        color=COLORS["deep_blue"],
    )

    # Central transitions and exclusion branches
    rows = [
        (s1, 0.767, top, retained_16, r1),
        (s2, 0.593, retained_16, retained_14, r2),
        (s3, 0.419, retained_14, retained_7, r3),
    ]
    for stage, y_mid, upper, lower, excluded in rows:
        arrow(ax, stage[0] + stage[2] + 0.006, y_mid, 0.486, y_mid, color=COLORS["ink"], scale=6.5)
        arrow(ax, 0.514, y_mid, excluded[0] - 0.010, y_mid, color=COLORS["ink"], scale=6.5)
        arrow(ax, 0.500, upper[1] - 0.006, 0.500, lower[1] + lower[3] + 0.006, color=COLORS["ink"], scale=7)

    output = Path(__file__).resolve().parent / "Figure_2_feature_screening_revised"
    width_in, height_in = width_mm / 25.4, height_mm / 25.4
    crop = Bbox.from_bounds(width_in * 0.100, height_in * 0.235, width_in * 0.800, height_in * 0.765)
    fig.savefig(output.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches=crop)
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
