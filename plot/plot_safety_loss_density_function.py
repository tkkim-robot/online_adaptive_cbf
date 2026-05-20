#!/usr/bin/env python3
"""Generate the paper-style safety-loss density function SVG."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


OBSTACLE_RADIUS = 0.42
ROOT = Path(__file__).resolve().parents[1]


def configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 5.8,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def safety_loss_density(
    x: np.ndarray,
    y: np.ndarray,
    peak_lambda: float,
    robot_heading: float,
    beta_1: float = 35.0,
    beta_2: float = 2.00,
) -> np.ndarray:
    angle_to_obstacle = np.arctan2(-y, -x)
    delta_theta = np.arctan2(
        np.sin(angle_to_obstacle - robot_heading),
        np.cos(angle_to_obstacle - robot_heading),
    )
    beta = beta_1 * np.exp(-beta_2 * (np.cos(delta_theta) + 1.0))
    distance = np.sqrt(x**2 + y**2)
    return peak_lambda / (beta * distance**2 + 1.0)


def axes_rects() -> dict[tuple[int, int], list[float]]:
    left = 0.128
    bottom = 0.155
    width = 0.216
    height = 0.208
    gap_x = 0.042
    gap_y = 0.048
    rects = {}
    for row in range(3):
        for col in range(3):
            rects[(row, col)] = [
                left + col * (width + gap_x),
                bottom + (2 - row) * (height + gap_y),
                width,
                height,
            ]
    return rects


def style_3d_axis(ax, z_max: float) -> None:
    ax.view_init(elev=27, azim=-52)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.set_zlim(0.0, z_max)
    ax.set_box_aspect((1.0, 1.0, 0.78))
    ax.set_xlabel("p_x [m]", labelpad=-7)
    ax.set_ylabel("p_y [m]", labelpad=-7)
    ax.set_zlabel("ϕ", labelpad=-7)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_zticks([0.0, z_max / 2.0, z_max])
    ax.tick_params(axis="both", which="major", pad=-5, width=0.35, length=1.8)
    ax.tick_params(axis="z", which="major", pad=-3, width=0.35, length=1.8)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.45)
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 0.36))
        axis.pane.set_edgecolor((0.88, 0.88, 0.88, 0.62))
        axis._axinfo["grid"]["linewidth"] = 0.28
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.42)


def add_obstacle_boundary(ax, robot_heading: float, peak_lambda: float) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    ring_x = OBSTACLE_RADIUS * np.cos(theta)
    ring_y = OBSTACLE_RADIUS * np.sin(theta)
    ring_z = safety_loss_density(ring_x, ring_y, peak_lambda, robot_heading) + 0.055
    ax.plot(ring_x, ring_y, ring_z, color="white", linewidth=2.0, zorder=16)
    ax.plot(ring_x, ring_y, ring_z, color="#D7191C", linewidth=1.25, zorder=17)


def add_parameter_arrows(fig) -> None:
    arrow_color = "0.43"
    x_arrow = FancyArrowPatch(
        (0.125, 0.105),
        (0.855, 0.105),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color=arrow_color,
        shrinkA=0,
        shrinkB=0,
    )
    y_arrow = FancyArrowPatch(
        (0.095, 0.135),
        (0.095, 0.90),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color=arrow_color,
        shrinkA=0,
        shrinkB=0,
    )
    fig.add_artist(x_arrow)
    fig.add_artist(y_arrow)
    fig.text(0.49, 0.048, "Relative Angle Δθ [rad]", ha="center", va="center", fontsize=8.4)
    fig.text(
        0.038,
        0.52,
        "Peak K Parameter λ₁",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.4,
    )


def remove_mask_like_svg_constructs(svg_path: Path) -> None:
    if svg_path.suffix.lower() != ".svg":
        return

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(svg_path)
    root = tree.getroot()

    for elem in root.iter():
        for attr in list(elem.attrib):
            local_attr = attr.split("}", 1)[-1]
            if local_attr in {"clip-path", "mask"}:
                del elem.attrib[attr]

    for parent in root.iter():
        for child in list(parent):
            local_tag = child.tag.split("}", 1)[-1]
            if local_tag in {"clipPath", "mask"}:
                parent.remove(child)

    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def plot_safety_loss_density_grid(output_path: Path, preview_path: Path | None = None) -> None:
    configure_ieee_style()

    lambda_values = [
        float(value)
        for value in os.environ.get("SAFETY_LOSS_LAMBDA_VALUES", "0.20,0.40,0.60").split(",")
    ]
    heading_values = [
        float(value)
        for value in os.environ.get("SAFETY_LOSS_HEADING_VALUES", "-0.10,-1.50,-2.90").split(",")
    ]

    x = np.linspace(-2.4, 2.4, int(os.environ.get("SAFETY_LOSS_GRID_SIZE", "45")))
    y = np.linspace(-2.4, 2.4, int(os.environ.get("SAFETY_LOSS_GRID_SIZE", "45")))
    x_grid, y_grid = np.meshgrid(x, y)

    z_max = max(lambda_values)
    fig = plt.figure(figsize=(7.16, 5.95), facecolor="white")
    rects = axes_rects()
    cmap = plt.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=0.0, vmax=z_max)

    for row, peak_lambda in enumerate(reversed(lambda_values)):
        for col, robot_heading in enumerate(heading_values):
            ax = fig.add_axes(rects[(row, col)], projection="3d")
            z_grid = safety_loss_density(x_grid, y_grid, peak_lambda, robot_heading)
            facecolors = cmap(norm(z_grid))
            ax.plot_surface(
                x_grid,
                y_grid,
                z_grid,
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                shade=True,
                edgecolor="none",
            )
            ax.contour(
                x_grid,
                y_grid,
                z_grid,
                zdir="z",
                offset=0.0,
                levels=7,
                cmap=cmap,
                linewidths=0.22,
                alpha=0.55,
            )
            add_obstacle_boundary(ax, robot_heading, peak_lambda)
            style_3d_axis(ax, z_max=z_max)

    add_parameter_arrows(fig)
    fig.suptitle("Safety Loss Density Function", fontsize=9.5, y=0.968)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    remove_mask_like_svg_constructs(output_path)
    print(f"Wrote {output_path}")

    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=450, bbox_inches="tight")
        print(f"Wrote {preview_path}")

    plt.close(fig)


def main() -> None:
    default_output = ROOT / "paper_figures" / "output" / "safety_loss_density_function.svg"
    output_path = Path(os.environ.get("SAFETY_LOSS_DENSITY_OUTPUT", str(default_output))).resolve()

    preview_env = os.environ.get("SAFETY_LOSS_DENSITY_PREVIEW", "").strip()
    if preview_env:
        preview_path = Path(preview_env).resolve()
    else:
        preview_path = output_path.with_suffix(".png")

    plot_safety_loss_density_grid(output_path=output_path, preview_path=preview_path)


if __name__ == "__main__":
    main()
