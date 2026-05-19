import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn_model.penn.nn_iccbf_predict import ProbabilisticEnsembleNN


def generate_new_dataset(
    gamma_pairs,
    distance_range,
    velocity_range,
    theta_range,
    num_samples_per_param=10,
):
    """
    Generate all combinations of gamma pairs and sampled state features.
    """
    distance_vals = np.linspace(distance_range[0], distance_range[1], num_samples_per_param)
    velocity_vals = np.linspace(velocity_range[0], velocity_range[1], num_samples_per_param)
    theta_vals = np.linspace(theta_range[0], theta_range[1], num_samples_per_param)

    dataset = []
    for gamma1, gamma2 in gamma_pairs:
        for distance in distance_vals:
            for velocity in velocity_vals:
                for theta in theta_vals:
                    dataset.append([distance, velocity, theta, gamma1, gamma2])
    return np.array(dataset)


def configure_ieee_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 6.2,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_3d_axis(ax):
    ax.view_init(elev=24, azim=-58)
    ax.invert_xaxis()
    ax.set_xlabel("dist(p, p_obs) [m]", labelpad=-6)
    ax.set_ylabel("v [m/s]", labelpad=-6)
    ax.set_zlabel("Δθ [rad]", labelpad=-7)
    ax.tick_params(axis="both", which="major", pad=-5, width=0.4, length=2)
    ax.tick_params(axis="z", which="major", pad=-3, width=0.4, length=2)
    ax.set_xticks([1, 2, 3])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks([0.0, 0.25, 0.5, 0.75])
    ax.set_box_aspect((1.0, 1.0, 0.85))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.55)
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 0.42))
        axis.pane.set_edgecolor((0.88, 0.88, 0.88, 0.65))
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.55)


def plot_3d_grid_mean_predictions(
    ax,
    x_original,
    mean_predictions,
    gamma1,
    gamma2,
    vmin,
    vmax,
    cmap="RdYlBu_r",
    tolerance=1e-4,
):
    mask = (
        np.isclose(x_original[:, 3], gamma1, atol=tolerance)
        & np.isclose(x_original[:, 4], gamma2, atol=tolerance)
    )
    x = x_original[mask][:, 0]
    y = x_original[mask][:, 1]
    z = x_original[mask][:, 2]
    preds = mean_predictions[mask]

    if len(x) == 0:
        ax.text2D(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(
            x,
            y,
            z,
            c=preds,
            cmap=cmap,
            alpha=0.74,
            vmin=vmin,
            vmax=vmax,
            s=9,
            linewidths=0.18,
            edgecolors="face",
            depthshade=False,
        )

    style_3d_axis(ax)
    return ax


def add_parameter_arrows(fig):
    arrow_style = "-|>"
    arrow_color = "0.43"
    x_arrow = FancyArrowPatch(
        (0.125, 0.105),
        (0.84, 0.105),
        transform=fig.transFigure,
        arrowstyle=arrow_style,
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
        arrowstyle=arrow_style,
        mutation_scale=12,
        linewidth=1.1,
        color=arrow_color,
        shrinkA=0,
        shrinkB=0,
    )
    fig.add_artist(x_arrow)
    fig.add_artist(y_arrow)
    fig.text(0.49, 0.048, "ICCBF Parameter 1: α̃₁", ha="center", va="center", fontsize=8.4)
    fig.text(
        0.038,
        0.52,
        "ICCBF Parameter 2: α̃₂",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.4,
    )


def axes_rects():
    left = 0.132
    bottom = 0.155
    width = 0.206
    height = 0.205
    gap_x = 0.043
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


def remove_mask_like_svg_constructs(svg_path):
    if not svg_path.lower().endswith(".svg"):
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


def main():
    configure_ieee_style()

    model_path = os.environ.get(
        "UNCERTAINTY_MODEL_PATH",
        "nn_model/checkpoint/DynamicUnicycle2D_1120_mlp_1230_epoch_400.pth",
    )
    scaler_path = os.environ.get(
        "UNCERTAINTY_SCALER_PATH",
        model_path.replace(".pth", ".save"),
    )

    penn = ProbabilisticEnsembleNN()
    penn.load_model(model_path)
    penn.load_scaler(scaler_path)
    penn.model.eval()

    # This range keeps the query inside a smoother region of the trained model
    # and preserves the expected trend: larger ICCBF parameters increase risk.
    gamma_values = [
        float(value)
        for value in os.environ.get("UNCERTAINTY_GAMMA_VALUES", "0.18,0.26,0.35").split(",")
    ]
    gamma_pairs = [
        (gamma_values[0], gamma_values[0]),
        (gamma_values[0], gamma_values[1]),
        (gamma_values[0], gamma_values[2]),
        (gamma_values[1], gamma_values[0]),
        (gamma_values[1], gamma_values[1]),
        (gamma_values[1], gamma_values[2]),
        (gamma_values[2], gamma_values[0]),
        (gamma_values[2], gamma_values[1]),
        (gamma_values[2], gamma_values[2]),
    ]

    distance_range = (0.6, 3.0)
    velocity_range = (0.01, 1.0)
    theta_range = (0.01, np.pi / 4)
    num_samples = int(os.environ.get("UNCERTAINTY_NUM_SAMPLES", "10"))

    new_dataset = generate_new_dataset(
        gamma_pairs=gamma_pairs,
        distance_range=distance_range,
        velocity_range=velocity_range,
        theta_range=theta_range,
        num_samples_per_param=num_samples,
    )
    print(f"Generated dataset shape: {new_dataset.shape}")

    unique_gamma_pairs = np.unique(new_dataset[:, 3:5], axis=0)
    print("Unique gamma pairs in the dataset:")
    for pair in unique_gamma_pairs:
        print(f"alpha1: {pair[0]:.2f}, alpha2: {pair[1]:.2f}")

    with torch.no_grad():
        y_pred_safety_loss, _, _ = penn.predict(new_dataset)

    safety_loss_mu = np.array([[pred[0] for pred in sample] for sample in y_pred_safety_loss])
    mean_safety_loss = np.mean(safety_loss_mu, axis=1)
    x_original = new_dataset

    global_min = float(os.environ.get("UNCERTAINTY_COLOR_MIN", "0.10"))
    global_max = float(os.environ.get("UNCERTAINTY_COLOR_MAX", "0.85"))
    print(f"Color scale min: {global_min}")
    print(f"Color scale max: {global_max}")

    fig = plt.figure(figsize=(7.16, 5.95), facecolor="white")
    sorted_gamma2 = sorted(set(pair[1] for pair in gamma_pairs))
    sorted_gamma1 = sorted(set(pair[0] for pair in gamma_pairs))
    rects = axes_rects()

    for gamma1 in sorted_gamma1:
        for gamma2 in sorted_gamma2:
            row = sorted_gamma2.index(gamma2)
            col = sorted_gamma1.index(gamma1)
            ax = fig.add_axes(rects[(len(sorted_gamma2) - row - 1, col)], projection="3d")
            plot_3d_grid_mean_predictions(
                ax=ax,
                x_original=x_original,
                mean_predictions=mean_safety_loss,
                gamma1=gamma1,
                gamma2=gamma2,
                vmin=global_min,
                vmax=global_max,
                cmap="RdYlBu_r",
                tolerance=1e-4,
            )

    cbar_ax = fig.add_axes([0.875, 0.205, 0.022, 0.635])
    norm = plt.Normalize(vmin=global_min, vmax=global_max)
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])
    cbar.ax.tick_params(labelsize=7.0, width=0.55, length=2.5)
    cbar.outline.set_linewidth(0.55)

    add_parameter_arrows(fig)
    fig.suptitle("Mean Predicted Risk Level", fontsize=9.5, y=0.968)

    output_path = os.environ.get("UNCERTAINTY_OUTPUT", str(ROOT / "paper_figures" / "output" / "uncertainties_grid.png"))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    remove_mask_like_svg_constructs(output_path)
    print(f"Saved uncertainty grid: {output_path}")

    preview_output_path = os.environ.get("UNCERTAINTY_PREVIEW_OUTPUT", "").strip()
    if preview_output_path:
        os.makedirs(os.path.dirname(preview_output_path) or ".", exist_ok=True)
        fig.savefig(preview_output_path, dpi=450, bbox_inches="tight")
        print(f"Saved uncertainty preview: {preview_output_path}")

    if os.environ.get("SHOW_PLOT", "").strip().lower() in ("1", "true", "yes"):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
