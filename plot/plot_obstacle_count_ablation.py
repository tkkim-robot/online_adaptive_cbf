#!/usr/bin/env python3
"""
Create the obstacle-count ablation figure from existing experiment CSVs.

This script is intentionally standalone: it does not import controller, model,
or training modules. It only reads saved CSV artifacts and writes figure/data
outputs under paper_figures/output by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


SEED = 5010
OBS_COUNT_LOW = 2
OBS_COUNT_HIGH_EXCLUSIVE = 10


@dataclass(frozen=True)
class ResultCase:
    robot: str
    method: str
    csv_path: str
    obs_id_filter_csv: str | None = None


SDF_CASES: tuple[ResultCase, ...] = (
    ResultCase(
        robot="Dynamic unicycle",
        method="OA-CBF w/ GAT",
        csv_path="epoch_exp/du2d_1113/"
        "sim_results_in_obs_sweep_Online_Adaptive_MPC-CBF_GAT_DynamicUnicycle2D_1120_1217.csv",
    ),
    ResultCase(
        robot="Dynamic unicycle",
        method="OA-CBF w/ FC",
        csv_path="epoch_exp/du2d_1113/"
        "sim_results_in_obs_sweep_Online_Adaptive_MPC-CBF_MLP_DynamicUnicycle2D_1120_2354.csv",
    ),
    ResultCase(
        robot="Quad2D",
        method="OA-CBF w/ GAT",
        csv_path="epoch_exp/quad2d_1111/"
        "sim_results_in_obs_sweep_Online_Adaptive_MPC-CBF_GAT_Quad2D_1109_0241.csv",
        obs_id_filter_csv="epoch_exp/quad2d_1111/"
        "sim_results_in_obs_sweep_MPC-CBF_low_fixed_param_Quad2D_1111_1322.csv",
    ),
    ResultCase(
        robot="Quad2D",
        method="OA-CBF w/ FC",
        csv_path="epoch_exp/quad2d_1111/"
        "sim_results_in_obs_sweep_Online_Adaptive_MPC-CBF_MLP_Quad2D_1121_0018.csv",
    ),
    ResultCase(
        robot="Quad3D",
        method="OA-CBF w/ GAT",
        csv_path="epoch_exp/quad3d_1103/"
        "sim_results_in_obs_sweep_Online_Adaptive_MPC-CBF_GAT_Quad3D_1115_0317.csv",
    ),
    ResultCase(
        robot="Quad3D",
        method="OA-CBF w/ FC",
        csv_path="epoch_exp/quad3d_1207/"
        "sim_results_in_obs_sweep_Online_Adaptive_MPC-CBF_MLP_Quad3D_1207_1413.csv",
    ),
)


DPCBF_CASES: tuple[ResultCase, ...] = (
    ResultCase(
        robot="DPCBF bicycle",
        method="OA-CBF w/ GAT",
        csv_path="epoch_exp/kinematicbicycle2D_1101/"
        "sim_results_in_obs_sweep_Online_Adaptive_CBF-QP_GAT_KinematicBicycle2D_DPCBF_1102_0215.csv",
        obs_id_filter_csv="epoch_exp/kinematicbicycle2D_1101/"
        "sim_results_in_obs_sweep_CBF-QP_low_fixed_param_KinematicBicycle2D_DPCBF_1103_0220.csv",
    ),
    ResultCase(
        robot="DPCBF bicycle",
        method="OA-CBF w/ FC",
        csv_path="epoch_exp/kinematicbicycle2D_1209/"
        "sim_results_in_obs_sweep_Online_Adaptive_CBF-QP_MLP_KinematicBicycle2D_DPCBF_1208_2025_epoch_200.csv",
    ),
)


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() == "true"


def obstacle_count_from_obs_id(obs_id: int) -> int:
    """Match adaptation_experiment.py: RandomState(SEED + obs_id).randint(2, 10)."""
    rng = np.random.RandomState(SEED + obs_id)
    return int(rng.randint(OBS_COUNT_LOW, OBS_COUNT_HIGH_EXCLUSIVE))


def resolve_data_path(repo_root: Path, path_text: str) -> Path:
    direct = repo_root / path_text
    if direct.exists():
        return direct
    dataset_path = repo_root / "dataset" / path_text
    if dataset_path.exists():
        return dataset_path
    return direct


def read_obs_id_filter(repo_root: Path, filter_csv: str | None) -> set[int] | None:
    if filter_csv is None:
        return None
    path = resolve_data_path(repo_root, filter_csv)
    with path.open(newline="") as f:
        return {int(row["obs_id"]) for row in csv.DictReader(f)}


def iter_rows(repo_root: Path, case: ResultCase) -> Iterable[dict[str, object]]:
    path = resolve_data_path(repo_root, case.csv_path)
    keep_obs_ids = read_obs_id_filter(repo_root, case.obs_id_filter_csv)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            obs_id = int(row["obs_id"])
            if keep_obs_ids is not None and obs_id not in keep_obs_ids:
                continue
            reached = bool_from_csv(row["reached"])
            collided = bool_from_csv(row["collided"])
            yield {
                "robot": case.robot,
                "method": case.method,
                "obs_id": obs_id,
                "n_obs": obstacle_count_from_obs_id(obs_id),
                "reached": reached,
                "collided": collided,
                "sim_t": float(row["sim_t"]),
                "csv_path": case.csv_path,
            }


def collect_rows(repo_root: Path, include_dpcbf: bool) -> list[dict[str, object]]:
    cases = list(SDF_CASES)
    if include_dpcbf:
        cases.extend(DPCBF_CASES)

    rows: list[dict[str, object]] = []
    for case in cases:
        case_rows = list(iter_rows(repo_root, case))
        if len(case_rows) != 100:
            raise RuntimeError(
                f"Expected 100 rows for {case.robot} {case.method}, got {len(case_rows)} "
                f"from {case.csv_path}"
            )
        rows.extend(case_rows)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in rows})
    counts = range(OBS_COUNT_LOW, OBS_COUNT_HIGH_EXCLUSIVE)

    for method in methods:
        for n_obs in counts:
            group = [row for row in rows if row["method"] == method and row["n_obs"] == n_obs]
            if not group:
                continue
            reached = sum(bool(row["reached"]) for row in group)
            collided = sum(bool(row["collided"]) for row in group)
            reach_times = [float(row["sim_t"]) for row in group if bool(row["reached"])]
            summary.append(
                {
                    "method": method,
                    "n_obs": n_obs,
                    "n": len(group),
                    "reach_rate": reached / len(group),
                    "collision_rate": collided / len(group),
                    "avg_reach_time": float(np.mean(reach_times)) if reach_times else np.nan,
                }
            )
    return summary


def write_summary_csv(summary: list[dict[str, object]], out_path: Path) -> None:
    fields = ["method", "n_obs", "n", "reach_rate", "collision_rate", "avg_reach_time"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def plot_summary(summary: list[dict[str, object]], out_base: Path, include_dpcbf: bool) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"OA-CBF w/ FC": "#B45309", "OA-CBF w/ GAT": "#2563EB"}
    markers = {"OA-CBF w/ FC": "s", "OA-CBF w/ GAT": "o"}
    methods = ["OA-CBF w/ FC", "OA-CBF w/ GAT"]
    x_values = list(range(OBS_COUNT_LOW, OBS_COUNT_HIGH_EXCLUSIVE))

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), sharex=True)
    metric_specs = [
        ("collision_rate", "Collision rate (%)", (0, 30)),
        ("reach_rate", "Goal-reaching rate (%)", (70, 102)),
    ]

    for ax, (metric, ylabel, ylim) in zip(axes, metric_specs):
        for method in methods:
            ys = []
            for n_obs in x_values:
                match = [
                    row
                    for row in summary
                    if row["method"] == method and int(row["n_obs"]) == n_obs
                ]
                if match:
                    ys.append(100.0 * float(match[0][metric]))
                else:
                    ys.append(np.nan)
            ax.plot(
                x_values,
                ys,
                color=colors[method],
                marker=markers[method],
                linewidth=2.0,
                markersize=5.0,
                label=method,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Number of obstacles")
        ax.set_xticks(x_values)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", color="#D4D4D4", linewidth=0.8)
        ax.grid(True, axis="x", color="#ECECEC", linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_title("(a) Collision")
    axes[1].set_title("(b) Goal reaching")
    axes[0].legend(frameon=False, loc="upper left", handlelength=2.0)
    fig.tight_layout(pad=0.8)

    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to this script's repository root.",
    )
    parser.add_argument(
        "--include-dpcbf",
        action="store_true",
        help="Include the kinematic-bicycle DPCBF experiment in addition to SDF benchmarks.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper_figures" / "output",
        help="Output directory for the figure and source-data CSV.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(repo_root, include_dpcbf=args.include_dpcbf)
    summary = summarize(rows)

    suffix = "sdf_dpcbf" if args.include_dpcbf else "sdf"
    out_base = out_dir / f"obstacle_count_ablation_{suffix}"
    write_summary_csv(summary, out_base.with_suffix(".csv"))
    plot_summary(summary, out_base, include_dpcbf=args.include_dpcbf)

    print(f"Wrote {out_base.with_suffix('.pdf')}")
    print(f"Wrote {out_base.with_suffix('.png')}")
    print(f"Wrote {out_base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
