#!/usr/bin/env python3
"""
Summarize BarrierNet experiment CSVs into the same "metrics summary" text format used for MLP summaries.

Input CSV format (from adaptation_experiment.py):
mode,robot,controller,idx,pose,obs_id,reached,collided,sim_t,deadlock

Outputs one summary .txt per CSV, placed alongside the CSV by default.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent


@dataclass
class Metrics:
    total: int
    reached: int
    collided: int
    avg_time_reached: float
    avg_time_all: float


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def summarize_csv(csv_path: Path) -> Tuple[str, str, Metrics]:
    reached_times: List[float] = []
    all_times: List[float] = []
    reached = collided = total = 0
    robot = controller = None

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if robot is None:
                robot = row.get("robot", "").strip()
            if controller is None:
                controller = row.get("controller", "").strip()

            total += 1
            r = _parse_bool(row.get("reached", "False"))
            c = _parse_bool(row.get("collided", "False"))
            t = _safe_float(row.get("sim_t", "0.0"), 0.0)
            all_times.append(t)
            if r:
                reached += 1
                reached_times.append(t)
            if c:
                collided += 1

    robot = robot or "UnknownRobot"
    controller = controller or "UnknownController"
    avg_reached = (sum(reached_times) / len(reached_times)) if reached_times else 0.0
    avg_all = (sum(all_times) / len(all_times)) if all_times else 0.0
    return robot, controller, Metrics(total=total, reached=reached, collided=collided, avg_time_reached=avg_reached, avg_time_all=avg_all)


def format_summary(robot: str, controller: str, m: Metrics, title: str) -> str:
    # Match the 1-11 line format the user showed (with slightly generalized title)
    collision_rate = (m.collided / m.total) if m.total else 0.0
    reach_rate = (m.reached / m.total) if m.total else 0.0

    collision_str = f"{collision_rate*100:.1f}% ({m.collided}/{m.total})"
    reach_str = f"{reach_rate*100:.1f}% ({m.reached}/{m.total})"

    lines = []
    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"ROBOT: {robot}")
    lines.append("=" * 100)
    lines.append(
        f"{'Controller':<40} {'Collision Rate':<20} {'Reach Rate':<20} {'Avg Time (Reached)':<20} {'Avg Time (All)':<20}"
    )
    lines.append("-" * 100)
    lines.append(
        f"{controller:<40} {collision_str:<20} {reach_str:<20} {m.avg_time_reached:>7.3f} sec{'':<12} {m.avg_time_all:>7.3f} sec"
    )
    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines) + "\n"


def iter_csv_files(base_dir: Path) -> Iterable[Path]:
    yield from sorted(base_dir.glob("**/*.csv"))


def main():
    base_dir = Path(os.environ.get("BN_RESULTS_DIR", str(ROOT / "barriernet_results"))).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    out_mode = os.environ.get("BN_SUMMARY_OUT", "next_to_csv").strip().lower()
    # next_to_csv: write summary next to each CSV
    # per_robot_dir: write to base_dir/<robot>/summaries/

    csv_files = list(iter_csv_files(base_dir))
    if not csv_files:
        raise RuntimeError(f"No CSV files found under: {base_dir}")

    wrote = 0
    for csv_path in csv_files:
        robot, controller, metrics = summarize_csv(csv_path)

        title = f"METRICS SUMMARY FOR BarrierNet ({csv_path.name})"
        summary_txt = format_summary(robot, controller, metrics, title)

        if out_mode == "per_robot_dir":
            out_dir = base_dir / robot / "summaries"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (csv_path.stem + "_summary.txt")
        else:
            out_path = csv_path.with_suffix(".summary.txt")

        out_path.write_text(summary_txt)
        wrote += 1

    print(f"✅ Wrote {wrote} summary files under {base_dir} (mode={out_mode})")


if __name__ == "__main__":
    main()






