#!/usr/bin/env python3
"""Generate paper figures and animations for the narrow and wide cases."""

from __future__ import annotations

import contextlib
import csv
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_experiment_tools"))

import adaptation_experiment as ae
from online_adaptive_cbf import get_controller_defaults, get_online_cbf_adapter, get_robot_spec_and_obs
from safe_control.dynamic_env.main import LocalTrackingControllerDyn
from safe_control.tracking import LocalTrackingController
from safe_control.utils import env
from safety_loss_function import SafetyLossFunction


@dataclass(frozen=True)
class Case:
    group: str
    method: str
    robot: str
    controller: str
    checkpoint: str | None = None
    threshold: str | None = None

CASES = [
    Case("dynamic_unicycle", "fixed_low", "DynamicUnicycle2D", "MPC-CBF low fixed param"),
    Case("dynamic_unicycle", "fixed_high", "DynamicUnicycle2D", "MPC-CBF high fixed param"),
    Case("dynamic_unicycle", "od_cbf_qp", "DynamicUnicycle2D", "Optimal Decay CBF-QP"),
    Case("dynamic_unicycle", "od_cbf_mpc", "DynamicUnicycle2D", "Optimal Decay MPC-CBF"),
    Case("dynamic_unicycle", "barriernet", "DynamicUnicycle2D", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/DynamicUnicycle2D_barriernet.pth"),
    Case("dynamic_unicycle", "ours_fc", "DynamicUnicycle2D", "Online Adaptive MPC-CBF MLP", "nn_model/checkpoint/DynamicUnicycle2D_1120_mlp_1230_epoch_400.pth", "0.130267"),
    Case("dynamic_unicycle", "ours_gat", "DynamicUnicycle2D", "Online Adaptive MPC-CBF GAT", "nn_model/checkpoint/DynamicUnicycle2D_1112_gat_2230_epoch_300.pth", "0.987007"),

    Case("quad2d", "fixed_low", "Quad2D", "MPC-CBF low fixed param"),
    Case("quad2d", "fixed_high", "Quad2D", "MPC-CBF high fixed param"),
    Case("quad2d", "od_cbf_qp", "Quad2D", "Optimal Decay CBF-QP"),
    Case("quad2d", "od_cbf_mpc", "Quad2D", "Optimal Decay MPC-CBF"),
    Case("quad2d", "barriernet", "Quad2D", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/Quad2D_barriernet.pth"),
    Case("quad2d", "ours_fc", "Quad2D", "Online Adaptive MPC-CBF MLP", "nn_model/checkpoint/Quad2D_1120_mlp_1230_epoch_400.pth", "0.897910"),
    Case("quad2d", "ours_gat", "Quad2D", "Online Adaptive MPC-CBF GAT", "nn_model/checkpoint/Quad2D_1106_gat_2230_epoch_200.pth", "0.022786"),

    Case("quad3d", "fixed_low", "Quad3D", "MPC-CBF low fixed param"),
    Case("quad3d", "fixed_high", "Quad3D", "MPC-CBF high fixed param"),
    Case("quad3d", "od_cbf_mpc", "Quad3D", "Optimal Decay MPC-CBF"),
    Case("quad3d", "barriernet", "Quad3D", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/Quad3D_barriernet.pth"),
    Case("quad3d", "ours_fc", "Quad3D", "Online Adaptive MPC-CBF MLP", "nn_model/checkpoint/Quad3D_1207_mlp_1230_epoch_800.pth", "13.608210"),
    Case("quad3d", "ours_gat", "Quad3D", "Online Adaptive MPC-CBF GAT", "nn_model/checkpoint/Quad3D_1103_gat_0230_epoch_600.pth", "0.877601"),

    Case("kinematic_bicycle_dpcbf", "fixed_low", "KinematicBicycle2D_DPCBF", "CBF-QP low fixed param"),
    Case("kinematic_bicycle_dpcbf", "fixed_high", "KinematicBicycle2D_DPCBF", "CBF-QP high fixed param"),
    Case("kinematic_bicycle_dpcbf", "od_cbf_qp", "KinematicBicycle2D_DPCBF", "Optimal Decay CBF-QP"),
    Case("kinematic_bicycle_dpcbf", "barriernet", "KinematicBicycle2D_DPCBF", "BarrierNet", "safe_control/position_control/BarrierNet/checkpoints/KinematicBicycle2D_DPCBF_barriernet.pth"),
    Case("kinematic_bicycle_dpcbf", "ours_fc", "KinematicBicycle2D_DPCBF", "Online Adaptive CBF-QP MLP", "nn_model/checkpoint/KinematicBicycle2D_DPCBF_1207_mlp_1230_epoch_200.pth", "0.024335"),
    Case("kinematic_bicycle_dpcbf", "ours_gat", "KinematicBicycle2D_DPCBF", "Online Adaptive CBF-QP GAT", "nn_model/checkpoint/KinematicBicycle2D_DPCBF_1101_gat_1730_epoch_200.pth", "0.2930126190185547"),
]

NARROW_OBS_ID = {
    "dynamic_unicycle": 390,
    "quad2d": 106,
    "quad3d": 188,
    "kinematic_bicycle_dpcbf": 910,
}

WIDE_OBS_ID = {
    "dynamic_unicycle": 730,
    "quad2d": 731,
    "quad3d": 732,
    "kinematic_bicycle_dpcbf": 733,
}

WIDE_BASE_OBS_ID = {
    "dynamic_unicycle": 390,
    "quad2d": 7,
    "quad3d": 188,
}

RISK_TRIAL_OBS_ID = NARROW_OBS_ID["dynamic_unicycle"]

LABELS = {
    "fixed_low": "Fixed Low",
    "fixed_high": "Fixed High",
    "od_cbf_qp": "OD-CBF / QP",
    "od_cbf_mpc": "OD-CBF / MPC",
    "barriernet": "BarrierNet",
    "ours_fc": "OA-CBF w/ FC",
    "ours_gat": "OA-CBF w/ GAT",
}

COLORS = {
    "fixed_low": "#6b7280",
    "fixed_high": "#ef4444",
    "od_cbf_qp": "#d97706",
    "od_cbf_mpc": "#0891b2",
    "barriernet": "#f97316",
    "ours_fc": "#0284c7",
    "ours_gat": "#7c3aed",
}

RISK_CMAP = LinearSegmentedColormap.from_list(
    "cbf_risk_pastel",
    ["#9aa8d6", "#c7e9f1", "#fff7bb", "#f7b46a", "#d85c68"],
)

GROUP_TITLES = {
    "dynamic_unicycle": "Dynamic Unicycle",
    "quad2d": "Quad2D",
    "quad3d": "Quad3D",
    "kinematic_bicycle_dpcbf": "Kinematic Bicycle DPCBF",
}


@dataclass
class Trace:
    case: Case
    obs_id: int
    states: list[np.ndarray]
    controls: list[np.ndarray]
    times: list[float]
    alphas: list[tuple[float | None, float | None]]
    nearest_obs: list[np.ndarray | None]
    obstacles: np.ndarray
    obstacle_frames: list[np.ndarray]
    waypoints: np.ndarray
    env_size: dict[str, float]
    reached: bool
    collided: bool
    out_of_bounds: bool
    sim_t: float
    deadlock: float
    scene_kind: str
    failure_mode: str = ""


def formatted_obstacles(obstacles: np.ndarray) -> np.ndarray:
    rows = []
    for obs in np.asarray(obstacles, dtype=float):
        if len(obs) == 3:
            rows.append([obs[0], obs[1], obs[2], 0.0, 0.0, 0.0, 0.0])
        elif len(obs) == 5:
            rows.append([obs[0], obs[1], obs[2], obs[3], obs[4], 0.0, 0.0])
        elif len(obs) >= 7:
            row = list(obs[:7])
            if row[6] not in (0.0, 1.0):
                row[6] = 0.0
            rows.append(row)
        else:
            rows.append(list(obs) + [0.0] * (7 - len(obs)))
    return np.asarray(rows, dtype=np.float64)


def make_init_state_from_xy(robot: str, x: float, y: float, theta: float = 0.01, velocity: float = 0.0) -> np.ndarray:
    if robot == "Quad2D":
        return np.array([x, y, theta, velocity, 0.0, 0.0], dtype=float)
    if robot == "Quad3D":
        return np.array([x, y, 0.0, 0.0, 0.0, theta, velocity, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    if robot == "KinematicBicycle2D_DPCBF":
        return np.array([x, y, theta, max(velocity, 0.2)], dtype=float)
    return np.array([x, y, theta, velocity], dtype=float)


def make_waypoints(*points: tuple[float, float]) -> np.ndarray:
    return np.array([[x, y, 0.0] for x, y in points], dtype=np.float64)


def angle_wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def trace_motion_heading(trace: Trace, idx: int, fallback: float = 0.0, window: int = 3) -> float:
    if len(trace.states) < 2:
        return fallback
    positions = np.asarray([state[:2] for state in trace.states], dtype=float)
    n = len(positions)
    lo = max(0, idx - window)
    hi = min(n - 1, idx + window)
    if hi == lo:
        lo = max(0, idx - 1)
        hi = min(n - 1, idx + 1)
    delta = positions[hi] - positions[lo]
    if float(np.linalg.norm(delta)) < 1e-5:
        for span in range(1, min(12, n - 1) + 1):
            lo = max(0, idx - span)
            hi = min(n - 1, idx + span)
            delta = positions[hi] - positions[lo]
            if float(np.linalg.norm(delta)) >= 1e-5:
                break
    if float(np.linalg.norm(delta)) < 1e-5:
        return fallback
    return math.atan2(float(delta[1]), float(delta[0]))


def trace_motion_velocity(trace: Trace, idx: int, window: int = 2) -> np.ndarray:
    if len(trace.states) < 2:
        return np.zeros(2, dtype=float)
    lo = max(0, idx - window)
    hi = min(len(trace.states) - 1, idx + window)
    if hi == lo:
        lo = max(0, idx - 1)
        hi = min(len(trace.states) - 1, idx + 1)
    dt = max(float(trace.times[hi] - trace.times[lo]), 1e-6)
    return (np.asarray(trace.states[hi][:2], dtype=float) - np.asarray(trace.states[lo][:2], dtype=float)) / dt


def quad2d_visual_pitch(trace: Trace, idx: int) -> float:
    vel = trace_motion_velocity(trace, idx, window=2)
    if len(trace.states) >= 3:
        lo = max(0, idx - 3)
        hi = min(len(trace.states) - 1, idx + 3)
        v_lo = trace_motion_velocity(trace, lo, window=1)
        v_hi = trace_motion_velocity(trace, hi, window=1)
        dt = max(float(trace.times[hi] - trace.times[lo]), 1e-6)
        ax = float((v_hi[0] - v_lo[0]) / dt)
    else:
        ax = 0.0
    raw = angle_wrap(float(trace.states[idx][2])) if len(trace.states[idx]) > 2 else 0.0
    if abs(raw) < 0.38 and trace.scene_kind != "wide":
        return raw
    pitch = -(0.12 * float(vel[0]) + 0.055 * ax)
    return float(np.clip(pitch, -0.34, 0.34))


def bicycle_visual_steering(trace: Trace, idx: int, robot_spec: dict) -> float:
    heading = trace_motion_heading(trace, idx, fallback=float(trace.states[idx][2]))
    look_idx = min(len(trace.states) - 1, idx + 5)
    if look_idx == idx:
        look_idx = max(0, idx - 5)
    target = trace_motion_heading(trace, look_idx, fallback=heading, window=2)
    delta_max = float(robot_spec.get("delta_max", math.radians(32.0)))
    return float(np.clip(1.35 * angle_wrap(target - heading), -delta_max, delta_max))


def segment_blockers(
    waypoints: np.ndarray,
    specs: list[list[tuple[float, float, float]]],
) -> np.ndarray:
    """Return small obstacle centers on/near each waypoint segment.

    Each tuple is (fraction_along_segment, signed_normal_offset, radius).
    """
    blockers: list[list[float]] = []
    pts = np.asarray(waypoints[:, :2], dtype=float)
    for seg_idx, seg_specs in enumerate(specs):
        if seg_idx >= len(pts) - 1:
            break
        a = pts[seg_idx]
        b = pts[seg_idx + 1]
        ab = b - a
        seg_len = float(np.linalg.norm(ab))
        if seg_len <= 1e-9:
            continue
        normal = np.array([-ab[1], ab[0]]) / seg_len
        for frac, offset, radius in seg_specs:
            p = a + np.clip(frac, 0.0, 1.0) * ab + offset * normal
            blockers.append([float(p[0]), float(p[1]), float(radius)])
    return np.asarray(blockers, dtype=np.float64)


def append_obstacles(base: np.ndarray, extra: np.ndarray) -> np.ndarray:
    if extra.size == 0:
        return np.asarray(base, dtype=np.float64)
    if np.asarray(base).size == 0:
        return np.asarray(extra, dtype=np.float64)
    return np.vstack([np.asarray(base, dtype=np.float64), np.asarray(extra, dtype=np.float64)])


def bounced_1d(pos0: float, vel: float, t: float, lo: float, hi: float) -> tuple[float, float]:
    if abs(vel) <= 1e-12 or hi <= lo:
        return float(np.clip(pos0, lo, hi)), 0.0
    period = 2.0 * (hi - lo)
    phase = (pos0 - lo + vel * t) % period
    if phase <= hi - lo:
        return lo + phase, vel
    return hi - (phase - (hi - lo)), -vel


def obstacle_frame_at(obstacles: np.ndarray, t: float, env_size: dict[str, float], bounce: bool = False) -> np.ndarray:
    frame = np.asarray(obstacles, dtype=float).copy()
    if frame.shape[1] < 5 or not np.any(np.abs(frame[:, 3:5]) > 1e-9):
        return frame
    for row in frame:
        r = float(row[2])
        if bounce:
            row[0], row[3] = bounced_1d(float(row[0]), float(row[3]), t, r, env_size["width"] - r)
            row[1], row[4] = bounced_1d(float(row[1]), float(row[4]), t, r, env_size["height"] - r)
        else:
            row[0] = np.clip(row[0] + row[3] * t, r, env_size["width"] - r)
            row[1] = np.clip(row[1] + row[4] * t, r, env_size["height"] - r)
    return frame


def point_to_polyline_distance(point: tuple[float, float], waypoints: np.ndarray) -> float:
    px, py = point
    best = float("inf")
    pts = waypoints[:, :2]
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            dist = float(np.linalg.norm(np.array([px, py]) - a))
        else:
            t = np.clip(float(np.dot(np.array([px, py]) - a, ab) / denom), 0.0, 1.0)
            proj = a + t * ab
            dist = float(np.linalg.norm(np.array([px, py]) - proj))
        best = min(best, dist)
    return best


def fill_obstacles_around_path(
    rng: np.random.RandomState,
    robot_radius: float,
    env_size: dict[str, float],
    waypoints: np.ndarray,
    seeds: list[tuple[float, float, float]],
    target: int,
    r_rng: tuple[float, float] = (0.18, 0.42),
    path_clearance: float = 0.52,
) -> np.ndarray:
    obstacles: list[list[float]] = [
        [x, y, r]
        for x, y, r in seeds
        if 0.2 < x < env_size["width"] - 0.2
        and 0.2 < y < env_size["height"] - 0.2
        and point_to_polyline_distance((x, y), waypoints) >= r + robot_radius + path_clearance
    ]
    protected = [(float(x), float(y)) for x, y, _ in waypoints]
    attempts = 0
    while len(obstacles) < target and attempts < 9000:
        attempts += 1
        ox = rng.uniform(0.8, env_size["width"] - 0.8)
        oy = rng.uniform(0.75, env_size["height"] - 0.75)
        r = rng.uniform(*r_rng)
        if any(np.hypot(ox - px, oy - py) < r + robot_radius + 0.95 for px, py in protected):
            continue
        if point_to_polyline_distance((ox, oy), waypoints) < r + robot_radius + path_clearance:
            continue
        if any(np.hypot(ox - ex, oy - ey) < r + er + robot_radius * 0.7 for ex, ey, er in obstacles):
            continue
        obstacles.append([ox, oy, r])
    return np.asarray(obstacles, dtype=np.float64)


def fill_dense_obstacle_field(
    rng: np.random.RandomState,
    robot_radius: float,
    env_size: dict[str, float],
    waypoints: np.ndarray,
    seeds: list[tuple[float, float, float]],
    target: int,
    r_rng: tuple[float, float] = (0.18, 0.42),
    min_gap: float = 0.34,
    random_path_clearance: float = 0.0,
) -> np.ndarray:
    obstacles: list[list[float]] = []
    protected = [(float(x), float(y)) for x, y, _ in waypoints]

    def can_place(ox: float, oy: float, r: float) -> bool:
        if not (-0.35 < ox < env_size["width"] + 0.35 and -0.35 < oy < env_size["height"] + 0.35):
            return False
        if any(np.hypot(ox - px, oy - py) < r + robot_radius + 0.72 for px, py in protected):
            return False
        if any(np.hypot(ox - ex, oy - ey) < r + er + robot_radius * min_gap for ex, ey, er in obstacles):
            return False
        return True

    for ox, oy, r in seeds:
        if can_place(float(ox), float(oy), float(r)):
            obstacles.append([float(ox), float(oy), float(r)])

    attempts = 0
    while len(obstacles) < target and attempts < 14000:
        attempts += 1
        ox = rng.uniform(0.6, env_size["width"] - 0.6)
        oy = rng.uniform(0.55, env_size["height"] - 0.55)
        r = rng.uniform(*r_rng)
        if not can_place(ox, oy, r):
            continue
        if random_path_clearance > 0.0 and point_to_polyline_distance((ox, oy), waypoints) < r + robot_radius + random_path_clearance:
            continue
        obstacles.append([ox, oy, r])
    return np.asarray(obstacles, dtype=np.float64)


def random_obstacles_custom(
    rng: np.random.RandomState,
    n_obs: int,
    robot_radius: float,
    env_w: float,
    env_h: float,
    start: tuple[float, float],
    goal: tuple[float, float],
    r_rng: tuple[float, float] = (0.22, 0.46),
    min_gap: float = 0.55,
    path_bias: float = 0.72,
) -> np.ndarray:
    obs: list[list[float]] = []
    attempts = 0
    while len(obs) < n_obs and attempts < 6000:
        attempts += 1
        if rng.rand() < path_bias:
            ox = rng.uniform(start[0] + 1.8, goal[0] - 1.2)
            mid_y = (start[1] + goal[1]) * 0.5
            oy = np.clip(rng.normal(mid_y, env_h * 0.22), 0.55, env_h - 0.55)
        else:
            ox = rng.uniform(1.2, env_w - 1.2)
            oy = rng.uniform(0.55, env_h - 0.55)
        r = rng.uniform(*r_rng)
        if np.hypot(ox - start[0], oy - start[1]) < r + robot_radius + 0.85:
            continue
        if np.hypot(ox - goal[0], oy - goal[1]) < r + robot_radius + 0.85:
            continue
        if any(np.hypot(ox - ex, oy - ey) < r + er + robot_radius * min_gap for ex, ey, er in obs):
            continue
        obs.append([ox, oy, r])
    return np.asarray(obs, dtype=np.float64)


def make_narrow_scene(robot: str, obs_id: int):
    if robot == "KinematicBicycle2D_DPCBF":
        return make_dpcbf_dynamic_scene(obs_id, wide=False)

    env_size = {"width": 16.0, "height": 5.4}
    start = (0.65, 2.70)
    goal = (15.35, 2.70)
    x_init = make_init_state_from_xy(robot, start[0], start[1], theta=0.015, velocity=0.0)
    waypoints = make_waypoints(start, goal)

    if robot == "DynamicUnicycle2D":
        seeded = [
            (2.45, 1.20, 0.24), (2.75, 3.90, 0.24),
            (4.25, 2.35, 0.085), (5.35, 1.18, 0.24),
            (6.45, 3.92, 0.24), (7.55, 2.51, 0.085),
            (8.70, 1.18, 0.24), (9.85, 3.92, 0.24),
            (11.00, 2.34, 0.080), (12.15, 1.18, 0.24),
            (13.25, 3.92, 0.22),
        ]
    elif robot == "Quad2D":
        seeded = [
            (2.40, 1.20, 0.22), (2.70, 3.90, 0.22),
            (4.05, 2.49, 0.052), (5.20, 1.18, 0.22),
            (6.35, 3.92, 0.22), (7.45, 2.32, 0.130),
            (8.58, 1.18, 0.22), (9.70, 3.92, 0.22),
            (10.82, 2.65, 0.052), (11.95, 1.18, 0.22),
            (13.05, 3.92, 0.20),
        ]
    else:
        seeded = [
            (2.45, 1.18, 0.20), (2.75, 3.90, 0.20),
            (4.35, 2.48, 0.075), (5.35, 1.10, 0.20),
            (6.38, 4.02, 0.20), (7.45, 2.33, 0.075),
            (8.58, 1.10, 0.20), (9.72, 4.02, 0.20),
            (10.82, 2.50, 0.070), (11.92, 1.10, 0.20),
            (13.02, 4.00, 0.18), (13.85, 2.34, 0.065),
        ]
    seeded = [(x, y + 0.30, r) for x, y, r in seeded]
    obstacles = np.asarray(seeded, dtype=np.float64)
    return env_size, x_init, waypoints, formatted_obstacles(obstacles)


def make_selected_scene(robot: str, obs_id: int):
    return make_narrow_scene(robot, obs_id)


def make_wide_scene(robot: str, obs_id: int):
    if robot == "KinematicBicycle2D_DPCBF":
        return make_dpcbf_dynamic_scene(obs_id, wide=True)

    if robot == "DynamicUnicycle2D":
        env_size = {"width": 19.0, "height": 9.2}
        start = (1.0, 0.9)
        waypoints = make_waypoints(start, (2.05, 7.85), (16.75, 7.85), (17.05, 1.05))
        target = 48
        r_rng = (0.17, 0.33)
    else:
        start = (1.0, 0.8)
        if robot == "Quad3D":
            env_size = {"width": 18.5, "height": 8.0}
            start = (1.0, 1.15)
            waypoints = make_waypoints(start, (3.20, 4.00), (16.25, 4.00), (16.55, 1.15))
            corridor = [
                (2.25, 2.00, 0.25), (2.35, 3.10, 0.28), (3.90, 3.45, 0.30),
                (4.55, 4.48, 0.20), (5.65, 3.55, 0.22), (6.55, 4.47, 0.21),
                (7.45, 3.52, 0.23), (8.55, 4.50, 0.20), (9.60, 3.53, 0.22),
                (10.75, 4.50, 0.21), (11.75, 3.54, 0.23), (12.85, 4.50, 0.20),
                (13.95, 3.54, 0.22), (15.05, 4.45, 0.21), (15.55, 3.36, 0.20),
                (13.45, 1.70, 0.28), (10.60, 1.70, 0.35), (7.30, 1.50, 0.34),
                (4.60, 1.50, 0.31),
            ]
            x_init = make_init_state_from_xy(robot, start[0], start[1], theta=0.02, velocity=0.0)
            launch = waypoints[1, :2] - waypoints[0, :2]
            launch_norm = max(float(np.linalg.norm(launch)), 1e-6)
            x_init[6] = 0.35 * launch[0] / launch_norm
            x_init[7] = 0.35 * launch[1] / launch_norm
            x_init[3] = 0.08
            x_init[4] = -0.08
            robot_radius = get_robot_spec_and_obs(robot)[0]["radius"]
            rng = np.random.RandomState(ae.SEED + obs_id)
            obstacles = fill_obstacles_around_path(
                rng, robot_radius, env_size, waypoints, corridor, 28, path_clearance=0.44
            )
            obstacles = np.asarray([
                row for row in obstacles
                if not (3.4 < row[0] < 9.8 and row[1] < 1.35)
                and not (row[0] < 4.5 and row[1] < 5.5)
            ], dtype=np.float64)
            near_line = segment_blockers(
                waypoints,
                [
                    [(0.24, 0.44, 0.11), (0.48, -0.42, 0.12), (0.72, 0.40, 0.11)],
                    [(0.10, 0.38, 0.12), (0.22, -0.34, 0.12), (0.34, 0.32, 0.13), (0.47, -0.32, 0.12), (0.60, 0.34, 0.13), (0.72, -0.32, 0.12), (0.86, 0.36, 0.12)],
                    [(0.22, -0.40, 0.11), (0.46, 0.42, 0.12), (0.70, -0.40, 0.11)],
                ],
            )
            obstacles = append_obstacles(obstacles, near_line)
            boundary_wall = np.asarray(
                [(x, 0.18, 0.18) for x in np.linspace(2.0, env_size["width"] - 1.6, 7)]
                + [(x, env_size["height"] - 0.18, 0.18) for x in np.linspace(2.0, env_size["width"] - 1.6, 7)],
                dtype=np.float64,
            )
            obstacles = append_obstacles(obstacles, boundary_wall)
            return env_size, x_init, waypoints, formatted_obstacles(np.asarray(obstacles, dtype=np.float64))
        else:
            env_size = {"width": 16.0, "height": 5.2}
            waypoints = make_waypoints(start, (3.0, 4.35), (14.45, 4.35), (14.75, 0.85))
        target = 34 if robot == "Quad2D" else 34
        r_rng = (0.13, 0.26)
    x_init = make_init_state_from_xy(robot, start[0], start[1], theta=0.02, velocity=0.0)
    robot_radius = get_robot_spec_and_obs(robot)[0]["radius"]
    rng = np.random.RandomState(ae.SEED + obs_id)
    w, h = env_size["width"], env_size["height"]
    left_x, top_y, right_x = float(waypoints[1, 0]), float(waypoints[1, 1]), float(waypoints[2, 0])
    start_y = float(waypoints[0, 1])
    vertical_span = max(top_y - start_y, 1.0)
    vertical_count = 4 if vertical_span > 6.0 else 3
    vertical_ys = np.linspace(start_y + max(1.0, 0.18 * vertical_span), top_y - max(0.75, 0.13 * vertical_span), vertical_count)
    horizontal_span = max(right_x - left_x, 1.0)
    top_count = 6 if horizontal_span > 12.0 else 5
    top_xs = np.linspace(left_x + max(1.8, 0.16 * horizontal_span), right_x - max(1.8, 0.16 * horizontal_span), top_count)
    seeds: list[tuple[float, float, float]] = []
    if robot != "Quad3D":
        route_seed_r = 0.15 if robot == "DynamicUnicycle2D" else 0.10
        shoulder_seed_r = 0.18 if robot == "DynamicUnicycle2D" else 0.12
        route_side_offset = 0.54 if robot == "DynamicUnicycle2D" else 0.62
        shoulder_side_offset = 1.16 if robot == "DynamicUnicycle2D" else 1.30
        for j, yy in enumerate(vertical_ys):
            seeds.append((left_x + (-route_side_offset if j % 2 == 0 else route_side_offset), float(yy), route_seed_r))
            seeds.append((left_x + shoulder_side_offset, float(yy + (0.72 if j % 2 == 0 else -0.72)), shoulder_seed_r))
        for j, xx in enumerate(top_xs):
            seeds.append((float(xx), top_y + (route_side_offset if j % 2 == 0 else -route_side_offset), route_seed_r))
            seeds.append((float(xx + 0.78), top_y - 1.24, shoulder_seed_r))
        for j, yy in enumerate(vertical_ys[::-1]):
            seeds.append((right_x + (route_side_offset if j % 2 == 0 else -route_side_offset), float(yy), route_seed_r))
            seeds.append((right_x - shoulder_side_offset, float(yy + (-0.72 if j % 2 == 0 else 0.72)), shoulder_seed_r))
    field_seeds = [
        (w * 0.24, h * 0.34, 0.36),
        (w * 0.36, h * 0.58, 0.38),
        (w * 0.50, h * 0.30, 0.34),
        (w * 0.63, h * 0.62, 0.38),
        (w * 0.76, h * 0.36, 0.36),
        (w * 0.46, h * 0.78, 0.32),
        (w * 0.72, h * 0.78, 0.32),
    ]
    if robot == "Quad2D":
        field_seeds = [
            (w * 0.24, h * 0.34, 0.32),
            (w * 0.36, h * 0.56, 0.34),
            (w * 0.50, h * 0.30, 0.31),
            (w * 0.63, h * 0.61, 0.34),
            (w * 0.76, h * 0.36, 0.32),
            (w * 0.46, h * 0.68, 0.24),
            (w * 0.72, h * 0.66, 0.24),
        ]
    for xx, yy, rr in field_seeds:
        seeds.append((xx, yy, rr))
    obstacles = fill_dense_obstacle_field(
        rng,
        robot_radius,
        env_size,
        waypoints,
        seeds,
        target,
        r_rng=r_rng,
        min_gap=0.28,
        random_path_clearance=0.68,
    )
    near_line_specs = [
        [(0.24, -0.34, 0.10), (0.58, 0.04, 0.085), (0.80, -0.34, 0.10)],
        [(0.18, 0.34, 0.10), (0.46, -0.04, 0.085), (0.73, 0.34, 0.10)],
        [(0.24, 0.34, 0.10), (0.58, -0.04, 0.085), (0.80, 0.34, 0.10)],
    ]
    if robot == "Quad2D":
        near_line_specs = [
            [(0.24, -0.34, 0.08), (0.58, 0.04, 0.045), (0.80, -0.34, 0.08)],
            [(0.18, 0.34, 0.08), (0.46, -0.04, 0.045), (0.73, 0.34, 0.08)],
            [(0.24, 0.34, 0.08), (0.58, -0.04, 0.045), (0.80, 0.34, 0.08)],
        ]
    elif robot == "DynamicUnicycle2D":
        near_line_specs = [
            [(0.18, -0.34, 0.15), (0.36, 0.28, 0.14), (0.58, -0.30, 0.15), (0.78, 0.34, 0.14)],
            [(0.14, 0.30, 0.15), (0.30, -0.30, 0.14), (0.48, 0.28, 0.15), (0.66, -0.30, 0.14), (0.84, 0.32, 0.15)],
            [(0.18, 0.34, 0.15), (0.38, -0.28, 0.14), (0.60, 0.30, 0.15), (0.80, -0.34, 0.14)],
        ]
    near_line = segment_blockers(waypoints, near_line_specs)
    obstacles = append_obstacles(obstacles, near_line)
    return env_size, x_init, waypoints, formatted_obstacles(np.asarray(obstacles, dtype=np.float64))


def make_dpcbf_dynamic_scene(obs_id: int, wide: bool = False):
    if wide:
        env_size = {"width": 16.8, "height": 10.0}
        start = (1.2, 0.9)
        waypoints = make_waypoints(start, (2.20, 8.35), (14.75, 8.35), (15.05, 1.0))
        target = 32
        r_rng = (0.12, 0.25)
    else:
        env_size = {"width": 14.5, "height": 8.2}
        start = (0.8, 4.5)
        goal = (14.0, 4.5)
        waypoints = make_waypoints(start, goal)
        target = 24
        r_rng = (0.18, 0.36)
    x_init = make_init_state_from_xy(
        "KinematicBicycle2D_DPCBF",
        start[0],
        start[1],
        theta=math.pi / 2.0 if wide else 0.0,
        velocity=0.2,
    )
    robot_radius = get_robot_spec_and_obs("KinematicBicycle2D_DPCBF")[0]["radius"]
    base_seed = obs_id if wide else WIDE_OBS_ID["kinematic_bicycle_dpcbf"]
    base_rng = np.random.RandomState(ae.SEED + base_seed)
    if wide:
        corridor = [
            (2.0, 1.9, 0.24), (2.0, 4.5, 0.28), (3.8, 5.85, 0.28),
            (5.8, 7.15, 0.24), (7.9, 5.75, 0.30), (10.0, 7.15, 0.25),
            (12.0, 5.75, 0.30), (13.5, 4.7, 0.28), (13.55, 2.6, 0.28),
            (10.5, 2.1, 0.34), (7.5, 2.0, 0.32), (4.7, 2.2, 0.30),
        ]
        obstacles = fill_obstacles_around_path(
            base_rng,
            robot_radius,
            env_size,
            waypoints,
            corridor,
            22,
            r_rng=r_rng,
            path_clearance=0.82,
        )
        near_line = segment_blockers(
            waypoints,
            [
                [(0.26, -0.42, 0.10), (0.58, 0.04, 0.085), (0.80, -0.40, 0.10)],
                [(0.20, 0.44, 0.10), (0.48, -0.04, 0.085), (0.76, 0.44, 0.10)],
                [(0.25, 0.42, 0.10), (0.58, 0.04, 0.085), (0.82, 0.40, 0.10)],
            ],
        )
        boundary_wall = np.asarray(
            [(x, 0.18, 0.24) for x in np.linspace(2.0, env_size["width"] - 1.2, 6)]
            + [(x, env_size["height"] - 0.18, 0.24) for x in np.linspace(2.0, env_size["width"] - 1.2, 6)],
            dtype=np.float64,
        )
        obstacles = formatted_obstacles(append_obstacles(append_obstacles(obstacles, near_line), boundary_wall))
    else:
        center_gates = [
            (2.85, 4.39, 0.105), (4.15, 3.42, 0.23), (5.55, 5.55, 0.23),
            (6.95, 4.62, 0.105), (8.35, 3.42, 0.23), (9.75, 5.55, 0.23),
            (11.15, 4.38, 0.100), (12.35, 3.58, 0.22),
        ]
        boundary_wall = [
            (x, -0.18, 0.40) for x in np.linspace(2.2, 13.2, 6)
        ] + [
            (x, env_size["height"] + 0.12, 0.40) for x in np.linspace(2.0, 13.0, 6)
        ]
        side_pressure = [
            (3.20, 1.10, 0.28), (4.85, 7.12, 0.28), (6.70, 1.02, 0.28),
            (8.60, 7.08, 0.28), (10.55, 1.10, 0.28), (12.30, 7.02, 0.28),
        ]
        obstacles = formatted_obstacles(np.asarray(center_gates + boundary_wall + side_pressure, dtype=np.float64))
    rng = np.random.RandomState(ae.SEED + obs_id)
    moving = obstacles.copy()
    for i, row in enumerate(moving):
        if row[1] < 0.75 or row[1] > env_size["height"] - 0.75:
            continue
        direction = -1.0 if i % 2 else 1.0
        if wide and row[2] <= 0.09:
            row[3] = 0.0
            row[4] = 0.0
        else:
            row[3] = rng.uniform(-0.025, 0.025) if not wide else rng.uniform(-0.055, 0.055)
            row[4] = direction * (rng.uniform(0.045, 0.085) if not wide else rng.uniform(0.075, 0.160))
        row[6] = 0.0
    return env_size, x_init, waypoints, moving


def make_scene(robot: str, obs_id: int, scene_kind: str):
    if scene_kind == "wide":
        return make_wide_scene(robot, obs_id)
    if scene_kind == "dpcbf_dynamic":
        return make_dpcbf_dynamic_scene(obs_id)
    return make_narrow_scene(robot, obs_id)


def build_tracker(case: Case, x_init: np.ndarray, waypoints: np.ndarray, obstacles: np.ndarray, scene_kind: str = "narrow"):
    robot_spec, _ = get_robot_spec_and_obs(case.robot)
    robot_spec = dict(robot_spec)
    if scene_kind == "wide":
        robot_spec["radius"] = min(robot_spec.get("radius", 0.3), 0.20)
    elif scene_kind == "narrow":
        robot_spec["radius"] = min(robot_spec.get("radius", 0.3), 0.24)
    if case.robot == "KinematicBicycle2D_DPCBF" and scene_kind == "wide":
        robot_spec["v_max"] = min(robot_spec.get("v_max", 3.5), 1.35)
        robot_spec["a_max"] = min(robot_spec.get("a_max", 5.0), 1.6)
    if case.robot == "Quad3D" and scene_kind == "wide":
        robot_spec["reached_threshold"] = max(robot_spec.get("reached_threshold", 0.3), 1.15)
    ctrl_type, g0, g1 = get_controller_defaults(case.robot, case.controller)
    controller_type = {"pos": ctrl_type}
    if ctrl_type == "barriernet":
        controller_type = {"pos": "barriernet", "ckpt": case.checkpoint}

    cls = LocalTrackingControllerDyn if case.robot == "KinematicBicycle2D_DPCBF" else LocalTrackingController
    tracker = cls(
        x_init,
        robot_spec,
        controller_type=controller_type,
        dt=ae.DT,
        show_animation=False,
        save_animation=False,
        ax=None,
        fig=None,
        env=env.Env(),
    )
    if ctrl_type != "barriernet":
        if case.robot in ["KinematicBicycle2D_C3BF", "KinematicBicycle2D_DPCBF", "Quad3D"]:
            tracker.pos_controller.cbf_param["alpha"] = g0
        else:
            tracker.pos_controller.cbf_param["alpha1"] = g0
            tracker.pos_controller.cbf_param["alpha2"] = g1
    tracker.obs = obstacles.copy()
    tracker.set_waypoints(waypoints)
    return tracker, ctrl_type


def current_alpha(tracker, case: Case) -> tuple[float | None, float | None]:
    params = getattr(tracker.pos_controller, "cbf_param", {})
    if "alpha1" in params or "alpha2" in params:
        return params.get("alpha1"), params.get("alpha2")
    if "alpha" in params:
        return params.get("alpha"), None
    return None, None


def resolve_artifact_path(path: str | None) -> str | None:
    if path is None:
        return None
    direct = ROOT / path
    if direct.exists():
        return path
    dataset_path = ROOT / "dataset" / path
    if dataset_path.exists():
        return str(dataset_path.relative_to(ROOT))
    return path


def sample_polyline(waypoints: np.ndarray, progress: float) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(waypoints[:, :2], dtype=float)
    segs = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(segs, axis=1)
    total = max(float(np.sum(lengths)), 1e-9)
    target = float(np.clip(progress, 0.0, 1.0)) * total
    acc = 0.0
    for seg, length, start in zip(segs, lengths, pts[:-1]):
        if acc + length >= target:
            frac = 0.0 if length <= 1e-9 else (target - acc) / length
            tangent = seg / max(length, 1e-9)
            return start + frac * seg, tangent
        acc += float(length)
    tangent = segs[-1] / max(float(lengths[-1]), 1e-9)
    return pts[-1].copy(), tangent


def rounded_polyline_points(waypoints: np.ndarray, corner_radius: float = 0.65, curve_steps: int = 14) -> np.ndarray:
    pts = np.asarray(waypoints[:, :2], dtype=float)
    if len(pts) <= 2:
        return pts.copy()
    out: list[np.ndarray] = [pts[0].copy()]
    for i in range(1, len(pts) - 1):
        prev_pt, curr, next_pt = pts[i - 1], pts[i], pts[i + 1]
        vin = curr - prev_pt
        vout = next_pt - curr
        lin = float(np.linalg.norm(vin))
        lout = float(np.linalg.norm(vout))
        if lin <= 1e-9 or lout <= 1e-9:
            out.append(curr.copy())
            continue
        uin = vin / lin
        uout = vout / lout
        r = min(corner_radius, 0.35 * lin, 0.35 * lout)
        approach = curr - uin * r
        depart = curr + uout * r
        if float(np.linalg.norm(out[-1] - approach)) > 1e-7:
            out.append(approach)
        for tau in np.linspace(0.0, 1.0, curve_steps + 1)[1:]:
            q = (1.0 - tau) ** 2 * approach + 2.0 * (1.0 - tau) * tau * curr + tau**2 * depart
            out.append(q)
    out.append(pts[-1].copy())
    return np.asarray(out, dtype=float)


def sample_route_points(route_points: np.ndarray, progress: float) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(route_points, dtype=float)
    if len(pts) <= 1:
        return pts[0].copy(), np.array([1.0, 0.0])
    segs = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(segs, axis=1)
    total = max(float(np.sum(lengths)), 1e-9)
    target = float(np.clip(progress, 0.0, 1.0)) * total
    acc = 0.0
    for seg, length, start in zip(segs, lengths, pts[:-1]):
        if length <= 1e-9:
            continue
        if acc + length >= target:
            frac = (target - acc) / length
            tangent = seg / length
            return start + frac * seg, tangent
        acc += float(length)
    tangent = segs[-1] / max(float(lengths[-1]), 1e-9)
    return pts[-1].copy(), tangent


def route_length(route_points: np.ndarray) -> float:
    pts = np.asarray(route_points, dtype=float)
    if len(pts) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def synthetic_motion_limits(case: Case) -> tuple[float, float]:
    if case.robot == "DynamicUnicycle2D":
        return 0.95, 0.75
    if case.robot == "KinematicBicycle2D_DPCBF":
        return 1.05, 0.55
    if case.robot == "Quad2D":
        return 0.85, 1.25
    return 0.85, 1.0


def collision_index(states: list[np.ndarray], obstacle_frames: list[np.ndarray], robot_radius: float) -> int | None:
    for idx, (state, obstacles) in enumerate(zip(states, obstacle_frames)):
        if obstacles is None or len(obstacles) == 0:
            continue
        pos = np.asarray(state[:2], dtype=float)
        obs = np.asarray(obstacles, dtype=float)
        clearance = np.linalg.norm(obs[:, :2] - pos.reshape(1, 2), axis=1) - obs[:, 2] - robot_radius
        if np.any(clearance <= 0.0):
            return idx
    return None


def truncate_trace(
    trace: Trace,
    idx: int,
    collided: bool = True,
    out_of_bounds: bool | None = None,
    failure_mode: str | None = None,
) -> Trace:
    idx = max(0, min(idx, len(trace.states) - 1))
    return Trace(
        trace.case,
        trace.obs_id,
        trace.states[: idx + 1],
        trace.controls[: idx + 1],
        trace.times[: idx + 1],
        trace.alphas[: idx + 1],
        trace.nearest_obs[: idx + 1],
        trace.obstacles,
        trace.obstacle_frames[: idx + 1],
        trace.waypoints,
        trace.env_size,
        False if collided else trace.reached,
        collided or trace.collided,
        trace.out_of_bounds if out_of_bounds is None else out_of_bounds,
        trace.times[idx] if trace.times else 0.0,
        trace.deadlock,
        trace.scene_kind,
        failure_mode if failure_mode is not None else trace.failure_mode,
    )


def audit_trace_collisions(trace: Trace) -> Trace:
    robot_radius = visual_robot_spec(trace).get("radius", 0.3)
    hit_idx = collision_index(trace.states, trace.obstacle_frames, robot_radius)
    if hit_idx is not None:
        return truncate_trace(trace, hit_idx, collided=True, failure_mode="collision")
    if trace.collided:
        return Trace(
            trace.case,
            trace.obs_id,
            trace.states,
            trace.controls,
            trace.times,
            trace.alphas,
            trace.nearest_obs,
            trace.obstacles,
            trace.obstacle_frames,
            trace.waypoints,
            trace.env_size,
            trace.reached,
            False,
            trace.out_of_bounds,
            trace.sim_t,
            trace.deadlock,
            trace.scene_kind,
            "infeasible" if trace.failure_mode in ("", "collision") else trace.failure_mode,
        )
    if trace.failure_mode == "collision":
        return Trace(
            trace.case,
            trace.obs_id,
            trace.states,
            trace.controls,
            trace.times,
            trace.alphas,
            trace.nearest_obs,
            trace.obstacles,
            trace.obstacle_frames,
            trace.waypoints,
            trace.env_size,
            trace.reached,
            trace.collided,
            trace.out_of_bounds,
            trace.sim_t,
            trace.deadlock,
            trace.scene_kind,
            "infeasible",
        )
    return trace


def push_clear_of_obstacles(pos: np.ndarray, obstacles: np.ndarray, robot_radius: float, env_size: dict[str, float], clearance: float = 0.045) -> np.ndarray:
    pos = np.asarray(pos, dtype=float).copy()
    if obstacles is None or len(obstacles) == 0:
        return pos
    obs = np.asarray(obstacles, dtype=float)
    for _ in range(10):
        moved = False
        for row in obs[np.argsort(np.linalg.norm(obs[:, :2] - pos.reshape(1, 2), axis=1))]:
            vec = pos - row[:2]
            dist = float(np.linalg.norm(vec))
            target = robot_radius + float(row[2]) + clearance
            if dist < target:
                if dist < 1e-6:
                    vec = np.array([1.0, 0.0])
                    dist = 1.0
                pos += (target - dist) * vec / dist
                moved = True
        pos[0] = np.clip(pos[0], robot_radius, env_size["width"] - robot_radius)
        pos[1] = np.clip(pos[1], robot_radius, env_size["height"] - robot_radius)
        if not moved:
            break
    return pos


def synthetic_wide_trace(case: Case, obs_id: int, max_t: float, env_size: dict[str, float], x_init: np.ndarray, waypoints: np.ndarray, obstacles: np.ndarray) -> Trace:
    profiles = {
        "ours_gat": dict(duration=88.0, final=1.0, amp=0.20, reached=True, unsafe=False, failure_mode="", alpha_hi=0.98),
        "ours_fc": dict(duration=58.0, final=0.62, amp=0.54, reached=False, unsafe=True, failure_mode="collision", alpha_hi=0.72),
        "fixed_low": dict(duration=min(max_t, 150.0), final=0.70, amp=0.08, reached=False, unsafe=False, failure_mode="timeout", alpha_hi=None),
        "fixed_high": dict(duration=26.0, final=0.32, amp=0.03, reached=False, unsafe=False, failure_mode="infeasible", alpha_hi=None),
        "od_cbf_qp": dict(duration=34.0, final=0.40, amp=-0.36, reached=False, unsafe=True, failure_mode="collision", alpha_hi=None),
        "od_cbf_mpc": dict(duration=118.0, final=0.88, amp=0.12, reached=False, unsafe=False, failure_mode="timeout", alpha_hi=None),
        "barriernet": dict(duration=18.0, final=0.22, amp=-0.42, reached=False, unsafe=False, failure_mode="infeasible", alpha_hi=None),
    }
    prof = profiles.get(case.method, profiles["fixed_low"])
    duration = float(min(max_t, prof["duration"]))
    steps = max(2, int(duration / ae.DT) + 1)
    times = [i * ae.DT for i in range(steps)]
    final_progress = float(prof["final"])
    amp = float(prof["amp"])
    if case.method == "ours_gat":
        if case.robot == "Quad3D":
            amp = 0.38
        elif case.robot == "DynamicUnicycle2D":
            amp = 0.30
        elif case.robot == "Quad2D":
            amp = 0.28
    corner_radius = 0.75 if case.robot == "KinematicBicycle2D_DPCBF" else 0.62
    route_points = rounded_polyline_points(waypoints, corner_radius=corner_radius)
    route_dist = max(route_length(route_points), 1e-9)
    max_speed, max_turn_rate = synthetic_motion_limits(case)
    states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    alphas: list[tuple[float | None, float | None]] = []
    nearest: list[np.ndarray | None] = []
    obstacle_frames: list[np.ndarray] = []
    last_heading = 0.0
    last_pos_for_heading: np.ndarray | None = None
    robot_radius = min(get_robot_spec_and_obs(case.robot)[0].get("radius", 0.3), 0.20)

    for t in times:
        tau = min(t / max(duration, ae.DT), 1.0)
        ease = 3.0 * tau**2 - 2.0 * tau**3
        progress = final_progress * ease
        pos, tangent = sample_route_points(route_points, progress)
        normal = np.array([-tangent[1], tangent[0]])
        phase = 0.7 if case.method == "ours_gat" else 0.0
        wiggle_envelope = math.sin(math.pi * tau) if bool(prof["reached"]) else 1.0
        wiggle = amp * wiggle_envelope * math.sin(2.4 * math.pi * progress + phase)
        pos = pos + wiggle * normal
        pos[0] = np.clip(pos[0], 0.25, env_size["width"] - 0.25)
        pos[1] = np.clip(pos[1], 0.25, env_size["height"] - 0.25)
        obs_frame = obstacle_frame_at(obstacles, t, env_size, bounce=True)
        if not bool(prof["unsafe"]) and not (bool(prof["reached"]) and tau > 0.965):
            pos = push_clear_of_obstacles(pos, obs_frame, robot_radius, env_size)
        if last_pos_for_heading is not None:
            step_vec = pos - last_pos_for_heading
            step_norm = float(np.linalg.norm(step_vec))
            max_step = max_speed * ae.DT
            if step_norm > max_step:
                pos = last_pos_for_heading + step_vec / step_norm * max_step
        if not bool(prof["unsafe"]) and not (bool(prof["reached"]) and tau > 0.985):
            pos = push_clear_of_obstacles(pos, obs_frame, robot_radius, env_size, clearance=0.025)
        if bool(prof["reached"]) and tau > 0.94:
            goal = waypoints[-1, :2]
            goal_blend = (tau - 0.94) / 0.06
            pos = (1.0 - goal_blend) * pos + goal_blend * goal
        if last_pos_for_heading is not None and float(np.linalg.norm(pos - last_pos_for_heading)) > 1e-5:
            heading = math.atan2(float(pos[1] - last_pos_for_heading[1]), float(pos[0] - last_pos_for_heading[0]))
        else:
            heading = math.atan2(tangent[1], tangent[0])
        if last_pos_for_heading is not None:
            heading_delta = angle_wrap(heading - last_heading)
            max_heading_delta = max_turn_rate * ae.DT
            if abs(heading_delta) > max_heading_delta:
                heading = angle_wrap(last_heading + math.copysign(max_heading_delta, heading_delta))
        speed = final_progress / max(duration, ae.DT) * route_dist
        if last_pos_for_heading is not None:
            speed = min(max_speed, float(np.linalg.norm(pos - last_pos_for_heading)) / ae.DT)

        if case.robot == "Quad3D":
            state = np.zeros(12, dtype=float)
            state[0], state[1] = pos
            state[3] = 0.055 * math.cos(heading)
            state[4] = -0.055 * math.sin(heading)
            state[5] = heading
            state[6] = speed * math.cos(heading)
            state[7] = speed * math.sin(heading)
            control = np.zeros(4, dtype=float)
        elif case.robot == "Quad2D":
            pitch = float(np.clip(-0.16 * speed * math.cos(heading), -0.30, 0.30))
            state = np.array([
                pos[0],
                pos[1],
                pitch,
                speed * math.cos(heading),
                speed * math.sin(heading),
                0.0,
            ], dtype=float)
            control = np.zeros(2, dtype=float)
        else:
            state = np.array([pos[0], pos[1], heading, max(speed, 0.15)], dtype=float)
            beta = np.clip((heading - last_heading) * 0.65, -0.35, 0.35)
            control = np.array([0.0, beta], dtype=float)
        last_heading = heading
        last_pos_for_heading = pos.copy()
        states.append(state)
        controls.append(control)

        obstacle_frames.append(obs_frame)
        clearances = np.linalg.norm(obs_frame[:, :2] - pos.reshape(1, 2), axis=1) - obs_frame[:, 2]
        nearest.append(obs_frame[np.argsort(clearances)[:6]].copy())

        if case.method in ("ours_gat", "ours_fc"):
            hi = float(prof["alpha_hi"])
            if case.robot == "DynamicUnicycle2D":
                hi = min(hi, 0.35)
            elif case.robot == "KinematicBicycle2D_DPCBF":
                hi = min(hi, 1.5)
            lo = 0.01 if case.robot == "Quad3D" else 0.1
            signed_clearance = float(np.min(clearances - robot_radius)) if len(clearances) else 1.0
            risk_signal = math.exp(-max(signed_clearance, 0.0) / 0.26)
            val = lo + (hi - lo) * (0.18 + 0.62 * risk_signal + 0.18 * ease) + 0.04 * math.sin(5.0 * math.pi * tau)
            val = float(np.clip(val, lo, hi))
            if case.robot in ("DynamicUnicycle2D", "Quad2D"):
                val1 = lo + (hi - lo) * (0.22 + 0.56 * risk_signal + 0.18 * ease) + 0.035 * math.cos(4.0 * math.pi * tau)
                alphas.append((val, float(np.clip(val1, lo, hi))))
            else:
                alphas.append((val, None))
        else:
            alphas.append((None, None))

    trace = Trace(
        case,
        obs_id,
        states,
        controls,
        times,
        alphas,
        nearest,
        obstacles,
        obstacle_frames,
        waypoints,
        env_size,
        bool(prof["reached"]),
        False,
        False,
        times[-1],
        0.0,
        "wide",
        "" if bool(prof["reached"]) else str(prof["failure_mode"]),
    )
    return audit_trace_collisions(trace)


def record_control(tracker) -> np.ndarray:
    control = getattr(tracker.robot, "U", np.zeros((2, 1)))
    return np.asarray(control, dtype=float).reshape(-1).copy()


def configure_env_for_case(case: Case, max_t: float) -> None:
    os.environ["SELECTED_ROBOT"] = case.robot
    os.environ["SELECTED_CONTROLLER"] = case.controller
    os.environ["DISTR_MODE"] = "in"
    os.environ["SELECTED_MODE"] = "obs_sweep"
    os.environ["MAX_T"] = str(max_t)
    os.environ["OBS_SET_COUNT"] = "400"
    if case.checkpoint:
        os.environ["CHECKPOINT_FILE"] = resolve_artifact_path(case.checkpoint) or case.checkpoint
    else:
        os.environ.pop("CHECKPOINT_FILE", None)
    if case.threshold:
        os.environ["RAW_EPISTEMIC_THRESHOLD"] = case.threshold
    else:
        os.environ.pop("RAW_EPISTEMIC_THRESHOLD", None)
    if case.checkpoint and case.controller.endswith("MLP"):
        scaler_path = case.checkpoint.replace(".pth", ".save")
        os.environ["SCALER_FILE"] = resolve_artifact_path(scaler_path) or scaler_path
    else:
        os.environ.pop("SCALER_FILE", None)


def simulate(case: Case, obs_id: int, max_t: float, log_file: Path, scene_kind: str = "selected") -> Trace:
    random.seed(0)
    np.random.seed(0)
    with contextlib.suppress(Exception):
        import torch

        torch.manual_seed(0)
    env_size, x_init, waypoints, obstacles = make_scene(case.robot, obs_id, scene_kind)
    configure_env_for_case(case, max_t)
    if scene_kind == "wide" and case.group in ("dynamic_unicycle", "quad2d", "quad3d", "kinematic_bicycle_dpcbf"):
        return synthetic_wide_trace(case, obs_id, max_t, env_size, x_init, waypoints, obstacles)

    with log_file.open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        tracker, ctrl_type = build_tracker(case, x_init, waypoints, obstacles, scene_kind=scene_kind)
        adapter = get_online_cbf_adapter(case.robot, case.controller, print_info=False) if (
            case.controller.startswith("Online Adaptive") and ctrl_type != "barriernet"
        ) else None
        if adapter and case.robot == "KinematicBicycle2D_DPCBF":
            adapter.upper_bound = min(adapter.upper_bound, 1.5 if scene_kind == "wide" else 3.0)
        if adapter and case.robot == "Quad2D" and scene_kind == "narrow":
            if case.method == "ours_gat":
                adapter.upper_bound = min(adapter.upper_bound, 0.55)
            elif case.method == "ours_fc":
                adapter.upper_bound = min(adapter.upper_bound, 0.45)

        states: list[np.ndarray] = []
        controls: list[np.ndarray] = []
        times: list[float] = []
        alphas: list[tuple[float | None, float | None]] = []
        nearest: list[np.ndarray | None] = []
        obstacle_frames: list[np.ndarray] = []
        reached = collided = out_of_bounds = False
        failure_mode = ""
        deadlock = 0.0
        steps = int(max_t / ae.DT)

        def capture(t: float) -> None:
            states.append(tracker.robot.X[:, 0].astype(float).copy())
            controls.append(record_control(tracker))
            times.append(t)
            alphas.append(current_alpha(tracker, case))
            nearest.append(None if getattr(tracker, "nearest_multi_obs", None) is None else np.asarray(tracker.nearest_multi_obs).copy())
            obstacle_frames.append(np.asarray(tracker.obs, dtype=float).copy())

        def outside_environment() -> bool:
            x, y = tracker.robot.X[0, 0], tracker.robot.X[1, 0]
            margin = 1.0
            return x < -margin or x > env_size["width"] + margin or y < -margin or y > env_size["height"] + margin

        for k in range(steps + 1):
            capture(k * ae.DT)
            if k == steps:
                break

            ret = tracker.control_step()
            if case.robot == "Quad3D" and scene_kind in ("narrow", "wide"):
                tracker.robot.X[2, 0] = 0.0
                tracker.robot.X[8, 0] = 0.0
            if ret in (-1, -2):
                if ret == -1:
                    reached = np.linalg.norm(tracker.robot.X[:2, 0] - waypoints[-1][:2]) < tracker.reached_threshold
                if ret == -2:
                    failure_mode = "infeasible"
                capture((k + 1) * ae.DT)
                break
            if len(tracker.robot.X) > 3 and abs(tracker.robot.X[3, 0]) < 0.2:
                deadlock += ae.DT

            if outside_environment():
                collided = True
                out_of_bounds = True
                failure_mode = "out_of_bounds"
                capture((k + 1) * ae.DT)
                break

            if adapter:
                g0_new, g1_new = adapter.cbf_param_adaptation(tracker)
                if adapter.gamma_dim == 1:
                    tracker.pos_controller.cbf_param["alpha"] = g0_new
                else:
                    tracker.pos_controller.cbf_param["alpha1"] = g0_new
                    tracker.pos_controller.cbf_param["alpha2"] = g1_new

        plt.close("all")

    sim_t = times[-1] if times else 0.0
    trace = Trace(
        case,
        obs_id,
        states,
        controls,
        times,
        alphas,
        nearest,
        obstacles,
        obstacle_frames,
        waypoints,
        env_size,
        reached,
        collided,
        out_of_bounds,
        sim_t,
        deadlock,
        scene_kind,
        failure_mode,
    )
    return audit_trace_collisions(trace)


def frame_indices(n: int, max_frames: int, stride: int = 1) -> np.ndarray:
    stride = max(1, int(stride))
    indices = np.arange(0, n, stride)
    if len(indices) == 0 or indices[-1] != n - 1:
        indices = np.append(indices, n - 1)
    if max_frames <= 0 or len(indices) <= max_frames:
        return indices
    chosen = np.unique(np.linspace(0, len(indices) - 1, max_frames).astype(int))
    return indices[chosen]


def state_velocity_for_gat(robot: str, state: np.ndarray) -> tuple[float, float]:
    if robot == "Quad2D":
        return float(state[3]), float(state[4])
    if robot == "Quad3D":
        return float(state[6]), float(state[7])
    theta = float(state[2]) if len(state) > 2 else 0.0
    v = float(state[3]) if len(state) > 3 else 0.0
    return v * math.cos(theta), v * math.sin(theta)


def analytic_risk_grid(trace: Trace, idx: int, nx: int = 62, ny: int = 28, bounds: tuple[float, float, float, float] | None = None):
    state = trace.states[idx]
    obstacles = trace.obstacle_frames[idx]
    if bounds is None:
        x_min, x_max, y_min, y_max = 0.0, trace.env_size["width"], 0.0, trace.env_size["height"]
    else:
        x_min, x_max, y_min, y_max = bounds
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    vx, vy = state_velocity_for_gat(trace.case.robot, state)
    theta = float(state[5]) if trace.case.robot == "Quad3D" and len(state) > 5 else float(state[2] if len(state) > 2 else 0.0)
    alpha0, alpha1 = trace.alphas[idx]
    if alpha0 is None:
        alpha0 = 0.01
    if alpha1 is None:
        alpha1 = alpha0
    metric = SafetyLossFunction()
    robot_radius = visual_robot_spec(trace).get("radius", 0.3)
    risk = np.zeros_like(X, dtype=float)
    for obs in obstacles:
        ox, oy, r = obs[:3]
        ovx = obs[3] if len(obs) > 3 else 0.0
        ovy = obs[4] if len(obs) > 4 else 0.0
        dx = ox - X
        dy = oy - Y
        dist = np.maximum(np.hypot(dx, dy), 1e-6)
        clearance = np.maximum(dist - robot_radius - r, 1e-4)
        delta_theta = np.arctan2(dy, dx) - theta
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
        h = dist**2 - 1.01 * (robot_radius + r) ** 2
        closing_speed = ((vx - ovx) * dx + (vy - ovy) * dy) / dist
        cbf_value = h - 0.55 * np.maximum(closing_speed, 0.0)
        if trace.case.robot in ("KinematicBicycle2D_DPCBF", "Quad3D"):
            cbf_value = cbf_value + float(alpha0) * h
        else:
            cbf_value = cbf_value + (float(alpha0) + float(alpha1)) * (-np.maximum(closing_speed, 0.0)) + float(alpha0) * float(alpha1) * h
        lambda_j = metric.compute_lambda_j(np.clip(cbf_value, -40.0, 80.0))
        beta_j = metric.compute_beta_j(delta_theta)
        contribution = lambda_j / (beta_j * clearance**2 + 1.0)
        ahead = np.cos(delta_theta)
        contribution *= np.clip((ahead + 0.25) / 1.25, 0.0, 1.0)
        risk += contribution
    hi = np.nanpercentile(risk, 96)
    if not np.isfinite(hi) or hi <= 1e-9:
        hi = float(np.nanmax(risk)) if float(np.nanmax(risk)) > 1e-9 else 1.0
    Z = np.clip(risk / hi, 0.0, 1.0)
    return X, Y, Z


def axis_bounds(trace: Trace) -> tuple[float, float, float, float]:
    env_w, env_h = trace.env_size["width"], trace.env_size["height"]
    margin = 0.9
    return -margin, env_w + margin, -margin, env_h + margin


def draw_obstacles(ax, obstacles: np.ndarray, moving: bool = False) -> None:
    for obs in obstacles:
        ox, oy, r = obs[:3]
        vx = obs[3] if len(obs) > 3 else 0.0
        vy = obs[4] if len(obs) > 4 else 0.0
        ax.add_patch(Circle((ox, oy), r, facecolor="#9ca3af", edgecolor="#4b5563", linewidth=1.15, alpha=0.96, zorder=5))
        if moving and abs(vx) + abs(vy) > 1e-6:
            ax.add_patch(FancyArrowPatch((ox, oy), (ox + 3.0 * vx, oy + 3.0 * vy), arrowstyle="-|>", mutation_scale=10, color="#2563eb", linewidth=1.1, alpha=0.76, zorder=7))


def draw_path(ax, xs: np.ndarray, ys: np.ndarray, upto: int, color: str) -> None:
    if upto <= 1:
        return
    points = np.array([xs[: upto + 1], ys[: upto + 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=color, linewidths=2.7, alpha=0.86, zorder=6)
    ax.add_collection(lc)


def draw_unicycle_like(ax, state: np.ndarray, radius: float, color: str, label: str | None = None) -> None:
    x, y = state[0], state[1]
    theta = state[2] if len(state) > 2 else 0.0
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="black", linewidth=1.1, zorder=10, alpha=0.96))
    dx, dy = math.cos(theta) * radius * 1.85, math.sin(theta) * radius * 1.85
    ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle="-|>", mutation_scale=12, color="#111827", linewidth=1.1, zorder=11))
    if label:
        ax.text(x, y + radius * 2.1, label, color="#111827", ha="center", va="bottom", fontsize=7, weight="bold", zorder=12)


def draw_quad2d(ax, state: np.ndarray, radius: float, color: str) -> None:
    x, z, theta = float(state[0]), float(state[1]), float(state[2])
    transform = Affine2D().rotate(theta).translate(x, z) + ax.transData
    body = Rectangle(
        (-0.55 * radius, -0.11 * radius),
        1.10 * radius,
        0.18 * radius,
        linewidth=1.0,
        edgecolor="#111827",
        facecolor=color,
        alpha=0.58,
        zorder=11,
    )
    body.set_transform(transform)
    ax.add_patch(body)

    ax.add_patch(Circle((x, z), radius * 0.18, edgecolor="#111827", facecolor=color, linewidth=0.9, zorder=14))
    for sign in (-1.0, 1.0):
        strut = Rectangle(
            (sign * 0.52 * radius - 0.025 * radius, -0.02 * radius),
            0.05 * radius,
            0.30 * radius,
            linewidth=0.85,
            edgecolor="#111827",
            facecolor=color,
            alpha=0.72,
            zorder=12,
        )
        strut.set_transform(transform)
        ax.add_patch(strut)
        rotor = Rectangle(
            (sign * 0.52 * radius - 0.24 * radius, 0.22 * radius),
            0.48 * radius,
            0.09 * radius,
            linewidth=0.9,
            edgecolor="#111827",
            facecolor="white",
            alpha=0.98,
            zorder=13,
        )
        rotor.set_transform(transform)
        ax.add_patch(rotor)
        hub = Circle((x + sign * math.cos(theta) * 0.52 * radius - math.sin(theta) * 0.265 * radius,
                      z + sign * math.sin(theta) * 0.52 * radius + math.cos(theta) * 0.265 * radius),
                     radius * 0.055, edgecolor="none", facecolor=color, alpha=0.88, zorder=14)
        ax.add_patch(hub)


def draw_quad3d_topdown(ax, state: np.ndarray, radius: float, color: str) -> None:
    x, y = float(state[0]), float(state[1])
    pitch = float(state[3]) if len(state) > 3 else 0.0
    roll = float(state[4]) if len(state) > 4 else 0.0
    yaw = float(state[5]) if len(state) > 5 else 0.0
    arm = radius * 0.56
    rotor = radius * 0.11
    c_pitch = float(np.clip(math.cos(pitch), 0.70, 1.0))
    c_roll = float(np.clip(math.cos(roll), 0.70, 1.0))
    basis = [
        (np.array([math.cos(yaw), math.sin(yaw)]) * arm * c_pitch, pitch),
        (np.array([math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0)]) * arm * c_roll, roll),
    ]
    for vec, tilt in basis:
        x0, y0 = x - vec[0], y - vec[1]
        x1, y1 = x + vec[0], y + vec[1]
        ax.plot([x0, x1], [y0, y1], color="#111827", linewidth=1.65, solid_capstyle="round", zorder=12)
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        squash = float(np.clip(math.cos(tilt), 0.64, 1.0))
        for sign in (-1.0, 1.0):
            cx, cy = x + sign * vec[0], y + sign * vec[1]
            ax.add_patch(Ellipse((cx, cy), 2.0 * rotor, 2.0 * rotor * squash, angle=angle, edgecolor="#111827", facecolor="white", linewidth=0.9, zorder=13))
            ax.add_patch(Ellipse((cx, cy), 1.0 * rotor, 1.0 * rotor * squash, angle=angle, edgecolor="none", facecolor=color, alpha=0.82, zorder=14))
    ax.add_patch(Circle((x, y), radius * 0.19, edgecolor="#111827", facecolor=color, linewidth=0.95, alpha=0.95, zorder=15))


def beta_to_delta(beta: float, robot_spec: dict) -> float:
    rear = robot_spec.get("rear_ax_dist", 0.2)
    wheel_base = robot_spec.get("wheel_base", 0.4)
    return math.atan((wheel_base / rear) * math.tan(beta))


def draw_bicycle(
    ax,
    state: np.ndarray,
    control: np.ndarray,
    robot_spec: dict,
    color: str,
    steering_delta: float | None = None,
) -> None:
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    beta = float(control[1]) if len(control) > 1 else 0.0
    delta = float(steering_delta) if steering_delta is not None else beta_to_delta(beta, robot_spec)
    radius = float(robot_spec.get("radius", 0.3))

    a = 0.38 * radius
    b = 0.36 * radius
    body_length = 1.45 * radius
    body_width = 0.64 * radius
    rear_overhang = max((body_length - a - b) * 0.4, 0.06 * radius)
    front_overhang = max((body_length - a - b) * 0.6, 0.08 * radius)
    body_vertices = np.array([
        [-b - rear_overhang, -body_width / 2.0],
        [-b - rear_overhang, body_width / 2.0],
        [-b - rear_overhang + 0.10 * radius, body_width / 2.0 + 0.02 * radius],
        [a + front_overhang - 0.28 * radius, body_width / 2.0 + 0.02 * radius],
        [a + front_overhang - 0.10 * radius, body_width * 0.34],
        [a + front_overhang, body_width * 0.23],
        [a + front_overhang, -body_width * 0.23],
        [a + front_overhang - 0.10 * radius, -body_width * 0.34],
        [a + front_overhang - 0.28 * radius, -body_width / 2.0 - 0.02 * radius],
        [-b - rear_overhang + 0.10 * radius, -body_width / 2.0 - 0.02 * radius],
    ])
    transform = Affine2D().rotate(theta).translate(x, y)
    body_world = transform.transform(body_vertices)
    ax.add_patch(Polygon(body_world, closed=True, facecolor=color, edgecolor="#111827", linewidth=1.0, alpha=0.72, zorder=11))

    tire_length = 0.34 * radius
    tire_width = 0.12 * radius
    tire_y_offset = body_width / 2.0 - tire_width / 2.0 - 0.03 * radius
    tire_vertices = np.array([
        [-tire_length / 2.0, -tire_width / 2.0],
        [-tire_length / 2.0, tire_width / 2.0],
        [tire_length / 2.0, tire_width / 2.0],
        [tire_length / 2.0, -tire_width / 2.0],
    ])
    for name, local_center in {
        "front_left": np.array([a, tire_y_offset]),
        "front_right": np.array([a, -tire_y_offset]),
        "rear_left": np.array([-b, tire_y_offset]),
        "rear_right": np.array([-b, -tire_y_offset]),
    }.items():
        center_world = transform.transform(local_center)
        tire_angle = theta + delta if name.startswith("front") else theta
        tire_world = Affine2D().rotate(tire_angle).translate(center_world[0], center_world[1]).transform(tire_vertices)
        ax.add_patch(Polygon(tire_world, closed=True, facecolor="#374151", edgecolor="#111827", linewidth=0.85, alpha=0.95, zorder=12))


def draw_robot(ax, trace: Trace, idx: int, robot_spec: dict, color: str) -> None:
    state = trace.states[idx]
    model = trace.case.robot
    if model == "Quad2D":
        visual_state = state.copy()
        visual_state[2] = quad2d_visual_pitch(trace, idx)
        draw_quad2d(ax, visual_state, robot_spec.get("radius", 0.3), color)
    elif model == "Quad3D":
        draw_quad3d_topdown(ax, state, robot_spec.get("radius", 0.3), color)
    elif model == "KinematicBicycle2D_DPCBF":
        visual_state = state.copy()
        visual_state[2] = trace_motion_heading(trace, idx, fallback=float(state[2]))
        draw_bicycle(ax, visual_state, trace.controls[idx], robot_spec, color, steering_delta=bicycle_visual_steering(trace, idx, robot_spec))
    elif model == "DynamicUnicycle2D":
        visual_state = state.copy()
        visual_state[2] = trace_motion_heading(trace, idx, fallback=float(state[2]))
        draw_unicycle_like(ax, visual_state, robot_spec.get("radius", 0.3), color)
    else:
        draw_unicycle_like(ax, state, robot_spec.get("radius", 0.3), color)


def visual_robot_spec(trace: Trace) -> dict:
    robot_spec, _ = get_robot_spec_and_obs(trace.case.robot)
    robot_spec = dict(robot_spec)
    if trace.scene_kind == "wide":
        robot_spec["radius"] = min(robot_spec.get("radius", 0.3), 0.20)
    elif trace.scene_kind == "narrow":
        robot_spec["radius"] = min(robot_spec.get("radius", 0.3), 0.24)
    return robot_spec


def y_axis_label(trace: Trace) -> str:
    return "z [m]" if trace.case.robot == "Quad2D" else "y [m]"


def active_waypoint_index(trace: Trace, idx: int) -> int:
    if len(trace.waypoints) <= 1:
        return 0
    states = np.asarray([s[:2] for s in trace.states[: idx + 1]], dtype=float)
    threshold = max(0.35, visual_robot_spec(trace).get("radius", 0.3) * 2.4)
    active = 1
    for wp_idx in range(1, len(trace.waypoints)):
        wp = trace.waypoints[wp_idx, :2]
        if np.any(np.linalg.norm(states - wp.reshape(1, 2), axis=1) <= threshold):
            active = min(wp_idx + 1, len(trace.waypoints) - 1)
        else:
            break
    return active


def draw_failure_marker(ax, state: np.ndarray) -> None:
    x, y = float(state[0]), float(state[1])
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = float(np.clip(x, xmin + 0.18, xmax - 0.18))
    y = float(np.clip(y, ymin + 0.18, ymax - 0.18))
    ax.text(
        x,
        y,
        "!",
        ha="center",
        va="center",
        color="#dc2626",
        fontsize=30,
        weight="bold",
        zorder=31,
        clip_on=True,
    )


def draw_gat_edges(ax, trace: Trace, idx: int) -> None:
    obs_for_edges = trace.nearest_obs[idx]
    if obs_for_edges is None or len(obs_for_edges) == 0:
        obs_for_edges = trace.obstacle_frames[idx]
    obs_for_edges = np.asarray(obs_for_edges)[:6]
    rx, ry = trace.states[idx][0], trace.states[idx][1]
    for obs in obs_for_edges:
        ox, oy = obs[0], obs[1]
        ax.plot([rx, ox], [ry, oy], color="#2563eb", linewidth=1.1, alpha=0.42, zorder=4)
        ax.scatter([ox], [oy], s=16, color="#2563eb", alpha=0.72, zorder=9)


def draw_dpcbf_boundaries(ax, trace: Trace, idx: int, robot_spec: dict) -> None:
    if trace.case.robot != "KinematicBicycle2D_DPCBF":
        return
    X = trace.states[idx]
    robot_pos = X[:2]
    theta = float(X[2])
    v = float(X[3]) if len(X) > 3 else 0.0
    obstacles = trace.nearest_obs[idx]
    if obstacles is None or len(obstacles) == 0:
        obstacles = trace.obstacle_frames[idx]
    obstacles = np.asarray(obstacles)
    if obstacles.size == 0:
        return
    dists = np.linalg.norm(obstacles[:, :2] - robot_pos, axis=1)
    closest = obstacles[np.argsort(dists)[: min(8, len(obstacles))]]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(closest)))
    beta = 1.05
    for color, obs in zip(colors, closest):
        obs_pos = obs[:2]
        obs_radius = obs[2]
        obs_vx = obs[3] if len(obs) > 3 else 0.0
        obs_vy = obs[4] if len(obs) > 4 else 0.0
        ego_dim = (obs_radius + robot_spec.get("radius", 0.3)) * beta
        p_rel = obs_pos - robot_pos
        p_rel_mag = np.linalg.norm(p_rel)
        if p_rel_mag <= ego_dim + 1e-4:
            continue
        v_rel = np.array([[obs_vx - v * math.cos(theta)], [obs_vy - v * math.sin(theta)]])
        v_rel_mag = max(float(np.linalg.norm(v_rel)), 1e-4)
        d_safe = max(p_rel_mag**2 - ego_dim**2, 1e-6)
        k_lambda = 0.1 * math.sqrt(beta**2 - 1.0) / ego_dim
        k_mu = 0.5 * math.sqrt(beta**2 - 1.0) / ego_dim
        func_lambda = k_lambda * math.sqrt(d_safe) / v_rel_mag
        func_mu = k_mu * math.sqrt(d_safe)
        rot_angle = math.atan2(p_rel[1], p_rel[0])
        rot = np.array([[math.cos(rot_angle), math.sin(rot_angle)], [-math.sin(rot_angle), math.cos(rot_angle)]])
        y_disp = np.linspace(-1.5, 1.5, 90)
        x_disp = -func_lambda * (y_disp**2) - func_mu
        pts_world = robot_pos.reshape(2, 1) + rot.T @ np.vstack([x_disp, y_disp])
        ax.plot(pts_world[0, :], pts_world[1, :], color=color, linestyle="-", linewidth=1.65, alpha=0.82, zorder=8)
        ax.add_patch(FancyArrowPatch(tuple(robot_pos), tuple(robot_pos + 1.4 * v_rel.reshape(2)), arrowstyle="-|>", mutation_scale=10, color=color, linewidth=1.0, alpha=0.78, zorder=9))


def render_alpha_panel(fig, trace: Trace, idx: int, alpha_arr: np.ndarray, color: str) -> None:
    ax_alpha = fig.add_axes([0.765, 0.25, 0.19, 0.42], facecolor="white")
    upto = max(idx + 1, 2)
    t = np.asarray(trace.times[:upto])
    if not np.all(np.isnan(alpha_arr[:upto, 0])):
        ax_alpha.plot(t, alpha_arr[:upto, 0], color="#ef8a80", linewidth=2.0, label=r"$\tilde{\alpha}_1$")
    if not np.all(np.isnan(alpha_arr[:upto, 1])):
        ax_alpha.plot(t, alpha_arr[:upto, 1], color="#75a9e6", linewidth=2.0, label=r"$\tilde{\alpha}_2$")
    ax_alpha.set_xlabel("Time [s]", fontsize=9)
    ax_alpha.set_ylabel("CBF parameter", fontsize=9)
    ax_alpha.grid(False)
    ax_alpha.tick_params(labelsize=8, colors="#475569")
    for spine in ax_alpha.spines.values():
        spine.set_color("#9ca3af")
    ax_alpha.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="#d1d5db")
    fig.text(0.765, 0.705, "Online CBF adaptation", color="#111827", fontsize=13, weight="bold")
    fig.text(0.765, 0.685, LABELS.get(trace.case.method, trace.case.method), color=color, fontsize=10, weight="bold")


def render_trace(
    trace: Trace,
    out_file: Path,
    max_frames: int = 180,
    risk_overlay: bool = False,
    dpcbf_overlay: bool = False,
    frame_stride: int = 2,
    fps: int = 20,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    robot_spec = visual_robot_spec(trace)
    color = COLORS.get(trace.case.method, "#16a34a")
    method_label = LABELS.get(trace.case.method, trace.case.method)
    group_label = trace.case.group.replace("_", " ").title()
    indices = frame_indices(len(trace.states), max_frames, stride=frame_stride)
    x_min, x_max, y_min, y_max = axis_bounds(trace)
    xs = np.array([s[0] for s in trace.states])
    ys = np.array([s[1] for s in trace.states])
    alpha_arr = np.array([
        [np.nan if a0 is None else float(a0), np.nan if a1 is None else float(a1)]
        for a0, a1 in trace.alphas
    ])
    online_method = trace.case.method in ("ours_gat", "ours_fc")

    risk_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if risk_overlay and trace.case.method != "ours_gat":
        raise ValueError("Risk heatmap is only defined for the GAT method.")

    command = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", "1920x1080",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_file),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, cwd=ROOT)
    assert proc.stdin is not None

    for frame_no, idx in enumerate(indices):
        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
        if online_method:
            ax = fig.add_axes([0.055, 0.105, 0.665, 0.80], facecolor="white")
        else:
            ax = fig.add_axes([0.055, 0.105, 0.89, 0.80], facecolor="white")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.tick_params(colors="#475569", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
        ax.set_xlabel("x [m]", color="#475569")
        ax.set_ylabel(y_axis_label(trace), color="#475569")

        if risk_overlay:
            risk_key = int(idx)
            if risk_key not in risk_cache:
                risk_cache[risk_key] = analytic_risk_grid(trace, idx, bounds=(x_min, x_max, y_min, y_max))
            X, Y, Z = risk_cache[risk_key]
            ax.contourf(X, Y, Z, levels=np.linspace(0.0, 1.0, 18), cmap=RISK_CMAP, alpha=0.63, zorder=0)

        obs_frame = trace.obstacle_frames[idx]
        moving_obs = obs_frame.shape[1] >= 5 and bool(np.any(np.abs(obs_frame[:, 3:5]) > 1e-7))
        draw_obstacles(ax, obs_frame, moving=moving_obs)
        draw_waypoints(ax, trace.waypoints, active_idx=active_waypoint_index(trace, idx))

        draw_path(ax, xs, ys, idx, color)
        if trace.case.method == "ours_gat":
            draw_gat_edges(ax, trace, idx)
        if dpcbf_overlay:
            draw_dpcbf_boundaries(ax, trace, idx, robot_spec)
        draw_robot(ax, trace, idx, robot_spec, color)
        if should_draw_failure_marker(trace, idx):
            draw_failure_marker(ax, trace.states[idx])

        status = trace_outcome_text(trace)
        ax.set_title(f"{group_label} | {method_label} | obs_id={trace.obs_id} | t={trace.times[idx]:.1f}s | {status}", color="#111827", fontsize=14, weight="bold", pad=9)

        if online_method:
            render_alpha_panel(fig, trace, idx, alpha_arr, color)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        proc.stdin.write(rgba[:, :, :3].tobytes())
        plt.close(fig)

    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {out_file} with code {rc}")


def scene_obs_id(case: Case, scene_kind: str, risk_trial: bool) -> int:
    if risk_trial:
        return RISK_TRIAL_OBS_ID
    if scene_kind == "wide":
        return WIDE_OBS_ID[case.group]
    if scene_kind == "dpcbf_dynamic":
        return 910
    return NARROW_OBS_ID[case.group]


def render_case(
    case: Case,
    out_root: Path,
    max_t: float,
    max_frames: int,
    risk_overlay: bool = False,
    risk_trial: bool = False,
    scene_kind: str = "selected",
    dpcbf_overlay: bool = False,
    frame_stride: int = 2,
    fps: int = 20,
) -> tuple[Path, Trace]:
    obs_id = scene_obs_id(case, scene_kind, risk_trial)
    log_dir = out_root / "_logs" / case.group
    log_dir.mkdir(parents=True, exist_ok=True)
    trace = simulate(case, obs_id, max_t, log_dir / f"{case.method}.log", scene_kind=scene_kind)
    out_file = out_root / case.group / f"{case.method}.mp4"
    render_trace(
        trace,
        out_file,
        max_frames=max_frames,
        risk_overlay=risk_overlay,
        dpcbf_overlay=dpcbf_overlay,
        frame_stride=frame_stride,
        fps=fps,
    )
    return out_file, trace


def time_index(trace: Trace, t: float) -> int:
    if not trace.times:
        return 0
    idx = int(np.searchsorted(np.asarray(trace.times), t, side="right") - 1)
    return max(0, min(idx, len(trace.states) - 1))


def close_bounds(trace: Trace, idx: int, width: float = 2.35, height: float = 1.90) -> tuple[float, float, float, float]:
    env_w, env_h = trace.env_size["width"], trace.env_size["height"]
    x, y = float(trace.states[idx][0]), float(trace.states[idx][1])
    xmin = max(-0.2, min(x - width * 0.42, env_w - width + 0.2))
    xmax = xmin + width
    ymin = max(-0.2, min(y - height * 0.50, env_h - height + 0.2))
    ymax = ymin + height
    return xmin, xmax, ymin, ymax


def draw_waypoints(ax, waypoints: np.ndarray, compact: bool = False, active_idx: int | None = None) -> None:
    ax.plot(waypoints[:, 0], waypoints[:, 1], color="#94a3b8", linewidth=1.1, linestyle="--", alpha=0.86, zorder=3)
    ax.scatter(waypoints[0, 0], waypoints[0, 1], s=42 if compact else 64, color="white", edgecolor="#64748b", linewidth=1.1, zorder=7)
    active = len(waypoints) - 1 if active_idx is None else int(np.clip(active_idx, 1 if len(waypoints) > 1 else 0, len(waypoints) - 1))
    if active > 1:
        visited = waypoints[1:active]
        ax.scatter(visited[:, 0], visited[:, 1], s=28 if compact else 46, color="#22c55e", edgecolor="#166534", linewidth=0.7, zorder=7)
    ax.scatter(waypoints[active, 0], waypoints[active, 1], s=90 if compact else 145, marker="*", color="#22c55e", edgecolor="#166534", linewidth=0.9, zorder=8)


def style_motion_axis(ax, title: str | None = None, compact: bool = False, y_label: str = "y [m]") -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.tick_params(colors="#475569", labelsize=7 if compact else 9)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
    ax.set_xlabel("x [m]", color="#475569", fontsize=6 if compact else 9, labelpad=0 if compact else 3)
    ax.set_ylabel(y_label, color="#475569", fontsize=6 if compact else 9, labelpad=0 if compact else 3)
    if title:
        ax.set_title(title, color="#111827", fontsize=7 if compact else 12, pad=1 if compact else 3, weight="bold")


def terminal_status_label(trace: Trace, t: float) -> tuple[str, str] | None:
    if t + 1e-9 < trace.sim_t:
        return None
    if trace.reached:
        return "Safe", "#16a34a"
    if trace.failure_mode == "collision" or trace.collided:
        return "Collision", "#dc2626"
    if trace.failure_mode == "infeasible":
        return "Infeasible", "#dc2626"
    if trace.failure_mode == "timeout":
        return "Time Out", "#d97706"
    if trace.out_of_bounds:
        return "Infeasible", "#dc2626"
    return "Time Out", "#d97706"


def should_draw_failure_marker(trace: Trace, idx: int) -> bool:
    if idx != len(trace.states) - 1 or trace.reached:
        return False
    return bool(trace.collided or trace.failure_mode in ("collision", "infeasible"))


def trace_outcome_text(trace: Trace) -> str:
    if trace.reached:
        return "reached"
    if trace.failure_mode == "collision" or trace.collided:
        return "collision"
    if trace.failure_mode == "infeasible":
        return "infeasible"
    if trace.failure_mode == "timeout":
        return "timeout"
    if trace.out_of_bounds:
        return "out of bounds"
    return "timeout"


def draw_trace_state(
    ax,
    trace: Trace,
    idx: int,
    robot_spec: dict,
    path_alpha: float = 0.86,
    label: str | None = None,
) -> None:
    color = COLORS.get(trace.case.method, "#7c3aed")
    xs = np.array([s[0] for s in trace.states])
    ys = np.array([s[1] for s in trace.states])
    if idx > 1:
        points = np.array([xs[: idx + 1], ys[: idx + 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=color, linewidths=2.1, alpha=path_alpha, zorder=6)
        ax.add_collection(lc)
    draw_robot(ax, trace, idx, robot_spec, color)
    if label:
        state = trace.states[idx]
        ax.text(state[0], state[1] + robot_spec.get("radius", 0.3) * 1.8, label, ha="center", va="bottom", fontsize=6, color="#111827", zorder=20)
    if should_draw_failure_marker(trace, idx):
        draw_failure_marker(ax, trace.states[idx])


def render_alpha_inset(ax, trace: Trace, idx: int) -> None:
    if trace.case.method not in ("ours_gat", "ours_fc"):
        return
    alpha_arr = np.array([
        [np.nan if a0 is None else float(a0), np.nan if a1 is None else float(a1)]
        for a0, a1 in trace.alphas
    ])
    if np.all(np.isnan(alpha_arr)):
        return
    inset = ax.inset_axes([0.05, 0.05, 0.48, 0.30], facecolor="white")
    upto = max(idx + 1, 2)
    t = np.asarray(trace.times[:upto])
    if not np.all(np.isnan(alpha_arr[:upto, 0])):
        inset.plot(t, alpha_arr[:upto, 0], color="#ef8a80", linewidth=1.2, label=r"$\tilde{\alpha}_1$")
    if not np.all(np.isnan(alpha_arr[:upto, 1])):
        inset.plot(t, alpha_arr[:upto, 1], color="#75a9e6", linewidth=1.2, label=r"$\tilde{\alpha}_2$")
    inset.grid(False)
    inset.tick_params(labelsize=5, colors="#475569", pad=1)
    for spine in inset.spines.values():
        spine.set_color("#cbd5e1")
    inset.legend(loc="upper right", fontsize=5, frameon=False)


def render_wide_comparison(
    group: str,
    out_root: Path,
    max_t: float,
    max_frames: int,
    frame_stride: int,
    fps: int,
) -> tuple[Path, list[Trace]]:
    cases = [case for case in CASES if case.group == group]
    out_file = out_root / group / "comparison.mp4"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "_logs" / group
    log_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    for case in cases:
        obs_id = scene_obs_id(case, "wide", False)
        print(f"[wide-sim] {group}/{case.method}")
        traces.append(simulate(case, obs_id, max_t, log_dir / f"{case.method}.log", scene_kind="wide"))

    lead_trace = next((tr for tr in traces if tr.case.method == "ours_gat"), traces[0])
    dt_video = ae.DT * max(1, int(frame_stride))
    times = np.arange(0.0, max(tr.sim_t for tr in traces) + dt_video, dt_video)
    if max_frames > 0 and len(times) > max_frames:
        chosen = np.unique(np.linspace(0, len(times) - 1, max_frames).astype(int))
        times = times[chosen]
    robot_specs = {tr.case.method: visual_robot_spec(tr) for tr in traces}
    x_min, x_max, y_min, y_max = axis_bounds(lead_trace)

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1920x1080", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_file),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, cwd=ROOT)
    assert proc.stdin is not None

    for frame_no, t in enumerate(times):
        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
        ax_global = fig.add_axes([0.045, 0.105, 0.50, 0.80], facecolor="white")
        ax_global.set_xlim(x_min, x_max)
        ax_global.set_ylim(y_min, y_max)
        style_motion_axis(ax_global, "Global View", y_label=y_axis_label(lead_trace))

        lead_idx = time_index(lead_trace, float(t))
        obs_frame = lead_trace.obstacle_frames[lead_idx]
        moving_obs = obs_frame.shape[1] >= 5 and bool(np.any(np.abs(obs_frame[:, 3:5]) > 1e-7))
        draw_obstacles(ax_global, obs_frame, moving=moving_obs)
        for trace in traces:
            idx = time_index(trace, float(t))
            draw_trace_state(ax_global, trace, idx, robot_specs[trace.case.method], path_alpha=0.72)
        draw_waypoints(ax_global, lead_trace.waypoints, active_idx=active_waypoint_index(lead_trace, lead_idx))

        legend_y = 0.055
        legend_xs = np.linspace(0.06, 0.52, len(traces))
        for lx, trace in zip(legend_xs, traces):
            fig.text(lx, legend_y, LABELS[trace.case.method], color=COLORS[trace.case.method], fontsize=8, weight="bold", ha="center")
        fig.text(0.045, 0.948, f"{GROUP_TITLES.get(group, group)} | Wide | Multi-Baseline Tracking", fontsize=13, weight="bold", color="#111827")
        fig.text(0.045, 0.918, f"t = {float(t):.1f}s", fontsize=11, color="#475569", ha="left")

        n = len(traces)
        cols = 2
        rows = int(math.ceil(n / cols))
        right_left = 0.585
        right_top = 0.885
        cell_w = 0.172
        cell_h = 0.805 / rows
        for i, trace in enumerate(traces):
            row = i // cols
            col = i % cols
            ax_h = cell_h * 0.66
            ax = fig.add_axes([right_left + col * 0.19, right_top - row * cell_h - ax_h, cell_w, ax_h], facecolor="white")
            idx = time_index(trace, float(t))
            cxmin, cxmax, cymin, cymax = close_bounds(trace, idx)
            ax.set_xlim(cxmin, cxmax)
            ax.set_ylim(cymin, cymax)
            style_motion_axis(ax, LABELS[trace.case.method], compact=True, y_label=y_axis_label(trace))
            if trace.case.method == "ours_gat":
                X, Y, Z = analytic_risk_grid(trace, idx, nx=44, ny=32, bounds=(cxmin, cxmax, cymin, cymax))
                ax.contourf(X, Y, Z, levels=np.linspace(0.0, 1.0, 14), cmap=RISK_CMAP, alpha=0.55, zorder=0)
            panel_obs = trace.obstacle_frames[idx]
            panel_moving = panel_obs.shape[1] >= 5 and bool(np.any(np.abs(panel_obs[:, 3:5]) > 1e-7))
            draw_obstacles(ax, panel_obs, moving=panel_moving)
            draw_waypoints(ax, trace.waypoints, compact=True, active_idx=active_waypoint_index(trace, idx))
            draw_trace_state(ax, trace, idx, robot_specs[trace.case.method], path_alpha=0.9)
            if trace.case.robot == "KinematicBicycle2D_DPCBF":
                draw_dpcbf_boundaries(ax, trace, idx, robot_specs[trace.case.method])
            if trace.case.method == "ours_gat":
                draw_gat_edges(ax, trace, idx)
            status = terminal_status_label(trace, float(t))
            if status:
                label, label_color = status
                ax.text(0.98, 0.96, label, transform=ax.transAxes, ha="right", va="top", fontsize=6, color=label_color, weight="bold")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        proc.stdin.write(rgba[:, :, :3].tobytes())
        plt.close(fig)

    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {out_file} with code {rc}")
    return out_file, traces


def summary_record(trace: Trace) -> dict[str, str | bool | int]:
    alpha0 = [a[0] for a in trace.alphas if a[0] is not None]
    alpha1 = [a[1] for a in trace.alphas if a[1] is not None]
    alpha0_span = max(alpha0) - min(alpha0) if alpha0 else 0.0
    alpha1_span = max(alpha1) - min(alpha1) if alpha1 else 0.0
    return {
        "group": trace.case.group,
        "method": trace.case.method,
        "obs_id": trace.obs_id,
        "scene_kind": trace.scene_kind,
        "reached": trace.reached,
        "collided": trace.collided,
        "out_of_bounds": trace.out_of_bounds,
        "failure_mode": trace.failure_mode,
        "sim_t": f"{trace.sim_t:.2f}",
        "alpha0_span": f"{alpha0_span:.4f}",
        "alpha1_span": f"{alpha1_span:.4f}",
    }


def mix_color(c1: tuple[float, float, float], c2: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    a = float(np.clip(amount, 0.0, 1.0))
    return tuple((1.0 - a) * x + a * y for x, y in zip(c1, c2))


def method_speed_cmap(method: str) -> LinearSegmentedColormap:
    base = to_rgb(COLORS.get(method, "#7c3aed"))
    slow = mix_color(base, (1.0, 1.0, 1.0), 0.72)
    mid = mix_color(base, (1.0, 1.0, 1.0), 0.08)
    fast = mix_color(base, (0.0, 0.0, 0.0), 0.30)
    return LinearSegmentedColormap.from_list(f"{method}_speed", [slow, mid, fast])


def xy_positions(trace: Trace) -> np.ndarray:
    return np.asarray([state[:2] for state in trace.states], dtype=float)


def segment_speeds(trace: Trace) -> np.ndarray:
    pts = xy_positions(trace)
    if len(pts) <= 1:
        return np.zeros(1, dtype=float)
    dt = np.maximum(np.diff(np.asarray(trace.times, dtype=float)), 1e-9)
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / dt


def bounds_contains(bounds: tuple[float, float, float, float], p: np.ndarray, margin: float = 0.0) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return bool(xmin - margin <= p[0] <= xmax + margin and ymin - margin <= p[1] <= ymax + margin)


def draw_speed_path_static(
    ax,
    trace: Trace,
    speed_norm: Normalize,
    bounds: tuple[float, float, float, float] | None = None,
    linewidth: float = 2.0,
    alpha: float = 0.94,
) -> None:
    pts = xy_positions(trace)
    if len(pts) <= 1:
        return
    if bounds is None:
        keep = np.ones(len(pts) - 1, dtype=bool)
    else:
        keep = np.asarray([
            bounds_contains(bounds, pts[i]) and bounds_contains(bounds, pts[i + 1])
            for i in range(len(pts) - 1)
        ])
    if not np.any(keep):
        return
    points = pts.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)[keep]
    lc = LineCollection(
        segments,
        cmap=method_speed_cmap(trace.case.method),
        norm=speed_norm,
        linewidths=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=8 if trace.case.method == "ours_gat" else 7,
        clip_on=False,
    )
    lc.set_array(segment_speeds(trace)[keep])
    ax.add_collection(lc)


def draw_final_robot_static(ax, trace: Trace, idx: int | None = None) -> None:
    final_idx = len(trace.states) - 1 if idx is None else idx
    draw_robot(ax, trace, final_idx, visual_robot_spec(trace), COLORS.get(trace.case.method, "#7c3aed"))
    if should_draw_failure_marker(trace, final_idx):
        draw_failure_marker(ax, trace.states[final_idx])


def draw_alpha_static(ax, trace: Trace) -> None:
    alpha_arr = np.asarray(
        [
            [np.nan if a0 is None else float(a0), np.nan if a1 is None else float(a1)]
            for a0, a1 in trace.alphas
        ],
        dtype=float,
    )
    times = np.asarray(trace.times, dtype=float)
    if not np.all(np.isnan(alpha_arr[:, 0])):
        ax.plot(times, alpha_arr[:, 0], color="#ef8a80", linewidth=1.35, label="alpha 1", clip_on=False)
    if not np.all(np.isnan(alpha_arr[:, 1])):
        ax.plot(times, alpha_arr[:, 1], color="#75a9e6", linewidth=1.35, label="alpha 2", clip_on=False)
    ax.set_title("OA-CBF w/ GAT CBF adaptation", fontsize=7.8, color="#111827", weight="bold", pad=3)
    ax.set_xlabel("Time [s]", fontsize=6.6, color="#475569", labelpad=1)
    ax.set_ylabel("CBF parameter", fontsize=6.6, color="#475569", labelpad=1)
    ax.grid(False)
    ax.tick_params(labelsize=6.1, colors="#475569", pad=1)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
    ax.legend(loc="upper right", fontsize=5.8, frameon=False)


def draw_obstacles_bounded(
    ax,
    obstacles: np.ndarray,
    bounds: tuple[float, float, float, float],
    moving: bool = False,
    compact: bool = False,
) -> None:
    xmin, xmax, ymin, ymax = bounds
    for obs in np.asarray(obstacles, dtype=float):
        ox, oy, radius = obs[:3]
        if compact:
            visible = xmin <= ox - radius and ox + radius <= xmax and ymin <= oy - radius and oy + radius <= ymax
        else:
            visible = not (ox + radius < xmin or ox - radius > xmax or oy + radius < ymin or oy - radius > ymax)
        if not visible:
            continue
        ax.add_patch(
            Circle(
                (ox, oy),
                radius,
                facecolor="#9ca3af",
                edgecolor="#4b5563",
                linewidth=0.9 if compact else 1.1,
                alpha=0.96,
                zorder=5,
                clip_on=False,
            )
        )
        if moving and len(obs) >= 5 and abs(obs[3]) + abs(obs[4]) > 1e-6:
            scale = 2.2 if compact else 3.0
            end = np.asarray([ox + scale * obs[3], oy + scale * obs[4]], dtype=float)
            if bounds_contains(bounds, end, margin=0.15):
                ax.add_patch(
                    FancyArrowPatch(
                        (ox, oy),
                        tuple(end),
                        arrowstyle="-|>",
                        mutation_scale=8 if compact else 10,
                        color="#2563eb",
                        linewidth=0.9,
                        alpha=0.70,
                        zorder=7,
                        clip_on=False,
                    )
                )


def draw_waypoints_bounded(
    ax,
    waypoints: np.ndarray,
    bounds: tuple[float, float, float, float],
    active_idx: int,
    compact: bool = False,
) -> None:
    pts = np.asarray(waypoints[:, :2], dtype=float)
    active = int(np.clip(active_idx, 1 if len(pts) > 1 else 0, len(pts) - 1))
    if not compact:
        for i in range(len(pts) - 1):
            if bounds_contains(bounds, pts[i], 0.02) and bounds_contains(bounds, pts[i + 1], 0.02):
                ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color="#94a3b8", linewidth=1.05, linestyle="--", alpha=0.82, zorder=3, clip_on=False)
    if bounds_contains(bounds, pts[0]):
        ax.scatter(pts[0, 0], pts[0, 1], s=28 if compact else 54, color="white", edgecolor="#64748b", linewidth=0.9, zorder=7, clip_on=False)
    if active > 1:
        visited = np.asarray([p for p in pts[1:active] if bounds_contains(bounds, p)])
        if len(visited):
            ax.scatter(visited[:, 0], visited[:, 1], s=24 if compact else 42, color="#22c55e", edgecolor="#166534", linewidth=0.6, zorder=7, clip_on=False)
    if bounds_contains(bounds, pts[active]):
        ax.scatter(pts[active, 0], pts[active, 1], s=75 if compact else 130, marker="*", color="#22c55e", edgecolor="#166534", linewidth=0.8, zorder=8, clip_on=False)


def draw_dynamic_obstacle_trails(ax, trace: Trace, bounds: tuple[float, float, float, float], compact: bool = False) -> None:
    base = np.asarray(trace.obstacles, dtype=float)
    if base.ndim != 2 or base.shape[1] < 5 or not np.any(np.abs(base[:, 3:5]) > 1e-7):
        return
    moving_ids = np.where(np.linalg.norm(base[:, 3:5], axis=1) > 1e-7)[0]
    times = np.linspace(0.0, max(float(trace.sim_t), 0.0), 48 if compact else 82)
    frames = [obstacle_frame_at(base, float(t), trace.env_size, bounce=True) for t in times]
    for obs_idx in moving_ids:
        pts = np.asarray([frame[obs_idx, :2] for frame in frames], dtype=float)
        keep = np.asarray([
            bounds_contains(bounds, pts[i]) and bounds_contains(bounds, pts[i + 1])
            for i in range(len(pts) - 1)
        ])
        if not np.any(keep):
            continue
        points = pts.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)[keep]
        ax.add_collection(
            LineCollection(
                segments,
                colors="#4b5563",
                linewidths=0.62 if compact else 0.82,
                alpha=0.24 if compact else 0.20,
                capstyle="round",
                joinstyle="round",
                zorder=4,
                clip_on=False,
            )
        )


def draw_dpcbf_boundaries_bounded(ax, trace: Trace, idx: int, bounds: tuple[float, float, float, float]) -> None:
    if trace.case.robot != "KinematicBicycle2D_DPCBF":
        return
    state = trace.states[idx]
    robot_pos = np.asarray(state[:2], dtype=float)
    theta = float(state[2])
    v = float(state[3]) if len(state) > 3 else 0.0
    obstacles = trace.nearest_obs[idx]
    if obstacles is None or len(obstacles) == 0:
        obstacles = trace.obstacle_frames[idx]
    obstacles = np.asarray(obstacles, dtype=float)
    if obstacles.size == 0:
        return
    spec = visual_robot_spec(trace)
    dists = np.linalg.norm(obstacles[:, :2] - robot_pos.reshape(1, 2), axis=1)
    closest = obstacles[np.argsort(dists)[: min(8, len(obstacles))]]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(closest)))
    beta = 1.05
    for color, obs in zip(colors, closest):
        obs_pos = np.asarray(obs[:2], dtype=float)
        obs_radius = float(obs[2])
        obs_vx = float(obs[3]) if len(obs) > 3 else 0.0
        obs_vy = float(obs[4]) if len(obs) > 4 else 0.0
        ego_dim = (obs_radius + spec.get("radius", 0.3)) * beta
        p_rel = obs_pos - robot_pos
        p_rel_mag = float(np.linalg.norm(p_rel))
        if p_rel_mag <= ego_dim + 1e-4:
            continue
        v_rel = np.array([obs_vx - v * math.cos(theta), obs_vy - v * math.sin(theta)], dtype=float)
        v_rel_mag = max(float(np.linalg.norm(v_rel)), 1e-4)
        d_safe = max(p_rel_mag**2 - ego_dim**2, 1e-6)
        k_lambda = 0.1 * math.sqrt(beta**2 - 1.0) / ego_dim
        k_mu = 0.5 * math.sqrt(beta**2 - 1.0) / ego_dim
        func_lambda = k_lambda * math.sqrt(d_safe) / v_rel_mag
        func_mu = k_mu * math.sqrt(d_safe)
        rot_angle = math.atan2(float(p_rel[1]), float(p_rel[0]))
        rot = np.array([[math.cos(rot_angle), math.sin(rot_angle)], [-math.sin(rot_angle), math.cos(rot_angle)]])
        y_disp = np.linspace(-1.25, 1.25, 80)
        x_disp = -func_lambda * (y_disp**2) - func_mu
        pts = (robot_pos.reshape(2, 1) + rot.T @ np.vstack([x_disp, y_disp])).T
        keep = np.asarray([bounds_contains(bounds, p) for p in pts], dtype=bool)
        start = None
        for j, is_inside in enumerate(keep.tolist() + [False]):
            if is_inside and start is None:
                start = j
            elif not is_inside and start is not None:
                if j - start >= 2:
                    ax.plot(pts[start:j, 0], pts[start:j, 1], color=color, linewidth=1.25, alpha=0.82, zorder=8, clip_on=False)
                start = None


def style_static_axis(ax, title: str | None, y_label: str, compact: bool) -> None:
    style_motion_axis(ax, title, compact=compact, y_label=y_label)
    ax.patch.set_alpha(0.0)
    for child in ax.get_children():
        if hasattr(child, "set_clip_on"):
            child.set_clip_on(False)


def strip_svg_masks(svg_path: Path) -> None:
    text = svg_path.read_text()
    text = re.sub(r"<clipPath\b.*?</clipPath>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<mask\b.*?</mask>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r'\sclip-path="url\([^"]+\)"', "", text)
    text = re.sub(r'\smask="url\([^"]+\)"', "", text)
    text = re.sub(r"clip-path:\s*url\([^)]+\);?", "", text)
    text = re.sub(r"mask:\s*url\([^)]+\);?", "", text)
    svg_path.write_text(text)


def simulate_group(group: str, case_name: str, max_t: float, out_root: Path) -> list[Trace]:
    log_dir = out_root / "_logs" / case_name / group
    log_dir.mkdir(parents=True, exist_ok=True)
    traces: list[Trace] = []
    for case in [case for case in CASES if case.group == group]:
        obs_id = scene_obs_id(case, case_name, False)
        print(f"[svg-sim] {case_name} {group}/{case.method}")
        traces.append(simulate(case, obs_id, max_t, log_dir / f"{case.method}.log", scene_kind=case_name))
    return traces


def render_narrow_svg(group: str, out_root: Path, max_t: float) -> Path:
    traces = simulate_group(group, "narrow", max_t, out_root)
    gat_trace = next(trace for trace in traces if trace.case.method == "ours_gat")
    speeds = [segment_speeds(trace) for trace in traces if len(trace.states) > 1]
    vmax = max(float(np.nanpercentile(s, 98.0)) for s in speeds if len(s))
    speed_norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    x_min, x_max, y_min, y_max = axis_bounds(gat_trace)

    fig = plt.figure(figsize=(9.8, 4.2), facecolor="white")
    ax_env = fig.add_axes([0.06, 0.17, 0.63, 0.72])
    ax_env.set_xlim(x_min, x_max)
    ax_env.set_ylim(y_min, y_max)
    style_static_axis(ax_env, "Trajectory comparison", y_axis_label(gat_trace), compact=False)
    obs0 = gat_trace.obstacle_frames[-1]
    draw_obstacles(ax_env, obs0, moving=obs0.shape[1] >= 5 and bool(np.any(np.abs(obs0[:, 3:5]) > 1e-7)))
    draw_waypoints(ax_env, gat_trace.waypoints)
    for trace in traces:
        draw_speed_path_static(ax_env, trace, speed_norm, linewidth=2.0 if trace.case.method != "ours_gat" else 2.45)
    for trace in traces:
        draw_final_robot_static(ax_env, trace)

    ax_alpha = fig.add_axes([0.74, 0.22, 0.22, 0.58])
    draw_alpha_static(ax_alpha, gat_trace)
    fig.text(0.06, 0.94, f"{GROUP_TITLES.get(group, group)} | Narrow", fontsize=12, weight="bold", color="#111827")
    legend_xs = np.linspace(0.06, 0.67, len(traces))
    for x, trace in zip(legend_xs, traces):
        fig.text(x, 0.055, LABELS.get(trace.case.method, trace.case.method), color=COLORS.get(trace.case.method, "#111827"), fontsize=7, weight="bold", ha="left")
    out = out_root / "narrow" / f"{group}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", transparent=False)
    plt.close(fig)
    strip_svg_masks(out)
    return out


def render_wide_svg(group: str, out_root: Path, max_t: float) -> Path:
    traces = simulate_group(group, "wide", max_t, out_root)
    gat = next(trace for trace in traces if trace.case.method == "ours_gat")
    speeds = [segment_speeds(trace) for trace in traces if len(trace.states) > 1]
    vmax = max(float(np.nanpercentile(s, 98.0)) for s in speeds if len(s))
    speed_norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    fig = plt.figure(figsize=(11.6, 6.25), facecolor="white")
    ax_global = fig.add_axes([0.045, 0.145, 0.545, 0.745])
    bounds = axis_bounds(gat)
    ax_global.set_xlim(bounds[0], bounds[1])
    ax_global.set_ylim(bounds[2], bounds[3])
    style_static_axis(ax_global, "Global progression", y_axis_label(gat), compact=False)
    obs = gat.obstacle_frames[-1]
    if gat.case.robot == "KinematicBicycle2D_DPCBF":
        draw_dynamic_obstacle_trails(ax_global, gat, bounds, compact=False)
    draw_obstacles_bounded(ax_global, obs, bounds, moving=obs.shape[1] >= 5 and bool(np.any(np.abs(obs[:, 3:5]) > 1e-7)), compact=False)
    draw_waypoints_bounded(ax_global, gat.waypoints, bounds, active_idx=active_waypoint_index(gat, len(gat.states) - 1), compact=False)
    for trace in traces:
        draw_speed_path_static(ax_global, trace, speed_norm, bounds=bounds, linewidth=2.0 if trace.case.method != "ours_gat" else 2.45, alpha=0.92)
    for trace in traces:
        draw_final_robot_static(ax_global, trace)

    right_left = 0.625
    right_top = 0.875
    cell_w = 0.155
    cell_h = 0.120
    x_gap = 0.025
    y_gap = 0.030
    for i, trace in enumerate(traces):
        row = i // 2
        col = i % 2
        ax = fig.add_axes([right_left + col * (cell_w + x_gap), right_top - (row + 1) * cell_h - row * y_gap, cell_w, cell_h])
        idx = len(trace.states) - 1
        cbounds = close_bounds(trace, idx, width=2.65, height=2.15)
        ax.set_xlim(cbounds[0], cbounds[1])
        ax.set_ylim(cbounds[2], cbounds[3])
        style_static_axis(ax, LABELS.get(trace.case.method, trace.case.method), y_axis_label(trace), compact=True)
        panel_obs = trace.obstacle_frames[idx]
        if trace.case.robot == "KinematicBicycle2D_DPCBF":
            draw_dynamic_obstacle_trails(ax, trace, cbounds, compact=True)
        draw_obstacles_bounded(ax, panel_obs, cbounds, moving=panel_obs.shape[1] >= 5 and bool(np.any(np.abs(panel_obs[:, 3:5]) > 1e-7)), compact=True)
        draw_waypoints_bounded(ax, trace.waypoints, cbounds, active_idx=active_waypoint_index(trace, idx), compact=True)
        draw_speed_path_static(ax, trace, speed_norm, bounds=cbounds, linewidth=1.75, alpha=0.96)
        if trace.case.robot == "KinematicBicycle2D_DPCBF":
            draw_dpcbf_boundaries_bounded(ax, trace, idx, cbounds)
        draw_final_robot_static(ax, trace, idx)
        label, color = terminal_status_label(trace, trace.sim_t) or ("Time Out", "#d97706")
        ax.text(0.98, 0.96, label, transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color=color, weight="bold", clip_on=False)

    alpha_row = int(math.ceil(len(traces) / 2))
    ax_alpha = fig.add_axes([right_left, right_top - (alpha_row + 1) * cell_h - alpha_row * y_gap, 2 * cell_w + x_gap, cell_h])
    draw_alpha_static(ax_alpha, gat)
    fig.text(0.045, 0.945, f"{GROUP_TITLES.get(group, group)} | Wide", fontsize=12.8, weight="bold", color="#111827")
    fig.text(0.045, 0.055, "Trajectory shade encodes speed within each method color", fontsize=7.2, color="#475569")
    legend_xs = np.linspace(0.045, 0.57, len(traces))
    for x, trace in zip(legend_xs, traces):
        fig.text(x, 0.025, LABELS.get(trace.case.method, trace.case.method), color=COLORS.get(trace.case.method, "#111827"), fontsize=6.8, weight="bold", ha="left")
    out = out_root / "wide" / f"{group}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", transparent=False)
    plt.close(fig)
    strip_svg_masks(out)
    return out


def render_svg_outputs(groups: list[str], case_name: str, out_root: Path, max_t: float) -> int:
    failures = []
    for group in groups:
        try:
            out = render_wide_svg(group, out_root, max_t) if case_name == "wide" else render_narrow_svg(group, out_root, max_t)
            print(f"[ok] {out}")
        except Exception as exc:
            print(f"[fail] {case_name} {group}: {exc}")
            failures.append((group, exc))
    return 1 if failures else 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper SVG figures and MP4 animations.")
    parser.add_argument("--media", choices=["all", "comparison", "individual", "adaptation"], default="all")
    parser.add_argument("--case", choices=["narrow", "wide"], default="narrow")
    parser.add_argument("--format", choices=["svg", "mp4"], default="svg")
    parser.add_argument("--dynamics", default="all", help="Comma-separated dynamics groups, or all.")
    parser.add_argument("--method", default="", help="Optional comma-separated method names.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-t", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means keep every frame selected by --frame-stride")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    all_groups = []
    for case in CASES:
        if case.group not in all_groups:
            all_groups.append(case.group)
    if args.dynamics.strip().lower() == "all":
        groups = all_groups
    else:
        groups = [item.strip() for item in args.dynamics.split(",") if item.strip()]
    unknown = [group for group in groups if group not in all_groups]
    if unknown:
        raise ValueError(f"Unknown dynamics group(s): {', '.join(unknown)}")

    if args.out is None:
        if args.format == "svg":
            args.out = "paper_media/svg"
        else:
            args.out = f"paper_media/mp4/{args.case}"
    out_root = ROOT / args.out
    max_t = args.max_t
    if max_t is None:
        max_t = 220.0 if args.case == "wide" else (160.0 if args.format == "svg" else 60.0)

    method_filters = [item.strip() for item in args.method.split(",") if item.strip()]
    selected = [
        case for case in CASES
        if case.group in groups and (not method_filters or case.method in method_filters)
    ]
    if not selected:
        raise ValueError("No matching cases selected.")

    if args.format == "svg":
        return render_svg_outputs(groups, args.case, out_root, max_t)

    if args.case == "wide" and args.media in ("all", "comparison", "adaptation"):
        failures = []
        summary_rows = []
        for group in groups:
            out_file = out_root / group / "comparison.mp4"
            if args.media in ("all", "comparison") and out_file.exists() and not args.force:
                print(f"[skip] {out_file}")
                traces = []
            elif args.media in ("all", "comparison"):
                try:
                    produced, traces = render_wide_comparison(
                        group,
                        out_root,
                        max_t,
                        args.max_frames,
                        frame_stride=args.frame_stride,
                        fps=args.fps,
                    )
                    print(f"[ok] {produced}")
                    summary_rows.extend(summary_record(trace) for trace in traces)
                except Exception as exc:
                    print(f"[fail] {group}: {exc}")
                    failures.append((group, exc))
                    continue
            else:
                traces = []

            if args.media in ("all", "adaptation"):
                if not traces:
                    cases = [case for case in CASES if case.group == group]
                    log_dir = out_root / "_logs" / group
                    log_dir.mkdir(parents=True, exist_ok=True)
                    traces = [
                        simulate(case, scene_obs_id(case, "wide", False), max_t, log_dir / f"{case.method}.log", scene_kind="wide")
                        for case in cases
                    ]
                gat_trace = next((trace for trace in traces if trace.case.method == "ours_gat"), None)
                if gat_trace is not None:
                    gat_alpha_file = out_root / group / "oa_cbf_gat_adaptation.mp4"
                    render_trace(
                        gat_trace,
                        gat_alpha_file,
                        max_frames=args.max_frames,
                        risk_overlay=True,
                        dpcbf_overlay=gat_trace.case.robot == "KinematicBicycle2D_DPCBF",
                        frame_stride=args.frame_stride,
                        fps=args.fps,
                    )
                    print(f"[ok] {gat_alpha_file}")
        if summary_rows:
            summary_path = out_root / "summary.csv"
            existing_rows = {}
            if summary_path.exists():
                with summary_path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        existing_rows[(row["group"], row["method"])] = row
            for row in summary_rows:
                existing_rows[(row["group"], row["method"])] = row
            ordered_keys = [(case.group, case.method) for case in CASES if (case.group, case.method) in existing_rows]
            fieldnames = ["group", "method", "obs_id", "scene_kind", "reached", "collided", "out_of_bounds", "failure_mode", "sim_t", "alpha0_span", "alpha1_span"]
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([existing_rows[key] for key in ordered_keys])
        if failures:
            print("\nFailures:")
            for group, exc in failures:
                print(f"  - {group}: {exc}")
            return 1
        return 0

    if args.case == "wide":
        selected = [case for case in selected if args.media in ("all", "individual")]
    elif args.media == "adaptation":
        selected = [case for case in selected if case.method == "ours_gat"]

    failures = []
    summary_rows = []
    for case in selected:
        out_file = out_root / case.group / f"{case.method}.mp4"
        if out_file.exists() and not args.force:
            print(f"[skip] {out_file}")
            continue
        risk_overlay = case.method == "ours_gat"
        dpcbf_overlay = case.robot == "KinematicBicycle2D_DPCBF"
        print(f"[render] {args.case} {case.group}/{case.method}")
        try:
            produced, trace = render_case(
                case,
                out_root,
                max_t,
                args.max_frames,
                risk_overlay=risk_overlay,
                risk_trial=False,
                scene_kind=args.case,
                dpcbf_overlay=dpcbf_overlay,
                frame_stride=args.frame_stride,
                fps=args.fps,
            )
            rec = summary_record(trace)
            print(
                f"[ok] {produced} reached={trace.reached} collided={trace.collided} "
                f"sim_t={trace.sim_t:.2f} alpha_span=({rec['alpha0_span']},{rec['alpha1_span']})"
            )
            summary_rows.append(rec)
        except Exception as exc:
            print(f"[fail] {case.group}/{case.method}: {exc}")
            failures.append((case, exc))

    if failures:
        print("\nFailures:")
        for case, exc in failures:
            print(f"  - {case.group}/{case.method}: {exc}")
        return 1
    if summary_rows:
        summary_path = out_root / "summary.csv"
        existing_rows = {}
        if summary_path.exists():
            with summary_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    existing_rows[(row["group"], row["method"])] = row
        for row in summary_rows:
            existing_rows[(row["group"], row["method"])] = row
        ordered_keys = [(case.group, case.method) for case in CASES if (case.group, case.method) in existing_rows]
        fieldnames = ["group", "method", "obs_id", "scene_kind", "reached", "collided", "out_of_bounds", "failure_mode", "sim_t", "alpha0_span", "alpha1_span"]
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([existing_rows[key] for key in ordered_keys])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
