#!/usr/bin/env python3
"""Run a live simulation for the narrow/wide paper-media scenarios."""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
from pathlib import Path

import matplotlib


ROOT = Path(__file__).resolve().parents[1]


def import_media_module(backend: str | None = None):
    if backend:
        matplotlib.use(backend)

    original_use = matplotlib.use

    def use_without_forcing_agg(name, *args, **kwargs):
        if str(name).lower() == "agg":
            return None
        return original_use(name, *args, **kwargs)

    matplotlib.use = use_without_forcing_agg
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if str(ROOT / "plot") not in sys.path:
            sys.path.insert(0, str(ROOT / "plot"))
        import generate_paper_media as media
    finally:
        matplotlib.use = original_use
    return media


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one paper-media scenario with the live safe_control renderer."
    )
    parser.add_argument("--case", choices=["narrow", "wide"], default="narrow")
    parser.add_argument(
        "--dynamics",
        required=False,
        help="One dynamics group, e.g. dynamic_unicycle, quad2d, quad3d, kinematic_bicycle_dpcbf.",
    )
    parser.add_argument(
        "--method",
        required=False,
        help="One method, e.g. fixed_low, fixed_high, od_cbf_qp, od_cbf_mpc, barriernet, ours_fc, ours_gat.",
    )
    parser.add_argument("--max-t", type=float, default=None)
    parser.add_argument("--pause", type=float, default=0.001)
    parser.add_argument("--backend", default=None, help="Optional Matplotlib backend, e.g. MacOSX or TkAgg.")
    parser.add_argument("--hold", action="store_true", help="Keep the figure window open after the run.")
    parser.add_argument("--list", action="store_true", help="Print available dynamics/method combinations and exit.")
    return parser.parse_args()


def available_cases(media) -> list:
    return list(media.CASES)


def find_case(media, group: str, method: str):
    matches = [case for case in media.CASES if case.group == group and case.method == method]
    if not matches:
        groups = sorted({case.group for case in media.CASES})
        methods = sorted({case.method for case in media.CASES if case.group == group})
        raise ValueError(
            f"No case for dynamics={group!r}, method={method!r}.\n"
            f"Available dynamics: {', '.join(groups)}\n"
            f"Available methods for {group!r}: {', '.join(methods) if methods else '(none)'}"
        )
    return matches[0]


def live_build_tracker(media, case, x_init, waypoints, obstacles, env_size, scene_kind: str):
    import matplotlib.pyplot as plt
    from safe_control.utils import env, plotting

    robot_spec, _ = media.get_robot_spec_and_obs(case.robot)
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

    ctrl_type, g0, g1 = media.get_controller_defaults(case.robot, case.controller)
    controller_type = {"pos": ctrl_type}
    if ctrl_type == "barriernet":
        controller_type = {"pos": "barriernet", "ckpt": media.resolve_artifact_path(case.checkpoint)}

    plotter = plotting.Plotting(
        width=env_size["width"],
        height=env_size["height"],
        known_obs=obstacles,
    )
    ax, fig = plotter.plot_grid(f"{media.GROUP_TITLES.get(case.group, case.group)} | {media.LABELS.get(case.method, case.method)} | {scene_kind}")
    cls = media.LocalTrackingControllerDyn if case.robot == "KinematicBicycle2D_DPCBF" else media.LocalTrackingController
    tracker = cls(
        x_init,
        robot_spec,
        controller_type=controller_type,
        dt=media.ae.DT,
        show_animation=True,
        save_animation=False,
        ax=ax,
        fig=fig,
        env=env.Env(width=env_size["width"], height=env_size["height"], known_obs=obstacles),
    )
    if ctrl_type != "barriernet":
        if case.robot in ["KinematicBicycle2D_C3BF", "KinematicBicycle2D_DPCBF", "Quad3D"]:
            tracker.pos_controller.cbf_param["alpha"] = g0
        else:
            tracker.pos_controller.cbf_param["alpha1"] = g0
            tracker.pos_controller.cbf_param["alpha2"] = g1
    tracker.obs = obstacles.copy()
    tracker.set_waypoints(waypoints)
    plt.ion()
    return tracker, ctrl_type


def make_adapter(media, case, ctrl_type: str, scene_kind: str):
    if not (case.controller.startswith("Online Adaptive") and ctrl_type != "barriernet"):
        return None
    adapter = media.get_online_cbf_adapter(case.robot, case.controller, print_info=False)
    if adapter and case.robot == "KinematicBicycle2D_DPCBF":
        adapter.upper_bound = min(adapter.upper_bound, 1.5 if scene_kind == "wide" else 3.0)
    if adapter and case.robot == "Quad2D" and scene_kind == "narrow":
        if case.method == "ours_gat":
            adapter.upper_bound = min(adapter.upper_bound, 0.55)
        elif case.method == "ours_fc":
            adapter.upper_bound = min(adapter.upper_bound, 0.45)
    return adapter


def apply_adapter_update(tracker, adapter) -> None:
    if adapter is None:
        return
    g0_new, g1_new = adapter.cbf_param_adaptation(tracker)
    if adapter.gamma_dim == 1:
        tracker.pos_controller.cbf_param["alpha"] = g0_new
    else:
        tracker.pos_controller.cbf_param["alpha1"] = g0_new
        tracker.pos_controller.cbf_param["alpha2"] = g1_new


def run_preview(media, args: argparse.Namespace) -> int:
    if args.list:
        for group in sorted({case.group for case in media.CASES}):
            methods = [case.method for case in media.CASES if case.group == group]
            print(f"{group}: {', '.join(methods)}")
        return 0
    if not args.dynamics or not args.method:
        raise ValueError("--dynamics and --method are required unless --list is used.")

    random.seed(0)
    media.np.random.seed(0)
    with contextlib.suppress(Exception):
        import torch

        torch.manual_seed(0)

    case = find_case(media, args.dynamics, args.method)
    max_t = args.max_t
    if max_t is None:
        max_t = 220.0 if args.case == "wide" else 60.0
    obs_id = media.scene_obs_id(case, args.case, False)
    env_size, x_init, waypoints, obstacles = media.make_scene(case.robot, obs_id, args.case)
    media.configure_env_for_case(case, max_t)

    tracker, ctrl_type = live_build_tracker(media, case, x_init, waypoints, obstacles, env_size, args.case)
    adapter = make_adapter(media, case, ctrl_type, args.case)
    pause = max(float(args.pause), 1e-4)

    print(f"Backend: {matplotlib.get_backend()}")
    print(f"Running {args.case} {case.group}/{case.method} for up to {max_t:.2f}s")
    print("Close the Matplotlib window or press Ctrl+C in the terminal to stop.")

    steps = int(max_t / media.ae.DT)
    try:
        tracker.draw_plot(pause=pause)
        for k in range(steps):
            ret = tracker.control_step()
            if case.robot == "Quad3D":
                tracker.robot.X[2, 0] = 0.0
                tracker.robot.X[8, 0] = 0.0
                tracker.robot.render_plot()
            apply_adapter_update(tracker, adapter)
            tracker.draw_plot(pause=pause)
            if ret in (-1, -2):
                status = "reached" if ret == -1 else "stopped"
                print(f"Preview ended at t={(k + 1) * media.ae.DT:.2f}s ({status}).")
                break
        else:
            print(f"Preview reached max_t={max_t:.2f}s.")
        if args.hold:
            import matplotlib.pyplot as plt

            print("Holding figure window. Close it to exit.")
            plt.ioff()
            plt.show()
    except KeyboardInterrupt:
        print("\nPreview interrupted.")
    return 0


def main() -> int:
    args = parse_args()
    media = import_media_module(args.backend)
    return run_preview(media, args)


if __name__ == "__main__":
    raise SystemExit(main())
