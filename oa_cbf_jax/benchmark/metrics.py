"""Exclusive physical outcomes and time to goal for matched navigation trials."""

from collections import Counter
from statistics import mean, median, stdev

import numpy as np

from .protocol import DT, STEPS

OUTCOMES = (
    "success",
    "solver_failure_rejection",
    "policy_admissibility_stop",
    "collision",
    "state_bound_violation",
    "timeout",
    "planner_other",
)
PLANAR_CODES = (
    "running",
    "goal_reached",
    "collision",
    "solver_infeasible",
    "timeout",
    "hocbf_inadmissible",
    "policy_rejected",
    "planner_failure",
    "state_bound_violation",
)
CODES = {
    "unicycle": PLANAR_CODES,
    "quad2d": PLANAR_CODES,
    "quad3d": (
        "running",
        "goal_reached",
        "hocbf_inadmissible",
        "solver_rejected",
        "collision",
        "state_bound_violation",
        "timeout",
    ),
    "bicycle": (
        "running",
        "goal_reached",
        "collision",
        "qp_rejected",
        "timeout",
        "barrier_inadmissible",
        "planner_failure",
        "state_bound_violation",
    ),
}
CATEGORIES = {
    "goal_reached": "success",
    "success": "success",
    "collision": "collision",
    "state_bound_violation": "state_bound_violation",
    "timeout": "timeout",
    "planner_failure": "planner_other",
    "solver_infeasible": "solver_failure_rejection",
    "solver_rejected": "solver_failure_rejection",
    "qp_rejected": "solver_failure_rejection",
    "hocbf_inadmissible": "policy_admissibility_stop",
    "policy_rejected": "policy_admissibility_stop",
    "barrier_inadmissible": "policy_admissibility_stop",
}


def _arrived(system, x, goal):
    tolerance = 1e-5
    if np.linalg.norm(x[: len(goal)] - goal) > 0.25 + tolerance:
        return False
    if system in ("unicycle", "bicycle"):
        return abs(x[3]) <= (0.35 if system == "bicycle" else 0.2) + tolerance
    if system == "quad2d":
        return (
            np.linalg.norm(x[3:5]) <= 0.2 + tolerance
            and abs(x[2]) <= 0.1 + tolerance
            and abs(x[5]) <= 0.2 + tolerance
        )
    return (
        np.linalg.norm(x[6:9]) <= 0.2 + tolerance
        and np.max(abs(x[3:6])) <= 0.1 + tolerance
        and np.max(abs(x[9:12])) <= 0.2 + tolerance
    )


def trial(case, method, raw, trace):
    """Classify the executed trajectory and measure only actually applied controls."""
    system = case["dynamics"]
    active = np.asarray(trace["active"], dtype=bool)
    steps = int(active.sum())
    reported_steps = raw["applied_steps"] if "applied_steps" in raw else raw["steps"]
    if not 0 <= steps <= STEPS or steps != int(reported_steps):
        raise ValueError("Applied control mask disagrees with the reported physical clock")
    status = raw["status"]
    if isinstance(status, (int, np.integer)):
        if not 0 <= int(status) < len(CODES[system]):
            raise ValueError(f"Unknown {system} status {status}")
        status = CODES[system][int(status)]
    outcome = CATEGORIES[status]
    minimum = float(raw.get("min_clearance", np.inf))
    bound = 0.0
    for key in (
        "clearance",
        "swept_clearance",
        "state_violation",
        "state_bound_violation",
        "envelope",
    ):
        if key in trace:
            values = np.asarray(trace[key])[active]
            values = values[np.isfinite(values)]
            if values.size:
                if key in ("clearance", "swept_clearance"):
                    minimum = min(minimum, float(values.min()))
                else:
                    bound = max(bound, float(values.max()))
    if minimum <= 0:
        outcome = "collision"
    elif bound > 1e-5 and outcome != "collision":
        outcome = "state_bound_violation"
    if outcome == "success":
        key = next(k for k in ("states", "next_state", "state") if k in trace)
        x = np.asarray(trace[key][-1], dtype=float)
        goal = case["prepared_arguments"]["goal"]
        if not np.isfinite(x).all() or not _arrived(system, x, goal):
            raise ValueError("Goal status disagrees with the final physical state")
    return dict(
        dynamics=system,
        method=method,
        trial_id=case["trial_id"],
        noise_scale=case["noise_scale"],
        outcome=outcome,
        applied_steps=steps,
        simulated_seconds=round(steps * DT, 2),
        reach_time_seconds=round(steps * DT, 2) if outcome == "success" else None,
    )


def summarize(rows):
    """Summarize each outcome over all trials and reach time over successes."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dynamics"], row["method"]), []).append(row)
    for (system, method), group in grouped.items():
        n = len(group)
        if len({r["trial_id"] for r in group}) != n:
            raise ValueError("Duplicate method trial")
        counts = Counter(r["outcome"] for r in group)
        if set(counts).difference(OUTCOMES):
            raise ValueError("Unknown physical outcome")
        times = [r["reach_time_seconds"] for r in group if r["outcome"] == "success"]
        result = dict(dynamics=system, method=method, trials=n)
        for key in OUTCOMES:
            result[key + "_count"] = counts[key]
            result[key + "_percent"] = 100 * counts[key] / n
        result["mean_reach_time_seconds"] = mean(times) if times else None
        result["median_reach_time_seconds"] = median(times) if times else None
        result["sample_sd_reach_time_seconds"] = (
            stdev(times) if len(times) > 1 else (0.0 if times else None)
        )
        yield result
