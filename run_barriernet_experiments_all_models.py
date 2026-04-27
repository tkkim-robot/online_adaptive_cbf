#!/usr/bin/env python3
"""
BarrierNet Experiments Runner (All 4 Models, Best-Window)
========================================================

Goal:
- Define a "best-window" evaluation subset per robot using the *high fixed param* baseline CSV
- Run BarrierNet on exactly those obs_ids via adaptation_experiment.py (OBS_IDS_FILTER)
- Verify CSV files were created
- Generate summaries and analyze p1/p2/alpha values

Best-window definition (window_size=100 by default):
- maximize success rate (reached)
- tie-breaker: minimize average reach time among reached in the window

This script is self-contained and does not require torch_geometric (BarrierNet-only).
"""

import os
import sys
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


ROOT = Path(__file__).resolve().parent


# =========================
# CONFIG (edit here)
# =========================
OUTPUT_DIR = "barriernet_results"
DISTR_MODE = "in"
SELECTED_MODE = "obs_sweep"
MAX_T = "100.0"
OBS_SET_COUNT = "400"

WINDOW_SIZE = int(os.environ.get("BN_WINDOW_SIZE", "100"))
LIMIT_OBS_IDS = int(os.environ.get("BN_LIMIT_OBS_IDS", "0"))  # 0 => no limit
DRY_RUN = os.environ.get("BN_DRY_RUN", "").strip().lower() in ("1", "true", "yes")

# BarrierNet checkpoints (produced by safe_control/run_barriernet_all.sh)
# Note: Quad3D excluded because agent_barrier raises NotImplementedError
CKPT: Dict[str, str] = {
    "DynamicUnicycle2D": "safe_control/position_control/BarrierNet/checkpoints/DynamicUnicycle2D_barriernet.pth",
    "Quad2D": "safe_control/position_control/BarrierNet/checkpoints/Quad2D_barriernet.pth",
    "KinematicBicycle2D_DPCBF": "safe_control/position_control/BarrierNet/checkpoints/KinematicBicycle2D_DPCBF_barriernet.pth",
}

# Per-robot baseline controller used to define best window
# Note: Quad3D excluded because agent_barrier raises NotImplementedError
HIGH_FIXED_PARAM_CONTROLLER: Dict[str, str] = {
    "DynamicUnicycle2D": "MPC-CBF high fixed param",
    "Quad2D": "MPC-CBF high fixed param",
    "KinematicBicycle2D_DPCBF": "CBF-QP high fixed param",
}

# Optional robot filter: set BN_ROBOTS="Quad2D" to run a subset
# Note: Quad3D excluded because agent_barrier raises NotImplementedError
_robots_env = os.environ.get("BN_ROBOTS", "").strip()
if _robots_env:
    ROBOT_MODELS = [s.strip() for s in _robots_env.split(",") if s.strip()]
else:
    ROBOT_MODELS = ["DynamicUnicycle2D", "Quad2D", "KinematicBicycle2D_DPCBF"]


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def _find_most_recent_baseline_csv(robot_model: str) -> Path:
    """
    Find the most recent baseline CSV for the given robot/model/controller.

    We search under common output folders:
    - epoch_exp/**  (older pipeline outputs)
    - sim_results/** (your current saved baseline runs)
    - repo root (fallback)
    """
    controller = HIGH_FIXED_PARAM_CONTROLLER[robot_model]
    controller_token = controller.replace(" ", "_")
    pattern = f"sim_results_{DISTR_MODE}_{SELECTED_MODE}_{controller_token}_{robot_model}_*.csv"
    roots = [ROOT / "epoch_exp", ROOT / "sim_results", ROOT]
    candidates: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        candidates.extend(list(r.glob(f"**/{pattern}")))
    if not candidates:
        raise FileNotFoundError(
            f"No baseline CSV found for robot={robot_model} using controller='{controller}'.\n"
            f"Searched patterns like: **/{pattern}\n"
            f"Expected under: epoch_exp/, sim_results/, or repo root.\n"
            f"Action: run adaptation_experiment.py with SELECTED_CONTROLLER='{controller}' and SELECTED_ROBOT='{robot_model}' "
            f"in DISTR_MODE={DISTR_MODE} SELECTED_MODE={SELECTED_MODE}."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _best_window_from_csv(csv_path: Path, window_size: int) -> Tuple[List[int], float, float]:
    """
    Returns (obs_ids, success_rate, collision_rate) for the best sliding window.
    Tie-breaker: fastest average reach time among reached.
    """
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                obs_id = int(row.get("obs_id", "-1"))
                reached = _parse_bool(row.get("reached", "False"))
                collided = _parse_bool(row.get("collided", "False"))
                sim_t = float(row.get("sim_t", "0.0"))
            except Exception:
                continue
            if obs_id < 0:
                continue
            rows.append({"obs_id": obs_id, "reached": reached, "collided": collided, "sim_t": sim_t})

    if len(rows) < window_size:
        raise ValueError(f"CSV too short for window_size={window_size}: {csv_path} has {len(rows)} rows")

    best_obs_ids: List[int] = []
    best_success = -1.0
    best_collision = 1.0
    best_avg_t = float("inf")

    for start in range(0, len(rows) - window_size + 1):
        window = rows[start : start + window_size]
        reached_count = sum(1 for r in window if r["reached"])
        collided_count = sum(1 for r in window if r["collided"])
        success = reached_count / window_size
        collision = collided_count / window_size
        reached_times = [r["sim_t"] for r in window if r["reached"]]
        avg_t = (sum(reached_times) / len(reached_times)) if reached_times else float("inf")

        if (success > best_success) or (success == best_success and avg_t < best_avg_t):
            best_success = success
            best_collision = collision
            best_avg_t = avg_t
            best_obs_ids = [r["obs_id"] for r in window]

    return best_obs_ids, best_success, best_collision


def run_one(robot_model: str) -> Optional[Path]:
    """
    Run BarrierNet experiment for one robot model.
    Returns the path to the created CSV file, or None if failed.
    """
    ckpt = (ROOT / CKPT[robot_model]).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for {robot_model}: {ckpt}")

    baseline_csv = _find_most_recent_baseline_csv(robot_model)
    obs_ids, success_rate, collision_rate = _best_window_from_csv(baseline_csv, WINDOW_SIZE)
    if LIMIT_OBS_IDS > 0:
        obs_ids = obs_ids[:LIMIT_OBS_IDS]

    env = os.environ.copy()
    env["SELECTED_ROBOT"] = robot_model
    env["SELECTED_CONTROLLER"] = "BarrierNet"
    env["DISTR_MODE"] = DISTR_MODE
    env["SELECTED_MODE"] = SELECTED_MODE
    env["MAX_T"] = MAX_T
    env["OBS_SET_COUNT"] = OBS_SET_COUNT
    env["OUTPUT_DIR"] = str((ROOT / OUTPUT_DIR / robot_model).resolve())
    env["CHECKPOINT_FILE"] = str(ckpt)
    env["OBS_IDS_FILTER"] = repr(obs_ids)

    print("\n" + "=" * 100)
    print(f"Running BarrierNet experiment: robot={robot_model}")
    print(f"checkpoint: {ckpt}")
    print(f"best-window baseline: {baseline_csv}")
    print(f"best-window stats (baseline): success={success_rate:.2%} collision={collision_rate:.2%} window={WINDOW_SIZE}")
    print(f"obs_ids passed to BarrierNet: {len(obs_ids)}")
    print("=" * 100)

    # Record timestamp before experiment
    start_time = datetime.now()
    timestamp_str = start_time.strftime("%m%d_%H%M")

    # Run adaptation_experiment.py from repo root
    if DRY_RUN:
        print("[DRY_RUN] Skipping subprocess.run()")
        return None
    result = subprocess.run([sys.executable, str(ROOT / "adaptation_experiment.py")], cwd=str(ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Experiment failed for {robot_model} (exit={result.returncode})")
    
    # Find the newly created CSV file
    output_dir = ROOT / OUTPUT_DIR / robot_model
    # Look for CSV files created after start_time (with some margin)
    # Pattern: sim_results_in_obs_sweep_BarrierNet_{robot_model}_*.csv
    pattern = f"sim_results_{DISTR_MODE}_{SELECTED_MODE}_BarrierNet_{robot_model}_*.csv"
    candidates = list(output_dir.glob(pattern))
    
    # Filter by modification time (should be after start_time)
    start_timestamp = start_time.timestamp()
    new_csvs = [c for c in candidates if c.stat().st_mtime >= start_timestamp - 60]  # 60s margin
    
    if not new_csvs:
        # Fallback: get the most recent CSV
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"⚠️  Warning: Could not find CSV created after experiment start. Using most recent: {candidates[0]}")
            return candidates[0]
        else:
            raise FileNotFoundError(f"No CSV file found in {output_dir} after experiment")
    
    # Return the most recent one (in case multiple were created)
    new_csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return new_csvs[0]


def verify_csv(csv_path: Path, expected_rows: int = 100) -> bool:
    """Verify CSV file exists and has expected number of rows."""
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    try:
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            n_rows = len(rows)
            if n_rows < expected_rows * 0.9:  # Allow 10% tolerance
                print(f"⚠️  CSV has {n_rows} rows, expected ~{expected_rows}")
                return False
            print(f"✅ CSV verified: {n_rows} rows in {csv_path.name}")
            return True
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False


def analyze_results(robot_model: str, csv_path: Path) -> None:
    """Run summary and p-value analysis for a robot model."""
    print(f"\n{'='*100}")
    print(f"Analyzing results for {robot_model}")
    print(f"{'='*100}\n")
    
    # 1. Generate summary
    print("Step 1: Generating summary...")
    summary_script = ROOT / "summarize_barriernet_results.py"
    if summary_script.exists():
        result = subprocess.run([sys.executable, str(summary_script)], cwd=str(ROOT))
        if result.returncode == 0:
            summary_file = csv_path.with_suffix(".summary.txt")
            if summary_file.exists():
                print(f"✅ Summary generated: {summary_file.name}")
                # Print summary content
                with summary_file.open("r") as f:
                    print(f.read())
            else:
                print(f"⚠️  Summary file not found: {summary_file}")
        else:
            print(f"⚠️  Summary generation failed (exit={result.returncode})")
    else:
        print(f"⚠️  Summary script not found: {summary_script}")
    
    # 2. Analyze p values
    print("\nStep 2: Analyzing p1/p2/alpha values...")
    analyze_script = ROOT / "analyze_barriernet_p_values.py"
    ckpt_path = ROOT / CKPT[robot_model]
    
    if analyze_script.exists() and ckpt_path.exists():
        result = subprocess.run(
            [sys.executable, str(analyze_script),
             "--robot_model", robot_model,
             "--checkpoint", str(ckpt_path)],
            cwd=str(ROOT)
        )
        if result.returncode == 0:
            print("✅ p-value analysis completed")
        else:
            print(f"⚠️  p-value analysis failed (exit={result.returncode})")
    else:
        if not analyze_script.exists():
            print(f"⚠️  Analysis script not found: {analyze_script}")
        if not ckpt_path.exists():
            print(f"⚠️  Checkpoint not found: {ckpt_path}")


def main():
    os.makedirs(ROOT / OUTPUT_DIR, exist_ok=True)
    
    results: Dict[str, Optional[Path]] = {}
    
    # Run experiments
    print("\n" + "="*100)
    print("PHASE 1: Running BarrierNet Experiments")
    print("="*100)
    
    for robot in ROBOT_MODELS:
        try:
            csv_path = run_one(robot)
            results[robot] = csv_path
            if csv_path:
                verify_csv(csv_path, expected_rows=WINDOW_SIZE if LIMIT_OBS_IDS == 0 else LIMIT_OBS_IDS)
        except Exception as e:
            print(f"❌ Failed for {robot}: {e}")
            results[robot] = None
    
    print("\n" + "="*100)
    print("PHASE 2: Analyzing Results")
    print("="*100)
    
    # Analyze results
    for robot, csv_path in results.items():
        if csv_path and csv_path.exists():
            analyze_results(robot, csv_path)
        else:
            print(f"\n⚠️  Skipping analysis for {robot}: no valid CSV file")
    
    print("\n" + "="*100)
    print("✅ BarrierNet experiments and analysis completed for all models.")
    print("="*100)
    
    # Final summary
    print("\nFinal Results Summary:")
    for robot, csv_path in results.items():
        if csv_path and csv_path.exists():
            print(f"  ✅ {robot}: {csv_path.name}")
        else:
            print(f"  ❌ {robot}: No CSV file generated")


if __name__ == "__main__":
    main()


