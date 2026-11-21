"""Run the matched CPU benchmark and write its trial and summary tables."""

import argparse
import csv
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from . import _worker_init, evaluate
from .metrics import summarize
from .protocol import CASES_PER_SYSTEM, METHODS, SYSTEMS, verify_models


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    from oa_cbf_jax.model_release import DEFAULT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT, help="downloaded model directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmark"),
        help="new directory for the results",
    )
    parser.add_argument("--dynamics", choices=SYSTEMS, help="run just one vehicle")
    parser.add_argument("--method", choices=METHODS["unicycle"], help="run just one method")
    parser.add_argument("--workers", type=int, default=4, help="number of scalar CPU workers")
    args = parser.parse_args(argv)
    systems = [args.dynamics] if args.dynamics else SYSTEMS
    groups = [
        (d, m) for d in systems for m in METHODS[d] if args.method is None or m == args.method
    ]
    if not groups:
        parser.error("The requested method is unsupported for this vehicle")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        parser.error(f"{args.output} already exists. Choose a new --output directory.")
    if not (args.models / "release.json").is_file():
        parser.error("Install the models with python -m oa_cbf_jax.model_release")
    verify_models(args.models)
    os.environ["JAX_PLATFORMS"] = "cpu"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    available = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    workers = min(args.workers, len(available))
    context = mp.get_context("spawn")
    cores = context.Queue()
    for cpu in available[:workers]:
        cores.put(cpu)

    # Interleave methods so an expensive final comparison cannot monopolize
    # one process while the other workers have no remaining episodes.
    batches = max(1, workers)
    width = (CASES_PER_SYSTEM + batches - 1) // batches
    jobs = [
        (d, m, start, min(start + width, CASES_PER_SYSTEM))
        for start in range(0, CASES_PER_SYSTEM, width)
        for d, m in groups
    ]
    rows = []
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=_worker_init, initargs=(cores,)
    ) as pool:
        results = [pool.submit(evaluate, args.models, *job) for job in jobs]
        for result in results:
            rows.extend(result.result())
    rows.sort(
        key=lambda r: (
            SYSTEMS.index(r["dynamics"]),
            METHODS[r["dynamics"]].index(r["method"]),
            r["trial_id"],
        )
    )
    if len(rows) != CASES_PER_SYSTEM * len(groups):
        raise ValueError("Incomplete benchmark")
    summary = list(summarize(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        completed = Path(temp) / "benchmark"
        completed.mkdir()
        _write_csv(completed / "trials.csv", rows)
        _write_csv(completed / "summary.csv", summary)
        completed.rename(args.output)
    print(f"Wrote {len(rows)} trials and {len(summary)} summary rows to {args.output}.")


if __name__ == "__main__":
    main()
