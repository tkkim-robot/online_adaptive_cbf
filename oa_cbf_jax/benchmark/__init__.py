"""Reproducible closed-loop navigation benchmarks."""

import importlib
import os

from .metrics import trial
from .protocol import CASES_PER_SYSTEM, configure_precision, requests


def _worker_init(cores):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cores.get()})


def evaluate(models, system, method, start=0, end=CASES_PER_SYSTEM):
    """Run a contiguous group of complete episodes starting at their initial states."""
    import jax
    from .scenarios import make_case

    configure_precision(system)
    learned = method in ("ours_gat", "ours_fc")
    if learned:
        controller = importlib.import_module(f".{system}", __package__).build(models, method)
    else:
        from . import baselines

        controller = baselines.build(models, system, method)
    rows = []
    for request in requests(system):
        ordinal = request["ordinal"]
        if ordinal >= end:
            break
        if ordinal < start:
            continue
        configure_precision(system)
        case = make_case(request, models)
        if not learned:
            # The native optimizers run in double precision. Their unicycle and
            # Quad2D plant and observation kernels retain single precision.
            jax.config.update("jax_enable_x64", True)
        raw, trace = controller(case)
        rows.append(trial(case, method, raw, trace))
    return rows
