"""Checks for the matched scene schedule and reported physical outcomes."""

from collections import Counter

import numpy as np
import pytest

from oa_cbf_jax.benchmark import metrics, protocol


@pytest.mark.parametrize("system", protocol.SYSTEMS)
def test_matched_inputs(system):
    rows = list(protocol.requests(system))
    assert rows == list(protocol.requests(system))
    assert len({r["case_id"] for r in rows}) == protocol.CASES_PER_SYSTEM
    assert len({r["sensor_seed"] for r in rows}) == len(rows)
    assert Counter(r["noise_scale"] for r in rows) == {scale: 250 for scale in protocol.NOISE}
    assert [r["ordinal"] for r in rows] == list(range(len(rows)))


def test_supported_methods():
    assert "optimal_decay_qp" in protocol.METHODS["quad2d"]
    assert "optimal_decay_qp" not in protocol.METHODS["bicycle"]
    assert sum(map(len, protocol.METHODS.values())) == 26


def _example(status, name="test", clearance=1.0):
    case = dict(
        dynamics="bicycle", trial_id=name, noise_scale=0.0, prepared_arguments={"goal": np.zeros(2)}
    )
    raw = dict(status=status, steps=2, min_clearance=clearance)
    trace = dict(
        active=np.array([True, True, False]),
        state=np.zeros((3, 4)),
        clearance=np.array([clearance, clearance, np.nan]),
    )
    return metrics.trial(case, "ours_gat", raw, trace)


def test_report_separates_policy_and_solver_stops():
    rows = [
        _example(1),
        _example(3, "solver"),
        _example(5, "policy"),
        _example(1, "collision", -0.01),
    ]
    (summary,) = metrics.summarize(rows)
    assert summary["trials"] == 4
    for outcome in (
        "success",
        "collision",
        "solver_failure_rejection",
        "policy_admissibility_stop",
    ):
        assert summary[outcome + "_percent"] == 25.0
    assert summary["mean_reach_time_seconds"] == 2 * protocol.DT
    assert rows[1]["reach_time_seconds"] is None


def test_undefined_reach_time_and_duplicate_detection():
    row = _example(4)
    assert next(metrics.summarize([row]))["mean_reach_time_seconds"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        list(metrics.summarize([row, row]))
