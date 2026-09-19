"""Causal obstacle-velocity forecasts for the flight guidance pilot.

Only identified raw tracks enter the observer. Current CBF rows still use raw
measurements; the estimate is used after the first step of nominal prediction.
The constant-velocity/bias assumption is explicit, not an acceleration bound.
"""
import numpy as np
import jax.numpy as jnp
from . import motion_observer


def update(memory, obstacles, mask, noise, dt, window=32):
    # Flight noise indices differ from the ground-robot observer's contract.
    mapped = jnp.zeros(6, noise.dtype).at[3].set(noise[4]).at[4].set(noise[5])
    return motion_observer.update(memory, obstacles, mask, mapped, dt, window)


def numpy_replay(raw, mask, noise, dt, window=32):
    """Independent NumPy reference using only the causal raw sensor sequence."""
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, bool)
    current = raw[0, :, :2].copy()
    previous = current.copy()
    estimates, bounds, lags, bads = [], [], [], []
    eps = np.finfo(np.float32).eps
    for k, reading in enumerate(raw):
        phase = k % window
        if k > 0 and phase == 0:
            previous, current = current.copy(), reading[:, :2].copy()
        anchor = previous if k >= window else current
        lag = window + phase if k >= window else k
        elapsed = max(lag, 1) * dt
        slope = (reading[:, :2] - anchor) / elapsed
        rounding = 16 * eps * (1 + abs(reading[:, :2]) + abs(anchor)) / elapsed
        error = .3 * noise[4] / elapsed + rounding
        sensor = np.full_like(slope, 1.15 * noise[5])
        lower = np.maximum(slope - error, reading[:, 3:5] - sensor)
        upper = np.minimum(slope + error, reading[:, 3:5] + sensor)
        bad = mask & (lag > 0) & np.any(lower > upper, axis=-1)
        use = mask & (lag > 0) & ~bad
        estimate = reading.copy()
        estimate[:, 3:5] = np.where(use[:, None], (lower + upper) / 2, reading[:, 3:5])
        estimate[~mask] = 0
        bound = np.where(use[:, None], (upper - lower) / 2, sensor)
        bound[~mask] = 0
        estimates.append(estimate); bounds.append(bound); lags.append(lag); bads.append(bad)
    return dict(obstacles=np.asarray(estimates), bound=np.asarray(bounds),
                lag=np.asarray(lags), inconsistent=np.asarray(bads))


def check_forecast_trace(data, guidance, dt=.05):
    """Reconstruct every recorded forecast; truth is used only for audit metrics."""
    reference = numpy_replay(data['observed_obstacles'], data['obstacle_mask'],
                             data['noise'], dt, guidance['motion_window'])
    for field, key in [('forecast_obstacles', 'obstacles'),
                       ('forecast_velocity_bound', 'bound')]:
        np.testing.assert_allclose(data[field], reference[key], rtol=1e-6, atol=2e-6)
    for field, key in [('forecast_lag', 'lag'), ('forecast_inconsistent', 'inconsistent')]:
        np.testing.assert_array_equal(data[field], reference[key])
    mask = data['obstacle_mask']; truth = data['true_obstacles'][mask, 3:5]
    residual = abs(data['forecast_obstacles'][:, mask, 3:5] - truth)
    excess = float(np.max(residual - data['forecast_velocity_bound'][:, mask], initial=0.))
    if excess > 3e-6:
        raise ValueError('Declared constant-velocity forecast interval missed physical truth')
    return dict(forecast_audit_passed=True, forecast_bound_excess=excess,
                forecast_inconsistent_tracks=int(data['forecast_inconsistent'].sum()))
