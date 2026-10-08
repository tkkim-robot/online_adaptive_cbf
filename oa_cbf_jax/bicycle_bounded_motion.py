"""Bounded causal velocity estimate for constant-velocity obstacle tracks.

Only observations, elapsed time and declared sensor ranges enter the estimate.
The constant position bias cancels in a secant; independent position innovations
and floating-point rounding remain. This is a sensor bound, not a CBF guarantee.
No deployed controller uses this diagnostic module yet.
"""
import numpy as np


EPS = float(np.finfo(np.float32).eps)


def numpy_estimate(current, past, mask, noise, elapsed):
    """Independent obstacle-loop implementation, including the output rounding."""
    current = np.asarray(current); past = np.asarray(past); mask = np.asarray(mask, bool)
    output = current.copy(); radius = np.zeros(len(mask)); raw_bounds = np.zeros(len(mask))
    used = np.zeros(len(mask), bool); contradiction = np.zeros(len(mask), bool)
    if not np.isfinite(elapsed) or elapsed < 0 or np.any(np.asarray(noise) < 0):
        raise ValueError('Nonnegative finite history and noise required')
    for i in np.flatnonzero(mask):
        raw = current[i, 3:5].astype(float)
        rb = 1.15*float(noise[4])+8*EPS*(1+float(np.hypot(*raw)))
        raw_bounds[i] = radius[i] = rb
        if elapsed <= 0: continue
        now = current[i, :2].astype(float); before = past[i].astype(float)
        secant = (now-before)/float(elapsed)
        rounded_endpoints = 8*EPS*(1+float(np.hypot(*now))+float(np.hypot(*before)))
        sb = (.3*float(noise[3])+rounded_endpoints)/float(elapsed)+8*EPS*(1+float(np.hypot(*secant)))
        contradiction[i] = np.hypot(*(secant-raw)) > rb+sb
        if noise[4] > 0 and sb <= rb and not contradiction[i]:
            used[i] = True; output[i, 3:5] = secant.astype(current.dtype); radius[i] = sb
    return dict(obstacles=output, radius=radius, raw_radius=raw_bounds, used=used, contradiction=contradiction)


def held_offset(current, past, mask, noise, elapsed):
    """Development branch only: hold a causally estimated bias correction.

    Unlike a sliding secant, a fixed correction preserves the original velocity
    innovation differences used by the next-observation margin. Reserve another
    0.3*velocity noise for the difference from the acquisition innovation to any
    future innovation. Reject corrections that cannot retain the original full
    absolute sensor error allowance. No physical value enters this calculation.
    """
    r = numpy_estimate(current, past, mask, noise, elapsed)
    offset = (r['obstacles'][:, 3:5].astype(float)-np.asarray(current)[:, 3:5]).astype(np.float32)
    rounding = 16*EPS*(1+np.linalg.norm(np.asarray(current)[:, 3:5], axis=-1)+np.linalg.norm(offset, axis=-1))
    future_bound = r['radius']+.3*float(noise[4])+rounding
    used = r['used'] & (future_bound <= 1.15*float(noise[4]))
    offset[~used] = 0.
    return dict(offset=offset, used=used, future_radius=np.where(used, future_bound, r['raw_radius']))
