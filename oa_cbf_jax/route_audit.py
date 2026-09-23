"""Independent route audit, including FP32 arithmetic and nearest-segment ties.

An argmin is discontinuous: different FP32 operation fusion can pick different
near-equal segments. On a failed ordinary check we admit only a segment whose
outward-rounded distance interval overlaps the smallest upper distance bound.
The target and committed cursor must still match that SAME segment at the
original tolerance. No physical or CBF audit tolerance is changed.
"""
import numpy as np


def _outward(lower, upper):
    return (np.nextafter(np.asarray(lower, np.float32), np.float32(-np.inf)).astype(float),
            np.nextafter(np.asarray(upper, np.float32), np.float32(np.inf)).astype(float))


def _exact(value):
    a = np.asarray(value, float)
    return a, a


def _add(a, b): return _outward(a[0]+b[0], a[1]+b[1])
def _sub(a, b): return _outward(a[0]-b[1], a[1]-b[0])


def _mul(a, b):
    products = np.stack((a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    return _outward(products.min(axis=0), products.max(axis=0))


def _square(a):
    low = np.where((a[0]<=0)&(a[1]>=0), 0., np.minimum(a[0]**2, a[1]**2))
    return _outward(low, np.maximum(a[0]**2, a[1]**2))


def _sum2(a): return _add((a[0][:, 0], a[1][:, 0]), (a[0][:, 1], a[1][:, 1]))


def distance_intervals(position, points):
    """Enclose FP32 point-to-segment squared distances using operation bounds.

    Endpoints are exact values of the stored inputs. Every arithmetic operation
    rounds both endpoints outward by one FP32 neighbor, also enclosing fused
    multiply/add variants. This is conservative numerical ambiguity accounting,
    not uncertainty in the physical robot or obstacle positions.
    """
    p = _exact(points[:-1]); v = _sub(_exact(points[1:]), p)
    length2 = _sum2(_square(v))
    length = _outward(np.sqrt(np.maximum(length2[0], 0.)), np.sqrt(np.maximum(length2[1], 0.)))
    denominator = _square(length); denominator = tuple(np.maximum(a, 1e-12) for a in denominator)
    numerator = _sum2(_mul(_sub(_exact(position), p), v))
    inverse = _outward(1/denominator[1], 1/denominator[0]); fraction = _mul(numerator, inverse)
    fraction = tuple(np.clip(a, 0., 1.)[:, None] for a in fraction)
    nearest = _add(p, _mul(fraction, v)); delta = _sub(nearest, _exact(position))
    low, high = _sum2(_square(delta))
    return np.maximum(low, 0.), high


def check_transition(x, points, mask, cursor, recorded_target, recorded_cursor, active, tolerance=1e-5):
    from .quad2d_mpc_experiment import numpy_target
    # Replay the exact stored values in one consistent reference precision.
    # Mixed FP64 observations / FP32 route lengths introduce an extra rounding
    # path in projection and interpolation, unrelated to the recorded kernel.
    # Keep the same absolute tolerance and explicit nearest-segment intervals.
    x = np.asarray(x, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    target, proposed = numpy_target(x, points, mask, cursor)
    def matches(t, p):
        return np.max(np.abs(recorded_target-t))<=tolerance and abs(recorded_cursor-(p if active else cursor))<=tolerance
    if matches(target, proposed): return False
    # A long segment can accumulate more than1e-5 of absolute interpolation
    # error even when its nearest-segment choice is unambiguous. The runtime
    # route kernel uses FP32, so also replay consistent FP32 arithmetic for
    # exactly representable stored inputs. Require BOTH target and committed
    # cursor from that same replay; keep the original comparison tolerance.
    x32, points32, cursor32 = np.asarray(x, np.float32), np.asarray(points, np.float32), np.float32(cursor)
    if (np.array_equal(x, x32) and np.array_equal(points, points32) and cursor == cursor32):
        target32, proposed32 = numpy_target(x32, points32, mask, cursor32)
        if matches(target32, proposed32): return True
    vectors = points[1:]-points[:-1]; valid = mask[:-1]&mask[1:]
    lengths = np.where(valid, np.linalg.norm(vectors, axis=1), 0.); cumulative = np.r_[0., np.cumsum(lengths)]
    fractions = np.clip(np.sum((x[:2]-points[:-1])*vectors, axis=1)/np.maximum(lengths**2, 1e-12), 0., 1.)
    projected = cumulative[:-1]+fractions*lengths
    eligible = valid&(projected>=cursor-.05)&(projected<=cursor+1.)
    low, high = distance_intervals(x[:2], points)
    candidates = np.flatnonzero(eligible&(low<=np.min(high[eligible], initial=np.inf)))
    for index in candidates:
        t, p = numpy_target(x, points, mask, cursor, projection_index=index)
        if matches(t, p): return True
    raise ValueError('Route target/cursor inconsistent with every numerically admissible nearest segment')
