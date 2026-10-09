"""Shared fp32 audit implementation."""

import numpy as np

from .route_audit import _exact, _outward, _sub, _square

def _norm2_interval(position,goal):
    delta=_sub(_exact(position),_exact(goal));squared=_square(delta)
    total=_outward(squared[0][...,0]+squared[0][...,1],squared[1][...,0]+squared[1][...,1])
    return _outward(np.sqrt(np.maximum(total[0],0.)),np.sqrt(np.maximum(total[1],0.)))

def goal_progress_interval(position,predicted_position,goal):
    """Enclose norm(position-goal)-norm(predicted_position-goal).

    Inputs are exact stored FP32 values. Outward rounding of each subtraction,
    square, addition, square root and final subtraction also encloses fused
    multiply-add variants. Unlike a fixed absolute/relative tolerance this
    bound follows the two distance magnitudes before their cancellation.
    """
    arrays=np.broadcast_arrays(*[np.asarray(v,float) for v in (position,predicted_position,goal)])
    if arrays[0].shape[-1:]!=(2,) or any(not np.isfinite(v).all() for v in arrays):raise ValueError('Finite XY arrays required')
    if any(not np.array_equal(v,v.astype(np.float32).astype(float)) for v in arrays):raise ValueError('Exact stored FP32 inputs required')
    return _sub(_norm2_interval(arrays[0],arrays[2]),_norm2_interval(arrays[1],arrays[2]))

def check_goal_progress(recorded,position,predicted_position,goal):
    low,high=goal_progress_interval(position,predicted_position,goal)
    value=np.asarray(recorded,float)
    if value.shape!=low.shape or not np.isfinite(value).all():raise AssertionError('Invalid recorded goal progress')
    if np.any((value<low)|(value>high)):raise AssertionError('Goal progress outside independent FP32 operation bounds')
    return low,high
