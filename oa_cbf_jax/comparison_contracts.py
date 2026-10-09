"""Shared comparison contracts implementation."""

import numpy as np

def physical_obstacle_scope(dynamics, obstacles, mask):
    """Accept actual physical obstacles, never nominal scene labels."""
    if dynamics not in ('unicycle', 'quad2d', 'bicycle', 'quad3d'):
        raise ValueError('Unknown benchmark dynamics')
    obstacles = np.asarray(obstacles)
    mask = np.asarray(mask, bool)
    if obstacles.shape[:-1] != mask.shape or obstacles.shape[-1] != 5:
        raise ValueError('Expected physical [x,y,r,vx,vy] obstacles and mask')
    if not np.isfinite(obstacles[mask]).all():
        raise ValueError('Nonfinite physical obstacle')
    if dynamics != 'bicycle' and np.any(obstacles[..., 3:5][mask] != 0.):
        raise ValueError('Moving physical obstacles are allowed only for bicycle DPCBF')
    return True
