"""Fail-closed checks for encoder-only ablations and physical benchmark scope."""

import copy
import numpy as np


def matched_training_settings(gat, fc):
    """Compare the training treatment, including splits, sampling and precision.

    Device identity is execution provenance: a separately qualified CPU/GPU
    choice does not change the ablation. Optimizer, precision and all model/data
    settings remain compared. The training runner records backend qualification.
    """
    left, right = copy.deepcopy(gat), copy.deepcopy(fc)
    if left['architecture']['encoder'] != 'gat' or right['architecture']['encoder'] != 'matched_fc':
        raise ValueError('Expected GAT and its encoder-only matched FC ablation')
    left['architecture'].pop('encoder')
    right['architecture'].pop('encoder')
    left.pop('device',None)
    right.pop('device',None)
    if left != right:
        differences = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
        raise ValueError('Unmatched training settings: ' + ', '.join(differences))
    return True


def matched_controller_settings(gat, fc):
    """Canonical descriptors exclude weights/fit values, include fit protocol.

    Model-specific fitted calibration values need not be equal. Their fitting
    parents and procedure, and every surrounding control choice, must match.
    """
    required = ('dynamics', 'observation', 'features', 'controller', 'guidance',
                'routing', 'gain_domain', 'initial_gain', 'fallback',
                'query_cadence', 'calibration_protocol', 'evaluation_inputs_sha256')
    for side in (gat, fc):
        missing = set(required) - side.keys()
        if missing:
            raise ValueError('Missing comparison contract: ' + ', '.join(sorted(missing)))
    if gat != fc:
        differences = sorted(k for k in gat.keys() | fc.keys() if gat.get(k) != fc.get(k))
        raise ValueError('Unmatched controller settings: ' + ', '.join(differences))
    return True


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
