"""Observed local constraints for a bounded, explicitly uncalibrated pilot.

The physical world and route stay intact. Graph/CBF/predictive witnesses use
the same current neighborhood, held during each guidance prediction. Keeping
original padded slots makes the all-obstacle default numerically unchanged.
"""
import numpy as np
import jax.numpy as jnp

from .obstacle_selection import contract, nearest_obstacles, nearest_numpy


def neighborhood_contract(count):
    result = contract(count)
    result.update(schema='quad2d_observed_neighborhood_diagnostic_v1',
        storage='Original padded slots; only the graph and controller mask change.',
        prediction='Hold the current observed neighborhood over the unchanged guidance horizon.',
        route='Original observed route unchanged.',
        calibration_coverage_valid=False, final_test=False)
    return result


def neighborhood_mask(position, obstacles, mask, count):
    _, present, indices = nearest_obstacles(position, obstacles, mask, count)
    # Padded -1 entries must not clear the valid entry at index zero.
    selected = jnp.zeros(mask.shape, jnp.int32).at[jnp.maximum(indices, 0)].add(present.astype(jnp.int32))
    return mask & (selected > 0)


def numpy_neighborhood_mask(position, obstacles, mask, count):
    _, present, indices = nearest_numpy(position, obstacles, mask, count)
    selected = np.zeros_like(mask, dtype=bool)
    selected[indices[present]] = True
    return selected


def audit_neighborhood(data, count):
    actual = np.asarray(data['controller_obstacle_mask'])
    expected = np.stack([numpy_neighborhood_mask(x, obs, data['obstacle_mask'], count)
        for x, obs in zip(data['observed_state'], data['observed_obstacles'])])
    np.testing.assert_array_equal(actual, expected)
    return dict(neighborhood_audit_passed=True, neighborhood_observations=len(actual),
        full_world_obstacles=int(np.sum(data['obstacle_mask'])),
        maximum_controller_obstacles=int(actual.sum(-1).max(initial=0)))
