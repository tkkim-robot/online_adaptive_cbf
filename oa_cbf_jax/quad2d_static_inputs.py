"""Quad2d static inputs functions and shared contracts."""

from collections import Counter

import json

from pathlib import Path

import numpy as np

from .comparison_contracts import physical_obstacle_scope

from .io import sha256

from .scenes import multiscale_scenes_scene as scene

from .quad2d_control import FlightConfig, flight_config_from_contract

from .scenes import DIVERSE_FAMILIES

SCHEMA = 'quad2d_static_source_reservation'

METHODS = ['gat', 'matched_fc', 'fixed_low', 'fixed_high', 'optimal_decay',
           'optimal_decay_qp', 'barriernet']

def read(path):
    return json.loads(Path(path).read_text())

def partitions(groups, seed, role):
    if role not in ('pilot', 'training', 'comparison') or groups < 64 or groups % 8:
        raise ValueError('Known role and balanced full families required')
    if role == 'comparison':
        if groups % 24:
            raise ValueError('Comparison requires equal family counts and a one-third calibration split')
        return ['trajectory_calibration' if (i // 8) % 3 == 0 else 'controller_validation'
                for i in range(groups)]
    rng = np.random.default_rng(seed)
    assignments = {}
    for family in DIVERSE_FAMILIES:
        for rank, index in enumerate(rng.permutation(groups // 8)):
            assignments[family, int(index)] = ('train' if rank < int(.7 * groups / 8)
                else 'validation' if rank < int(.85 * groups / 8) else 'development_calibration')
    result = [assignments[DIVERSE_FAMILIES[i % 8], i // 8] for i in range(groups)]
    if role == 'training' and Counter(result)['development_calibration'] < 200:
        raise ValueError('Full training needs at least 200 independent calibration parents')
    return result

def seeded_fields(seed, index):
    """Reconstruct physical design independently of saved routes and outcomes."""
    local_seed = seed * 100000 + index
    family = DIVERSE_FAMILIES[index % 8]
    rng = np.random.default_rng(local_seed + 103)
    geometry = scene(local_seed, family)
    state = np.r_[geometry.initial_state[:2], rng.uniform(-.1, .1),
                  rng.uniform(-.25, .25, 2), rng.uniform(-.15, .15)].astype(np.float32)
    obstacles = geometry.obstacles.astype(np.float32)
    obstacles[:, 3:5] = 0.
    scale = float(rng.choice([0., .5, 1., 2.]))
    noise = scale * np.array([.015, .01, .015, .015, .02, .02, .008], np.float32)
    return dict(group_id=f'quad2d_multiscale_v1:{family}:{local_seed}', family=family,
                seed=local_seed, initial_state=state.tolist(), goal=geometry.goal.astype(np.float32).tolist(),
                obstacles=obstacles.tolist(), obstacle_mask=geometry.obstacle_mask.tolist(), noise=noise.tolist())

def verify(directory):
    root = Path(directory)
    reservation = read(root / 'reservation.json')
    if reservation.get('schema') == 'quad2d_static_coverage_reservation':
        from .quad2d_training import verify as verify_coverage
        return verify_coverage(root)
    manifest, rows = read(root / 'manifest.json'), read(root / 'scenes.json')
    if reservation['schema'] != SCHEMA or reservation['manifest_sha256'] != sha256(root / 'manifest.json'):
        raise ValueError('Changed static source manifest')
    if manifest['scenes_sha256'] != sha256(root / 'scenes.json') or reservation['scenes_sha256'] != manifest['scenes_sha256']:
        raise ValueError('Changed static source rows')
    if flight_config_from_contract(manifest['config']) != FlightConfig(stationary_obstacles=True):
        raise ValueError('Changed static physical/sensor/task contract')
    role, groups, seed = reservation['role'], manifest['groups'], manifest['seed']
    expected_parts = partitions(groups, seed, role)
    if (len(rows) != groups or manifest['data_role'] != role
            or manifest['weight_fit_authorized'] != (role == 'training')
            or manifest['training_use'] != (role != 'comparison') or manifest['final_test'] is not False):
        raise ValueError('Changed source role or parent denominator')
    for i, row in enumerate(rows):
        expected = seeded_fields(seed, i)
        if any(row[k] != v for k, v in expected.items()) or row['partition'] != expected_parts[i]:
            raise ValueError('Source differs from prespecified seed/partition rule')
        physical_obstacle_scope('quad2d', row['obstacles'], row['obstacle_mask'])
    counts = dict(Counter(expected_parts))
    if reservation['partitions'] != counts:
        raise ValueError('Changed reserved partition counts')
    if role == 'comparison' and (reservation['methods'] != METHODS or reservation['steps'] != 1600):
        raise ValueError('Incomplete baseline or episode-budget reservation')
    return dict(source=str(root.resolve()), role=role, parents=groups, partitions=counts,
                manifest_sha256=sha256(root / 'manifest.json'), scenes_sha256=sha256(root / 'scenes.json'),
                seeded_geometry_static_truth_and_splits_verified=True,
                routes=dict(Counter(r['route']['status'] for r in rows)), benchmark_complete=False)


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


import hashlib


from .quad2d_ood_scenes import FAMILIES as TEMPLATES, geometry

FAMILIES = ('alternating_gates', 'zigzag_channel', 'offset_rooms',
            'nested_open_boxes', 'interleaved_rows', 'three_lanes',
            'narrow_gate', 'large_disks')

def static_topologies_seeded_fields(seed, index):
    local = seed*100000+index
    x, goal, obstacles, mask = geometry(local, TEMPLATES[index % 8])
    obstacles[:, 3:5] = 0.
    rng = np.random.default_rng(local+233)
    noise = float(rng.choice([0., .5, 1., 2.]))*np.array(
        [.015, .01, .015, .015, .02, .02, .008], np.float32)
    fingerprint = hashlib.sha256(b''.join(a.tobytes() for a in
        (x, goal, obstacles, mask, noise))).hexdigest()
    family = FAMILIES[index % 8]
    return dict(group_id=f'quad2d_static_topology:{family}:{local}',
        family=family, template=TEMPLATES[index % 8], seed=local,
        partition='frozen_forward_topology', initial_state=x.tolist(), goal=goal.tolist(),
        obstacles=obstacles.tolist(), obstacle_mask=mask.tolist(), noise=noise.tolist(),
        solvability='unknown', scene_fingerprint=fingerprint)


def validate_parent(parent):
    total=parent['waypoint_count'];goals=np.asarray(parent['waypoint_goals']);r=parent['waypoint_routes']
    if isinstance(total,bool) or not isinstance(total,int) or not 1<=total<=3 or goals.shape!=(3,2) or np.shape(r['points'])!=(3,64,2) or np.shape(r['mask'])!=(3,64) or np.shape(r['ready'])!=(3,):raise ValueError('Invalid ordered flight task')
    if not np.array_equal(np.asarray(parent['goal'],np.float32),goals[total-1].astype(np.float32)):raise ValueError('Final waypoint mismatch')

def numpy_arrived(x,goal,config,noise=None):
    e=np.zeros(4) if noise is None else 1.15*np.asarray(noise,float)[:4]
    return bool(np.linalg.norm(x[:2]-goal)+np.sqrt(2)*e[0]<=config.goal_tolerance
        and np.linalg.norm(x[3:5])+np.sqrt(2)*e[2]<=config.terminal_speed
        and abs(x[2])+e[1]<=config.terminal_pitch and abs(x[5])+e[3]<=config.terminal_pitch_rate)
