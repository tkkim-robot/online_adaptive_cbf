"""Prespecified stationary flight inputs for matched learning and comparison.

Geometry generators keep their historical names. Actual physical obstacle
velocities are zero, including under the saved noisy-sensor contract. Failed
routes and every declared parent remain in their original denominators.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .comparison_contracts import physical_obstacle_scope
from .dataset import sha256
from .io import write_json
from .multiscale_scenes import scene
from .quad2d_control import FlightConfig, flight_config_from_contract
from .quad2d_data import prepare as prepare_geometry
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


def prior_inventory(artifacts, ids, own):
    records = []
    paths = sorted(Path(artifacts).glob('experiments/*/scenes.json'))
    paths += sorted(Path(artifacts).glob('datasets/*/manifest.json'))
    for path in paths:
        if path.parent.resolve() == own.resolve():
            continue
        data = read(path)
        rows = data.get('groups', []) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            continue
        previous = {r.get('group_id', r.get('scene', {}).get('scene_id'))
                    for r in rows if isinstance(r, dict)} - {None}
        if ids & previous:
            raise ValueError('Previously used physical parent: ' + str(path))
        if previous:
            records.append(dict(path=str(path.resolve()), sha256=sha256(path), parents=len(previous)))
    return records


def verify(directory):
    root = Path(directory)
    reservation = read(root / 'reservation.json')
    if reservation.get('schema') == 'quad2d_static_coverage_reservation':
        from .quad2d_coverage_inputs import verify as verify_coverage
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


def prepare(output, groups=768, seed=9741, role='comparison', workers=1, artifacts='artifacts'):
    expected_parts = partitions(groups, seed, role)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed * 100000 + groups >= 2**32:
        raise ValueError('Nonnegative seed within the physical PRNG range required')
    root = Path(output)
    if root.exists():
        raise ValueError('Use a fresh reservation directory')
    ids = {f'quad2d_multiscale_v1:{DIVERSE_FAMILIES[i % 8]}:{seed * 100000 + i}'
           for i in range(groups)}
    inventory = prior_inventory(artifacts, ids, root)
    prepare_geometry(root, groups, seed, workers, stationary_obstacles=True)
    rows, manifest = read(root / 'scenes.json'), read(root / 'manifest.json')
    for row, partition in zip(rows, expected_parts):
        row['partition'] = partition
    write_json(root / 'scenes.json', rows)
    manifest.update(data_role=role, training_use=role != 'comparison', weight_fit_authorized=role == 'training',
                    stage='prespecified_static_' + role, scenes_sha256=sha256(root / 'scenes.json'),
                    selection='Consecutive seeds, balanced families, all failed routes retained; no model outcomes inspected.')
    write_json(root / 'manifest.json', manifest)
    reservation = dict(schema=SCHEMA, role=role, manifest_sha256=sha256(root / 'manifest.json'),
                       scenes_sha256=sha256(root / 'scenes.json'), partitions=dict(Counter(expected_parts)),
                       prior_sources=inventory, methods=METHODS if role == 'comparison' else None,
                       steps=1600 if role == 'comparison' else None,
                       noise='Saved parent noise ranges, identical innovations/latent seeds across methods.',
                       statistical_unit='Physical parent; family-stratified paired bootstrap.',
                       benchmark_complete=False, whole_goal_complete=False)
    write_json(root / 'reservation.json', reservation)
    return verify(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--groups', type=int, default=768)
    p.add_argument('--seed', type=int, default=9741)
    p.add_argument('--role', choices=['pilot', 'training', 'comparison'], default='comparison')
    p.add_argument('--workers', type=int, default=1)
    print(json.dumps(prepare(**vars(p.parse_args()))), flush=True)
