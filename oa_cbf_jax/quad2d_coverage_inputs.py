"""Fresh, balanced static geometry for diagnosing and expanding label coverage.

Previously evaluated physical parents remain excluded. Reusing a procedural
family for learning does not make that family an untouched generalization test.
"""


from collections import Counter


from dataclasses import asdict
from pathlib import Path

import numpy as np

from .comparison_contracts import physical_obstacle_scope
from .dataset import sha256, source_fingerprint


from .quad2d_control import FlightConfig
from .quad2d_static_inputs import read, prior_inventory, seeded_fields as multiscale_fields
from .quad2d_static_topologies import FAMILIES as LAYOUT_FAMILIES, seeded_fields as layout_fields


from .scenes import DIVERSE_FAMILIES

SCHEMA = 'quad2d_static_coverage_reservation'
FAMILIES = tuple(DIVERSE_FAMILIES) + tuple(LAYOUT_FAMILIES)
FRESH_ROLE = 'fresh_predictive_calibration'


def role_contract(role, seed, groups):
    result=dict(data_role=role,training_use=role in ('pilot','training'),
        weight_fit_authorized=role=='training',final_test=False)
    if role==FRESH_ROLE:
        result['calibration_reservation']=dict(schema='quad2d_coverage_fresh_calibration_v1',
            seed=seed,parents=groups,selection='All prespecified parents retained, no outcome filtering.',
            partition_interpretation='Legacy storage tags only; every parent is reserved from weight fitting.',
            intended_use='Fresh prediction fit/gate/audit on disjoint parents; no trajectory coverage or final promotion.')
    return result


def fields(seed, index, groups):
    if groups < 256 or groups % 32 or not 0 <= index < groups:
        raise ValueError('Balanced sixteen-family source with at least256 parents required')
    generator = multiscale_fields if index < groups // 2 else layout_fields
    result = generator(seed, index)
    result.pop('partition', None)
    result['coverage_stratum'] = 'multiscale' if index < groups // 2 else 'layout'
    return result


def partitions(groups, seed, role):
    if role not in ('pilot', 'training', 'comparison', FRESH_ROLE) or groups < 256 or groups % 32:
        raise ValueError('Known coverage role with balanced families required')
    if role == 'comparison':
        # At least32 reference parents per family keeps the existing95% finite
        # sample rank attainable. The remaining two thirds are forward trials.
        if groups < 1536 or groups % 96:
            raise ValueError('Comparison needs at least32 reference parents per family')
        return ['trajectory_calibration' if (i//8)%3 == 0 else 'controller_validation'
                for i in range(groups)]
    rng = np.random.default_rng(seed)
    result = [None] * groups
    for stratum in range(2):
        for family in range(8):
            indices = np.arange(stratum * groups//2 + family, (stratum+1)*groups//2, 8)
            for rank, index in enumerate(rng.permutation(indices)):
                result[index] = ('train' if rank < int(.7*len(indices)) else
                    'validation' if rank < int(.85*len(indices)) else 'development_calibration')
    if role == 'training' and Counter(result)['development_calibration'] < 200:
        raise ValueError('Training requires at least200 independent calibration parents')
    return result


def verify(directory):
    root = Path(directory)
    reservation, manifest, rows = [read(root/n) for n in ('reservation.json','manifest.json','scenes.json')]
    if (reservation['schema'] != SCHEMA or manifest['schema'] != SCHEMA
            or reservation['manifest_sha256'] != sha256(root/'manifest.json')
            or reservation['scenes_sha256'] != sha256(root/'scenes.json')
            or manifest['scenes_sha256'] != reservation['scenes_sha256']):
        raise ValueError('Changed coverage reservation')
    role, groups, seed = reservation['role'], manifest['groups'], manifest['seed']
    expected_parts = partitions(groups, seed, role)
    if (manifest['config'] != asdict(FlightConfig(stationary_obstacles=True))
            or manifest['families'] != list(FAMILIES) or len(rows) != groups
            or any(manifest.get(k)!=v for k,v in role_contract(role,seed,groups).items())):
        raise ValueError('Changed physical scope or reserved data role')
    for i, row in enumerate(rows):
        if any(row[k] != v for k,v in fields(seed,i,groups).items()) or row['partition'] != expected_parts[i]:
            raise ValueError('Changed prespecified parent, family or partition')
        physical_obstacle_scope('quad2d',row['obstacles'],row['obstacle_mask'])
    if len({r['group_id'] for r in rows}) != groups or len({r['seed'] for r in rows}) != groups:
        raise ValueError('Duplicate coverage parent')
    counts = dict(Counter(expected_parts))
    if reservation['partitions'] != counts:raise ValueError('Changed split counts')
    return dict(source=str(root.resolve()),role=role,parents=groups,partitions=counts,
        families=dict(Counter(r['family'] for r in rows)),
        manifest_sha256=sha256(root/'manifest.json'),scenes_sha256=sha256(root/'scenes.json'),
        seeded_geometry_static_truth_and_splits_verified=True,
        routes=dict(Counter(r['route']['status'] for r in rows)),benchmark_complete=False)
