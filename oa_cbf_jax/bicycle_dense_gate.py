"""Fresh, outcome-independent trajectory references for dense bicycle refits.

Neither learned weights nor prediction calibration is fitted here. All reserved
parents, including early physical rejections, enter the trajectory-max ranks.
"""


from pathlib import Path


import numpy as np
from .bicycle_experiment import read, control_config
from .bicycle_random_density import geometry, geometry_metrics
from .dataset import sha256


SCHEMA='bicycle_dense_readout_gate_reference'
ROLE='reserved_dense_readout_trajectory_calibration'
SEED=2910044000
GROUPS=256
PHASE='gate_calibration'


def indices():return [12*cell+replica for cell in range(64) for replica in range(4)]


def validate_declaration(m):
    if (m.get('schema')!=SCHEMA or m.get('data_role')!=ROLE or m.get('groups')!=GROUPS
            or m.get('training_use') is not False or m.get('weight_fit_authorized') is not False
            or m.get('final_test') is not False or m['reservation_sha256']!=sha256(m['reservation'])):
        raise ValueError('Explicit fresh trajectory-reference reservation required')
    s=read(m['reservation']);info=s['models'][m['encoder']]
    if (s['seed']!=SEED or s['selected_generator_indices']!=indices() or s['quantile']!=.95 or s['parents_per_family']!=64
            or m['phase_order']!={PHASE:np.random.default_rng(SEED+144).permutation(GROUPS).tolist()}
            or any(m[k]!=s[k] for k in ('config','controller','runtime_guidance'))
            or m['policy_config']!=info['policy_config'] or m['prediction_fit_sha256']!=sha256(info['prediction_fit'])
            or m['weights_sha256']!=sha256(Path(info['bundle'])/'weights.msgpack')):
        raise ValueError('Changed trajectory-reference controller or role')
    for path,digest in s['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed trajectory-reference source')
    return s


def validate_source(source):
    source=Path(source);m=read(source/'manifest.json');validate_declaration(m);parents=read(source/'scenes.json')
    if len(parents)!=GROUPS or sha256(source/'scenes.json')!=m['scenes_sha256']:raise ValueError('Changed gate parents')
    for p,i in zip(parents,indices(),strict=True):
        if p['partition']!=PHASE or p['calibration_role']!='trajectory_gate' or p['weight_fit_allowed']:
            raise ValueError('Trajectory-reference role changed')
        for k,v in geometry(i,seed=SEED,schema=SCHEMA).items():np.testing.assert_array_equal(p[k],v)
    return m,parents
