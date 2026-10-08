"""Reserved dense bicycle observations for prediction diagnosis and later learning.

No benchmark trace enters fitting. Both frozen encoders visit the same new
parents; their histories will be shared. Collection does not promote a model.
"""


from collections import Counter


from pathlib import Path


import numpy as np
from .bicycle_experiment import read, control_config


from .dataset import sha256


PHASE='dense_prediction_acquisition'
SCHEMA='bicycle_reserved_dense_prediction'
ROLE='reserved_dense_bicycle_prediction_acquisition'
SEED=2810032000
GROUPS=320
ROLES=(('train','none'),('train','none'),('validation','none'),
       ('development_calibration','prediction_fit'),('development_calibration','prediction_audit'))


def indices():
    return [cell*12+replica for cell in range(64) for replica in range(5)]


def validate_parent_role(p):
    replica=p['replica']
    if type(replica) is not int or not 0<=replica<5:
        raise ValueError('Unknown reserved replica')
    if ((p['partition'],p['calibration_role'])!=ROLES[replica]
            or p.get('weight_fit_allowed') is not (replica<2)):
        raise ValueError('Reserved parent role or weight-fit permission changed')


def validate_source(directory):
    root=Path(directory);m=read(root/'manifest.json');p=root/'scenes.json';parents=read(p)
    if (m.get('schema')!=SCHEMA or m.get('data_role')!=ROLE or m.get('groups')!=GROUPS
            or m.get('training_use') is not True or m.get('weight_fit_authorized') is not True
            or m.get('final_test') is not False or m['scenes_sha256']!=sha256(p)
            or m['reservation_sha256']!=sha256(m['reservation'])):
        raise ValueError('Unreserved dense acquisition source')
    reservation=read(m['reservation']);info=reservation['models'][m['encoder']]
    if (reservation['seed']!=SEED or reservation['selected_generator_indices']!=indices()
            or reservation['roles']!=[list(r) for r in ROLES] or len(parents)!=GROUPS
            or m['phase_order']!={PHASE:np.random.default_rng(SEED+144).permutation(GROUPS).tolist()}
            or any(m[k]!=reservation[k] for k in ('config','controller','runtime_guidance'))
            or m['policy_config']!=info['policy_config'] or m['prediction_fit_sha256']!=sha256(info['prediction_fit'])
            or m['weights_sha256']!=sha256(Path(info['bundle'])/'weights.msgpack')):
        raise ValueError('Changed dense acquisition contract')
    for p,i in zip(parents,indices(),strict=True):
        validate_parent_role(p)
        if p['replica']!=i%12 or p['group_id']!=f'{SCHEMA}:{SEED+i}' or p['seed']!=SEED+200000+i or p['layout_seed']!=SEED+(i//12)*32+i%12:
            raise ValueError('Changed reserved physical identity')
    if len({p['group_id'] for p in parents})!=GROUPS:raise ValueError('Duplicate parent')
    if (m['partitions']!=dict(Counter(p['partition'] for p in parents))
            or m['calibration_roles']!=dict(Counter(p['calibration_role'] for p in parents))):
        raise ValueError('Changed role accounting')
    return m,parents
