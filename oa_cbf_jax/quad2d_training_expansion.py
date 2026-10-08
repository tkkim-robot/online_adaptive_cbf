"""Append audited independent TRAIN parents without changing held-out roles."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from .dataset import load_dataset, sha256
from .quad2d_reflection_training import load_pair, check_training_rows
from .quad2d_static_inputs import read


def disjoint_reservations(primary, additional):
    for manifest in (primary, additional):
        if (manifest.get('weight_fit_authorized') is not True or manifest.get('final_test') is not False
                or manifest.get('data_role') != 'training'):
            raise ValueError('Only reserved training datasets may supply extra TRAIN rows')
        groups = manifest['groups']
        if len({r['group_id'] for r in groups}) != len(groups) or len({r['seed'] for r in groups}) != len(groups):
            raise ValueError('Duplicate reserved parent/seed')
        if any(r['partition'] not in ('train','validation','development_calibration') for r in groups):
            raise ValueError('Unknown parent role')
    for field in ('group_id','seed'):
        if {r[field] for r in primary['groups']} & {r[field] for r in additional['groups']}:
            raise ValueError('Additional data overlaps an existing training or held-out parent: '+field)
    for key in ('schema','config','controller','queries','replicas','horizon_steps','targets','events',
                'gain_domain','graph_schema','graph_features','capacity','route_capacity'):
        if primary[key] != additional[key]:
            raise ValueError('Changed extra-data contract: '+key)


def concatenate_train(primary, additional, expected_primary, expected_additional):
    for data, expected in ((primary,expected_primary),(additional,expected_additional)):
        if data['group_id'].tolist() != expected or not np.all(data['partition']=='train'):
            raise ValueError('Exact reserved TRAIN order required; holdouts cannot be appended')
    if len(set(expected_primary+expected_additional)) != len(expected_primary+expected_additional):
        raise ValueError('Repeated TRAIN identity')
    if set(primary) != set(additional):
        raise ValueError('Different raw training array fields')
    for key in primary:
        a,b = primary[key],additional[key]
        if a.shape[1:] != b.shape[1:] or (a.dtype != b.dtype and not (a.dtype.kind == b.dtype.kind == 'U')):
            raise ValueError('Changed TRAIN array shape or dtype: '+key)
    return {k:np.concatenate((primary[k],additional[k]),axis=0) for k in primary}


def extend(dataset, extra_dataset, extra_reflection, original, original_reflected):
    from .quad2d_static_learning import validate_training_dataset
    a,b,c = map(Path,(dataset,extra_dataset,extra_reflection))
    ma,mb = read(a/'manifest.json'),read(b/'manifest.json')
    disjoint_reservations(ma,mb)
    validate_training_dataset(b,b.parent/'data_review.json',dict(ma,dataset=str(a.resolve())))
    report = read(b.parent/'review.json')
    if (report.get('schema') != 'quad2d_shared_policy_training_collection_v1'
            or not report.get('all_context_selections_recomputed')
            or not report.get('all_parent_failures_retained')
            or report['data_review_sha256'] != sha256(b.parent/'data_review.json')):
        raise ValueError('Reviewed, outcome-independent shared policy collection required')
    new = load_dataset(b,'train')
    reflected,pair = load_pair(b,c,new)
    expected_a = [r['group_id'] for r in ma['groups'] if r['partition']=='train']
    expected_b = [r['group_id'] for r in mb['groups'] if r['partition']=='train']
    raw = concatenate_train(original,new,expected_a,expected_b)
    mirrored = concatenate_train(original_reflected,reflected,expected_a,expected_b)
    check_training_rows(raw,mirrored,expected_a+expected_b)
    paths = [p/f for p in (a,b,c) for f in ('manifest.json','index.json','independent_replay.json','complete.json')]
    paths += [b.parent/'data_review.json',b.parent/'review.json',b.parent/'protocol.json',
              c.parent/'review.json',c.parent/'protocol.json',c.parent/'qualification.json']
    expected = expected_a+expected_b
    proof = dict(schema='quad2d_shared_train_expansion_v1',primary_dataset=str(a.resolve()),
        additional_dataset=str(b.resolve()),additional_manifest_sha256=sha256(b/'manifest.json'),
        additional_reflection=str(c.resolve()),additional_pair=pair,
        bindings={str(p.resolve()):sha256(p) for p in paths},
        primary_train_parents=len(expected_a),additional_train_parents=len(expected_b),
        independent_train_parents=len(expected),orientations_per_parent=2,
        training_parent_ids_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
        additional_partitions=dict(Counter(r['partition'] for r in mb['groups'])),
        normalization='Original primary TRAIN only; unchanged units for both encoders.',
        selection='Original primary validation only; additional validation reported separately.',
        calibration='Neither primary nor additional calibration parents enter weight fitting.',
        weighting='Same gain-opportunity rule recomputed on the shared TRAIN union, followed by same-seed parent bootstrap.',
        sampling='Shared TRAIN union; independent Bernoulli0.5 original/actually simulated reflection per parent per update.',
        all_roles_disjoint=True,models_share_labels=True)
    return raw,mirrored,proof
