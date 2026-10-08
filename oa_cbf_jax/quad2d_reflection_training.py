"""Paired, actually simulated flight orientations within one parent bootstrap.

No outcome is copied under reflection. Validation and calibration stay outside
the augmentation; the original TRAIN-only normalization and weights are kept.
"""
from pathlib import Path
import hashlib
import jax
import jax.numpy as jnp
import numpy as np

from .dataset import load_dataset, sha256
from .quad2d_static_inputs import read

KEYS=('features','node_mask','gains','target','target_mask','events','event_mask')
PREFIX='reflected_'


def check_training_rows(original,reflected,expected):
    if (original['group_id'].tolist()!=expected or reflected['group_id'].tolist()!=expected
            or len(set(expected))!=len(expected)):
        raise ValueError('Reflection must preserve unique TRAIN parent order')
    for data in (original,reflected):
        if not np.all(data['partition']=='train'):
            raise ValueError('Only TRAIN orientations may enter fitting')
    np.testing.assert_array_equal(original['gains'],reflected['gains'])
    for key in KEYS:
        if original[key].shape!=reflected[key].shape or original[key].dtype!=reflected[key].dtype:
            raise ValueError('Changed paired feature/target layout: '+key)


def load_pair(original_dataset,reflected_dataset,original):
    a,b=map(Path,(original_dataset,reflected_dataset));ma,mb=read(a/'manifest.json'),read(b/'manifest.json')
    if any(m.get('weight_fit_authorized') is not True or m.get('final_test') is not False
            or m['schema']!='oa_cbf_quad2d_guided_hurdle_v1' for m in (ma,mb)):
        raise ValueError('Audited TRAIN-authorized flight data required')
    for key in ('groups','config','controller','queries','replicas','horizon_steps','targets','events',
            'graph_features','graph_schema','capacity','route_capacity','gain_domain'):
        if ma[key]!=mb[key]:raise ValueError('Different paired learning contract: '+key)
    for path in (a,b):
        audit=read(path/'independent_replay.json');done=read(path/'complete.json')
        for flag in ('audit_passed','all_collision_bound_branches_audited',
                'all_observed_graph_features_independently_checked','all_guidance_approvals_checked',
                'all_physical_task_target_inputs_checked'):
            if audit.get(flag) is not True:raise ValueError('Incomplete paired physical audit')
        for key,file in [('manifest_sha256','manifest.json'),('index_sha256','index.json')]:
            if done[key]!=sha256(path/file) or audit[key]!=done[key]:
                raise ValueError('Changed paired labels')
    report=read(b.parent/'review.json')
    if (not report['actual_new_labels_not_inherited'] or not report['original_parent_partitions_preserved']
            or report['dataset']!=str(b.resolve()) or report['manifest_sha256']!=sha256(b/'manifest.json')
            or report['index_sha256']!=sha256(b/'index.json')
            or report['audit_sha256']!=sha256(b/'independent_replay.json')):
        raise ValueError('Unbound actual reflected outcome review')
    spec=read(b.parent/'protocol.json')
    if spec['original_dataset']!=str(a.resolve()):raise ValueError('Different original donor dataset')
    for path,digest in spec['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed reflection donor binding')
    reflected=load_dataset(b,'train')
    expected=[g['group_id'] for g in ma['groups'] if g['partition']=='train']
    check_training_rows(original,reflected,expected)
    paths=[p/f for p in (a,b) for f in ('manifest.json','index.json','independent_replay.json','complete.json')]
    paths += [b.parent/'review.json',b.parent/'protocol.json',b.parent/'qualification.json']
    proof=dict(schema='quad2d_actual_paired_orientation_v1',original_dataset=str(a.resolve()),
        reflected_dataset=str(b.resolve()),bindings={str(p.resolve()):sha256(p) for p in paths},
        independent_training_parents=len(expected),orientations_per_parent=2,
        training_parent_ids_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
        sampling='Original parent bootstrap, one Bernoulli(0.5) orientation per sampled parent and optimizer update; all gain/replica branches share that orientation and its ACTUAL targets/masks/events.',
        normalization='Unchanged original TRAIN-only target units, shared between encoders.',
        weighting='Unchanged original TRAIN gain-opportunity weights, shared between encoders.',
        validation='Original validation only for checkpoint selection; reflected validation is secondary, never independent added parents.',
        calibration='Original reserved parents excluded from fitting; no reflected calibration enters fitting.',
        seed_rule='fold_in(PRNGKey(member_seed+37001), optimizer_step)',copied_targets=False)
    return reflected,proof


def pack(original,reflected):
    if set(original)!=set(KEYS) or set(reflected)!=set(KEYS):
        raise ValueError('Paired augmentation only supports the original flight heads')
    return dict(original,**{PREFIX+k:reflected[k] for k in KEYS})


def orientation_mask(count,seed,step):
    return jax.random.bernoulli(jax.random.fold_in(jax.random.PRNGKey(seed),step),.5,(count,))


def select_orientation(batch,seed,step):
    mirrored=orientation_mask(batch['features'].shape[0],seed,step)
    return {key:jnp.where(mirrored.reshape((-1,)+(1,)*(batch[key].ndim-1)),
                         batch[PREFIX+key],batch[key]) for key in KEYS}
