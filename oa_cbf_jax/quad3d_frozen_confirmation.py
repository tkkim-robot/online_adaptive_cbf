"""New randomized scenes for a policy frozen before any new outcomes.

This changes neither the physical distribution nor either learned policy. The
existing all-method development cohort stays separate. New native comparisons
must use these same parents; old native outcomes cannot fill their cells.
"""


from functools import lru_cache


from pathlib import Path

from .dataset import sha256


from .quad3d_learning_contract import read
from . import quad3d_density_study as density

FROZEN = Path('artifacts/experiments/quad3d_failure_requery_navigation')
GEOMETRY_SEED = 2609301000
STATE_SEED = 2609304000
NAMESPACE = 'quad3d_frozen_density_confirmation'
GROUPS = 2048


def reservation():
    return dict(schema=NAMESPACE,geometry_seed=GEOMETRY_SEED,state_seed=STATE_SEED,
        parents=GROUPS,families=['random_field','paired_clusters','staggered_layers','random_pockets'],
        obstacle_counts=[24,36,48,60],noise_levels=[0.,.5,1.,2.],replicas=32,
        primary_endpoint='Paired complete-cohort goal success; report collision/state exits and all negative strata.',
        stopping_rule='Run every reserved parent for both methods. No outcome-dependent early stopping, resampling, route retries, model/gate fitting or policy changes.',
        baseline_rule='Old native outcomes are on different scenes and must not be used as confirmation baseline results.',
        use='Prospective check of the frozen policy; development and confirmation results remain separate.')


@lru_cache(maxsize=GROUPS)
def fresh_parent(index,initial_gain):
    from .quad2d_random_density import fields
    row=fields(index,seed=GEOMETRY_SEED,schema=NAMESPACE+'_geometry')
    return density.project_parent(row,index,initial_gain,seed=STATE_SEED,namespace=NAMESPACE)


def check_parents(parents,raw,spec):
    from .comparison_contracts import physical_obstacle_scope
    indices=spec['pilot_indices'] if spec['pilot'] else list(range(GROUPS))
    if len(parents)!=len(indices) or len(raw)!=len(indices):raise ValueError('Incomplete frozen reservation')
    for i,p,r in zip(indices,parents,raw,strict=True):
        expected=fresh_parent(i,spec['policy_config']['initial_gain'])
        if spec['pilot']:expected=dict(expected,study_slot=0)
        if r!=expected or any(p[k]!=v for k,v in r.items()):raise ValueError('Changed prospective parent')
        physical_obstacle_scope('quad3d',p['obstacles'],p['mask'])
    if [sum(p['study_slot']==i for p in parents) for i in range(4)]!=spec['slot_counts']:
        raise ValueError('Changed balanced worker allocation')


def verify_source(root,spec=None):
    from .quad3d_policy_experiment import INPUT_SUPPORT_SCHEMA
    root=Path(root);s=read(root/'manifest.json') if spec is None else spec
    if (s['schema']!=INPUT_SUPPORT_SCHEMA or not s.get('prospective_confirmation')
        or not s['input_box_admission'] or not s['failure_requery'] or s['training_use']
        or s['weight_fit_authorized'] or s['reference_parents']!=0):
        raise ValueError('Wrong frozen confirmation contract')
    if s['origin']!=str((FROZEN/'inputs').resolve()):raise ValueError('Wrong frozen policy')
    original=read(FROZEN/'inputs/manifest.json')
    for key in ('config','policy_config','models','gates','steps','capacity','batch'):
        if s[key]!=original[key]:raise ValueError('Changed frozen setting: '+key)
    for p,h in s['evidence_bindings'].items():
        if sha256(p)!=h:raise ValueError('Changed frozen evidence: '+p)
    if s['origin_manifest_sha256']!=sha256(FROZEN/'inputs/manifest.json'):
        raise ValueError('Changed frozen source')
    rp=Path(s['reservation'])
    if sha256(rp)!=s['reservation_sha256'] or read(rp)['protocol']!=reservation():
        raise ValueError('Changed prospective protocol')
    for file,key in (('parents.json','parents_sha256'),('preplanning_parents.json','preplanning_sha256')):
        if sha256(root/file)!=s[key]:raise ValueError('Changed new parent inventory')
    parents=read(root/'parents.json');raw=read(root/'preplanning_parents.json')
    check_parents(parents,raw,s)
    return parents
