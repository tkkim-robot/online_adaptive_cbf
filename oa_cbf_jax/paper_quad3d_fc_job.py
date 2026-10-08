"""Bind and evaluate one-nearest-obstacle FC on the frozen static flight task."""


from collections import Counter


from pathlib import Path


from .comparison_contracts import matched_controller_settings, physical_obstacle_scope
from .dataset import sha256


from .nearest_fc import validate_metadata
from .nearest_fc_qualification import read


def verify(root, spec, metadata):
    root=Path(root)
    if (spec['schema']!='paper_nearest_fc_quad3d_inputs' or set(spec['models'])!={'nearest_fc'}
            or spec.get('encoder_only') is not False or spec.get('matched_encoders_only') is not False):
        raise ValueError('Explicit single-nearest paper comparison scope required')
    for path,digest in spec['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('Changed frozen paper input: '+path)
    original=read(Path(spec['original_comparison_inputs'])/'manifest.json')
    if original['schema']!='quad3d_matched_static_policy':raise ValueError('Wrong original physical task')
    matched_controller_settings(original['comparison_contracts']['gat'],original['comparison_contracts']['matched_fc'])
    for key in ('config','policy_config','parents_sha256','preplanning_sha256','steps','batch','capacity',
                'families','noise_levels','parents','reference_parents','adaptive_parents','stationary_physical_obstacles'):
        if spec[key]!=original[key]:raise ValueError('Changed physical task: '+key)
    if not spec['stationary_physical_obstacles'] or 'configs' in spec or 'policy_configs' in spec:
        raise ValueError('Shared static controller required')
    for filename,key in (('parents.json','parents_sha256'),('preplanning_parents.json','preplanning_sha256')):
        if sha256(root/filename)!=spec[key]:raise ValueError('Changed physical parent bytes')
    contract=validate_metadata(metadata)
    if contract!=spec['paper_nearest_fc_contract'] or contract['dynamics']!='quad3d':
        raise ValueError('Wrong nearest-obstacle flight model')
    old_bundle=Path(original['models']['gat']['bundle'])
    if old_bundle.resolve()!=Path(spec['frozen_oa_bundle']).resolve() or sha256(old_bundle/'manifest.json')!=original['models']['gat']['bundle_manifest_sha256']:
        raise ValueError('Changed frozen OA comparator')
    oa=read(old_bundle/'manifest.json')
    for key in ('controller','quad3d_contract','gain_domain','gain_dimension','normalization','targets','events','dataset_manifest_sha256'):
        if oa[key]!=metadata[key]:raise ValueError('Changed shared flight learning/controller contract: '+key)
    parents=read(root/'parents.json')
    if len(parents)!=spec['parents'] or len({p['id'] for p in parents})!=len(parents):
        raise ValueError('Wrong physical denominator')
    for phase,key in (('gate_calibration','reference_parents'),('policy_audit','adaptive_parents')):
        group=[p for p in parents if p['partition']==phase]
        if len(group)!=spec[key] or Counter(p['family'] for p in group)!=dict.fromkeys(spec['families'],24):
            raise ValueError('Changed reference/policy allocation')
    for parent in parents:
        physical_obstacle_scope('quad3d',parent['obstacles'],parent['mask'])
        if parent['gains']!=[spec['policy_config']['initial_gain']]*4:
            raise ValueError('Changed initial gain')
    return spec
