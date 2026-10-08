"""Parent-balanced, reserved calibration for the local-observation unicycle.

Four disjoint calibration roles are assigned before observing their outcomes.
Predictive labels retain eight-second held gains. Trajectory groups are only
reserved here, never generated or used for fitting this predictive calibration.
"""


from pathlib import Path
import json


from .dataset import sha256, load_dataset


from .local_unicycle_dataset import SCHEMA, FAMILIES, scene, validate

CONTRACT_KEYS=('schema','config','nominal','neighborhood','acquisition_gain','held_gain_seconds',
    'adaptation_interval_seconds','observation','future_noise','reservation_sha256')
CAL_SCHEMA='unicycle_local_reserved_prediction_v1'
ROLES=('predictive_fit','predictive_audit','trajectory_gate','trajectory_audit')


def read(path):return json.loads(Path(path).read_text())


def training_contract(manifest):
    result={k:manifest[k] for k in CONTRACT_KEYS}
    from .local_unicycle_candidates import candidate_contract
    candidates=candidate_contract(manifest)
    if candidates is not None:result['candidate_bank_contract']=candidates
    return result


def validate_bundle(bundle,dataset):
    m=validate(dataset);b=read(Path(bundle)/'manifest.json')
    from .local_unicycle_candidates import validate_bank
    validate_bank(dataset,m)
    if (b['dataset_manifest_sha256']!=sha256(Path(dataset)/'manifest.json')
        or b.get('local_unicycle_contract')!=training_contract(m)
        or b['controller']!=m['controller'] or b['targets']!=m['targets'] or b['events']!=m['events']
        or b['gain_domain']!=m['gain_domain'] or b['architecture']['encoder'] not in ('gat','nearest_fc')):
        raise ValueError('Wrong local observation model/controller/target contract')
    if sha256(Path(bundle)/'weights.msgpack')!=b['weights_sha256']:raise ValueError('Changed weights')
    if b['architecture'].get('unicycle_constraint_features',False):
        from .unicycle_constraint_features import contract as coefficient_contract
        if b.get('unicycle_constraint_features_contract')!=coefficient_contract():
            raise ValueError('Changed observed unicycle coefficient feature contract')
    if 'local_unicycle_supervision' in b:
        from .local_unicycle_supervision import CONTRACT
        if b['local_unicycle_supervision']!=CONTRACT:raise ValueError('Unknown live-query supervision')
    if 'local_unicycle_stop_contrast' in b:
        from .local_unicycle_stop_contrast import contract as stop_contract
        if b['local_unicycle_stop_contrast']!=stop_contract():
            raise ValueError('Unknown stop-contrast supervision')
    if 'local_unicycle_prediction_selection' in b:
        from .local_unicycle_prediction_selection import contract as selection_contract
        if b['local_unicycle_prediction_selection']!=selection_contract():
            raise ValueError('Unknown checkpoint selection')
    if 'local_unicycle_reflection' in b:
        from .local_unicycle_reflection_training import check_proof
        check_proof(b['local_unicycle_reflection'])
    if 'local_unicycle_expansion' in b:
        from .local_unicycle_expansion import SCHEMA as EXPANSION_SCHEMA
        from .local_unicycle_coverage_union import SCHEMA as UNION_SCHEMA, validate as validate_union
        proof=b['local_unicycle_expansion']
        if (proof['schema'] not in (EXPANSION_SCHEMA,UNION_SCHEMA) or proof['primary_manifest_sha256']!=b['dataset_manifest_sha256']
            or proof['validation_weight_fitting'] or proof['forward_parents_used']):
            raise ValueError('Unknown or leaky shared-parent expansion')
        if proof['schema']==UNION_SCHEMA:
            expected,_=validate_union(proof['primary_dataset'],proof['additional_dataset'])
            if proof!=expected:raise ValueError('Changed retained-generation training proof')
        for p,d in proof['bound_files'].items():
            if sha256(p)!=d:raise ValueError('Changed expanded training evidence')
    return b


def assign_roles(groups):
    """Order-only allocation; neither model outputs nor physical outcomes used."""
    ids=[g['group_id'] for g in groups]
    seeds=[g['physical_seed'] for g in groups]
    if len(ids)!=len(set(ids)) or len(seeds)!=len(set(seeds)):raise ValueError('Duplicate reserved parent')
    result=[]
    for family in FAMILIES:
        rows=sorted((g for g in groups if g['family']==family and g['partition']=='reserved_calibration'),key=lambda g:g['physical_seed'])
        if len(rows)!=64:raise ValueError('Expected64reserved calibration parents per family')
        for i,row in enumerate(rows):result.append(dict(row,partition=ROLES[i//16]))
    if len(result)!=512:raise ValueError('Wrong calibration reservation size')
    return result


def validate_calibration_data(dataset):
    root=Path(dataset);m=read(root/'manifest.json');source=Path(m['training_dataset']);tm=validate(source)
    if (m['schema']!=CAL_SCHEMA or m['weight_fit_authorized'] or m['pilot']
        or sha256(source/'manifest.json')!=m['training_manifest_sha256']
        or tm['reservation_sha256']!=m['original_reservation_sha256']):raise ValueError('Changed predictive data contract')
    for k in CONTRACT_KEYS:
        if k not in ('schema','reservation_sha256') and m[k]!=tm[k]:raise ValueError('Different physical calibration contract')
    from .local_unicycle_candidates import validate_bank
    if validate_bank(root,m)!=validate_bank(source,tm):
        raise ValueError('Predictive calibration uses a different candidate bank')
    if read(root/'reserved_groups.json')!=assign_roles(read(source/'reserved_groups.json')):raise ValueError('Changed calibration roles')
    for name,key in (('records.json','records_sha256'),('source.npz','source_sha256'),('reserved_groups.json','reservation_sha256')):
        if sha256(root/name)!=m[key]:raise ValueError('Changed predictive source')
    a=read(root/'audit.json')
    if (a['manifest_sha256']!=sha256(root/'manifest.json') or a['index_sha256']!=sha256(root/'index.json')
        or not a['all_saved_bindings_checked'] or not a['all_applied_branches_audited_at_collection']):raise ValueError('Unaudited calibration labels')
    if tm['controller'].get('position_observer') is not None:
        from .local_unicycle_observer_data import qualified_source
        qualified_source(root)
        replay=read(root/'compact_replay.json')
        if (m['controller']!=tm['controller'] or not a['all_observer_memories_independently_checked']
            or replay['audit_sha256']!=sha256(root/'audit.json') or not replay['exact_reconstruction']):
            raise ValueError('Unqualified observer calibration/replay contract')
    return m
