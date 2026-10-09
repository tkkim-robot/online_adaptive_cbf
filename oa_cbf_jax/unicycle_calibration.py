"""Unicycle calibration functions and shared contracts."""

from pathlib import Path

import json

from .io import sha256

CONTRACT_KEYS=('schema','config','nominal','neighborhood','acquisition_gain','held_gain_seconds',
    'adaptation_interval_seconds','observation','future_noise','reservation_sha256')

CAL_SCHEMA='unicycle_local_reserved_prediction_v1'

ROLES=('predictive_fit','predictive_audit','trajectory_gate','trajectory_audit')

def read(path):return json.loads(Path(path).read_text())

def training_contract(manifest):
    result={k:manifest[k] for k in CONTRACT_KEYS}
    from .unicycle_policy import candidate_contract
    candidates=candidate_contract(manifest)
    if candidates is not None:result['candidate_bank_contract']=candidates
    return result

def validate_bundle(bundle,dataset):
    from .unicycle_data import dataset_validate as validate
    m=validate(dataset);b=read(Path(bundle)/'manifest.json')
    from .unicycle_policy import validate_bank
    validate_bank(dataset,m)
    if (b['dataset_manifest_sha256']!=sha256(Path(dataset)/'manifest.json')
        or b.get('local_unicycle_contract')!=training_contract(m)
        or b['controller']!=m['controller'] or b['targets']!=m['targets'] or b['events']!=m['events']
        or b['gain_domain']!=m['gain_domain'] or b['architecture']['encoder'] not in ('gat','nearest_fc')):
        raise ValueError('Wrong local observation model/controller/target contract')
    if sha256(Path(bundle)/'weights.msgpack')!=b['weights_sha256']:raise ValueError('Changed weights')
    if b['architecture'].get('unicycle_constraint_features',False):
        from .unicycle_features import contract as coefficient_contract
        if b.get('unicycle_constraint_features_contract')!=coefficient_contract():
            raise ValueError('Changed observed unicycle coefficient feature contract')
    if 'local_unicycle_supervision' in b:
        from .unicycle_training import CONTRACT
        if b['local_unicycle_supervision']!=CONTRACT:raise ValueError('Unknown live-query supervision')
    if 'local_unicycle_stop_contrast' in b:
        from .unicycle_training import stop_contrast_contract as stop_contract
        if b['local_unicycle_stop_contrast']!=stop_contract():
            raise ValueError('Unknown stop-contrast supervision')
    if 'local_unicycle_prediction_selection' in b:
        from .unicycle_training import contract as selection_contract
        if b['local_unicycle_prediction_selection']!=selection_contract():
            raise ValueError('Unknown checkpoint selection')
    if 'local_unicycle_reflection' in b:
        from .unicycle_training import check_proof
        check_proof(b['local_unicycle_reflection'])
    if 'local_unicycle_expansion' in b:
        from .unicycle_data import EXPANSION_SCHEMA
        from .unicycle_data import SCHEMA as UNION_SCHEMA, validate as validate_union
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
    from .unicycle_data import FAMILIES
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
    from .unicycle_data import dataset_validate as validate
    root=Path(dataset);m=read(root/'manifest.json');source=Path(m['training_dataset']);tm=validate(source)
    if (m['schema']!=CAL_SCHEMA or m['weight_fit_authorized'] or m['pilot']
        or sha256(source/'manifest.json')!=m['training_manifest_sha256']
        or tm['reservation_sha256']!=m['original_reservation_sha256']):raise ValueError('Changed predictive data contract')
    for k in CONTRACT_KEYS:
        if k not in ('schema','reservation_sha256') and m[k]!=tm[k]:raise ValueError('Different physical calibration contract')
    from .unicycle_policy import validate_bank
    if validate_bank(root,m)!=validate_bank(source,tm):
        raise ValueError('Predictive calibration uses a different candidate bank')
    if read(root/'reserved_groups.json')!=assign_roles(read(source/'reserved_groups.json')):raise ValueError('Changed calibration roles')
    for name,key in (('records.json','records_sha256'),('source.npz','source_sha256'),('reserved_groups.json','reservation_sha256')):
        if sha256(root/name)!=m[key]:raise ValueError('Changed predictive source')
    a=read(root/'audit.json')
    if (a['manifest_sha256']!=sha256(root/'manifest.json') or a['index_sha256']!=sha256(root/'index.json')
        or not a['all_saved_bindings_checked'] or not a['all_applied_branches_audited_at_collection']):raise ValueError('Unaudited calibration labels')
    if tm['controller'].get('position_observer') is not None:
        from .unicycle_data import qualified_source
        qualified_source(root)
        replay=read(root/'compact_replay.json')
        if (m['controller']!=tm['controller'] or not a['all_observer_memories_independently_checked']
            or replay['audit_sha256']!=sha256(root/'audit.json') or not replay['exact_reconstruction']):
            raise ValueError('Unqualified observer calibration/replay contract')
    return m


def check_bindings(bindings):
    for path, digest in bindings.items():
        if sha256(path) != digest:
            raise ValueError(f'Changed diagnostic source: {path}')


import numpy as np

POLICY_CALIBRATION_DATA_SCHEMA='local_unicycle_reserved_policy_coverage_v1'

MODES=('gat','nearest_fc')

VISITS=8

def policy_calibration_data_read(path):return json.loads(Path(path).read_text())

def validate_source(dataset):
    from .unicycle_policy import contract
    root=Path(dataset);m=policy_calibration_data_read(root/'manifest.json');base=Path(m['base_dataset']);validate_calibration_data(base)
    check_bindings(m['bindings'])
    if (m['schema']!=POLICY_CALIBRATION_DATA_SCHEMA or m['modes']!=list(MODES) or m['visits']!=VISITS or m['position_observer']!=contract()
        or m['weight_fit_authorized'] or m['forward_parents_used'] or m['trajectory_gate_parents_used']
        or m['collector_sha256']!=sha256(__file__) or m['numpy_version']!=np.__version__):raise ValueError('Changed reserved collection contract')
    for name,digest in m['helper_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=digest:raise ValueError('Changed physical helper')
    records=policy_calibration_data_read(base/'records.json')
    if len(set(m['indices']))!=len(m['indices']) or any(records[i]['partition'] not in ('predictive_fit','predictive_audit') for i in m['indices']):
        raise ValueError('Duplicate or forbidden parent')
    if m['pilot'] and any(records[i]['partition']!='predictive_fit' for i in m['indices']):raise ValueError('Pilot cannot select on audit parents')
    return m,base,records


SCHEMA = 'local_unicycle_shared_policy_stop_calibration_v1'

def policy_calibration_read(path):
    return json.loads(Path(path).read_text())

def collection_contract(collection):
    root = Path(collection)
    m, base, records = validate_source(root)
    audit = policy_calibration_read(root / 'audit.json')
    proof = policy_calibration_read(root / 'independent_completion_review.json')
    if (m['pilot'] or not m['predictive_calibration_fit_authorized']
            or m['variants_per_policy'] != 1024 or m['selected_parent_count'] != 256
            or m['held_gain_seconds'] != 8 or m['adaptation_interval_seconds'] != .2
            or m['replicas'] != 2 or m['queries'] != 32
            or not proof['independently_reproduced'] or proof['pilot']
            or proof['audit_sha256'] != sha256(root / 'audit.json')
            or proof['observations'] != 16384 or proof['branches'] != 1048576
            or proof['physical_parents'] != 256
            or proof['roles'] != {'predictive_fit': 8192, 'predictive_audit': 8192}
            or audit['index_sha256'] != sha256(root / 'index.json')
            or audit['manifest_sha256'] != sha256(root / 'manifest.json')):
        raise ValueError('Complete, independently reviewed reserved policy collection required')
    paths = [root / n for n in ('manifest.json', 'index.json', 'audit.json',
                               'independent_completion_review.json', 'independent_compact_replay.json')]
    paths += [base / n for n in ('manifest.json', 'index.json', 'audit.json', 'records.json', 'reserved_groups.json')]
    return m, base, records, {str(p.resolve()): sha256(p) for p in paths}

def validate_supplement(info):
    """Optional runtime extension; historical calibrations take the old path."""
    proof = info['policy_visited_stop_calibration']
    if (proof['schema'] != SCHEMA or proof['weight_fitting'] or proof['forward_parents_used']
            or proof['changed_parameters'] != ['stop_temperature', 'stop_bias']):
        raise ValueError('Unknown policy-visited calibration extension')
    check_bindings(proof['bound_files'])
    _, base, _, _ = collection_contract(proof['collection'])
    if str(base.resolve()) != info['calibration_dataset']:
        raise ValueError('Different predictive reservation')
    previous = policy_calibration_read(proof['previous_calibration'])
    unchanged = ('bundle', 'bundle_manifest_sha256', 'weights_sha256', 'dataset',
                 'dataset_manifest_sha256', 'calibration_dataset', 'calibration_manifest_sha256',
                 'calibration_audit_sha256', 'local_unicycle_contract', 'variance_scale',
                 'controller', 'targets', 'events', 'gain_domain', 'qualification',
                 'qualification_sha256', 'group_ids')
    if (any(info[k] != previous[k] for k in unchanged)
            or info['event_calibration'][0] != previous['event_calibration'][0]):
        raise ValueError('Stop-only refit changed frozen model or calibration fields')
    report = policy_calibration_read(proof['report'])
    if (report['event_calibration'] != info['event_calibration']
            or report['variance_scale'] != info['variance_scale']
            or report['fit_groups'] != info['group_ids']['fit']
            or report['audit_groups'] != info['group_ids']['audit']
            or set(report['fit_groups']) & set(report['audit_groups'])):
        raise ValueError('Changed stop fit or parent roles')
