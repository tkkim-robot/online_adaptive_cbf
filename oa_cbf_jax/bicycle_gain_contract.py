"""Bicycle gain contract functions and shared contracts."""

import copy

import json

from pathlib import Path

import numpy as np

from .io import sha256

RAW_SCHEMA = 'oa_cbf_bicycle_expanded_scalar_observed_labels'

TRAIN_SCHEMA = 'oa_cbf_bicycle_expanded_scalar_training_view'

LEGACY_SCHEMA = 'oa_cbf_bicycle_acquired_history_hurdle_v67'

def candidate_bank():
    return np.r_[np.geomspace(.5, 8., 8).astype(np.float32),
                 np.asarray([.0625, .125, .25, .375, 12., 16., 32., 64.], np.float32)]

def contract():
    return dict(schema='bicycle_reviewed_wide_scalar_gain',
                candidates=candidate_bank().tolist(), lower=.0625, upper=64.,
                replicas=2, horizon_steps=40, original_candidates=8,
                training='Fresh matched encoders; no extrapolation of legacy weights.',
                deployment='Separate exported-weight qualification and fresh calibration required.')

def model_bank(metadata):
    """Resolve only explicitly declared supported domains; never extrapolate."""
    if metadata.get('offline_wide_gain_pilot') or metadata.get('dataset_schema')==TRAIN_SCHEMA:
        if metadata.get('architecture', {}).get('encoder') == 'nearest_fc':
            from .nearest_fc import validate_metadata
            if (metadata.get('dataset_schema') != TRAIN_SCHEMA or metadata.get('offline_wide_gain_pilot') is not True
                    or metadata.get('gain_domain') != dict(lower=.0625, upper=64.)
                    or metadata['architecture'].get('nearest_dynamics') != 'bicycle'):
                raise ValueError('Nearest-FC requires the exact reviewed bicycle gain domain')
        else:
            from .bicycle_features import validate_metadata
        validate_metadata(metadata)
        if metadata.get('bicycle_gain_contract')!=contract():
            raise ValueError('Missing reviewed wide-gain model contract')
        return candidate_bank()[:,None]
    if metadata.get('gain_domain')!=dict(lower=.5,upper=8.):
        raise ValueError('Unqualified bicycle model gain domain')
    return np.geomspace(.5,8,8).astype(np.float32)[:,None]

def validate_bank(bank):
    a=np.asarray(bank,np.float32)
    if not any(np.array_equal(a,b) for b in (candidate_bank()[:,None],candidate_bank()[:8,None])):
        raise ValueError('Unsupported or reordered bicycle candidate bank')
    return a

def read(path):
    return json.loads(Path(path).read_text())

def checked(path, digest):
    if sha256(path) != digest:
        raise ValueError('Changed wide-gain evidence: ' + str(path))

def validate_manifest(m):
    from .bicycle_gain_contract import validate_target_metadata
    validate_target_metadata(m)
    if ('bicycle_task_progress_contract' in m)!=('task_progress_derivative' in m):
        raise ValueError('Explicit task target derivative binding required')
    if (m.get('schema') != TRAIN_SCHEMA or m.get('weight_fit_authorized') is not True
            or m.get('production_eligible') is not False or m.get('final_test') is not False
            or m.get('bicycle_gain_contract') != contract()
            or m.get('gain_domain') != dict(lower=.0625, upper=64.)
            or m.get('gain_dimension') != 1 or m.get('queries') != 16
            or m.get('replicas') != 2 or m.get('horizon_steps') != 40
            or m.get('gain_candidates') != candidate_bank().tolist()):
        raise ValueError('Unreviewed or changed wide-gain training contract')

def _reviewed_raw(dataset, review):
    root=Path(dataset).resolve(); proof=read(review)
    if (proof.get('status')!='passed' or proof.get('permits_separate_training_view') is not True
            or Path(proof['report']).resolve()!=root/'report.json'):
        raise ValueError('Independent wide-gain dataset review required')
    for key in ('all_observed_geometry_and_selection_independently_recomputed',
                'all_original_parent_roles_and_labels_preserved',
                'all_added_physical_replay_and_dataset_bindings_verified'):
        if proof.get(key) is not True: raise ValueError('Incomplete wide-gain review: '+key)
    checked(proof['report'],proof['report_sha256']); checked(proof['protocol'],proof['protocol_sha256'])
    m=read(root/'manifest.json'); report=read(root/'report.json'); spec=read(proof['protocol'])
    if (m['schema']!=RAW_SCHEMA or m['weight_fit_authorized'] is not False
            or report['status']!='completed' or report['protocol_sha256']!=proof['protocol_sha256']):
        raise ValueError('Wrong raw wide-gain evidence')
    checked(root/'manifest.json',report['manifest_sha256']);checked(root/'index.json',report['index_sha256'])
    for path,digest in spec['bound_files'].items():checked(path,digest)
    for part in report['parts']:
        p=Path(part['directory'])
        for field,name in (('complete_sha256','complete.json'),('index_sha256','index.json'),('audit_sha256','independent_replay.json')):
            checked(p/name,part[field])
        audit=read(p/'independent_replay.json')
        if audit['status']!='passed' or audit['protocol_sha256']!=proof['protocol_sha256']:
            raise ValueError('Changed part physical proof')
    return root,m,report,proof

def validate_training_view(directory):
    view=Path(directory);m=read(view/'manifest.json');validate_manifest(m)
    if 'task_progress_derivative' in m:
        from .bicycle_gain_contract import validate_view
        return validate_view(view)
    binding=m['reviewed_training_view'];checked(binding['review'],binding['review_sha256'])
    root,original,report,proof=_reviewed_raw(binding['dataset'],binding['review'])
    checked(root/'report.json',binding['report_sha256'])
    expected=copy.deepcopy(original)
    expected.update({k:m[k] for k in ('schema','weight_fit_authorized','stage','bicycle_gain_contract','reviewed_training_view')})
    if m!=expected:raise ValueError('Training view changed labels, parents or controller semantics')
    auth=read(view/'authorization.json')
    checked(view/'manifest.json',auth['manifest_sha256']);checked(view/'index.json',auth['index_sha256'])
    if (auth['review_sha256']!=binding['review_sha256'] or auth['unchanged_payloads'] is not True
            or sha256(view/'index.json')!=report['index_sha256']):raise ValueError('Changed raw query index')
    entries=read(view/'index.json')
    if len(entries)!=proof['queries'] or len(m['groups'])!=proof['parents']:raise ValueError('Missing original evidence')
    roles={g['group_id']:g['partition'] for g in m['groups']}
    if len(roles)!=len(m['groups']) or set(roles.values())!={'train','validation','development_calibration'}:
        raise ValueError('Invalid original parent partitions')
    for e in entries:
        if not Path(e['file']).is_absolute() or e['group_id'] not in roles or e['branches']!=32:
            raise ValueError('Wrong shared wide-gain query')
        checked(e['file'],e['sha256'])
    return m


def task_progress_contract():
    return dict(schema='bicycle_physical_route_to_go_progress',
        potential='Distance to the nearest local route projection plus remaining route arclength.',
        projection='FP64 physical position; valid positive-length segments clipped to cursor+-1 metre; first nearest-distance tie.',
        target='Initial potential minus final potential, divided by horizon*dt*cruise_speed.',
        task_stop='Final potential is zero ONLY on a recorded physically verified GOAL event.',
        censoring='Actual retained prefix on every branch, including adverse termination; never invent later progress.',
        new_features=False, gain_search=False, controller_changed=False,
        limitation='Geometric task progress, not a reachability, safety or navigation-success certificate.')


from .bicycle_control import read as task_dataset_read


TARGET='physical_route_to_go_reduction_div_horizon_cruise'

def validate_target_metadata(metadata):
    value=metadata.get('bicycle_task_progress_contract')
    targets=metadata.get('targets',[])
    if value is not None or TARGET in targets:
        if value!=task_progress_contract() or len(targets)!=2 or targets[1]!=TARGET:
            raise ValueError('Missing or incompatible bicycle task-progress semantics')

def manifest_for(source, base, review):
    if 'task_progress_derivative' in base or 'bicycle_task_progress_contract' in base:
        raise ValueError('Nested or reinterpreted progress derivatives are forbidden')
    result=copy.deepcopy(base)
    result['stage']='reviewed_physical_task_progress_derivative'
    result['targets'][1]=TARGET
    result['bicycle_task_progress_contract']=task_progress_contract()
    result['task_progress_derivative']=dict(source=str(Path(source).resolve()),
        source_manifest_sha256=sha256(Path(source)/'manifest.json'),
        source_index_sha256=sha256(Path(source)/'index.json'),
        development_review=str(Path(review).resolve()),development_review_sha256=sha256(review),
        changed_fields=['target[...,1]'],physical_trajectories_unchanged=True)
    return result

def validate_view(directory):
    from .bicycle_gain_contract import validate_training_view, validate_manifest
    root=Path(directory);m=task_dataset_read(root/'manifest.json');validate_manifest(m)
    binding=m['task_progress_derivative'];source=Path(binding['source'])
    base=task_dataset_read(source/'manifest.json')
    if 'task_progress_derivative' in base:raise ValueError('Nested derivative forbidden')
    validate_training_view(source)
    review=binding['development_review']
    expected=manifest_for(source,base,review)
    if m!=expected:raise ValueError('Progress derivative changed its source, controller or input contract')
    proof=task_dataset_read(review)
    if proof.get('status')!='passed' or not proof.get('original_development_queries_statuses_roles_and_independent_statistics_checked'):
        raise ValueError('Missing independent development diagnosis')
    audit=task_dataset_read(root/'independent_replay.json');auth=task_dataset_read(root/'authorization.json')
    for file,key in [('manifest.json','manifest_sha256'),('index.json','index_sha256')]:
        if sha256(root/file)!=audit[key] or audit[key]!=auth[key]:raise ValueError('Changed derivative authorization')
    if (auth['audit_sha256']!=sha256(root/'independent_replay.json') or audit['status']!='passed'
            or not audit['all_nonprogress_fields_exact'] or not audit['all_targets_independently_recomputed']):
        raise ValueError('Missing independently audited target derivative')
    original,entries=task_dataset_read(source/'index.json'),task_dataset_read(root/'index.json')
    if len(original)!=len(entries) or len(entries)!=len(audit['rows']):raise ValueError('Missing original query')
    for old,new,checked in zip(original,entries,audit['rows'],strict=True):
        expected=dict(old,file=new['file'],sha256=new['sha256'],target_source_file=old['file'],target_source_sha256=old['sha256'])
        if new!=expected or sha256(new['file'])!=new['sha256'] or checked['sha256']!=new['sha256'] or checked['source_sha256']!=old['sha256']:
            raise ValueError('Changed original or derived query binding')
    return m
