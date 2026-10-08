"""An explicitly bound progress-target derivative; original futures stay intact."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import numpy as np

from .bicycle_experiment import read
from .bicycle_task_progress import contract, physical_progress, reference_potential
from .dataset import sha256
from .io import write_json

TARGET='physical_route_to_go_reduction_div_horizon_cruise'


def validate_target_metadata(metadata):
    value=metadata.get('bicycle_task_progress_contract')
    targets=metadata.get('targets',[])
    if value is not None or TARGET in targets:
        if value!=contract() or len(targets)!=2 or targets[1]!=TARGET:
            raise ValueError('Missing or incompatible bicycle task-progress semantics')


def manifest_for(source, base, review):
    if 'task_progress_derivative' in base or 'bicycle_task_progress_contract' in base:
        raise ValueError('Nested or reinterpreted progress derivatives are forbidden')
    result=copy.deepcopy(base)
    result['stage']='reviewed_physical_task_progress_derivative'
    result['targets'][1]=TARGET
    result['bicycle_task_progress_contract']=contract()
    result['task_progress_derivative']=dict(source=str(Path(source).resolve()),
        source_manifest_sha256=sha256(Path(source)/'manifest.json'),
        source_index_sha256=sha256(Path(source)/'index.json'),
        development_review=str(Path(review).resolve()),development_review_sha256=sha256(review),
        changed_fields=['target[...,1]'],physical_trajectories_unchanged=True)
    return result


def derive_one(task):
    entry,output,config,horizon=task
    source=Path(entry['file']);dest=Path(output)
    if sha256(source)!=entry['sha256']:raise ValueError('Changed audited original query')
    with np.load(source) as z:data={k:z[k] for k in z.files}
    if len(data['group_id'])!=1:raise ValueError('Expected original one-query payload')
    progress=physical_progress(data['initial_state'][0],data['final_state'][0],data['points'][0],data['route_mask'][0],
        data['cursor'][0],data['final_cursor'][0],data['status'][0],data['goal'][0],config)
    normalizer=horizon*config['robot']['dt']*config['cruise_speed']
    data['target']=data['target'].copy();data['target'][0,:,1]=(progress/normalizer).astype(np.float32)
    np.savez_compressed(dest,**data)
    return dict(entry,file=str(dest.resolve()),sha256=sha256(dest),target_source_file=str(source.resolve()),target_source_sha256=entry['sha256'])


def audit_one(task):
    old,new,config,horizon=task
    for entry in (old,new):
        if sha256(entry['file'])!=entry['sha256']:raise ValueError('Changed target evidence')
    with np.load(old['file']) as z:a={k:z[k] for k in z.files}
    with np.load(new['file']) as z:b={k:z[k] for k in z.files}
    if set(a)!=set(b):raise ValueError('Derivative changed field set')
    for key in a:
        if a[key].shape!=b[key].shape or a[key].dtype!=b[key].dtype:
            raise ValueError('Changed field dtype/shape: '+key)
        np.testing.assert_array_equal(a[key][...,0] if key=='target' else a[key],b[key][...,0] if key=='target' else b[key])
    # Scalar geometry is independent of the vectorized derivative calculation.
    points=a['points'][0];mask=a['route_mask'][0];goal=a['goal'][0]
    start=reference_potential(a['initial_state'][0,:2],points,mask,a['cursor'][0])
    ends=[]
    for state,cursor,status in zip(a['final_state'][0],a['final_cursor'][0],a['status'][0],strict=True):
        if status==1:
            if np.linalg.norm(state[:2]-goal)>config['goal_tolerance'] or state[3]>config['terminal_speed']:
                raise ValueError('False physical goal label')
            ends.append(0.)
        else:ends.append(reference_potential(state[:2],points,mask,cursor))
    expected=((start-np.asarray(ends))/(horizon*config['robot']['dt']*config['cruise_speed'])).astype(np.float32)
    np.testing.assert_allclose(b['target'][0,:,1],expected,atol=2e-7,rtol=2e-7)
    return dict(file=new['file'],sha256=new['sha256'],source_sha256=old['sha256'],
        group_id=str(a['group_id'][0]),partition=str(a['partition'][0]),branches=len(ends),
        max_target_error=float(np.max(abs(b['target'][0,:,1]-expected))),all_other_fields_exact=True)


def validate_view(directory):
    from .bicycle_gain_contract import validate_training_view, validate_manifest
    root=Path(directory);m=read(root/'manifest.json');validate_manifest(m)
    binding=m['task_progress_derivative'];source=Path(binding['source'])
    base=read(source/'manifest.json')
    if 'task_progress_derivative' in base:raise ValueError('Nested derivative forbidden')
    validate_training_view(source)
    review=binding['development_review']
    expected=manifest_for(source,base,review)
    if m!=expected:raise ValueError('Progress derivative changed its source, controller or input contract')
    proof=read(review)
    if proof.get('status')!='passed' or not proof.get('original_development_queries_statuses_roles_and_independent_statistics_checked'):
        raise ValueError('Missing independent development diagnosis')
    audit=read(root/'independent_replay.json');auth=read(root/'authorization.json')
    for file,key in [('manifest.json','manifest_sha256'),('index.json','index_sha256')]:
        if sha256(root/file)!=audit[key] or audit[key]!=auth[key]:raise ValueError('Changed derivative authorization')
    if (auth['audit_sha256']!=sha256(root/'independent_replay.json') or audit['status']!='passed'
            or not audit['all_nonprogress_fields_exact'] or not audit['all_targets_independently_recomputed']):
        raise ValueError('Missing independently audited target derivative')
    original,entries=read(source/'index.json'),read(root/'index.json')
    if len(original)!=len(entries) or len(entries)!=len(audit['rows']):raise ValueError('Missing original query')
    for old,new,checked in zip(original,entries,audit['rows'],strict=True):
        expected=dict(old,file=new['file'],sha256=new['sha256'],target_source_file=old['file'],target_source_sha256=old['sha256'])
        if new!=expected or sha256(new['file'])!=new['sha256'] or checked['sha256']!=new['sha256'] or checked['source_sha256']!=old['sha256']:
            raise ValueError('Changed original or derived query binding')
    return m


def create(source,review,output,workers=28):
    from .bicycle_gain_contract import validate_training_view
    start=time.monotonic();root=Path(source).resolve();base=validate_training_view(root)
    proof=read(review)
    if proof.get('status')!='passed' or proof['source_manifest_sha256']!=sha256(root/'manifest.json'):
        raise ValueError('Reviewed original development diagnosis required')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    original=read(root/'index.json');estimate=2*sum(Path(e['file']).stat().st_size for e in original)+128*2**20
    free=shutil.disk_usage(out).free
    if free-estimate<150.35*2**30:raise ValueError('Target derivative would exhaust storage reserve')
    m=manifest_for(root,base,review);write_json(out/'manifest.json',m)
    tasks=[(e,str(out/f'query_{i:05d}.npz'),m['config'],m['horizon_steps']) for i,e in enumerate(original)]
    with ProcessPoolExecutor(workers,mp_context=get_context('spawn')) as pool:entries=list(pool.map(derive_one,tasks,chunksize=16))
    write_json(out/'index.json',entries)
    print(dict(stage='independent_all_field_and_target_audit',queries=len(entries)),flush=True)
    with ProcessPoolExecutor(workers,mp_context=get_context('spawn')) as pool:
        checks=list(pool.map(audit_one,[(a,b,m['config'],m['horizon_steps']) for a,b in zip(original,entries,strict=True)],chunksize=16))
    audit=dict(status='passed',manifest_sha256=sha256(out/'manifest.json'),index_sha256=sha256(out/'index.json'),
        source_manifest_sha256=sha256(root/'manifest.json'),source_index_sha256=sha256(root/'index.json'),
        all_nonprogress_fields_exact=True,all_targets_independently_recomputed=True,
        queries=len(entries),branches=sum(c['branches'] for c in checks),parents=len(m['groups']),
        maximum_target_error=max(c['max_target_error'] for c in checks),rows=checks,
        calibration_roles_unchanged=True,calibration_outcome_statistics_reported=False,
        storage=dict(projected_gib=estimate/2**30,reserve_gib=150.35,free_gib=free/2**30),
        elapsed_seconds=time.monotonic()-start)
    write_json(out/'independent_replay.json',audit)
    write_json(out/'authorization.json',dict(manifest_sha256=audit['manifest_sha256'],index_sha256=audit['index_sha256'],audit_sha256=sha256(out/'independent_replay.json')))
    validate_view(out)
    print(dict(status='completed',queries=len(entries),branches=audit['branches'],elapsed_seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','review','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=28);create(**vars(p.parse_args()))
