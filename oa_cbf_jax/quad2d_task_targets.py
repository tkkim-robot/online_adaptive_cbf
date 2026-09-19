"""Explicit physical task-progress targets and a matched route-target ablation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .dataset import sha256
from .io import write_json

ROUTE_NAME='observed_prefix_route_progress_div_horizon_cruise_distance'
TASK_NAME='physical_terminal_blended_progress_plus_observed_early_arrival_v1'


def contract(kind):
    if kind not in ('route','terminal_task'):raise ValueError('Unknown flight performance target')
    return dict(schema='quad2d_terminal_performance_v1',kind=kind,
        target=ROUTE_NAME if kind=='route' else TASK_NAME,
        initial_blend='clip(1-10*observed_graph_ego_feature28/terminal_transition_distance,0,1)' if kind=='terminal_task' else 'unused',
        physical_progress='(1-initial_blend)*physical_route_progress+initial_blend*physical_goal_distance_decrease' if kind=='terminal_task' else 'physical_route_progress',
        normalization='full label horizon times dt times cruise speed',
        arrival_reward='goal_reached*(1-applied_steps/full_horizon)' if kind=='terminal_task' else 'none',
        censoring='Only integrated prefix progress and actually observed goal arrival; no invented post-stop progress or success')


def values(shard,manifest):
    c=manifest['config'];h=manifest['horizon_steps'];kind=manifest['controller']['performance_target']['kind']
    progress=np.asarray(shard['route_progress'],float)
    if kind=='terminal_task':
        distance=manifest['controller']['predictive_guidance']['terminal_transition_distance']
        blend=np.clip(1-10*np.asarray(shard['features'][:,0,28],float)/distance,0.,1.)[:,None]
        progress=(1-blend)*progress+blend*np.asarray(shard['goal_progress'],float)
    result=progress/(h*c['robot']['dt']*c['cruise_speed'])
    if kind=='terminal_task':result+=(shard['status']==1)*(1-np.asarray(shard['steps'],float)/h)
    return result.astype(np.float32)


def check_physical_target_inputs(shard,manifest):
    """Check every candidate's physical distance and actual-arrival provenance."""
    from .fp32_audit import check_goal_progress
    c=manifest['config'];status=shard['status'];steps=shard['steps'];h=manifest['horizon_steps']
    before=shard['physical_initial_state'];after=shard['final_state'];goal=shard['goal'][:,None,:]
    if before.shape!=(*status.shape,6) or after.shape!=before.shape:raise ValueError('Missing physical endpoint provenance')
    if not np.isfinite(before).all() or not np.isfinite(after).all() or np.any(steps<0) or np.any(steps>h):raise ValueError('Invalid physical prefix')
    shaped=before.reshape(len(before),manifest['queries'],manifest['replicas'],6)
    np.testing.assert_array_equal(shaped,np.broadcast_to(shaped[:,:1],shaped.shape))
    noise=shard['noise'];ranges=noise[:,[0,0,1,2,2,3]][:,None,:]
    if np.any(np.abs(before.astype(float)-shard['initial_state'][:,None,:])>ranges+2e-5):raise ValueError('Physical initial state outside declared latent range')
    check_goal_progress(shard['goal_progress'],before[...,:2],after[...,:2],goal)
    reached=status==1;distance=np.linalg.norm(after[...,:2].astype(float)-goal,axis=-1)
    speed=np.linalg.norm(after[...,3:5].astype(float),axis=-1)
    arrived=(distance<=c['goal_tolerance']+1e-6)&(speed<=c['terminal_speed']+1e-6)&(np.abs(after[...,2])<=c['terminal_pitch']+1e-6)&(np.abs(after[...,5])<=c['terminal_pitch_rate']+1e-6)
    if np.any(reached&~arrived):raise ValueError('Arrival reward without actual physical arrival')


def retarget(source,output,kind='route'):
    source=Path(source);root=Path(output);m=json.loads((source/'manifest.json').read_text());a=json.loads((source/'independent_replay.json').read_text())
    if not a['audit_passed'] or not a.get('all_physical_task_target_inputs_checked') or a['manifest_sha256']!=sha256(source/'manifest.json') or a['index_sha256']!=sha256(source/'index.json'):raise ValueError('Exactly audited terminal task source required')
    root.mkdir(parents=True,exist_ok=False);m=deepcopy(m);m['controller']['performance_target']=contract(kind);m['targets'][1]=contract(kind)['target']
    m.update(performance_retarget_source=str(source.resolve()),performance_retarget_manifest_sha256=sha256(source/'manifest.json'),performance_retarget_index_sha256=sha256(source/'index.json'))
    index=[]
    for entry in json.loads((source/'index.json').read_text()):
        if sha256(source/entry['file'])!=entry['sha256']:raise ValueError('Changed source shard')
        with np.load(source/entry['file']) as f:d=dict(f)
        d['target']=d['target'].copy();d['target'][...,1]=values(d,m)
        path=root/entry['file'];np.savez_compressed(path,**d);index.append(dict(entry,sha256=sha256(path)))
        for file,digest in [(entry['trace_file'],entry['trace_sha256'])]+[(t['file'],t['sha256']) for t in entry.get('event_traces',[])]:
            if sha256(source/file)!=digest:raise ValueError('Changed physical trace')
            (root/file).hardlink_to(source/file)
    write_json(root/'manifest.json',m);write_json(root/'index.json',index)
    # Full target, endpoint, graph and physical re-audit is mandatory before use.
    from .dataset import load_dataset
    audit=deepcopy(json.loads((source/'contract_audit.json').read_text()));audit.update(retarget_kind=kind,physical_branches_preserved=True)
    for partition,row in audit['partitions'].items():
        d=load_dataset(root,partition)
        row['target_std']=[float(d['target'][...,i][d['target_mask'][...,i]].std()) for i in range(2)]
    audit['contract_valid']=audit['contract_valid'] and all(min(r['target_std'])>1e-6 for r in audit['partitions'].values())
    write_json(root/'contract_audit.json',audit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['route','terminal_task'],default='route');retarget(**vars(p.parse_args()))
