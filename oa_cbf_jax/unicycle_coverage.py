"""Unicycle coverage functions and shared contracts."""

import os

import subprocess

import sys

def launch_child(job,name,args,backend,slot):
    env=os.environ.copy();env.pop('JAX_ENABLE_X64',None)
    tmp=job/'tmp';tmp.mkdir(exist_ok=True)
    env.update(JAX_PLATFORMS=backend,CUDA_VISIBLE_DEVICES=str(slot),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',TMPDIR=str(tmp.resolve()))
    with (job/(name+'.log')).open('w') as log:
        subprocess.run(['taskset','-c',f'{slot*14}-{slot*14+13}',sys.executable,'-m',*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


import argparse

from collections import Counter

from concurrent.futures import ThreadPoolExecutor

import json

from pathlib import Path

import shutil

import time

import numpy as np

import jax

import jax.numpy as jnp

from .io import sha256

from .io import write_json

SCHEMA = 'unicycle_local_observed_shared_coverage_v1'

def read(path):
    return json.loads(Path(path).read_text())

def reserve(navigation_job, output, pilot=False):
    from .unicycle_data import (
        FRACTIONS)
    from .unicycle_data import (
        MODES)
    from .unicycle_policy import contract
    from .unicycle_data import dataset_validate as validate_base
    job = Path(navigation_job); spec = read(job/'protocol.json'); done = read(job/'complete.json')
    base = Path(spec['dataset']); bm = validate_base(base)
    proof = read(Path(spec['output'])/'independent_completion_review.json')
    if (done['status'] != 'completed' or not proof['independently_reproduced']
        or proof['review_sha256'] != done['review_sha256']
        or bm['controller'].get('position_observer') != contract()
        or sha256(base/'manifest.json') != spec['dataset_manifest_sha256']):
        raise ValueError('Independently reviewed observer navigation required')
    records = read(base/'records.json'); indices = list(range(len(records)))
    if pilot:
        parents = {min(r['group_id'] for r in records if r['family'] == f and r['partition'] == 'train')
                   for f in sorted({r['family'] for r in records})}
        indices = [i for i in indices if records[i]['group_id'] in parents]
    if len(indices) % 4 or {records[i]['partition'] for i in indices} - {'train', 'validation'}:
        raise ValueError('Only original training and validation roles allowed')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=SCHEMA, base_dataset=str(base.resolve()),
        bound_base_files={name:sha256(base/name) for name in
                         ('manifest.json','index.json','audit.json','source.npz','records.json','reserved_groups.json')},
        navigation_job=str(job.resolve()), navigation_protocol_sha256=sha256(job/'protocol.json'),
        navigation_review_sha256=done['review_sha256'], models=spec['models'], modes=list(MODES),
        position_observer=contract(), snapshot_fractions=list(FRACTIONS), indices=indices,
        horizon_steps=160, held_gain_seconds=8., adaptation_interval_seconds=.2,
        acquisition_steps=1600, replicas=2, queries=32, variants_per_policy=len(indices),
        selected_parent_count=len({records[i]['group_id'] for i in indices}),
        pilot=pilot, weight_fit_authorized=not pilot, validation_weight_fitting=False,
        forward_parents_used=False, calibration_parents_used=False, whole_goal_complete=False,
        collector_sha256=sha256(__file__), numpy_version=np.__version__,
        helper_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                      ('unicycle_policy.py','unicycle_data.py','unicycle_data.py')},
        retention='All acquisitions/branches independently audited in memory; every outcome, snapshot, pre-reading memory, RNG and full trace digest retained for exact reconstruction.',
        weighting='Same physical-parent bootstrap across the original data and every shared actor/snapshot; correlated variants are not additional independent parents.')
    write_json(root/'manifest.json', manifest)
    return manifest

def validate_source(dataset):
    from .unicycle_data import (
        FRACTIONS)
    from .unicycle_data import (
        MODES)
    from .unicycle_policy import contract
    from .unicycle_data import dataset_validate as validate_base
    root = Path(dataset); m = read(root/'manifest.json'); base = Path(m['base_dataset'])
    bm = validate_base(base)
    for name, digest in m['bound_base_files'].items():
        if sha256(base/name) != digest:raise ValueError('Changed base evidence: '+name)
    if (m['schema'] != SCHEMA or m['position_observer'] != contract()
        or bm['controller'].get('position_observer') != contract()
        or m['snapshot_fractions'] != list(FRACTIONS) or m['modes'] != list(MODES)
        or m['horizon_steps'] != 160 or m['held_gain_seconds'] != 8.
        or m['adaptation_interval_seconds'] != .2 or m['replicas'] != 2 or m['queries'] != 32
        or m['forward_parents_used'] or m['calibration_parents_used']):
        raise ValueError('Changed observed coverage contract')
    if sha256(__file__) != m['collector_sha256'] or np.__version__ != m['numpy_version']:
        raise ValueError('Use the immutable collector and recorded RNG')
    for name, digest in m['helper_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != digest:raise ValueError('Changed observer/RNG helper')
    records = read(base/'records.json')
    if len(m['indices']) != len(set(m['indices'])) or any(records[i]['partition'] not in ('train','validation') for i in m['indices']):
        raise ValueError('Invalid acquired parent population')
    return m, base, records

def acquisition(m, mode):
    from .unicycle_policy import LocalPolicy
    from .unicycle_policy import policy_make_rollout as make_rollout
    if mode == 'fixed_reference':
        rollout = make_rollout(position_observer=True); params = calibration = None
    else:
        model = m['models'][mode]
        p = LocalPolicy(model['bundle'], model['prediction_calibration'], gate=model['gate'])
        if not p.position_observer or json.loads(json.dumps(p.metadata)) != model['policy']:
            raise ValueError('Changed acquisition policy')
        rollout, params, calibration = p.rollout, p.params, p.calibration
    return jax.jit(jax.vmap(rollout, in_axes=(None,None,0,0,0,0,None))), params, calibration

def acquisition_record(ids, errors, result, trace):
    from .unicycle_data import (
        array_digest)
    from .unicycle_data import snapshot
    from .unicycle_data import (
        snapshot_ticks)
    from .unicycle_data import (
        tree_digest)
    ticks = np.stack([snapshot_ticks({k:v[b] for k,v in trace.items()}) for b in range(4)])
    snapshots = [snapshot(trace, ticks[:,v]) for v in range(4)]
    return dict(indices=ids, ticks=ticks, errors_sha256=np.array([array_digest(e) for e in errors]),
        full_trace_sha256=np.array([tree_digest({k:v[b] for k,v in trace.items()}) for b in range(4)]),
        snapshot_initial=np.stack([s[0] for s in snapshots],1),
        snapshot_prediction=np.stack([s[1].predicted_position for s in snapshots],1),
        snapshot_centers=np.stack([s[1].obstacle_centers for s in snapshots],1),
        snapshot_ready=np.stack([s[1].ready for s in snapshots],1), **result)

def branch_audits(x, world, goal, errors, memory, result, trace):
    from .local_unicycle_audit import audit_trajectory
    return [audit_trajectory(x[b],world[b],goal[b],errors[b,q],
        {k:v[b,q] for k,v in result.items()}, {k:v[b,q] for k,v in trace.items()},
        filtered_observation=True, observer_memory=jax.tree.map(lambda a:np.asarray(a[b]),memory))
        for b in range(4) for q in range(64)]

def collect(dataset, mode, slot=0):
    from .unicycle_data import (
        MODES)
    from .unicycle_data import (
        acquisition_errors)
    from .unicycle_data import (
        array_digest)
    from .local_unicycle_audit import audit_trajectory
    from .unicycle_data import branch_inputs
    from .unicycle_data import (
        future_errors)
    from .unicycle_data import graph_inputs
    from .unicycle_data import label_kernel
    from .unicycle_data import snapshot
    from .unicycle_data import (
        tree_digest)
    root = Path(dataset); m,base,records = validate_source(root)
    if mode not in MODES:raise ValueError('Unknown shared actor')
    out = root/mode; out.mkdir(exist_ok=True)
    with np.load(base/'source.npz') as z:initial,world,goal,bank = (z[k] for k in ('initial','world','goal','gains'))
    acq,params,calibration = acquisition(m,mode)
    lab = label_kernel(); graph = jax.jit(jax.vmap(graph_inputs))
    start = time.monotonic(); timing = Counter(); index = []
    assigned = list(range(slot*4,len(m['indices']),16))
    for number,first in enumerate(assigned):
        ids = np.asarray(m['indices'][first:first+4]); rows = [records[i] for i in ids]
        errors = np.stack([acquisition_errors(r) for r in rows])
        args = (params,calibration,*map(jnp.asarray,(initial[ids],world[ids],goal[ids],errors,bank)))
        if number == 0:
            t=time.monotonic(); acq_exe=acq.lower(*args).compile(); timing['compile']+=time.monotonic()-t
        t=time.monotonic(); ar,at=jax.device_get(acq_exe(*args)); timing['acquisition']+=time.monotonic()-t
        t=time.monotonic(); aa=[audit_trajectory(initial[i],world[i],goal[i],errors[b],
            {k:v[b] for k,v in ar.items()},{k:v[b] for k,v in at.items()},filtered_observation=True) for b,i in enumerate(ids)]
        timing['audit']+=time.monotonic()-t
        acquired=out/f'acquisition_{first:04d}.npz'; a=acquisition_record(ids,errors,ar,at)
        np.savez_compressed(acquired,**a)
        for visit in range(4):
            ticks=a['ticks'][:,visit]; x,memory=snapshot(at,ticks); current=errors[np.arange(4),ticks]
            future=np.stack([future_errors(r,mode,visit,ticks[b],current[b]) for b,r in enumerate(rows)])
            args,gains,expanded=branch_inputs(x,world[ids],goal[ids],bank,future,memory)
            ga=(*map(jnp.asarray,(x,world[ids],goal[ids],current)),jax.tree.map(jnp.asarray,memory))
            if number == 0 and visit == 0:
                t=time.monotonic(); lab_exe=lab.lower(*args).compile(); graph_exe=graph.lower(*ga).compile(); timing['compile']+=time.monotonic()-t
            t=time.monotonic(); result,trace=jax.device_get(lab_exe(*args)); features,mask,observed,selected=jax.device_get(graph_exe(*ga)); timing['labels']+=time.monotonic()-t
            np.testing.assert_array_equal(observed,at['observed'][np.arange(4),ticks])
            np.testing.assert_array_equal(observed,trace['observed'][:,0,0]); np.testing.assert_array_equal(selected,trace['selected_ids'][:,0,0])
            t=time.monotonic(); audits=branch_audits(x,world[ids],goal[ids],expanded,memory,result,trace); timing['audit']+=time.monotonic()-t
            s=result['status']; valid=np.stack((np.isin(s,(1,2,4)),np.isin(s,(1,4))),-1)
            target=np.where(valid,np.stack((-np.minimum(result['min_clearance'],.6)/.3,result['progress']/12),-1),0.).astype(np.float32)
            events=np.stack((s==2,np.isin(s,(3,5,8))),-1).astype(np.float32)
            name=f'labels_{first:04d}_{visit}.npz'; audit_name=f'audit_{first:04d}_{visit}.npz'
            np.savez_compressed(out/name,features=features,node_mask=mask,gains=gains,target=target,target_mask=valid,
                events=events,event_mask=np.ones_like(events,bool),group_id=np.array([r['group_id'] for r in rows]),
                partition=np.array([r['partition'] for r in rows]),initial_state=x,prior_status=np.zeros(4,np.int32),**result)
            values={k:np.array([a[k] if a[k] is not None else np.nan for a in audits]).reshape(4,64) for k in audits[0]}
            np.savez_compressed(out/audit_name,indices=ids,ticks=ticks,current_errors=current,
                future_errors_sha256=np.array([array_digest(v) for v in future]),
                full_trace_sha256=np.array([tree_digest({k:v[b] for k,v in trace.items()}) for b in range(4)]),**values)
            index.append(dict(file=str(Path(mode)/name),sha256=sha256(out/name),audit_file=str(Path(mode)/audit_name),
                audit_sha256=sha256(out/audit_name),acquisition_file=str(acquired.relative_to(root)),acquisition_sha256=sha256(acquired),
                indices=ids.tolist(),ticks=ticks.tolist(),mode=mode,visit=visit,acquisition_audits=aa,
                status_counts={str(s):int(c) for s,c in zip(*np.unique(s,return_counts=True))},
                branches=int(s.size),applied_steps=int(result['steps'].sum())))
        write_json(out/f'index_{slot}.json',index)
        write_json(out/f'progress_{slot}.json',dict(completed_batches=number+1,total_batches=len(assigned),elapsed_seconds=time.monotonic()-start,timings=dict(timing)))
    if acq._cache_size() or lab._cache_size() or graph._cache_size():raise ValueError('Implicit collection compilation')
    write_json(out/f'complete_{slot}.json',dict(status='completed',mode=mode,backend=jax.default_backend(),
        index_sha256=sha256(out/f'index_{slot}.json'),manifest_sha256=sha256(root/'manifest.json'),
        implicit_signatures=0,explicit_signatures=3,elapsed_seconds=time.monotonic()-start,timings=dict(timing)))

def review(dataset):
    from .unicycle_data import (
        MODES)
    root=Path(dataset);m,base,records=validate_source(root)
    index=[];seen=set();counts=Counter();steps=0;acquisitions={}
    for mode in MODES:
        for slot in range(4):
            done=read(root/mode/f'complete_{slot}.json')
            if (done['status']!='completed' or done['backend']!='cpu' or done['implicit_signatures']
                or done['manifest_sha256']!=sha256(root/'manifest.json')
                or done['index_sha256']!=sha256(root/mode/f'index_{slot}.json')):
                raise ValueError('Incomplete collection lane')
            for row in read(root/mode/f'index_{slot}.json'):
                for f,k in (('file','sha256'),('audit_file','audit_sha256'),('acquisition_file','acquisition_sha256')):
                    if sha256(root/row[f])!=row[k]:raise ValueError('Changed acquired evidence')
                if not all(a['audit_passed'] and a['independent_observer_audit_passed'] for a in row['acquisition_audits']):
                    raise ValueError('Unaudited acquisition')
                with np.load(root/row['file']) as z,np.load(root/row['audit_file']) as a:
                    np.testing.assert_array_equal(z['group_id'],[records[i]['group_id'] for i in row['indices']])
                    np.testing.assert_array_equal(z['partition'],[records[i]['partition'] for i in row['indices']])
                    if np.any(z['prior_status']!=0) or not a['audit_passed'].all() or not a['independent_observer_audit_passed'].all():
                        raise ValueError('Terminal or unaudited query')
                    np.testing.assert_array_equal(a['applied_steps'],z['steps'])
                    s=z['status'];count={str(k):int(v) for k,v in zip(*np.unique(s,return_counts=True))}
                    if count!=row['status_counts'] or int(z['steps'].sum())!=row['applied_steps']:
                        raise ValueError('Incorrect branch summary')
                    valid=np.stack((np.isin(s,(1,2,4)),np.isin(s,(1,4))),-1)
                    np.testing.assert_array_equal(z['target_mask'],valid)
                    np.testing.assert_array_equal(z['target'],np.where(valid,np.stack((-np.minimum(z['min_clearance'],.6)/.3,z['progress']/12),-1),0.).astype(np.float32))
                    np.testing.assert_array_equal(z['events'],np.stack((s==2,np.isin(s,(3,5,8))),-1))
                    if not z['event_mask'].all():raise ValueError('Failure label censored')
                    for i in row['indices']:
                        key=(mode,i,row['visit'])
                        if key in seen:raise ValueError('Duplicate observation')
                        seen.add(key)
                counts.update(count);steps+=row['applied_steps'];index.append(row)
                if row['acquisition_file'] not in acquisitions:
                    with np.load(root/row['acquisition_file']) as z:
                        acquisitions[row['acquisition_file']]=dict(mode=mode,indices=z['indices'].tolist(),
                            statuses=z['status'].tolist(),steps=z['steps'].tolist(),ticks=z['ticks'].tolist())
    if seen!={(mode,i,v) for mode in MODES for i in m['indices'] for v in range(4)}:
        raise ValueError('Missing shared coverage')
    index.sort(key=lambda r:(MODES.index(r['mode']),r['indices'][0],r['visit']))
    write_json(root/'index.json',index);write_json(root/'acquisitions.json',acquisitions)
    report=dict(schema=SCHEMA,manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),
        acquisitions_sha256=sha256(root/'acquisitions.json'),observations=len(seen),branches=sum(counts.values()),
        branch_statuses=dict(counts),applied_steps=steps,physical_parent_count=m['selected_parent_count'],
        all_outcomes_retained=True,all_saved_bindings_checked=True,all_branches_audited=True,all_observer_memories_audited=True,
        pilot=m['pilot'],weight_fit_authorized=m['weight_fit_authorized'],forward_parents_used=False,whole_goal_complete=False)
    write_json(root/'audit.json',report)
    return report

def replay(dataset,entry):
    from .unicycle_data import (
        acquisition_errors)
    from .unicycle_data import (
        array_digest)
    from .unicycle_data import branch_inputs
    from .unicycle_data import (
        future_errors)
    from .unicycle_data import label_kernel
    from .unicycle_data import snapshot
    from .unicycle_data import (
        tree_digest)
    root=Path(dataset); m,base,records=validate_source(root); row=entry; ids=row['indices']; mode=row['mode']; visit=row['visit']
    if jax.default_backend()!='cpu':raise ValueError('Exact replay requires CPU')
    for f,k in (('file','sha256'),('audit_file','audit_sha256'),('acquisition_file','acquisition_sha256')):
        if sha256(root/row[f])!=row[k]:raise ValueError('Changed replay input')
    with np.load(base/'source.npz') as z:initial,world,goal,bank=(z[k] for k in ('initial','world','goal','gains'))
    errors=np.stack([acquisition_errors(records[i]) for i in ids]);acq,params,calibration=acquisition(m,mode)
    args=(params,calibration,*map(jnp.asarray,(initial[ids],world[ids],goal[ids],errors,bank)))
    ar,at=jax.device_get(acq.lower(*args).compile()(*args));rebuilt=acquisition_record(np.asarray(ids),errors,ar,at)
    with np.load(root/row['acquisition_file']) as saved:
        if set(saved.files)!=set(rebuilt):raise ValueError('Changed compact acquisition fields')
        for k,v in rebuilt.items():np.testing.assert_array_equal(v,saved[k])
    ticks=rebuilt['ticks'][:,visit];x,memory=snapshot(at,ticks);current=errors[np.arange(4),ticks]
    future=np.stack([future_errors(records[i],mode,visit,ticks[b],current[b]) for b,i in enumerate(ids)])
    args,gains,expanded=branch_inputs(x,world[ids],goal[ids],bank,future,memory)
    fn=label_kernel();result,trace=jax.device_get(fn.lower(*args).compile()(*args))
    with np.load(root/row['file']) as saved:
        for k,v in result.items():np.testing.assert_array_equal(v,saved[k])
        np.testing.assert_array_equal(x,saved['initial_state'])
    with np.load(root/row['audit_file']) as saved:
        np.testing.assert_array_equal(current,saved['current_errors'])
        np.testing.assert_array_equal([array_digest(v) for v in future],saved['future_errors_sha256'])
        np.testing.assert_array_equal([tree_digest({k:v[b] for k,v in trace.items()}) for b in range(4)],saved['full_trace_sha256'])
    checked=branch_audits(x,world[ids],goal[ids],expanded,memory,result,trace)
    if acq._cache_size() or fn._cache_size():raise ValueError('Implicit replay compilation')
    return dict(file=row['file'],branches=len(checked),exact_acquisition_and_memory=True,exact_results_and_full_traces=True,
        applied_steps=sum(a['applied_steps'] for a in checked),independent_physical_reaudit=True)

def run(directory,navigation_job,output,pilot=False,qualification=None):
    from .unicycle_data import (
        MODES)
    job=Path(directory);root=Path(output);start=time.monotonic()
    allowance=32*2**20
    if not pilot:
        if not qualification:raise ValueError('Full collection needs measured replay qualification')
        q=read(qualification)
        if not q['qualified'] or q['navigation_protocol_sha256']!=sha256(Path(navigation_job)/'protocol.json') or q['collector_sha256']!=sha256(__file__):
            raise ValueError('Changed pilot qualification')
        for p,d in q['bound_files'].items():
            if sha256(p)!=d:raise ValueError('Changed pilot evidence')
        allowance=q['full_storage_forecast_bytes']+32*2**20
    if shutil.disk_usage('.').free<90*2**30+allowance:raise RuntimeError('Measured compact allowance plus90GiB reserve required')
    manifest=reserve(navigation_job,root,pilot)
    write_json(job/'protocol.json',dict(navigation_job=str(Path(navigation_job).resolve()),output=str(root.resolve()),
        manifest_sha256=sha256(root/'manifest.json'),pilot=pilot,qualification=qualification,storage_allowance_bytes=allowance,
        workers=4,threads_per_worker=14,cpu_buffer=8,forward_parents_used=False,whole_goal_complete=False))
    mode_elapsed={}
    for mode in MODES:
        t=time.monotonic();write_json(job/'progress.json',dict(stage=mode,mode_elapsed=mode_elapsed,estimated_remaining_seconds=3600 if not pilot else 180))
        def child(slot):
            launch_child(job,mode+'_'+str(slot),['oa_cbf_jax.unicycle_coverage','collect','--dataset',root,'--mode',mode,'--slot',slot],'cpu',slot)
        with ThreadPoolExecutor(4) as pool:list(pool.map(child,range(4)))
        mode_elapsed[mode]=time.monotonic()-t
    report=review(root);replayed=[]
    for mode in MODES:
        rows=read(root/mode/'index_0.json')
        for row in (rows[0],rows[3]):replayed.append(replay(root,row))
    write_json(root/'independent_compact_replay.json',dict(exact_reconstruction=True,replayed=replayed,
        manifest_sha256=sha256(root/'manifest.json'),audit_sha256=sha256(root/'audit.json'),whole_goal_complete=False))
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if pilot:
        write_json(root/'qualification.json',dict(qualified=True,navigation_protocol_sha256=sha256(Path(navigation_job)/'protocol.json'),
            collector_sha256=manifest['collector_sha256'],branches=report['branches'],applied_steps=report['applied_steps'],
            measured_bytes=size,full_storage_forecast_bytes=size*(3072/manifest['variants_per_policy'])*1.5,
            mode_elapsed=mode_elapsed,pilot_parents=manifest['selected_parent_count'],exact_replay_batches=len(replayed),
            bound_files={str((root/n).resolve()):sha256(root/n) for n in ('manifest.json','index.json','audit.json','independent_compact_replay.json')},
            scope='Predetermined TRAIN-only numeric/replay/storage pilot; no fitting or model selection.',whole_goal_complete=False))
    write_json(job/'complete.json',dict(status='completed',output=str(root.resolve()),elapsed_seconds=time.monotonic()-start,
        audit_sha256=sha256(root/'audit.json'),mode_elapsed=mode_elapsed,measured_bytes=size,pilot=pilot,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('run')
    for k in ('directory','navigation-job','output'):a.add_argument('--'+k,required=True)
    a.add_argument('--pilot',action='store_true');a.add_argument('--qualification')
    a=sub.add_parser('collect');a.add_argument('--dataset',required=True);a.add_argument('--mode',required=True);a.add_argument('--slot',type=int,default=0)
    a=sub.add_parser('review');a.add_argument('--dataset',required=True)
    args=vars(p.parse_args());action=args.pop('action');globals()[action](**args)
