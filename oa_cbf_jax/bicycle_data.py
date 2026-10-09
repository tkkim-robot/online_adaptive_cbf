"""Bicycle data functions and shared contracts."""

import argparse

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from dataclasses import asdict

import json

from multiprocessing import get_context

from pathlib import Path

import shutil

import time

import jax

import jax.numpy as jnp

import numpy as np

from .bicycle_control import BicycleControlConfig

from .bicycle_guidance import BicycleGuidanceConfig

from .bicycle_observation import sample_bias, observe, BASE_NOISE, SCHEMA as SENSOR_SCHEMA

from .bicycle_observed_rollout import make_observed_episode

from .bicycle_features import bicycle_graph, SCHEMA as GRAPH_SCHEMA

from .bicycle_control import read, control_config

from .bicycle_rollout import NAMES, GOAL, COLLISION, TIMEOUT, STATE_BOUND

from .io import sha256, source_fingerprint

from .io import write_json

from .io import sanitize

from .bicycle_trace_storage import open_trace, write_query_traces, verify_index_dependencies, STORAGE_SCHEMA

SCHEMA='oa_cbf_bicycle_acquired_history_hurdle_v67'

FIELDS=('initial','goal','obstacles','mask','alpha','points','route_mask','ready','cursor','first_x','first_o','bias_x','bias_o','noise','key')

def prepare(output,groups=64,seed=6671,acquisition_mode='fixed2'):
    from .scenes import multiscale_scenes_scene as scene
    from .scenes import DIVERSE_FAMILIES
    from .routing import plan_route
    if groups<64 or groups%8:raise ValueError('At least64 family-balanced physical parents required')
    if acquisition_mode not in ('fixed2','balanced8'):raise ValueError('Unknown acquisition mode')
    if acquisition_mode=='balanced8' and groups%256:raise ValueError('Balanced acquisition requires complete family/noise/gain cells (multiple of256)')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);c=BicycleControlConfig();rng=np.random.default_rng(seed)
    assignments={}
    for family in DIVERSE_FAMILIES:
        order=rng.permutation(groups//8)
        for k,i in enumerate(order):assignments[family,int(i)]='train' if k<int(.7*len(order)) else 'validation' if k<int(.85*len(order)) else 'development_calibration'
    calibration_roles={}
    for family in DIVERSE_FAMILIES:
        reserved=[j for j in range(groups//8) if assignments[family,j]=='development_calibration']
        for k,j in enumerate(rng.permutation(reserved)):
            calibration_roles[family,int(j)]='prediction_fit' if k<len(reserved)//2 else 'prediction_audit'
    raw=[]
    for i in range(groups):
        family=DIVERSE_FAMILIES[i%8];local=seed*100000+i;s=scene(local,family);r=np.random.default_rng(local+337)
        x=s.initial_state.astype(np.float32);x[3]=r.uniform(.25,1.)
        acquisition_gain=float(np.geomspace(.5,8,8).astype(np.float32)[(i//32)%8]) if acquisition_mode=='balanced8' else 2.
        raw.append(dict(group_id=f'bicycle_acquired_v67:{family}:{local}',family=family,seed=local,partition=assignments[family,i//8],
            acquisition_gain=acquisition_gain,calibration_role=calibration_roles.get((family,i//8),'none'),
            initial=x,goal=s.goal.astype(np.float32),obstacles=s.obstacles.astype(np.float32),mask=s.obstacle_mask,
            noise=BASE_NOISE*np.float32([0.,.5,1.,2.][(i//8)%4])))
    excluded=[]
    for name in ('bicycle_v63_mechanics_inputs','bicycle_v63_rounded_diverse_inputs','bicycle_v66_id_inputs','bicycle_v66_ood_inputs'):
        path=Path('artifacts/experiments')/name/'scenes.json'
        if not path.exists():raise ValueError('Missing prior evaluation lineage')
        excluded.extend(read(path))
    prior=Path('artifacts/experiments/bicycle_v67_acquired_inputs/scenes.json')
    if acquisition_mode=='balanced8' and prior.exists():excluded.extend(read(prior))
    if {r['seed'] for r in raw}&{r['seed'] for r in excluded}:raise ValueError('Training/evaluation seed overlap')
    bias_fn=jax.jit(jax.vmap(sample_bias));keys=jnp.stack([jax.random.PRNGKey(r['seed']+1) for r in raw])
    bx,bo=jax.device_get(bias_fn(keys,jnp.asarray(np.array([r['noise'] for r in raw])),jnp.asarray(np.array([r['mask'] for r in raw]))))
    seen_fn=jax.jit(jax.vmap(lambda x,o,m,bx,bo,n:observe(x,o,m,bx,bo,n,jnp.zeros(4),jnp.zeros_like(o))))
    first_x,first_o=jax.device_get(seen_fn(*[jnp.asarray(a) for a in (np.array([r['initial'] for r in raw]),np.array([r['obstacles'] for r in raw]),np.array([r['mask'] for r in raw]),bx,bo,np.array([r['noise'] for r in raw]))]))
    def finish(i):
        r=raw[i];route=plan_route(first_x[i,:2],r['goal'],first_o[i],r['mask'],c,capacity=64,visibility_batch_nodes=32)
        value={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}
        value.update(first_x=first_x[i].tolist(),first_o=first_o[i].tolist(),bias_x=bx[i].tolist(),bias_o=bo[i].tolist(),
            route={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(route).items()},solvability='unknown')
        return value
    with ThreadPoolExecutor(32) as pool:parents=list(pool.map(finish,range(groups)))
    write_json(root/'scenes.json',parents)
    write_json(root/'manifest.json',dict(schema=SCHEMA,config=asdict(c),source_fingerprint=source_fingerprint(),seed=seed,groups=groups,scenes_sha256=sha256(root/'scenes.json'),
        training_use=True,weight_fit_authorized=True,final_test=False,data_role='acquired_history_training_development',acquisition_mode=acquisition_mode,sensor_schema=SENSOR_SCHEMA,noise_fields=['ego_xy','ego_heading','ego_speed','obstacle_xy','obstacle_velocity','obstacle_radius'],
        acquisition='Actual physical parents and persistent bounded biases. Initial map and route use observed state/obstacles. Position/velocity biases are isotropic uniform disks; angle/speed/radius are uniform scalar intervals. Future innovations independent,15percent of declared ranges. No clipping physical states/radii.',
        split='Family-stratified70/15/15 integer allocation before acquisition. All visits/gains/replicas inherit the physical parent partition. Reserved development-calibration parents split within each family into prediction_fit/prediction_audit before acquisition; neither used for weight fitting. Final trajectory calibration requires new parents after policy freeze.',
        distribution='Fresh multiscale physical geometry, initialspeed.25..1, noise0/.5/1/2 balanced within each family. No outcome filtering or reference hero coordinates.',
        limitation='One latent history per physical parent, future-innovation replicas; not an exact posterior conditioned only on a noisy query. Pilot, not final calibration/generalization evidence.'))

def args(row):
    return tuple(jnp.asarray(row[k],dtype=(jnp.float64 if k in ('initial','obstacles') else bool if k in ('mask','route_mask','ready') else jnp.uint32 if k=='key' else jnp.float32)) for k in FIELDS)

def trace_payload(row,summary,trace,horizon):
    status=int(summary['status']);steps=int(summary['steps']);length=max(1,min(horizon,steps+int(status not in (GOAL,COLLISION,TIMEOUT,STATE_BOUND))))
    value={k:np.asarray(v)[:length] for k,v in trace.items()}
    for k in FIELDS:value[k]=np.asarray(row[k],dtype=(np.float64 if k in ('initial','obstacles') else bool if k in ('mask','route_mask','ready') else np.uint32 if k=='key' else np.float32))
    value.update(final_status=np.int32(status),expected_steps=np.int32(steps),horizon=np.int32(horizon))
    return value

def targets(summary,horizon,config):
    status=np.asarray(summary['status']);adverse=~np.isin(status,[GOAL,TIMEOUT]);collision=status==COLLISION;valid=~adverse|collision
    progress=np.asarray(summary['route_progress'])/(horizon*config.robot.dt*config.cruise_speed)
    y=np.stack((np.where(valid,-np.minimum(summary['min_clearance'],.6)/.3,0.),progress),axis=-1).astype(np.float32)
    return dict(target=y,target_mask=np.stack((valid,np.ones_like(valid)),axis=-1),events=np.stack((collision,adverse),axis=-1).astype(np.float32),event_mask=np.stack((valid,np.ones_like(valid)),axis=-1))

def collection_layout(replicas,acquisition_steps,snapshot_ticks):
    ticks=tuple(snapshot_ticks)
    if replicas<2 or replicas%2 or acquisition_steps<1:raise ValueError('Require an even replica count >=2 and positive acquisition horizon')
    if not ticks or ticks[0]!=0 or sorted(set(ticks))!=list(ticks) or ticks[-1]>=acquisition_steps:
        raise ValueError('Acquisition ticks must start at zero, increase uniquely, and precede horizon')
    gains=np.geomspace(.5,8.,8).astype(np.float32)
    return ticks,np.repeat(gains,replicas)

def parent_indices(total,shard,shards,layout='round_robin'):
    if shards<1 or not 0<=shard<shards:raise ValueError('Invalid worker index')
    if layout=='round_robin':return [i for i in range(total) if i%shards==shard]
    if layout!='balanced_cells' or shards!=4 or total%256:
        raise ValueError('Balanced cells require four workers and complete256parent blocks')
    # Prepared source order: family fastest, then noise, then acquisition gain,
    # then cell replicate. A1024 block gives every cell to each lane; smaller
    # complete256 blocks still balance family/noise/gain marginals per lane.
    return [i for i in range(total) if (i%8+(i//8)%4+(i//32)%8+i//256)%4==shard]

def collect(source,output,shard=0,shards=4,horizon=80,acquisition_steps=160,replicas=2,snapshot_ticks=(0,40,120),min_free_gib=0.,shard_layout='round_robin',observation_margin=False,shared_prefix_storage=False):
    snapshot_ticks,canonical=collection_layout(replicas,acquisition_steps,snapshot_ticks)
    if horizon<1 or shards<1 or not 0<=shard<shards:raise ValueError('Invalid label horizon or shard')
    branches=len(canonical)
    root=Path(output);root.mkdir(parents=True,exist_ok=False);source=Path(source);sm=read(source/'manifest.json');all_parents=read(source/'scenes.json')
    if sm['schema']!=SCHEMA or sm.get('weight_fit_authorized') is not True or sha256(source/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Invalid training source')
    if shard_layout=='balanced_cells' and sm.get('acquisition_mode')!='balanced8':raise ValueError('Balanced cells require the balanced acquisition source')
    c=control_config(sm['config']);guidance=BicycleGuidanceConfig(observation_margin=observation_margin);parents=[all_parents[i] for i in parent_indices(len(all_parents),shard,shards,shard_layout)]
    manifest=dict(schema=SCHEMA,stage='bicycle_acquired_history_pilot',capacity=64,source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_fingerprint=source_fingerprint(),config=asdict(c),
        controller=dict(predictive_guidance=asdict(guidance),speed_rows='Worst-case speed in current observation interval;1.15timesknown speed range+5e-7roundoff ifnonzero. Originalphysical.2..3.5envelope unchanged.'),
        sensor_schema=SENSOR_SCHEMA,graph_schema=GRAPH_SCHEMA,graph_features=35,gain_dimension=1,gain_domain=dict(lower=.5,upper=8),queries=8,replicas=replicas,horizon_steps=horizon,acquisition_steps=acquisition_steps,
        acquisition_gain=2. if sm.get('acquisition_mode','fixed2')=='fixed2' else 'per_parent_observed',acquisition_mode=sm.get('acquisition_mode','fixed2'),snapshot_ticks=list(snapshot_ticks),shard=shard,shards=shards,shard_layout=shard_layout,training_use=True,weight_fit_authorized=True,production_eligible=False,final_test=False,
        groups=[{k:r[k] for k in ('group_id','family','seed','partition')} for r in all_parents],targets=['conditional_negative_physical_clearance_div_.3','retained_route_progress_div_horizon_cruise'],events=['collision_first','any_adverse_termination'],
        conditioning='Physical state/obstacle time/persistent biases and exact acquired first observation retained; only later innovations resampled, paired across all gains. Same observation-only graph per query. No true state, biases, future innovations, family or identity supplied to controller/graph.',
        censoring='Risk/collision negative labels masked on earlier adverse termination. Adverse-event and retained-prefix progress supervised. Label horizon timeout is horizon end, not full-task success. No post-stop future.',
        limitation='Small noisy acquired-history pilot; each query has one latent history, not a full posterior. Observed CBF rows are not robust barrier certificates. All actual physical failures retained; no calibration or learned-policy claim.')
    if shared_prefix_storage:manifest['trace_storage']=STORAGE_SCHEMA
    if 'routing_contract' in sm:manifest['controller']['routing']=sm['routing_contract']
    write_json(root/'manifest.json',manifest);index=[];traces=[];visited=[];start=time.monotonic()
    acquisition_fn=jax.jit(make_observed_episode(c,acquisition_steps,guidance));branch_fn=jax.jit(jax.vmap(make_observed_episode(c,horizon,guidance)));graph_fn=jax.jit(bicycle_graph)
    acq_exec=branch_exec=graph_exec=None;compile_seconds=0.
    for number,parent in enumerate(parents):
        if shutil.disk_usage(root).free/2**30<min_free_gib:raise ValueError('Collection disk safety buffer reached; retained completed evidence')
        acquisition_gain=np.float32(parent.get('acquisition_gain',2.))
        row=dict(initial=np.array(parent['initial'],np.float64),goal=np.array(parent['goal'],np.float32),obstacles=np.array(parent['obstacles'],np.float64),mask=np.array(parent['mask'],bool),alpha=acquisition_gain,
            points=np.array(parent['route']['points'],np.float32),route_mask=np.array(parent['route']['mask'],bool),ready=parent['route']['status']=='ready',cursor=np.float32(0.),
            first_x=np.array(parent['first_x'],np.float32),first_o=np.array(parent['first_o'],np.float32),bias_x=np.array(parent['bias_x'],np.float32),bias_o=np.array(parent['bias_o'],np.float32),noise=np.array(parent['noise'],np.float32),key=np.array(jax.random.PRNGKey(parent['seed']+2)))
        acq_args=args(row)
        if acq_exec is None:
            t=time.monotonic();acq_exec=acquisition_fn.lower(*acq_args).compile();compile_seconds+=time.monotonic()-t
            print(json.dumps(dict(stage='acquisition_compiled',seconds=compile_seconds,device=str(jax.devices()[0]))),flush=True)
        summary,history=jax.device_get(acq_exec(*acq_args));d=trace_payload(row,summary,history,acquisition_steps)
        file=f'acquisition_{number:03d}.npz';np.savez_compressed(root/file,**d)
        acquisition=dict(file=file,sha256=sha256(root/file),group_id=parent['group_id'],kind='acquisition',steps=int(summary['steps']),status=NAMES[int(summary['status'])]);traces.append(acquisition)
        eligible=[t for t in snapshot_ticks if t==0 or (t<len(d['active']) and (d['active'][t] or d['status'][t] in (3,5)))]
        for tick in eligible:
            query=dict(row);query.update(initial=d['state_before'][tick],obstacles=row['obstacles'].copy(),first_x=d['observed_state'][tick],first_o=d['observed_obstacles'][tick],cursor=d['cursor_before'][tick])
            query['obstacles'][:,:2]+=tick*c.robot.dt*query['obstacles'][:,3:5]
            previous=np.zeros(2,np.float32) if tick==0 else d['control'][tick-1]
            graph_args=tuple(jnp.asarray(v) for v in (query['first_x'],query['goal'],query['first_o'],query['mask'],query['points'],query['route_mask'],query['cursor'],previous,acquisition_gain,query['noise']))
            if graph_exec is None:
                t=time.monotonic();graph_exec=graph_fn.lower(*graph_args).compile();compile_seconds+=time.monotonic()-t
            features,node_mask=jax.device_get(graph_exec(*graph_args));sums=[];entries=[];query_payloads=[]
            keys=np.array(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(parent['seed']+3),tick),replicas));branch_rows=[]
            for candidate,alpha in enumerate(canonical):branch_rows.append(dict(query,alpha=alpha,key=keys[candidate%replicas]))
            for offset in range(0,branches,8):
                batch_rows=[args(r) for r in branch_rows[offset:offset+8]]
                batch=tuple(jnp.stack([r[i] for r in batch_rows]) for i in range(len(FIELDS)))
                if branch_exec is None:
                    t=time.monotonic();branch_exec=branch_fn.lower(*batch).compile();compile_seconds+=time.monotonic()-t
                    print(json.dumps(dict(stage='labels_compiled',total_compile_seconds=compile_seconds,device=str(jax.devices()[0]))),flush=True)
                summaries,histories=jax.device_get(branch_exec(*batch))
                for i in range(8):
                    candidate=offset+i;ss={k:v[i] for k,v in summaries.items()};hh={k:v[i] for k,v in histories.items()};sums.append(ss)
                    file=f'branch_{number:03d}_t{tick:03d}_c{candidate:02d}.npz';payload=trace_payload(branch_rows[candidate],ss,hh,horizon)
                    if shared_prefix_storage:query_payloads.append(payload)
                    else:np.savez_compressed(root/file,**payload)
                    entry=dict(file=file,group_id=parent['group_id'],kind='label',query_tick=tick,candidate=candidate,steps=int(ss['steps']),status=NAMES[int(ss['status'])],acquisition_file=acquisition['file'],acquisition_sha256=acquisition['sha256'])
                    if not shared_prefix_storage:entry['sha256']=sha256(root/file)
                    traces.append(entry);entries.append(entry)
            if shared_prefix_storage:
                records=write_query_traces(root,[e['file'] for e in entries],query_payloads,replicas,f'query_{number:03d}_t{tick:03d}')
                for entry,record in zip(entries,records,strict=True):entry.update(record)
                del query_payloads
            sums={k:np.asarray([s[k] for s in sums]) for k in sums[0]};labels=targets(sums,horizon,c)
            payload=dict(features=features[None],node_mask=node_mask[None],gains=canonical[None,:,None],group_id=np.array([parent['group_id']]),partition=np.array([parent['partition']]),query_tick=np.array([tick]),
                initial_state=query['initial'][None],goal=query['goal'][None],obstacles=query['obstacles'][None],obstacle_mask=query['mask'][None],points=query['points'][None],route_mask=query['route_mask'][None],
                observed_state=query['first_x'][None],observed_obstacles=query['first_o'][None],cursor=np.array([query['cursor']]),previous_control=previous[None],previous_gain=np.array([acquisition_gain],np.float32),noise=query['noise'][None],calibration_role=np.array([parent.get('calibration_role','unassigned')]),
                **{k:v[None] for k,v in sums.items()},**{k:v[None] for k,v in labels.items()})
            file=f'shard_{number:03d}_t{tick:03d}.npz';np.savez_compressed(root/file,**payload)
            index.append(dict(file=file,sha256=sha256(root/file),groups=1,branches=branches,observed_steps=int(sums['steps'].sum()),group_id=parent['group_id'],query_tick=tick,traces=entries))
        visited.append(dict(group_id=parent['group_id'],available_snapshot_ticks=eligible,acquisition=acquisition))
        write_json(root/'index.json',index);write_json(root/'trace_index.json',traces);write_json(root/'visitation.json',visited)
        print(json.dumps(dict(completed_parents=number+1,total_parents=len(parents),queries=len(index),branches=branches*len(index),label_physical_steps=sum(e['observed_steps'] for e in index),elapsed_seconds=time.monotonic()-start)),flush=True)
    if any(f._cache_size()!=0 for f in (acquisition_fn,branch_fn,graph_fn)):raise ValueError('Unexpected runtime JIT during collection')
    write_json(root/'summary.json',dict(complete=True,compiled_signatures=3,implicit_jit_cache_entries=0,compile_seconds=compile_seconds,execution_seconds=time.monotonic()-start,parents=len(parents),queries=len(index),branches=branches*len(index),physical_steps=sum(e['steps'] for e in traces)))

def audit_one(payload):
    root,entry,config,margin=payload
    from .bicycle_observed_audit import audit_trace
    with open_trace(root/entry['file'],entry['sha256']) as d:pass
    c=control_config(config);result=audit_trace(d,c)
    if margin:
        from .bicycle_margin_audit import audit_margin_trace
        result['margin']=audit_margin_trace(d,c)
    return sanitize(dict(file=entry['file'],sha256=entry['sha256'],**result))

def audit(directory,workers=12):
    from .bicycle_observed_audit import check_graph, check_route_progress
    root=Path(directory);m=read(root/'manifest.json');source=Path(m['source']);sm=read(source/'manifest.json');c=control_config(m['config'])
    if sha256(source/'manifest.json')!=m['source_manifest_sha256'] or sha256(source/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Changed physical/acquisition source')
    parents={r['group_id']:r for r in read(source/'scenes.json')};entries=read(root/'index.json');traces=read(root/'trace_index.json');visits=read(root/'visitation.json')
    storage_proof=verify_index_dependencies(root,traces) if m.get('trace_storage')==STORAGE_SCHEMA else None
    by_trace={r['file']:r for r in traces}
    ticks,canonical=collection_layout(m['replicas'],m['acquisition_steps'],m['snapshot_ticks']);branches=len(canonical)
    source_rows=read(source/'scenes.json')
    expected=[source_rows[i]['group_id'] for i in parent_indices(len(source_rows),m['shard'],m['shards'],m.get('shard_layout','round_robin'))]
    if [r['group_id'] for r in visits]!=expected:raise ValueError('Dropped/reordered parent')
    for visit in visits:
        with np.load(root/visit['acquisition']['file']) as acq:
            wanted=[t for t in ticks if t==0 or (t<len(acq['active']) and (acq['active'][t] or acq['status'][t] in (3,5)))]
        if visit['available_snapshot_ticks']!=wanted:raise ValueError('Acquired query censored or fabricated')
    from .bicycle_guidance import guidance_from_controller
    margin=guidance_from_controller(m['controller']).observation_margin
    results=[]
    with ProcessPoolExecutor(workers,mp_context=get_context('spawn')) as pool:
        for result in pool.map(audit_one,[(root,e,m['config'],margin) for e in traces],chunksize=1):
            results.append(result)
            if len(results)%512==0 or len(results)==len(traces):
                print(json.dumps(dict(stage='independent_physical_replay',checked=len(results),total=len(traces))),flush=True)
    by_file={r['file']:r for r in results}
    query_keys=set();progress_rounding_checks=0
    for entry_number,entry in enumerate(entries):
        if sha256(root/entry['file'])!=entry['sha256']:raise ValueError('Changed training shard')
        with np.load(root/entry['file']) as f:d={k:f[k] for k in f}
        parent=parents[entry['group_id']];tick=entry['query_tick'];query_keys.add((entry['group_id'],tick))
        assert d['group_id'][0]==entry['group_id'] and d['partition'][0]==parent['partition']
        if 'calibration_role' in d:assert d['calibration_role'][0]==parent.get('calibration_role','unassigned')
        np.testing.assert_array_equal(d['gains'][0,:,0],canonical)
        args=(d['observed_state'][0],d['goal'][0],d['observed_obstacles'][0],d['obstacle_mask'][0],d['points'][0],d['route_mask'][0],d['noise'][0],c,float(d['cursor'][0]),d['previous_control'][0],float(d['previous_gain'][0]))
        check_graph(d['features'][0],d['node_mask'][0],args)
        expected_labels=targets({k:d[k][0] for k in ('status','min_clearance','route_progress')},m['horizon_steps'],c)
        for k,v in expected_labels.items():np.testing.assert_array_equal(d[k][0],v)
        if len(entry['traces'])!=branches or entry['branches']!=branches:raise ValueError('Missing replica/candidate')
        seen_keys={};common_innovations={}
        for candidate,tr in enumerate(entry['traces']):
            if tr!=by_trace[tr['file']]:raise ValueError('Trace/query index mismatch')
            with open_trace(root/tr['file'],tr['sha256']) as b,np.load(root/tr['acquisition_file']) as acq:
                if sha256(root/tr['acquisition_file'])!=tr['acquisition_sha256']:raise ValueError('Changed acquisition lineage')
                for field,parent_field in [('initial','initial'),('goal','goal'),('obstacles','obstacles'),('mask','mask'),('bias_x','bias_x'),('bias_o','bias_o'),('noise','noise'),('first_x','first_x'),('first_o','first_o')]:
                    np.testing.assert_array_equal(acq[field],np.asarray(parent[parent_field],acq[field].dtype))
                np.testing.assert_array_equal(acq['points'],np.asarray(parent['route']['points'],np.float32));np.testing.assert_array_equal(acq['route_mask'],parent['route']['mask'])
                np.testing.assert_array_equal(acq['alpha'],np.float32(parent.get('acquisition_gain',2.)))
                np.testing.assert_array_equal(d['previous_gain'][0],acq['alpha'])
                np.testing.assert_array_equal(d['previous_control'][0],np.zeros(2,np.float32) if tick==0 else acq['control'][tick-1])
                np.testing.assert_array_equal(b['initial'],acq['state_before'][tick]);np.testing.assert_array_equal(b['first_x'],acq['observed_state'][tick]);np.testing.assert_array_equal(b['first_o'],acq['observed_obstacles'][tick])
                moving=acq['obstacles'].copy();moving[:,:2]+=tick*c.robot.dt*moving[:,3:5];np.testing.assert_array_equal(b['obstacles'],moving)
                for field in ('bias_x','bias_o','mask','goal','noise','points','route_mask'):np.testing.assert_array_equal(b[field],acq[field])
                np.testing.assert_array_equal(b['cursor'],acq['cursor_before'][tick]);np.testing.assert_array_equal(d['initial_state'][0],b['initial']);np.testing.assert_array_equal(d['observed_state'][0],b['first_x']);np.testing.assert_array_equal(d['observed_obstacles'][0],b['first_o'])
                assert float(b['alpha'])==float(d['gains'][0,candidate,0]) and int(b['expected_steps'])==int(d['steps'][0,candidate]) and int(b['final_status'])==int(d['status'][0,candidate])
                np.testing.assert_array_equal(d['final_state'][0,candidate],b['state'][-1]);np.testing.assert_array_equal(d['final_cursor'][0,candidate],b['route_progress'][-1])
                independent=by_file[tr['file']]
                np.testing.assert_allclose(d['min_clearance'][0,candidate],independent['min_clearance'],atol=1e-4,rtol=0)
                progress_rounding_checks+=check_route_progress(d['route_progress'][0,candidate],
                    b['initial'].astype(np.float32)[:2],b['state'][-1].astype(np.float32)[:2],b['points'],b['route_mask'],
                    float(b['cursor']),float(b['route_progress'][-1]))
                key=tuple(b['key']);replica=candidate%m['replicas']
                if replica in seen_keys:assert key==seen_keys[replica]
                seen_keys[replica]=key
                innovation=(b['innovation_x'],b['innovation_o'])
                if replica in common_innovations:
                    old=common_innovations[replica];n=min(len(old[0]),len(innovation[0]))
                    for left,right in zip(old,innovation):np.testing.assert_array_equal(left[:n],right[:n])
                if replica not in common_innovations or len(innovation[0])>len(common_innovations[replica][0]):common_innovations[replica]=tuple(v.copy() for v in innovation)
        assert len(set(seen_keys.values()))==m['replicas']
        if (entry_number+1)%128==0 or entry_number+1==len(entries):
            print(json.dumps(dict(stage='independent_label_lineage',checked=entry_number+1,total=len(entries))),flush=True)
    expected_queries={(r['group_id'],t) for r in visits for t in r['available_snapshot_ticks']}
    if query_keys!=expected_queries or len(query_keys)!=len(entries):raise ValueError('Missing or duplicate acquired query')
    result=dict(audit_passed=True,manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),trace_index_sha256=sha256(root/'trace_index.json'),
        source_manifest_sha256=m['source_manifest_sha256'],all_physical_prefixes_replayed=True,all_original_observed_rows_checked=True,all_graphs_independently_checked=True,all_acquired_history_bindings_checked=True,
        parents=len(visits),queries=len(entries),branches=branches*len(entries),physical_steps=sum(r['steps'] for r in results),feasible_qp_rejections=sum(r['feasible_qp_rejected'] for r in results),all_recorded_margin_bounds_checked=margin,rows=results)
    result['route_progress_rounding_checks']=int(progress_rounding_checks)
    if storage_proof is not None:result['trace_storage_verification']=storage_proof
    write_json(root/'independent_replay.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

def validate_training_dataset(directory):
    root=Path(directory);m=read(root/'manifest.json')
    from .bicycle_gain_contract import TRAIN_SCHEMA, validate_training_view
    if m['schema']==TRAIN_SCHEMA:
        return validate_training_view(root)
    if 'shared_observation_union' in m:
        from .bicycle_shared_data import validate_union
        return validate_union(root)
    a=read(root/'independent_replay.json');done=read(root/'complete.json')
    if (m['schema']!=SCHEMA or m.get('weight_fit_authorized') is not True or m.get('gain_dimension')!=1
            or done.get('status')!='completed' or not done.get('audit_passed') or done['audit_sha256']!=sha256(root/'independent_replay.json')
            or a['manifest_sha256']!=sha256(root/'manifest.json') or a['index_sha256']!=sha256(root/'index.json')):raise ValueError('Unaudited bicycle dataset')
    for key in ('audit_passed','all_physical_prefixes_replayed','all_original_observed_rows_checked','all_graphs_independently_checked','all_acquired_history_bindings_checked'):
        if a.get(key) is not True:raise ValueError('Incomplete bicycle audit: '+key)
    from .bicycle_guidance import guidance_from_controller
    margin=guidance_from_controller(m['controller']).observation_margin
    if margin and a.get('all_recorded_margin_bounds_checked') is not True:raise ValueError('Missing bicycle observation-margin audit')
    if sha256(Path(m['source'])/'manifest.json')!=m['source_manifest_sha256']:raise ValueError('Changed source reservation')
    for proof in a['parts']:
        p=root/proof['directory']
        for field,file in [('manifest_sha256','manifest.json'),('index_sha256','index.json'),('audit_sha256','independent_replay.json')]:
            if sha256(p/file)!=proof[field]:raise ValueError('Changed part evidence')
        pm=read(p/'manifest.json');pa=read(p/'independent_replay.json')
        if pm['controller']!=m['controller']:raise ValueError('Mixed bicycle guidance label semantics')
        if margin and pa.get('all_recorded_margin_bounds_checked') is not True:raise ValueError('Missing part observation-margin audit')
        if pm.get('trace_storage')!=m.get('trace_storage'):raise ValueError('Mixed trace storage contracts')
        if pm.get('trace_storage')==STORAGE_SCHEMA:
            if pa.get('trace_storage_verification',{}).get('all_shared_dependencies_verified') is not True or pa['trace_index_sha256']!=sha256(p/'trace_index.json'):
                raise ValueError('Missing or changed shared trace audit')
            verify_index_dependencies(p,read(p/'trace_index.json'))
    for e in read(root/'index.json'):
        if sha256(root/e['file'])!=e['sha256']:raise ValueError('Changed training labels')
    return m



import copy


from .scenes import DIVERSE_FAMILIES

PHASE = 'training_acquisition'

def select_training_parents(parents):
    """One lowest-seed TRAIN parent in each prespecified alternating gain cell."""
    train = [p for p in parents if p['partition'] == 'train']
    noises = sorted({p['noise'][0] for p in train})
    bank = np.geomspace(.5, 8, 8).astype(np.float32)
    if len(noises) != 4 or len({p['group_id'] for p in parents}) != len(parents):
        raise ValueError('Unique parents and all four training noise levels required')
    selected = []
    for fi, family in enumerate(DIVERSE_FAMILIES):
        for ni, noise in enumerate(noises):
            for gi, gain in enumerate(bank):
                if (fi+ni+gi) % 2:
                    continue
                cell = [p for p in train if p['family'] == family and p['noise'][0] == noise
                        and np.float32(p['acquisition_gain']) == gain]
                if not cell:
                    raise ValueError('Missing prespecified training cell')
                selected.append(copy.deepcopy(min(cell, key=lambda p: p['seed'])))
    if len(selected) != 128 or any(p['calibration_role'] != 'none' for p in selected):
        raise ValueError('128 train-only parents with no reserved calibration role required')
    return selected

def select_parents(parents):
    selected = select_training_parents(parents)
    validation = [p for p in parents if p['partition']=='validation']
    noises = sorted({p['noise'][0] for p in validation})
    if len(noises)!=4: raise ValueError('Four validation noise levels required')
    for family in DIVERSE_FAMILIES:
        for noise in noises:
            cell=[p for p in validation if p['family']==family and p['noise'][0]==noise]
            if not cell: raise ValueError('Missing validation family/noise cell')
            selected.append(copy.deepcopy(min(cell,key=lambda p:p['seed'])))
    if len(selected)!=160 or any(p['calibration_role']!='none' for p in selected):
        raise ValueError('Use128TRAIN/32validation parents, no calibration roles')
    return selected

def validate_source(directory):
    directory=Path(directory);m=read(directory/'manifest.json')
    if (m.get('data_role')!='reserved_bicycle_learned_policy_acquisition'
            or m.get('training_use') is not True or m.get('weight_fit_authorized') is not True
            or m.get('final_test') is not False or m.get('groups')!=160
            or sha256(directory/'scenes.json')!=m['scenes_sha256']):
        raise ValueError('Reserved acquisition source required')
    source=Path(m['acquisition_parent_source']);original=read(source/'manifest.json')
    if (sha256(source/'manifest.json')!=m['acquisition_parent_manifest_sha256']
            or sha256(source/'scenes.json')!=original['scenes_sha256']
            or original.get('training_use') is not True or original.get('weight_fit_authorized') is not True
            or original.get('final_test') is not False):
        raise ValueError('Changed or unauthorized original acquisition parents')
    parents=read(directory/'scenes.json')
    if parents!=select_parents(read(source/'scenes.json')):
        raise ValueError('Changed original physical parents, roles or selection')
    if m['config']!=original['config'] or m['routing_contract']!=original['routing_contract']:
        raise ValueError('Acquisition physics/routing changed')
    if m['phase_order']!={PHASE:list(range(160))}:
        raise ValueError('Missing or reordered acquisition parents')
    return m,parents


def source_map(dataset):
    root=Path(dataset);m=read(root/'manifest.json');union=m['shared_observation_union'];result={};bindings={}
    for base in [Path(union['base']),*[Path(p['directory']) for p in union['parts']]]:
        index=base/'index.json';bindings[str(index)]=sha256(index)
        for e in read(index):
            file=str((base/e['file']).resolve());t=e['traces'][0]
            path=(base/t['acquisition_file']).resolve()
            result[file]=dict(path=str(path),sha256=t['acquisition_sha256'],query_sha256=e['sha256'],
                group_id=e['group_id'],query_tick=e['query_tick'],query_origin=e.get('acquisition_encoder','fixed'))
    return result,bindings


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())

def verify_part(path):
    from .bicycle_policy_labels import HORIZON
    from .bicycle_policy_labels import SCHEMA as LABEL_VALIDATION_SCHEMA
    from .bicycle_policy_labels import REPLICAS
    p = Path(path); m = read(p/'manifest.json'); a = read(p/'independent_replay.json'); s = read(p/'summary.json')
    flags = ('audit_passed', 'all_physical_prefixes_replayed', 'all_original_observed_rows_checked',
             'all_graphs_independently_checked', 'all_acquired_history_bindings_checked', 'all_recorded_margin_bounds_checked')
    if (m['schema'] != LABEL_VALIDATION_SCHEMA or m['horizon_steps'] != HORIZON or m['replicas'] != REPLICAS
            or not all(a.get(k) is True for k in flags) or a['feasible_qp_rejections']
            or not a['trace_storage_verification']['all_shared_dependencies_verified']
            or not s['complete'] or s['compiled_signatures'] != 2 or s['implicit_jit_cache_entries'] != 0):
        raise ValueError('Incomplete or failed physical/compilation audit')
    for field, file in (('manifest_sha256','manifest.json'), ('index_sha256','index.json'),
                        ('trace_index_sha256','trace_index.json'), ('selected_queries_sha256','selected_queries.json')):
        if a[field] != sha256(p/file): raise ValueError('Changed audited evidence')
    for k in ('queries','branches','physical_steps','parents'):
        if a[k] != s[k]: raise ValueError('Different collection/audit totals')
    from .bicycle_trace_storage import verify_index_dependencies
    verify_index_dependencies(p, read(p/'trace_index.json'))
    return dict(directory=str(p.resolve()), parents=a['parents'], queries=a['queries'], branches=a['branches'],
        physical_steps=a['physical_steps'], bytes=directory_bytes(p), manifest_sha256=sha256(p/'manifest.json'),
        index_sha256=sha256(p/'index.json'), audit_sha256=sha256(p/'independent_replay.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True);q.add_argument('--groups',type=int,default=64);q.add_argument('--seed',type=int,default=6671)
    q.add_argument('--acquisition-mode',choices=['fixed2','balanced8'],default='fixed2')
    q=sub.add_parser('collect');q.add_argument('--source',required=True);q.add_argument('--output',required=True)
    for k,v in [('shard',0),('shards',4),('horizon',80),('acquisition-steps',160)]:q.add_argument('--'+k,type=int,default=v)
    q.add_argument('--replicas',type=int,default=2);q.add_argument('--snapshot-ticks',type=int,nargs='+',default=[0,40,120])
    q.add_argument('--min-free-gib',type=float,default=0.)
    q.add_argument('--shard-layout',choices=['round_robin','balanced_cells'],default='round_robin')
    q.add_argument('--observation-margin',action='store_true')
    q.add_argument('--shared-prefix-storage',action='store_true')
    q=sub.add_parser('audit');q.add_argument('--directory',required=True);q.add_argument('--workers',type=int,default=12)
    args_=vars(p.parse_args());command=args_.pop('command');dict(prepare=prepare,collect=collect,audit=audit)[command](**args_)
