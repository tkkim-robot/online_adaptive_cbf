"""Native flight BarrierNet on common noisy physical tasks and ordered missions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .quad2d_control import FlightConfig
from .quad2d_mpc_experiment import FlightPhysicalKernels
from .quad2d_barriernet import contract
from .quad2d_barriernet_inference import make_policy,load_bundle
from .quad2d_barriernet_audit import numpy_features,numpy_constraints,numpy_nominal,numpy_network,check_deployment
from .quad2d_audit import check_trace
from .quad2d_waypoints import CONTRACT,validate_parent,numpy_arrived,check_episode
from .quad2d_hero_experiment import counts
from .quad2d_rollout import NAMES
from .route_audit import check_transition
from .dataset import sha256,source_fingerprint
from .io import write_json
from .cli import sanitize


class Controller:
    def __init__(self,bundle,capacity=64,config=FlightConfig()):
        self.config=config;fn,self.manifest=make_policy(bundle);start=time.perf_counter()
        self.execute=fn.lower(jnp.zeros(6,jnp.float64),jnp.zeros(2,jnp.float64),jnp.zeros((capacity,5),jnp.float64),jnp.zeros(capacity,bool)).compile()
        self.compile_seconds=time.perf_counter()-start

    def solve(self,state,target,obstacles,mask):
        start=time.perf_counter();result=jax.device_get(self.execute(jnp.asarray(state,jnp.float64),jnp.asarray(target,jnp.float64),jnp.asarray(obstacles,jnp.float64),jnp.asarray(mask)))
        c=self.config.robot;u=result['candidate'].astype(np.float32)
        _,ctx=numpy_features(np.asarray(state,float),np.asarray(target,float),np.asarray(obstacles,float),mask,c.radius)
        A,b=numpy_constraints(ctx,result['gains'],c.radius,c.force_min,c.force_max)
        stored=float(np.max(A@u-b)) if np.isfinite(u).all() else np.inf
        result.update(feasible=bool(result['valid'] and stored<=c.qp_tolerance),stored_violation=stored,solve_seconds=time.perf_counter()-start)
        return result


def command_residual(x,u,obs,mask,gains,config):
    _,ctx=numpy_features(x,np.zeros(2),obs,mask,config.robot.radius)
    A,b=numpy_constraints(ctx,np.asarray(gains).reshape(5,2),config.robot.radius,config.robot.force_min,config.robot.force_max)
    return b-A@u,np.inf,np.inf


def episode(parent,kernels,solver,steps,config=FlightConfig()):
    c=config.robot;ordered='waypoint_count' in parent
    observed,final_goal,obs,mask,noise=(np.asarray(parent[k],bool if k=='obstacle_mask' else np.float32) for k in ['initial_state','goal','obstacles','obstacle_mask','noise'])
    if ordered:
        validate_parent(parent);goals=np.asarray(parent['waypoint_goals'],np.float32);total=parent['waypoint_count']
        routes=np.asarray(parent['waypoint_routes']['points'],np.float32);route_masks=np.asarray(parent['waypoint_routes']['mask'],bool);ready=np.asarray(parent['waypoint_routes']['ready'],bool)
    else:
        goals=final_goal[None];total=1;routes=np.asarray(parent['route']['points'],np.float32)[None];route_masks=np.asarray(parent['route']['mask'],bool)[None];ready=[parent['route']['status']=='ready']
    leg=0;key=np.asarray(jax.random.PRNGKey(parent['seed']+7193));initial,truth,xb,ob,xs,os,innovations=kernels.prepare(observed,obs,mask,noise,key)
    x=initial;previous=np.zeros(2,np.float32);cursor=np.float32(0);count=0;minimum=float(kernels.clearance(x,truth,mask))
    status=1 if total==1 and bool(kernels.arrived(x,goals[0])) else 0
    if float(kernels.bound(x))>c.qp_tolerance:status=8
    if minimum<=0:status=2
    if not ready[0]:status=7
    records=[];reason=NAMES[status];start=time.perf_counter()
    for k in range(steps):
        sensed,seen=map(np.asarray,kernels.sense(x,truth,xb,ob,xs,os,innovations[k],np.int32(k)))
        handoff=bool(status==0 and leg<total-1 and numpy_arrived(sensed.astype(float),goals[leg].astype(float),config,noise))
        if handoff:leg+=1;cursor=np.float32(0)
        goal=goals[leg];points=routes[leg];rm=route_masks[leg]
        if status==0 and not ready[leg]:status=7;reason=NAMES[status]
        before_cursor=cursor;before_control=previous.copy();target,proposed,remaining=map(np.asarray,kernels.target(sensed,points,rm,cursor))
        attempted=status==0;accepted=False;u=np.zeros(2,np.float32);clear=bound=np.nan
        result=dict(candidate=np.full(2,np.nan),raw_control=np.full(2,np.nan),reference=np.full(2,np.nan),u_nom=np.full(2,np.nan),gains=np.full((5,2),np.nan),
            valid=False,solver_valid=False,violation=np.nan,raw_violation=np.nan,stored_violation=np.nan,solve_seconds=0.,feasible=False)
        if attempted:
            result=solver.solve(sensed,target,seen,mask);accepted=result['feasible']
            if not accepted:
                status=3;reason='native_solver_rejected' if not result['solver_valid'] else 'native_post_clip_or_fp32_rejected'
            else:
                u=result['candidate'].astype(np.float32);x,clear,bound=kernels.advance(x,u,truth,mask,np.int32(k));clear=float(clear);bound=float(bound);minimum=min(minimum,clear)
                count+=1;cursor=np.float32(proposed);previous=u
                if leg==total-1 and bool(kernels.arrived(x,goal)):status=1
                if bound>c.qp_tolerance:status=8
                if clear<=0:status=2
                reason=NAMES[status]
        mission=dict(waypoint_index=leg,waypoint_handoff=handoff,mission_goal=goal,mission_previous_control=before_control,mission_route_cursor_before=before_cursor,waypoints_visited=leg+int(status==1)) if ordered else {}
        records.append(dict(state=np.asarray(x),control=u,active=accepted,status=status,observed_state=sensed,observed_obstacles=seen,clearance=clear,state_bound_violation=bound,
            route_progress=cursor,route_target=target,route_remaining=remaining,solver_attempted=attempted,**{'barriernet_'+k:v for k,v in result.items()},**mission))
        if status:break
    if not status:status=4;reason=NAMES[status]
    data={k:np.asarray([r[k] for r in records]) for k in records[0]}
    data.update(true_initial_state=np.asarray(initial),true_obstacles=np.asarray(truth),initial_observation=observed,observed_obstacles_initial=obs,obstacle_mask=mask,noise=noise,goal=final_goal,key=key)
    if not ordered:data.update(points=routes[0],route_mask=route_masks[0])
    row=dict(group_id=parent['group_id'],family=parent['family'],obstacles=int(mask.sum()),noise_scale=float(parent.get('noise_scale',round(float(noise[0]/.015),6))),status=NAMES[status],status_code=status,
        steps=count,min_clearance=minimum,final_state=np.asarray(x).tolist(),route_progress=float(cursor),termination_reason=reason,execution_seconds=time.perf_counter()-start,
        solver_attempts=int(data['solver_attempted'].sum()),solver_seconds=float(data['barriernet_solve_seconds'].sum()),solver_setup_seconds=solver.compile_seconds,
        waypoint_index=leg,waypoints_visited=leg+int(status==1),required_waypoints=total,waypoint_handoffs=sum(int(r.get('waypoint_handoff',False)) for r in records))
    if ordered:row.update(original_kind=parent['original_kind'],variant_index=parent['variant_index'])
    return sanitize(row),data


def audit_episode(parent,data,row,weights,mean,std,config=FlightConfig()):
    c=config.robot;ordered='waypoint_count' in parent
    summary=dict(steps=row['steps'],status=row['status_code'],min_clearance=np.inf if row['min_clearance'] is None else row['min_clearance'])
    physical_result=check_trace(data,summary,data['initial_observation'],data['observed_obstacles_initial'],data['obstacle_mask'],data['noise'],data['barriernet_gains'].reshape((-1,10)),config,command_residual=command_residual)
    if not physical_result['audit_passed']:raise ValueError('Independent native flight physical audit failed')
    for field,original in [('initial_observation','initial_state'),('observed_obstacles_initial','obstacles'),('obstacle_mask','obstacle_mask'),('noise','noise'),('goal','goal')]:
        np.testing.assert_array_equal(data[field],np.asarray(parent[original],data[field].dtype))
    np.testing.assert_array_equal(data['key'],np.asarray(jax.random.PRNGKey(parent['seed']+7193)))
    x0=data['true_initial_state'].astype(float);initial_clear=np.min(np.linalg.norm(x0[:2]-data['true_obstacles'][data['obstacle_mask'],:2],axis=1)-c.radius-data['true_obstacles'][data['obstacle_mask'],2],initial=np.inf)
    total=parent['waypoint_count'] if ordered else 1
    goals=np.asarray(parent['waypoint_goals'] if ordered else [parent['goal']],np.float32)
    ready=parent['waypoint_routes']['ready'] if ordered else [parent['route']['status']=='ready']
    prev_status=1 if total==1 and numpy_arrived(x0,goals[0],config) else 0
    if max(np.max(np.abs(x0[3:5]))-config.velocity_limit,abs(x0[2])-config.pitch_limit,abs(x0[5])-config.pitch_rate_limit)>c.qp_tolerance:prev_status=8
    if initial_clear<=0:prev_status=2
    if not ready[0]:prev_status=7
    cursor=np.float32(0);max_raw=max_stored=max_kkt=0.;ambiguous=0;clipped=0;feasible_rejections=0
    for k,active in enumerate(data['active']):
        x=data['observed_state'][k].astype(float);obs=data['observed_obstacles'][k].astype(float);mask=data['obstacle_mask'];target=data['route_target'][k]
        if ordered:
            leg=int(data['waypoint_index'][k]);goal=goals[leg]
            if not ready[leg] and prev_status==0:prev_status=7
        else:
            goal=goals[0];leg=0
            ambiguous+=int(check_transition(data['observed_state'][k],np.asarray(parent['route']['points'],np.float32),np.asarray(parent['route']['mask'],bool),cursor,target,data['route_progress'][k],active))
        attempted=prev_status==0
        if data['solver_attempted'][k]!=attempted:raise ValueError('Missing/extra native solve')
        if attempted:
            z,ctx=numpy_features(x,target.astype(float),obs,mask,c.radius)
            ref=numpy_nominal(x,target.astype(float),config);u_nom,p=numpy_network(weights,mean,std,z,ctx,ref)
            result={key.removeprefix('barriernet_'):value[k] for key,value in data.items() if key.startswith('barriernet_')}
            for key,expected in [('reference',ref),('u_nom',u_nom),('gains',p)]:
                np.testing.assert_allclose(result[key],expected,atol=1e-10,rtol=1e-12)
            checks=check_deployment(result,ctx,p,u_nom,config,classify=True)
            feasible=checks['accepted'];raw=checks['raw_violation'];stored=checks['stored_violation']
            if feasible!=bool(result['feasible']) or active!=feasible:raise ValueError('Incorrect native acceptance/censoring')
            if np.isfinite(stored):np.testing.assert_allclose(result['stored_violation'],stored,atol=1e-9,rtol=1e-10)
            max_kkt=max(max_kkt,checks['stationarity']);clipped+=int(checks['clipped'])
            feasible_rejections+=int(not result['solver_valid'] and checks['independently_feasible'])
            if not feasible:prev_status=3
        if active:
            max_raw=max(max_raw,raw);max_stored=max(max_stored,stored)
            np.testing.assert_array_equal(data['control'][k],data['barriernet_candidate'][k].astype(np.float32))
            expected=1 if leg==total-1 and numpy_arrived(data['state'][k].astype(float),goal,config) else 0
            if data['state_bound_violation'][k]>c.qp_tolerance:expected=8
            if data['clearance'][k]<=0:expected=2
            prev_status=expected;cursor=data['route_progress'][k]
        else:
            np.testing.assert_array_equal(data['control'][k],np.zeros(2))
            np.testing.assert_array_equal(data['state'][k],data['true_initial_state'] if k==0 else data['state'][k-1])
        if int(data['status'][k])!=prev_status:raise ValueError('Wrong physical termination status')
        if prev_status and k!=len(data['active'])-1:raise ValueError('Record after terminal decision')
    if row['status_code']!=(4 if prev_status==0 else prev_status):raise ValueError('Censored stop misreported as completion/timeout')
    np.testing.assert_array_equal(data['state'][-1],np.asarray(row['final_state'],np.float32))
    physical_result.update(all_attempted_qps_checked=True,all_attempted_network_outputs_checked=True,max_raw_qp_violation=max_raw,max_stored_qp_violation=max_stored,max_stationarity=max_kkt,clipped_commands=clipped,feasible_raw_solver_rejections=feasible_rejections,roundoff_ambiguous_route_decisions=ambiguous)
    if ordered:physical_result.update(check_episode(parent,data,row,config,adaptive=False))
    return physical_result


def run(source,bundle,output,steps=1600,shard_index=0,shards=1):
    source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False);c=FlightConfig()
    sm=json.loads((source/'manifest.json').read_text());parents=json.loads((source/'scenes.json').read_text());ordered=any('waypoint_count' in p for p in parents)
    legacy_held={'quad2d_v30_policy_pilot_inputs','quad2d_v31_fresh_inputs'}
    if sm['scenes_sha256']!=sha256(source/'scenes.json') or sm['config']!=asdict(c) or (sm.get('training_use') is not False and source.name not in legacy_held):raise ValueError('Audited-format held evaluation source required')
    if ordered:
        if sm['waypoint_contract']!=CONTRACT:raise ValueError('Unknown ordered task')
        for p in parents:validate_parent(p)
    if steps<1 or not 0<=shard_index<shards or not parents[shard_index::shards]:raise ValueError('Invalid episode/shard budget')
    bm=json.loads((Path(bundle)/'manifest.json').read_text());ba=json.loads((Path(bundle)/'independent_inference_audit.json').read_text())
    if not ba['audit_passed'] or ba['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json'):raise ValueError('Independent native inference audit required')
    with np.load(Path(bm['dataset'])/'data.npz') as f:training_groups=set(f['group_id'])
    if training_groups&{p['group_id'] for p in parents}:raise ValueError('Native evaluation overlaps training')
    chosen=parents[shard_index::shards];m=dict(schema='quad2d_native_barriernet_evaluation_v1',bundle=str(Path(bundle).resolve()),bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),weights_sha256=bm['weights_sha256'],source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_fingerprint=source_fingerprint(),config=asdict(c),
        controller=contract(c),ordered=ordered,waypoint_contract=CONTRACT if ordered else None,steps=steps,shard_index=shard_index,shards=shards,selected_parents=[p['group_id'] for p in chosen],final_test=False,
        limitations='Native default-trained five-obstacle BarrierNet with declared common flight task adapter. Every physical obstacle checked. No tuning or safety inference after censored rejection. Synchronous development evaluation only.')
    write_json(root/'manifest.json',m);kernels=FlightPhysicalKernels(64,64,steps,c);solver=Controller(bundle,64,c);start=time.perf_counter();index=[]
    print(json.dumps(dict(stage='compiled',seconds=kernels.compile_seconds+solver.compile_seconds,device=str(jax.devices()[0]))),flush=True)
    for i,p in enumerate(chosen):
        row,data=episode(p,kernels,solver,steps,c);path=root/f'episode_{shard_index+i*shards:05d}.npz';np.savez_compressed(path,**data);row.update(file=path.name,sha256=sha256(path));index.append(row);write_json(root/'index.json',index)
        print(json.dumps(dict(completed=len(index),total=len(chosen),status=row['status'],steps=row['steps'])),flush=True)
    write_json(root/'summary.json',dict(complete=True,physical_audit_pending=True,compile_seconds=kernels.compile_seconds+solver.compile_seconds,execution_seconds=time.perf_counter()-start,aggregate=counts(index)))


def audit(directory):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text());source=Path(m['source']);sm=json.loads((source/'manifest.json').read_text());c=FlightConfig()
    if m['schema']!='quad2d_native_barriernet_evaluation_v1' or m['config']!=asdict(c) or m['controller']!=contract(c):raise ValueError('Changed native flight method contract')
    if sha256(source/'manifest.json')!=m['source_manifest_sha256'] or sha256(source/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Changed task source')
    bm,_,params,mean,std=load_bundle(m['bundle']);weights=jax.device_get(params);mean,std=map(np.asarray,(mean,std))
    if sha256(Path(m['bundle'])/'manifest.json')!=m['bundle_manifest_sha256'] or bm['weights_sha256']!=m['weights_sha256']:raise ValueError('Changed evaluated native weights')
    parents=json.loads((source/'scenes.json').read_text())[m['shard_index']::m['shards']];rows=json.loads((root/'index.json').read_text());audits=[]
    if [r['group_id'] for r in rows]!=[p['group_id'] for p in parents]:raise ValueError('Lost/extra/reordered task')
    for p,r in zip(parents,rows):
        if sha256(root/r['file'])!=r['sha256']:raise ValueError('Changed physical trace')
        with np.load(root/r['file']) as f:d=dict(f)
        if r['status_code']==4 and r['steps']!=m['steps']:raise ValueError('Short prefix reported as timeout')
        audits.append(dict(group_id=r['group_id'],**audit_episode(p,d,r,weights,mean,std,c)))
    report=dict(audit_passed=bool(audits) and all(a['audit_passed'] for a in audits),manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),auditor_source_fingerprint=source_fingerprint(),episodes=len(rows),steps=sum(r['steps'] for r in rows),rows=audits,
        scope='Every physical/sensor prefix, attempted native NN output/QP, original1..10 solve, wrapper clip and actual FP32 command; nearest-five static rows, original nominal, route/ordered goals and censoring independently checked.')
    write_json(root/'independent_replay.json',sanitize(report));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if not report['audit_passed']:raise ValueError('Native flight episode audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','audit']);p.add_argument('--source');p.add_argument('--bundle');p.add_argument('--output');p.add_argument('--directory');p.add_argument('--steps',type=int,default=1600);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    run(a.source,a.bundle,a.output,a.steps,a.shard_index,a.shards) if a.action=='run' else audit(a.directory)
