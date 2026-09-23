"""Default original flight OD-QP on paired noisy physical tasks and missions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np
from .quad2d_control import FlightConfig,flight_config_from_contract
from .quad2d_mpc_experiment import FlightPhysicalKernels
from .quad2d_odqp import Quad2DODQP,contract,problem,nominal,nearest
from .quad2d_audit import check_trace,physical_flow
from .quad2d_waypoints import CONTRACT,validate_parent,numpy_arrived,check_episode
from .quad2d_hero_experiment import counts
from .quad2d_rollout import NAMES
from .route_audit import check_transition
from .dataset import sha256,source_fingerprint
from .io import write_json
from .cli import sanitize


def command_residual(x,u,obs,mask,omega,config):
    """Independent original continuous derivative from actual nonlinear flow."""
    c=config.robot;index=nearest(x,obs,mask);residual=[]
    if index>=0:
        delta=x[:2]-obs[index,:2];v=x[3:5];acceleration=physical_flow(x,u,c)[3:5]
        h=np.dot(delta,delta)-1.01*(c.radius+c.clearance_buffer+obs[index,2])**2
        hd=2*np.dot(delta,v);hdd=2*np.dot(v,v)+2*np.dot(delta,acceleration)
        residual.append(hdd+omega[0]*hd+.25*omega[1]*h)
    return np.r_[residual,u-c.force_min,c.force_max-u],np.inf,np.inf


def episode(parent,kernels,steps,config=FlightConfig()):
    c=config.robot;solver=Quad2DODQP(config);ordered='waypoint_count' in parent
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
        result=dict(solution=np.full(4,np.nan),reference=np.r_[nominal(sensed,target,config),1.,1.],selected_obstacle=nearest(sensed,seen,mask),solver_success=False,solver_status='not_attempted',status_value=0,
            iterations=0,solve_seconds=0.,primal_residual=np.nan,dual_residual=np.nan,raw_violation=np.nan,stored_violation=np.nan,feasible=False)
        if attempted:
            result=solver.solve(sensed,target,seen,mask);accepted=result['feasible']
            if not accepted:
                status=3;reason=('solver_reported_failure:'+result['solver_status'] if not result['solver_success'] else 'independent_stored_qp_rejected')
            else:
                u=result['solution'][:2].astype(np.float32);x,clear,bound=kernels.advance(x,u,truth,mask,np.int32(k));clear=float(clear);bound=float(bound);minimum=min(minimum,clear)
                count+=1;cursor=np.float32(proposed);previous=u
                if leg==total-1 and bool(kernels.arrived(x,goal)):status=1
                if bound>c.qp_tolerance:status=8
                if clear<=0:status=2
                reason=NAMES[status]
        mission=dict(waypoint_index=leg,waypoint_handoff=handoff,mission_goal=goal,mission_previous_control=before_control,mission_route_cursor_before=before_cursor,waypoints_visited=leg+int(status==1)) if ordered else {}
        records.append(dict(state=np.asarray(x),control=u,active=accepted,status=status,observed_state=sensed,observed_obstacles=seen,clearance=clear,state_bound_violation=bound,
            route_progress=cursor,route_target=target,route_remaining=remaining,solver_attempted=attempted,**{'odqp_'+k:v for k,v in result.items()},**mission))
        if status:break
    if not status:status=4;reason=NAMES[status]
    data={k:np.asarray([r[k] for r in records]) for k in records[0]}
    data.update(true_initial_state=np.asarray(initial),true_obstacles=np.asarray(truth),initial_observation=observed,observed_obstacles_initial=obs,obstacle_mask=mask,noise=noise,goal=final_goal,key=key)
    if not ordered:data.update(points=routes[0],route_mask=route_masks[0])
    row=dict(group_id=parent['group_id'],family=parent['family'],obstacles=int(mask.sum()),noise_scale=float(parent.get('noise_scale',round(float(noise[0]/.015),6))),status=NAMES[status],status_code=status,
        steps=count,min_clearance=minimum,final_state=np.asarray(x).tolist(),route_progress=float(cursor),termination_reason=reason,execution_seconds=time.perf_counter()-start,
        solver_attempts=int(data['solver_attempted'].sum()),solver_seconds=float(data['odqp_solve_seconds'].sum()),solver_setup_seconds=solver.setup_seconds,
        waypoint_index=leg,waypoints_visited=leg+int(status==1),required_waypoints=total,waypoint_handoffs=sum(int(r.get('waypoint_handoff',False)) for r in records))
    if ordered:row.update(original_kind=parent['original_kind'],variant_index=parent['variant_index'])
    return sanitize(row),data


def audit_episode(parent,data,row,config=FlightConfig()):
    c=config.robot;ordered='waypoint_count' in parent
    summary=dict(steps=row['steps'],status=row['status_code'],min_clearance=np.inf if row['min_clearance'] is None else row['min_clearance'])
    result=check_trace(data,summary,data['initial_observation'],data['observed_obstacles_initial'],data['obstacle_mask'],data['noise'],data['odqp_solution'][:,2:],config,command_residual=command_residual)
    if not result['audit_passed']:raise ValueError('Independent OD-QP physical audit failed')
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
    cursor=np.float32(0);max_raw=max_stored=0.;ambiguous=0
    for k,active in enumerate(data['active']):
        x=data['observed_state'][k].astype(float);obs=data['observed_obstacles'][k].astype(float);mask=data['obstacle_mask'];target=data['route_target'][k]
        if ordered:
            leg=int(data['waypoint_index'][k]);goal=goals[leg]
            if not ready[leg] and prev_status==0:prev_status=7
        else:
            goal=goals[0];leg=0
            ambiguous+=int(check_transition(data['observed_state'][k],np.asarray(parent['route']['points'],np.float32),np.asarray(parent['route']['mask'],bool),cursor,target,data['route_progress'][k],active))
        ref,A,b,selected=problem(x,target,obs,mask,config)
        np.testing.assert_allclose(data['odqp_reference'][k],ref,atol=1e-12,rtol=1e-12)
        if selected!=data['odqp_selected_obstacle'][k]:raise ValueError('Changed original nearest-obstacle policy')
        attempted=prev_status==0
        if data['solver_attempted'][k]!=attempted:raise ValueError('Missing/extra solver attempt')
        solution=data['odqp_solution'][k];finite=np.isfinite(solution).all();success=bool(data['odqp_status_value'][k] in (1,2) and finite)
        if success!=data['odqp_solver_success'][k]:raise ValueError('Wrong solver success')
        candidate=solution.copy();candidate[:2]=solution[:2].astype(np.float32)
        raw=float(np.max(A@solution-b)) if finite else np.inf;stored=float(np.max(A@candidate-b)) if finite else np.inf
        feasible=success and max(raw,stored)<=c.qp_tolerance
        if attempted:
            if finite:
                np.testing.assert_allclose([data['odqp_raw_violation'][k],data['odqp_stored_violation'][k]],[raw,stored],atol=1e-9,rtol=1e-12)
            if feasible!=data['odqp_feasible'][k] or active!=feasible:raise ValueError('Incorrect QP acceptance/censoring')
            if not feasible:prev_status=3
        if active:
            max_raw=max(max_raw,raw);max_stored=max(max_stored,stored)
            np.testing.assert_array_equal(data['control'][k],solution[:2].astype(np.float32))
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
    result.update(all_attempted_qps_checked=True,max_raw_qp_violation=max_raw,max_stored_qp_violation=max_stored,roundoff_ambiguous_route_decisions=ambiguous)
    if ordered:result.update(check_episode(parent,data,row,config,adaptive=False))
    return result


def run(source,output,steps=1600,shard_index=0,shards=1):
    source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    sm=json.loads((source/'manifest.json').read_text());parents=json.loads((source/'scenes.json').read_text());ordered=any('waypoint_count' in p for p in parents)
    c=flight_config_from_contract(sm['config'])
    legacy_held={'quad2d_v30_policy_pilot_inputs','quad2d_v31_fresh_inputs'}
    if sm['scenes_sha256']!=sha256(source/'scenes.json') or (sm.get('training_use') is not False and source.name not in legacy_held):raise ValueError('Audited-format held evaluation source required')
    if ordered:
        if sm['waypoint_contract']!=CONTRACT:raise ValueError('Unknown ordered task')
        for p in parents:validate_parent(p)
    if steps<1 or not 0<=shard_index<shards or not parents[shard_index::shards]:raise ValueError('Invalid episode/shard budget')
    chosen=parents[shard_index::shards];m=dict(schema='quad2d_default_odqp_evaluation_v1',source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),source_fingerprint=source_fingerprint(),config=asdict(c),
        controller=contract(c),ordered=ordered,waypoint_contract=CONTRACT if ordered else None,steps=steps,shard_index=shard_index,shards=shards,selected_parents=[p['group_id'] for p in chosen],final_test=False,
        limitations='Default original single-obstacle OD-QP with declared common task adapter. No OA guidance, baseline tuning, unseen-future certification or native BarrierNet comparison.')
    write_json(root/'manifest.json',m);kernels=FlightPhysicalKernels(64,64,steps,c);start=time.perf_counter();index=[]
    print(json.dumps(dict(stage='compiled',seconds=kernels.compile_seconds,device=str(jax.devices()[0]))),flush=True)
    for i,p in enumerate(chosen):
        row,data=episode(p,kernels,steps,c);path=root/f'episode_{shard_index+i*shards:05d}.npz';np.savez_compressed(path,**data);row.update(file=path.name,sha256=sha256(path));index.append(row);write_json(root/'index.json',index)
        print(json.dumps(dict(completed=len(index),total=len(chosen),status=row['status'],steps=row['steps'])),flush=True)
    write_json(root/'summary.json',dict(complete=True,physical_audit_pending=True,compile_seconds=kernels.compile_seconds,execution_seconds=time.perf_counter()-start,aggregate=counts(index)))


def audit(directory):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text());source=Path(m['source']);sm=json.loads((source/'manifest.json').read_text());c=flight_config_from_contract(m['config'])
    if m['schema']!='quad2d_default_odqp_evaluation_v1' or flight_config_from_contract(sm['config'])!=c or m['controller']!=contract(c):raise ValueError('Changed OD-QP/default-solver contract')
    if sha256(source/'manifest.json')!=m['source_manifest_sha256'] or sha256(source/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Changed task source')
    parents=json.loads((source/'scenes.json').read_text())[m['shard_index']::m['shards']];rows=json.loads((root/'index.json').read_text());audits=[]
    if [r['group_id'] for r in rows]!=[p['group_id'] for p in parents]:raise ValueError('Lost/extra/reordered task')
    for p,r in zip(parents,rows):
        if sha256(root/r['file'])!=r['sha256']:raise ValueError('Changed physical trace')
        with np.load(root/r['file']) as f:d=dict(f)
        if r['status_code']==4 and r['steps']!=m['steps']:raise ValueError('Short prefix reported as timeout')
        audits.append(dict(group_id=r['group_id'],**audit_episode(p,d,r,c)))
    report=dict(audit_passed=bool(audits) and all(a['audit_passed'] for a in audits),manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),auditor_source_fingerprint=source_fingerprint(),episodes=len(rows),steps=sum(r['steps'] for r in rows),rows=audits,
        scope='Every physical/sensor prefix, attempted original QP and actual FP32 command; single-nearest selection and original nominal; route/ordered goals and failure censoring independently checked.')
    write_json(root/'independent_replay.json',sanitize(report));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if not report['audit_passed']:raise ValueError('OD-QP episode audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','audit']);p.add_argument('--source');p.add_argument('--output');p.add_argument('--directory');p.add_argument('--steps',type=int,default=1600);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    run(a.source,a.output,a.steps,a.shard_index,a.shards) if a.action=='run' else audit(a.directory)
