"""Native untuned BarrierNet in the common continuously sensed physical task.

Five obstacle CBF rows are the original method contract. Every physical obstacle
still participates in collision checking. No OA nominal, observer, gate or
speed-bound correction is added to this comparator.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .barriernet_inference import make_policy
from .barriernet_audit import numpy_features,numpy_constraints
from .cli import sanitize
from .config import UnicycleConfig
from .unicycle_inputs import source_robot
from .dataset import source_fingerprint,sha256
from .io import write_json
from .physical_kernels import PhysicalKernels
from .simulation import RUNNING,GOAL,COLLISION,INFEASIBLE,TIMEOUT,STATUS_NAMES
from .stochastic import STATE_BOUND_VIOLATION
from .waypoint_tasks import CONTRACT

PLANNER_FAILURE=7
NAMES={**STATUS_NAMES,PLANNER_FAILURE:'planner_failure',STATE_BOUND_VIOLATION:'state_bound_violation'}
NAMES[INFEASIBLE]='solver_rejected'
CONDITIONS='Identical observed initial scenes, routes, physical prior, sensor noise stream and limits for every method. Synchronous compute; no real-delay or final-test claim.'


class Controller:
    def __init__(self,bundle,capacity,robot=UnicycleConfig()):
        self.robot=robot;fn,self.manifest=make_policy(bundle,robot.radius)
        begin=time.perf_counter()
        self.execute=fn.lower(jnp.zeros(4,jnp.float64),jnp.zeros(2,jnp.float64),jnp.zeros((capacity,5),jnp.float64),jnp.zeros(capacity,bool)).compile()
        self.compile_seconds=time.perf_counter()-begin

    def solve(self,state,goal,obstacles,mask):
        begin=time.perf_counter()
        u,feasible,violation,p,reference=jax.device_get(self.execute(
            jnp.asarray(state,jnp.float64),jnp.asarray(goal,jnp.float64),jnp.asarray(obstacles,jnp.float64),jnp.asarray(mask)))
        # Shared plant command storage is FP32. Independently recheck that exact
        # command; never change the QP, tolerance, or gains to make it pass.
        applied=np.asarray(u,np.float32)
        _,ctx=numpy_features(np.asarray(state,float),np.asarray(goal,float),np.asarray(obstacles,float),mask,self.robot.radius)
        A,b=numpy_constraints(ctx,p,self.robot.radius)
        applied_violation=float(np.max(A@applied-b))
        accepted=bool(feasible and np.isfinite(applied).all() and applied_violation<=self.robot.qp_tolerance)
        return dict(control=applied,feasible=accepted,solver_feasible=bool(feasible),gains=p,objective_reference=reference,
            violation=applied_violation,solver_violation=float(violation),seconds=time.perf_counter()-begin,
            solver_status='solved' if accepted else 'post_command_rejected' if feasible else 'solver_rejected')


def episode(record,controller,kernels,steps,noise_scale,ordered=False):
    robot=controller.robot;scene=record['scene'];f=lambda a:np.asarray(a,np.float32)
    mask=np.asarray(scene['obstacle_mask'],bool);observed=f(scene['initial_state']);obstacles=f(scene['obstacles'])
    if ordered:
        count=record['waypoint_count'];goals=f(record['waypoint_goals']);routes=record['waypoint_routes']
        points=f(routes['points']);rmask=np.asarray(routes['mask'],bool);ready=np.asarray(routes['ready'],bool)
    else:
        count=1;goals=f([scene['goal']]);points=f([record['route']['points']]);rmask=np.asarray([record['route']['mask']],bool)
        ready=np.asarray([record['route']['status']=='ready'])
    noise=noise_scale*np.array([.02,.03,.02,.03,.025,.01]);fn=f(noise)
    seed=int.from_bytes(hashlib.sha256(scene['scene_id'].encode()).digest()[:4],'little');key=np.asarray(jax.random.PRNGKey(seed))
    x,truth_obs,xb,ob,xs,os,innovations=kernels.prepare(observed,obstacles,mask,fn,key)
    initial=np.asarray(x);truth=np.asarray(truth_obs);minimum=float(kernels.clearance(x,truth_obs,mask))
    status=PLANNER_FAILURE if not ready[0] else COLLISION if minimum<=0 else GOAL if count==1 and bool(kernels.arrived(x,goals[0],np.zeros(6,np.float32))) else RUNNING
    leg=0;cursor=np.float32(0);traces=[];applied=0;solve_times=[];step_times=[];solver_statuses={};start=time.perf_counter()
    for k in range(steps):
        tick=time.perf_counter();sx,so=kernels.sense(x,truth_obs,xb,ob,xs,os,innovations[k],np.int32(k));sx,so=np.asarray(sx),np.asarray(so)
        handoff=status==RUNNING and leg<count-1 and bool(kernels.arrived(sx,goals[leg],fn))
        if handoff:leg+=1;cursor=np.float32(0)
        if status==RUNNING and not ready[leg]:status=PLANNER_FAILURE
        attempt=status==RUNNING;result=None;u=np.zeros(2,np.float32);clear=bound=violation=np.nan
        target,proposal,remaining=kernels.target(sx,points[leg],rmask[leg],cursor)
        if attempt:
            result=controller.solve(sx,goals[leg],so,mask);solve_times.append(result['seconds'])
            label=result['solver_status'];solver_statuses[label]=solver_statuses.get(label,0)+1
            accepted=result['feasible'];violation=result['violation']
            if accepted:
                u=result['control'];x,clear,bound=kernels.advance(x,u,truth_obs,mask,np.int32(k))
                clear,bound=float(clear),float(bound);minimum=min(minimum,clear);cursor=np.float32(proposal);applied+=1
                if leg==count-1 and bool(kernels.arrived(x,goals[leg],np.zeros(6,np.float32))):status=GOAL
                if bound>robot.qp_tolerance:status=STATE_BOUND_VIOLATION
                if clear<=0:status=COLLISION
            else:status=INFEASIBLE
        else:accepted=False
        traces.append(dict(state=np.asarray(x),control=u,active=accepted,status=status,
            gains=np.full((5,2),np.nan) if result is None else result['gains'],
            objective_reference=np.full(2,np.nan) if result is None else result['objective_reference'],
            observed_state=sx,observed_obstacles=so,clearance=clear,state_bound_violation=bound,qp_violation=violation,
            route_target=np.asarray(target),route_remaining=float(remaining),route_progress=cursor,
            selection_tick=attempt,selection_accepted=accepted,solver_status='' if result is None else result['solver_status'],
            solver_success=False if result is None else result['solver_feasible'],solver_seconds=np.nan if result is None else result['seconds'],
            waypoint_index=leg,waypoint_handoff=handoff,waypoints_visited=leg+int(status==GOAL),mission_goal=goals[leg],
            collision_event=accepted and clear<=0,state_bound_event=accepted and bound>robot.qp_tolerance))
        step_times.append(time.perf_counter()-tick)
        if status!=RUNNING:break
    if status==RUNNING:status=TIMEOUT
    trace={k:np.stack([t[k] for t in traces]) for k in traces[0]}
    row=dict(scene_id=scene['scene_id'],family=scene['family'],mode='barriernet',status=NAMES[status],steps=applied,min_clearance=minimum,
        goal_progress=float(np.linalg.norm(initial[:2]-goals[count-1])-np.linalg.norm(np.asarray(x)[:2]-goals[count-1])),
        final_route_coordinate=float(kernels.coordinate(x[:2],points[leg],rmask[leg],cursor)),
        applied_source_counts={'native_barriernet':applied},selection_attempts=len(solve_times),selection_rejections=int(status==INFEASIBLE),
        reactive_reselections=0,gate_stage_totals=[],gain_total_variation=float(np.abs(np.diff(trace['gains'][trace['selection_tick']],axis=0)).sum()),
        max_physical_bound_violation=float(np.max(trace['state_bound_violation'][trace['active']])) if applied else None,
        solver_status_counts=solver_statuses,elapsed_seconds=time.perf_counter()-start,solver_seconds=sum(solve_times),
        tick_seconds_p50_p99_max=np.quantile(step_times,[.5,.99,1]).tolist(),
        collision_event=bool(np.any(trace['collision_event'])) or minimum<=0,state_bound_event=bool(np.any(trace['state_bound_event'])))
    if status==INFEASIBLE:row['termination_reason']=result['solver_status']
    if ordered:row.update(waypoint_index=leg,waypoints_visited=leg+int(status==GOAL),required_waypoints=count,waypoint_handoffs=int(trace['waypoint_handoff'].sum()))
    return row,trace,dict(true_initial_state=initial,true_obstacles=truth,scene_id=scene['scene_id'],noise=noise,key=key)


def run(source,output,bundle,noise_scale=0.,steps=800,limit=None,ordered_waypoints=False,shard_index=0,shards=1):
    if steps<1 or shards<1 or not 0<=shard_index<shards or noise_scale<0:raise ValueError('Invalid evaluation budget/shard/noise')
    audit=json.loads((Path(bundle)/'independent_inference_audit.json').read_text())
    if not audit['audit_passed'] or audit['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json'):raise ValueError('Audited native bundle required')
    records=json.loads((Path(source)/'scenes.json').read_text())
    if any('waypoint_goals' in r for r in records) and not ordered_waypoints:raise ValueError('Ordered task cannot bypass intermediate goals')
    if limit is not None:records=[r for r in records if not r['scene']['scene_id'].startswith('fixture:')][:limit]+[r for r in records if r['scene']['scene_id'].startswith('fixture:')]
    records=records[shard_index::shards]
    if not records:raise ValueError('Empty evaluation shard')
    capacity=len(records[0]['scene']['obstacles']);field='waypoint_routes' if ordered_waypoints else 'route'
    route_capacity=len(records[0][field]['points'][0] if ordered_waypoints else records[0][field]['points'])
    for r in records:
        if len(r['scene']['obstacles'])!=capacity:raise ValueError('Mixed obstacle capacities')
        if np.shape(r[field]['points'])!=((3,route_capacity,2) if ordered_waypoints else (route_capacity,2)):raise ValueError('Mixed route capacities')
        if ordered_waypoints and (not 1<=r['waypoint_count']<=3 or not np.array_equal(r['waypoint_goals'][r['waypoint_count']-1],r['scene']['goal'])):raise ValueError('Invalid ordered goal contract')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);robot=source_robot(source)
    controller=Controller(bundle,capacity,robot);kernels=PhysicalKernels(capacity,route_capacity,steps,robot)
    contract=dict(name='pinned_native_barriernet_jax',obstacle_rows=5,barrier='static-center HOCBF,1.01 squared-radius multiplier',
        weights_sha256=controller.manifest['weights_sha256'],bundle=str(Path(bundle).resolve()),bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),
        input_bounds=[robot.a_max,robot.w_max],gain_bounds=[0.,4.],qp_diagonal=1+1e-6,qp_tolerance=1e-5,training_defaults=controller.manifest['settings'],
        no_deployment_slack=True,nominal='Original DynamicUnicycle2D default current-waypoint feedback plus learned residual')
    manifest=dict(stage='development_native_barriernet',final_test=False,source_fingerprint=source_fingerprint(),policy=dict(mode='barriernet',motion_observer_window=0,sensor_margin_scale=0.),robot=asdict(robot),barriernet=contract,
        conditions=CONDITIONS+(' Ordered waypoint contract: '+CONTRACT['name'] if ordered_waypoints else ''),noise_scale=noise_scale,steps=steps,batch=1,
        input_scene_sha256=sha256(Path(source)/'scenes.json'),input_scenes=str(Path(source).resolve()),shard_index=shard_index,shards=shards,limit=limit,scene_ids=[r['scene']['scene_id'] for r in records],
        initialization_compile_seconds=kernels.compile_seconds,solver_setup_seconds=controller.compile_seconds,
        runtime_compilation='All seven shared FP32 physical helpers and native FP64 BarrierNet compiled before episodes. Fixed executable signatures only.',
        capacity='Native nearest-five CBF retained. Every sensed obstacle available; all physical obstacles tested for collision.',
        censoring='Rejected solve applies no action. Physical speed violations and collisions count as failures. No safety inference from unobserved evolution after termination.')
    if ordered_waypoints:manifest['waypoint_contract']=CONTRACT
    write_json(root/'manifest.json',manifest);rows=[];start=time.perf_counter()
    print(json.dumps(dict(stage='ready',groups=len(records),physical_compile_seconds=kernels.compile_seconds,policy_compile_seconds=controller.compile_seconds)),flush=True)
    for i,r in enumerate(records):
        row,trace,truth=episode(r,controller,kernels,steps,noise_scale,ordered_waypoints);rows.append(row)
        np.savez_compressed(root/f'traces_{i:05d}.npz',**{k:v[None] for k,v in trace.items()},**{k:np.asarray(v)[None] for k,v in truth.items()})
        write_json(root/'results.json',sanitize(rows));progress=dict(completed_groups=i+1,total_groups=len(records),elapsed_seconds=time.perf_counter()-start,last_status=row['status'],last_steps=row['steps'])
        write_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    evaluated=[r for r in rows if not r['scene_id'].startswith('fixture:')]
    summary=dict(aggregate=dict(groups=len(evaluated),outcomes={s:sum(r['status']==s for r in evaluated) for s in sorted({r['status'] for r in evaluated})}),
        elapsed_seconds=time.perf_counter()-start,jit_signatures=8,runtime_jit_compiles=0,solver_seconds=sum(r['solver_seconds'] for r in rows),applied_steps=sum(r['steps'] for r in rows))
    write_json(root/'summary.json',summary);print(json.dumps(dict(stage='completed',**summary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output','bundle'):p.add_argument('--'+name,required=True)
    p.add_argument('--noise-scale',type=float,default=0.);p.add_argument('--steps',type=int,default=800);p.add_argument('--limit',type=int)
    p.add_argument('--ordered-waypoints',action='store_true');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    run(**vars(p.parse_args()))
