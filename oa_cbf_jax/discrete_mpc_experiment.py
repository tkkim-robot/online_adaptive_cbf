"""Original default discrete MPC controllers in the shared sensed physical task.

The CPU optimizer is synchronous. JAX sensor/plant/route kernels are compiled
once before any episode. No controller receives latent state or future noise.
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
from .cli import sanitize
from .config import UnicycleConfig
from .dataset import source_fingerprint,sha256
from .discrete_mpc import DiscreteMPC,DEFAULTS,numpy_barrier
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .io import write_json
from .routing import route_target,physical_route_coordinate
from .simulation import RUNNING,GOAL,COLLISION,INFEASIBLE,TIMEOUT,STATUS_NAMES
from .stochastic import conditioned_sensor_model,STATE_BOUND_VIOLATION
from .waypoint_tasks import CONTRACT

PLANNER_FAILURE=7
NAMES={**STATUS_NAMES,PLANNER_FAILURE:'planner_failure',STATE_BOUND_VIOLATION:'state_bound_violation'}
CONDITIONS='Identical observed initial scenes, routes, physical prior, sensor noise stream and limits for every method. Synchronous compute; no real-delay or final-test claim.'


from .physical_kernels import PhysicalKernels


def check_applied(sensed,control,obstacles,mask,omega,solver,speed_error):
    """Recheck the actual FP32 command; reject rather than clip a bad action."""
    robot=solver.robot
    residual=numpy_barrier(sensed,control,obstacles,solver.gains,omega,robot)
    low=max(0.,float(sensed[3])-speed_error)+robot.dt*float(control[0])
    high=min(robot.v_max,float(sensed[3])+speed_error)+robot.dt*float(control[0])
    violation=max(float(np.max(-residual[np.asarray(mask,bool)],initial=-np.inf)),
        float(np.max(np.abs(control)-[robot.a_max,robot.w_max])),-low,high-robot.v_max)
    return bool(np.isfinite(control).all() and np.isfinite(omega).all() and violation<=robot.qp_tolerance),violation


def episode(record,solver,kernels,steps,noise_scale,ordered=False):
    robot=solver.robot;scene=record['scene'];f=lambda a:np.asarray(a,np.float32)
    mask=np.asarray(scene['obstacle_mask'],bool);observed=f(scene['initial_state']);obstacles=f(scene['obstacles'])
    if ordered:
        count=record['waypoint_count'];goals=f(record['waypoint_goals']);routes=record['waypoint_routes']
        points=f(routes['points']);rmask=np.asarray(routes['mask'],bool);ready=np.asarray(routes['ready'],bool)
    else:
        count=1;goals=f([scene['goal']]);points=f([record['route']['points']])
        rmask=np.asarray([record['route']['mask']],bool);ready=np.asarray([record['route']['status']=='ready'])
    noise=noise_scale*np.array([.02,.03,.02,.03,.025,.01]);fn=f(noise)
    seed=int.from_bytes(hashlib.sha256(scene['scene_id'].encode()).digest()[:4],'little');key=np.asarray(jax.random.PRNGKey(seed))
    x,truth_obs,xb,ob,xs,os,innovations=kernels.prepare(observed,obstacles,mask,fn,key)
    initial=np.asarray(x);truth=np.asarray(truth_obs);minimum=float(kernels.clearance(x,truth_obs,mask))
    status=PLANNER_FAILURE if not ready[0] else COLLISION if minimum<=0 else GOAL if count==1 and bool(kernels.arrived(x,goals[0],np.zeros(6,np.float32))) else RUNNING
    leg=0;cursor=np.float32(0);previous=np.zeros(2,np.float32);solver.last_omega=np.zeros(2)
    traces=[];applied=0;solve_times=[];step_times=[];solver_statuses={};start=time.perf_counter()
    for k in range(steps):
        tick=time.perf_counter();sx,so=kernels.sense(x,truth_obs,xb,ob,xs,os,innovations[k],np.int32(k))
        sx,so=np.asarray(sx),np.asarray(so)
        handoff=status==RUNNING and leg<count-1 and bool(kernels.arrived(sx,goals[leg],fn))
        if handoff:leg+=1;cursor=np.float32(0)
        if status==RUNNING and not ready[leg]:status=PLANNER_FAILURE
        attempt=status==RUNNING;result=None;u=np.zeros(2,np.float32);omega=np.ones(2);clear=bound=violation=np.nan
        target,proposal,remaining=kernels.target(sx,points[leg],rmask[leg],cursor)
        if attempt:
            # Pinned tracking.update_goal and MPCCBF.solve_control_problem use
            # the current mandatory waypoint, not a receding route lookahead.
            result=solver.solve(sx,goals[leg],so,mask,previous,speed_error=1.15*float(fn[2]))
            solve_times.append(result['solve_seconds']);label=result['solver_status'];solver_statuses[label]=solver_statuses.get(label,0)+1
            candidate=f(result['control']);omega=result['omega']
            accepted,violation=check_applied(sx,candidate,so,mask,omega,solver,1.15*float(fn[2]))
            accepted=accepted and result['feasible']
            if accepted:
                u=candidate;x,clear,bound=kernels.advance(x,u,truth_obs,mask,np.int32(k))
                clear,bound=float(clear),float(bound);minimum=min(minimum,clear)
                cursor=np.float32(proposal);previous=u.copy();solver.last_omega=omega.copy();applied+=1
                if leg==count-1 and bool(kernels.arrived(x,goals[leg],np.zeros(6,np.float32))):status=GOAL
                if bound>robot.qp_tolerance:status=STATE_BOUND_VIOLATION
                if clear<=0:status=COLLISION
            else:status=INFEASIBLE
        else:accepted=False
        traces.append(dict(state=np.asarray(x),control=u,active=accepted,status=status,gains=solver.gains,
            omega=omega,observed_state=sx,observed_obstacles=so,clearance=clear,state_bound_violation=bound,
            qp_violation=violation,route_target=np.asarray(target),mpc_reference=goals[leg],route_remaining=float(remaining),route_progress=cursor,
            selection_tick=attempt,selection_accepted=accepted,solver_status='' if result is None else result['solver_status'],
            solver_success=False if result is None else result['solver_success'],
            solver_iterations=-1 if result is None else result['iterations'],solver_seconds=np.nan if result is None else result['solve_seconds'],
            predicted_states=np.full((11,4),np.nan) if result is None else result['states'],
            predicted_controls=np.full((10,2),np.nan) if result is None else result['controls'],
            predicted_omegas=np.full((10,2),np.nan) if result is None else result['omegas'],
            solver_equality_error=np.nan if result is None else result['max_equality_error'],
            solver_constraint_violation=np.nan if result is None else result['max_constraint_violation'],
            waypoint_index=leg,waypoint_handoff=handoff,waypoints_visited=leg+int(status==GOAL),mission_goal=goals[leg],
            collision_event=accepted and clear<=0,state_bound_event=accepted and bound>robot.qp_tolerance))
        step_times.append(time.perf_counter()-tick)
        if status!=RUNNING:break
    if status==RUNNING:status=TIMEOUT
    trace={k:np.stack([t[k] for t in traces]) for k in traces[0]}
    row=dict(scene_id=scene['scene_id'],family=scene['family'],mode='discrete_mpc_'+solver.method,status=NAMES[status],steps=applied,
        min_clearance=minimum,goal_progress=float(np.linalg.norm(initial[:2]-goals[count-1])-np.linalg.norm(np.asarray(x)[:2]-goals[count-1])),
        final_route_coordinate=float(kernels.coordinate(x[:2],points[leg],rmask[leg],cursor)),
        applied_source_counts={'default_discrete_mpc':applied},selection_attempts=len(solve_times),
        selection_rejections=int(status==INFEASIBLE),reactive_reselections=0,gate_stage_totals=[],gain_total_variation=0.,
        max_physical_bound_violation=float(np.max(trace['state_bound_violation'][trace['active']])) if applied else None,
        solver_status_counts=solver_statuses,elapsed_seconds=time.perf_counter()-start,
        solver_seconds=sum(solve_times),tick_seconds_p50_p99_max=np.quantile(step_times,[.5,.99,1]).tolist(),
        collision_event=bool(np.any(trace['collision_event'])) or minimum<=0,state_bound_event=bool(np.any(trace['state_bound_event'])))
    if status==INFEASIBLE:
        row['termination_reason']=('solver_reported_failure:'+result['solver_status'] if not result['solver_success'] else
            'independent_prediction_residual_rejection' if not result['feasible'] else 'applied_command_residual_rejection')
    if ordered:row.update(waypoint_index=leg,waypoints_visited=leg+int(status==GOAL),required_waypoints=count,waypoint_handoffs=int(trace['waypoint_handoff'].sum()))
    return row,trace,dict(true_initial_state=initial,true_obstacles=truth,scene_id=scene['scene_id'],noise=noise,key=key)


def run(source,output,method='fixed_low',noise_scale=0.,steps=800,limit=None,ordered_waypoints=False,shard_index=0,shards=1):
    if steps<1 or shards<1 or not 0<=shard_index<shards or noise_scale<0:raise ValueError('Invalid evaluation budget/shard/noise')
    records=json.loads((Path(source)/'scenes.json').read_text())
    if any('waypoint_goals' in r for r in records) and not ordered_waypoints:raise ValueError('Ordered tasks require ordered evaluation; no final-goal bypass')
    if limit is not None:records=[r for r in records if not r['scene']['scene_id'].startswith('fixture:')][:limit]+[r for r in records if r['scene']['scene_id'].startswith('fixture:')]
    records=records[shard_index::shards]
    if not records:raise ValueError('Empty evaluation shard')
    capacity=len(records[0]['scene']['obstacles']);field='waypoint_routes' if ordered_waypoints else 'route'
    route_capacity=len(records[0][field]['points'][0] if ordered_waypoints else records[0][field]['points'])
    for r in records:
        if len(r['scene']['obstacles'])!=capacity:raise ValueError('Mixed obstacle capacities')
        shape=np.shape(r[field]['points'])
        if shape!=((3,route_capacity,2) if ordered_waypoints else (route_capacity,2)):raise ValueError('Mixed route capacities')
        if ordered_waypoints and (not 1<=r['waypoint_count']<=3 or not np.array_equal(r['waypoint_goals'][r['waypoint_count']-1],r['scene']['goal'])):
            raise ValueError('Invalid ordered goal count/final goal')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);robot=UnicycleConfig()
    solver=DiscreteMPC(capacity,method,robot);kernels=PhysicalKernels(capacity,route_capacity,steps,robot)
    manifest=dict(stage='development_default_discrete_mpc',final_test=False,source_fingerprint=source_fingerprint(),
        policy=dict(mode='discrete_mpc_'+method,motion_observer_window=0,sensor_margin_scale=0.,gains=solver.gains.tolist()),robot=asdict(robot),
        discrete_mpc=solver.contract(),conditions=CONDITIONS+(' Ordered waypoint contract: '+CONTRACT['name'] if ordered_waypoints else ''),
        noise_scale=noise_scale,steps=steps,batch=1,input_scene_sha256=sha256(Path(source)/'scenes.json'),input_scenes=str(Path(source).resolve()),
        shard_index=shard_index,shards=shards,limit=limit,scene_ids=[r['scene']['scene_id'] for r in records],
        reference='Pinned current mandatory waypoint XY target; pinned MPC goal-state cost. Shared route is available and logged for progress only. No route-lookahead substitution, nominal turn bank, observer, gain search, model or baseline tuning.',
        initialization_compile_seconds=kernels.compile_seconds,solver_setup_seconds=solver.setup_seconds,
        runtime_compilation='All seven JAX sensor/plant/route helpers lowered and compiled before any episode; compiled executables only in runtime loop.',
        censoring='Rejected/numerically failed solve applies no action. Future physical evolution is unobserved; no safety claim after a stop.')
    if ordered_waypoints:manifest['waypoint_contract']=CONTRACT
    write_json(root/'manifest.json',manifest);rows=[];start=time.perf_counter()
    print(json.dumps(dict(stage='ready',groups=len(records),jit_compile_seconds=kernels.compile_seconds,solver_setup_seconds=solver.setup_seconds)),flush=True)
    for i,r in enumerate(records):
        row,trace,truth=episode(r,solver,kernels,steps,noise_scale,ordered_waypoints);rows.append(row)
        np.savez_compressed(root/f'traces_{i:05d}.npz',**{k:v[None] for k,v in trace.items()},**{k:np.asarray(v)[None] for k,v in truth.items()})
        write_json(root/'results.json',sanitize(rows))
        progress=dict(completed_groups=i+1,total_groups=len(records),elapsed_seconds=time.perf_counter()-start,last_scene=row['scene_id'],last_status=row['status'],last_steps=row['steps'])
        write_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    evaluated=[r for r in rows if not r['scene_id'].startswith('fixture:')]
    aggregate=dict(groups=len(evaluated),outcomes={s:sum(r['status']==s for r in evaluated) for s in sorted({r['status'] for r in evaluated})})
    write_json(root/'summary.json',dict(aggregate=aggregate,elapsed_seconds=time.perf_counter()-start,jit_signatures=7,runtime_jit_compiles=0,
        solver_seconds=sum(r['solver_seconds'] for r in rows),applied_steps=sum(r['steps'] for r in rows)))
    print(json.dumps(dict(stage='completed',aggregate=aggregate,elapsed_seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--method',choices=list(DEFAULTS),default='fixed_low');p.add_argument('--noise-scale',type=float,default=0.)
    p.add_argument('--steps',type=int,default=800);p.add_argument('--limit',type=int);p.add_argument('--ordered-waypoints',action='store_true')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    run(**vars(p.parse_args()))
