"""Barriernet inference functions and shared contracts."""

import argparse

import json

from pathlib import Path

import time

import jax

import jax.numpy as jnp

import numpy as np

from flax import serialization

from .barriernet import BarrierNet, features, nominal, constraints, hard_deployment_qp, require_x64

from .io import sha256

from .io import write_json

def load_bundle(bundle):
    require_x64();root=Path(bundle);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['schema']!='barriernet_unicycle_jax_v1':raise ValueError('Not native BarrierNet')
    for filename,field in [('weights.msgpack','weights_sha256'),('normalization.npz','normalization_sha256')]:
        if sha256(root/filename)!=manifest[field]:raise ValueError('Changed BarrierNet bundle')
    model=BarrierNet();template=model.init(jax.random.PRNGKey(0),jnp.zeros(25),jnp.zeros(4),jnp.zeros(2),jnp.zeros(2))['params']
    params=serialization.from_bytes(template,(root/'weights.msgpack').read_bytes())
    with np.load(root/'normalization.npz') as f:mean,std=f['mean'],f['std']
    if mean.shape!=(25,) or std.shape!=(25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.all(std>0):raise ValueError('Invalid normalization')
    return manifest,model,params,jnp.asarray(mean),jnp.asarray(std)

def make_policy(bundle,radius=None):
    manifest,model,params,mean,std=load_bundle(bundle)
    radius=manifest['radius'] if radius is None else radius
    def policy(state,goal,obstacles,mask):
        z,ctx=features(state,goal,obstacles,mask,radius)
        reference=nominal(state,goal)
        u_nom,p=model.apply({'params':params},(z-mean)/std,state,goal,reference)
        G,h=constraints(state,ctx[6:].reshape(5,7),p,radius)
        control,valid,violation=hard_deployment_qp(u_nom,G,h)
        return control,valid,violation,p,u_nom
    return jax.jit(policy),manifest

def audit(bundle,samples=512):
    from .barriernet_inference import numpy_features, numpy_nominal, numpy_constraints, check_optimality
    from scipy.optimize import linprog
    manifest,_,params,mean,std=load_bundle(bundle);weights=jax.device_get(params)
    dataset=Path(manifest['dataset']);dm=json.loads((dataset/'manifest.json').read_text())
    if sha256(dataset/'manifest.json')!=manifest['dataset_manifest_sha256'] or sha256(dataset/'data.npz')!=dm['data_sha256']:raise ValueError('Changed training provenance')
    da=json.loads((dataset/'independent_audit.json').read_text())
    if not da['audit_passed'] or da['manifest_sha256']!=sha256(dataset/'manifest.json'):raise ValueError('Missing dataset audit')
    with np.load(dataset/'data.npz') as f:data={k:f[k] for k in f.files}
    # Normalization is independently recomputed from valid training parents.
    train=data['z'][(data['split']==0)&data['valid']]
    expected_mean=train.mean(0);expected_std=train.std(0);expected_std=np.where(expected_std==0,1.,expected_std)
    np.testing.assert_array_equal(mean,expected_mean);np.testing.assert_array_equal(std,expected_std)
    pool=np.flatnonzero(data['split']==2);chosen=np.random.default_rng(904).choice(pool,min(samples,len(pool)),replace=False)
    source=Path(dm['source']);provenance={p['group_id']:p for p in dm['provenance']};entries={e['visitation_file']:e for e in json.loads((source/'index.json').read_text())}
    by_group={}
    for row in chosen:by_group.setdefault(str(data['group_id'][row]),[]).append(int(row))
    fn,_=make_policy(bundle);execute=None;maximum_p=maximum_nominal=maximum_primal=maximum_kkt=0.;invalid=0;feasible_rejections=0;checked=[]
    def linear(x,name):return x@weights[name]['kernel']+weights[name]['bias']
    for group,rows in by_group.items():
        e=entries[provenance[group]['source_trace']]
        if sha256(source/e['visitation_file'])!=e['visitation_sha256']:raise ValueError('Changed raw observations')
        with np.load(source/e['visitation_file']) as trace:
            states=trace['observed_state'];obstacles=trace['raw_observed_obstacles'];mask=trace['obstacle_mask']
            for row in rows:
                k=int(data['tick'][row]);x=states[k].astype(float);obs=obstacles[k].astype(float);goal=data['ctx'][row,4:6]
                z,ctx=numpy_features(x,goal,obs,mask,manifest['radius']);reference=numpy_nominal(x,goal)
                encoded=np.maximum(linear(((z-expected_mean)/expected_std).reshape(5,5),'obs_fc1'),0.)
                encoded=np.maximum(linear(encoded,'obs_fc2'),0.);p=4/(1+np.exp(-linear(encoded,'fc_p')))
                hidden=np.maximum(linear(np.r_[encoded.mean(0),x,goal,reference],'u_fc1'),0.)
                u_nom=reference+linear(hidden,'u_out')
                args=tuple(jnp.asarray(v) for v in (x,goal,obs,mask))
                if execute is None:execute=fn.lower(*args).compile()
                u,valid,violation,jp,ju=jax.device_get(execute(*args))
                maximum_p=max(maximum_p,float(np.max(np.abs(jp-p))));maximum_nominal=max(maximum_nominal,float(np.max(np.abs(ju-u_nom))))
                if maximum_p>1e-10 or maximum_nominal>1e-10:raise ValueError('NumPy network/inference mismatch')
                A,b=numpy_constraints(ctx,p,manifest['radius'])
                if valid:
                    primal,kkt=check_optimality(u,u_nom,A,b,1+1e-6)
                    if primal>1e-5 or kkt>2e-6:raise ValueError(f'Invalid deployed optimum {primal}, {kkt}')
                    maximum_primal=max(maximum_primal,primal);maximum_kkt=max(maximum_kkt,kkt)
                else:
                    result=linprog(np.zeros(2),A_ub=A,b_ub=b,bounds=[(None,None)]*2,method='highs')
                    if result.status not in (0,2):raise ValueError('Independent deployment feasibility unresolved')
                    if result.status==0:
                        feasible_rejections+=1
                        if np.isfinite(u).all() and np.max(A@u-b)<=1e-5:raise ValueError('Rejected finite command has no independent constraint violation')
                    invalid+=1
                checked.append(dict(row=row,group_id=group,tick=k,valid=bool(valid)))
    report=dict(audit_passed=True,bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),samples=len(checked),groups=len(by_group),invalid_solves_checked=invalid,
        feasible_solver_rejections_checked=feasible_rejections,independently_infeasible_solves_checked=invalid-feasible_rejections,
        max_gain_error=maximum_p,max_nominal_error=maximum_nominal,max_primal_violation=maximum_primal,max_stationarity_error=maximum_kkt,
        train_only_normalization_verified=True,runtime_compilations=0,source='Raw recorded pre-action observations from independent development-audit parents. Original nominal reference only; no expert label at deployment.',
        limitation='Algebraic deployment/lineage audit, not closed-loop performance or generalization evidence.',samples_detail=checked)
    write_json(Path(bundle)/'independent_inference_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='samples_detail'}),flush=True)

def benchmark(bundle,output,samples=256):
    fn,_=make_policy(bundle)
    # Fixed-capacity whole policy: feature selection, MLP, CBF and bounded QP.
    rng=np.random.default_rng(905);obs=np.zeros((64,5));obs[:,:2]=rng.uniform(-10,10,(64,2));obs[:,2]=.3
    args=tuple(jnp.asarray(v) for v in (np.array([0.,0.,.2,.3]),np.array([5.,2.]),obs,np.ones(64,bool)))
    begin=time.perf_counter();execute=fn.lower(*args).compile();jax.block_until_ready(execute(*args));cold=time.perf_counter()-begin
    times=[]
    for _ in range(samples):
        t=time.perf_counter();jax.device_get(execute(*args));times.append(time.perf_counter()-t)
    report=dict(device=str(jax.devices()[0]),samples=samples,cold_seconds=cold,milliseconds=(1000*np.quantile(times,[.5,.95,.99,1])).tolist(),
        runtime_compilations=0,deadline50ms_misses=int(np.sum(np.asarray(times)>.05)),weights_sha256=sha256(Path(bundle)/'weights.msgpack'),
        scope='Synchronized native FP64 complete policy from64 raw obstacle slots to returned command/status/per-obstacle gains; excludes physical simulator, observation transfer and real compute delay.')
    write_json(output,report);print(json.dumps(report),flush=True)



from scipy.optimize import nnls

def numpy_features(x,goal,obs,mask,radius):
    valid=obs[np.asarray(mask,bool)]
    order=np.argsort(np.linalg.norm(valid[:,:2]-x[:2],axis=1)-valid[:,2]-radius,kind='stable')[:5]
    selected=np.pad(valid[order],((0,0),(0,7-obs.shape[-1])))
    selected=np.vstack((selected,np.tile([100.,100.,0.,0.,0.,0.,0.],(5-len(selected),1))))
    delta=selected[:,:2]-x[:2]
    angle=(np.arctan2(delta[:,1],delta[:,0])-x[2]+np.pi)%(2*np.pi)-np.pi
    z=np.column_stack((delta,angle,np.linalg.norm(delta,axis=1)-selected[:,2]-radius,np.full(5,x[3]))).reshape(25)
    return z,np.r_[x,goal,selected.ravel()]

def numpy_constraints(ctx,p,radius):
    x=ctx[:4];obs=ctx[6:].reshape(5,7);rows=[];rhs=[]
    for obstacle,gain in zip(obs,p):
        dx,dy=x[:2]-obstacle[:2];c=np.cos(x[2]);s=np.sin(x[2]);v=x[3]
        barrier=dx*dx+dy*dy-1.01*(obstacle[2]+radius)**2
        along=dx*c+dy*s
        rows.append([-2*along,-2*v*(-dx*s+dy*c)])
        rhs.append(2*v*v+(gain[0]+gain[1])*2*v*along+gain[0]*gain[1]*barrier)
    return np.vstack((rows,np.eye(2),-np.eye(2))),np.r_[rhs,[.5]*4]

def numpy_nominal(x,goal):
    dx,dy=goal-x[:2];distance=max(np.hypot(dx,dy)-.05,0.)
    angle=(np.arctan2(dy,dx)-x[2]+np.pi)%(2*np.pi)-np.pi
    speed=0. if abs(angle)>np.pi/2 else min(distance*np.cos(angle),1.)
    return np.array([speed-x[3],2*angle])

def check_optimality(u,reference,A,b,diagonal=1.):
    residual=A@u-b
    active=residual>=-1e-6
    gradient=diagonal*u-reference
    if active.any():
        multipliers,_=nnls(A[active].T,-gradient,maxiter=1000)
        stationarity=np.linalg.norm(gradient+A[active].T@multipliers,ord=np.inf)
    else:stationarity=np.linalg.norm(gradient,ord=np.inf)
    return float(np.max(residual)),float(stationarity)


import hashlib


from .config import UnicycleConfig

from .simulation import RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT, STATUS_NAMES

from .stochastic import STATE_BOUND_VIOLATION

PLANNER_FAILURE=7

NAMES={**STATUS_NAMES,PLANNER_FAILURE:'planner_failure',STATE_BOUND_VIOLATION:'state_bound_violation'}

NAMES[INFEASIBLE]='solver_rejected'

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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','benchmark']);p.add_argument('--bundle',required=True);p.add_argument('--output')
    a=p.parse_args();audit(a.bundle) if a.action=='audit' else benchmark(a.bundle,a.output)
