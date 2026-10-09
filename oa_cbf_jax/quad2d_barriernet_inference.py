"""Quad2d barriernet inference functions and shared contracts."""

import numpy as np

from scipy.optimize import linprog

from scipy.special import expit

from .barriernet_inference import check_optimality

from .quad2d_odqp import nominal as numpy_nominal

from .quad2d_control import FlightConfig

def numpy_features(x,goal,obs,mask,radius=.3):
    valid=obs[np.asarray(mask,bool)]
    order=np.argsort(np.linalg.norm(valid[:,:2]-x[:2],axis=1)-valid[:,2]-radius,kind='stable')[:5]
    selected=np.pad(valid[order],((0,0),(0,7-obs.shape[-1])))
    selected=np.vstack((selected,np.tile([100.,100.,0.,0.,0.,0.,0.],(5-len(selected),1))))
    delta=selected[:,:2]-x[:2]
    angle=(np.arctan2(delta[:,1],delta[:,0])-x[2]+np.pi)%(2*np.pi)-np.pi
    z=np.column_stack((delta,angle,np.linalg.norm(delta,axis=1)-selected[:,2]-radius,np.full(5,np.hypot(x[3],x[4])))).reshape(25)
    return z,np.r_[x,goal,selected.ravel()]

def numpy_constraints(ctx,p,radius=.3,lower=1.,upper=10.):
    x=ctx[:6];obs=ctx[8:].reshape(5,7);rows=[];rhs=[]
    for obstacle,gain in zip(obs,p):
        dx,dz=x[:2]-obstacle[:2]
        barrier=dx*dx+dz*dz-1.01*(obstacle[2]+radius)**2
        derivative=2*(dx*x[3]+dz*x[4])
        drift=2*(x[3]*x[3]+x[4]*x[4])-2*9.81*dz
        coefficient=2*(-dx*np.sin(x[2])+dz*np.cos(x[2]))
        rows.append([-coefficient,-coefficient])
        rhs.append(drift+(gain[0]+gain[1])*derivative+gain[0]*gain[1]*barrier)
    return np.vstack((rows,np.eye(2),-np.eye(2))),np.r_[rhs,[upper]*2,[-lower]*2]

def numpy_network(weights,mean,std,z,ctx,reference):
    def linear(x,name):return x@weights[name]['kernel']+weights[name]['bias']
    encoded=np.maximum(linear(((z-mean)/std).reshape(5,5),'obs_fc1'),0.)
    encoded=np.maximum(linear(encoded,'obs_fc2'),0.)
    gains=4*expit(linear(encoded,'fc_p'))
    hidden=np.maximum(linear(np.r_[encoded.mean(0),ctx[:8],reference],'u_fc1'),0.)
    return reference+linear(hidden,'u_out'),gains

def feasibility(A,b):
    result=linprog(np.zeros(2),A_ub=A,b_ub=b,bounds=[(None,None)]*2,method='highs')
    if result.status not in (0,2):raise ValueError('Independent native QP feasibility unresolved')
    return result.status==0

def check_deployment(result,ctx,p,u_nom,config=FlightConfig(),*,classify=False):
    """Raw native optimum, wrapper clipping and final FP32 acceptance separately."""
    c=config.robot;A,b=numpy_constraints(ctx,p,c.radius)
    shared_A,shared_b=numpy_constraints(ctx,p,c.radius,c.force_min,c.force_max)
    raw=np.asarray(result['raw_control']);candidate=np.asarray(result['candidate']);solver_valid=bool(result['solver_valid'])
    clipped=np.clip(raw,c.force_min,c.force_max)
    np.testing.assert_array_equal(candidate,clipped)
    stationarity=0.;raw_violation=np.inf
    if solver_valid:
        if not np.isfinite(raw).all():raise ValueError('Nonfinite successful native solve')
        raw_violation,stationarity=check_optimality(raw,u_nom,A,b,1+1e-6)
        if raw_violation>c.qp_tolerance+1e-9 or stationarity>2e-6:raise ValueError(f'Invalid native raw optimum {raw_violation}, {stationarity}')
        np.testing.assert_allclose(result['raw_violation'],raw_violation,atol=1e-9,rtol=1e-10)
    elif np.isfinite(raw).any():raise ValueError('Missing native action must retain NaNs')
    violation=float(np.max(shared_A@candidate-shared_b)) if np.isfinite(candidate).all() else np.inf
    valid=solver_valid and np.isfinite(candidate).all() and violation<=c.qp_tolerance
    if valid!=bool(result['valid']):raise ValueError('Native post-clip acceptance changed')
    if np.isfinite(violation):np.testing.assert_allclose(result['violation'],violation,atol=1e-9,rtol=1e-10)
    stored_violation=float(np.max(shared_A@candidate.astype(np.float32).astype(float)-shared_b)) if np.isfinite(candidate).all() else np.inf
    accepted=valid and stored_violation<=c.qp_tolerance
    return dict(accepted=bool(accepted),raw_violation=float(raw_violation),stored_violation=stored_violation,stationarity=stationarity,
        independently_feasible=feasibility(A,b) if classify and not solver_valid else bool(solver_valid),
        clipped=bool(solver_valid and not np.array_equal(raw,candidate)))


import argparse

import json

from pathlib import Path

import time

import jax

import jax.numpy as jnp


from flax import serialization
from .barriernet import BarrierNet, require_x64
from .quad2d_barriernet import contract, features, nominal, constraints, deployment_qp


from .io import sha256

from .io import write_json

def load_bundle(bundle):
    require_x64();root=Path(bundle);m=json.loads((root/'manifest.json').read_text())
    if m['schema']!='barriernet_quad2d_jax_v1' or m['task_contract']!=contract():raise ValueError('Wrong native flight model/task contract')
    for file,key in [('weights.msgpack','weights_sha256'),('normalization.npz','normalization_sha256')]:
        if sha256(root/file)!=m[key]:raise ValueError('Changed native flight weights/normalization')
    model=BarrierNet();template=model.init(jax.random.PRNGKey(0),jnp.zeros(25),jnp.zeros(6),jnp.zeros(2),jnp.zeros(2))['params']
    params=serialization.from_bytes(template,(root/'weights.msgpack').read_bytes())
    if not all(np.isfinite(v).all() for v in jax.tree.leaves(params)):raise ValueError('Nonfinite native weights')
    with np.load(root/'normalization.npz') as f:mean,std=f['mean'],f['std']
    if mean.shape!=(25,) or std.shape!=(25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.all(std>0):raise ValueError('Invalid native normalization')
    return m,model,params,jnp.asarray(mean),jnp.asarray(std)

def policy_function(model,params,mean,std,config=FlightConfig()):
    def policy(state,goal,obstacles,mask):
        z,ctx=features(state,goal,obstacles,mask,config.robot.radius)
        reference=nominal(state,goal,config)
        u_nom,p=model.apply({'params':params},(z-mean)/std,state,goal,reference)
        G,h=constraints(state,ctx[8:].reshape(5,7),p,config.robot.radius)
        candidate,valid,violation,raw,solver_valid,raw_violation=deployment_qp(u_nom,G,h,config)
        return dict(candidate=candidate,valid=valid,violation=violation,raw_control=raw,solver_valid=solver_valid,raw_violation=raw_violation,
            gains=p,u_nom=u_nom,reference=reference)
    return jax.jit(policy)

def make_policy(bundle):
    m,model,params,mean,std=load_bundle(bundle)
    return policy_function(model,params,mean,std),m

def audit(bundle,samples=512):
    m,model,params,mean,std=load_bundle(bundle);dataset=Path(m['dataset']);dm=json.loads((dataset/'manifest.json').read_text())
    if sha256(dataset/'manifest.json')!=m['dataset_manifest_sha256'] or sha256(dataset/'data.npz')!=dm['data_sha256']:raise ValueError('Changed native training provenance')
    da=json.loads((dataset/'independent_audit.json').read_text())
    if not da['audit_passed'] or da['manifest_sha256']!=sha256(dataset/'manifest.json'):raise ValueError('Missing native dataset audit')
    with np.load(dataset/'data.npz') as f:d=dict(f)
    train=d['z'][(d['split']==0)&d['valid']];expected_mean=train.mean(0);expected_std=train.std(0);expected_std=np.where(expected_std==0,1.,expected_std)
    np.testing.assert_array_equal(mean,expected_mean);np.testing.assert_array_equal(std,expected_std)
    pool=np.flatnonzero(d['split']==2);chosen=np.random.default_rng(904).choice(pool,min(samples,len(pool)),replace=False)
    source=Path(dm['source']);entries={e['group_id']:e for e in json.loads((source/'index.json').read_text())};by_group={}
    for row in chosen:by_group.setdefault(str(d['group_id'][row]),[]).append(int(row))
    weights=jax.device_get(params);fn=policy_function(model,params,mean,std);execute=None;details=[];maximum_p=maximum_u=maximum_kkt=0.;invalid=clipped=feasible_rejections=0
    for group,rows in by_group.items():
        e=entries[group];path=source/e['file']
        if sha256(path)!=e['sha256']:raise ValueError('Changed raw deployment audit trace')
        with np.load(path) as trace:
            states=trace['observed_state'];obs=trace['observed_obstacles'];targets=trace['route_target'];mask=trace['obstacle_mask']
            for row in rows:
                tick=int(d['tick'][row]);x=states[tick].astype(float);goal=targets[tick].astype(float);o=obs[tick].astype(float)
                z,ctx=numpy_features(x,goal,o,mask,m['radius']);np.testing.assert_allclose(z,d['z'][row],atol=1e-10,rtol=0)
                ref=numpy_nominal(x,goal);u_nom,p=numpy_network(weights,expected_mean,expected_std,z,ctx,ref)
                args=tuple(jnp.asarray(v) for v in (x,goal,o,mask))
                if execute is None:execute=fn.lower(*args).compile()
                result=jax.device_get(execute(*args))
                maximum_p=max(maximum_p,float(np.max(np.abs(result['gains']-p))));maximum_u=max(maximum_u,float(np.max(np.abs(result['u_nom']-u_nom))))
                if maximum_p>1e-10 or maximum_u>1e-10:raise ValueError('Independent native network mismatch')
                checks=check_deployment(result,ctx,p,u_nom,classify=True);maximum_kkt=max(maximum_kkt,checks['stationarity']);clipped+=int(checks['clipped'])
                invalid+=int(not checks['accepted']);feasible_rejections+=int(not result['solver_valid'] and checks['independently_feasible'])
                details.append(dict(row=row,group_id=group,tick=tick,raw_solver_valid=bool(result['solver_valid']),post_clip_valid=bool(result['valid']),actual_fp32_accepted=checks['accepted'],clipped=checks['clipped']))
    report=dict(audit_passed=True,bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),samples=len(details),groups=len(by_group),invalid_applied_commands=invalid,clipped_commands=clipped,
        feasible_raw_solver_rejections=feasible_rejections,max_gain_error=maximum_p,max_nominal_error=maximum_u,max_stationarity_error=maximum_kkt,train_only_normalization_verified=True,runtime_compilations=0,
        scope='Raw pre-action observations from disjoint development-audit parents, native NumPy network and geometry, raw QP KKT, original wrapper clip and actual FP32 recheck. No expert label at inference.',samples_detail=details)
    write_json(Path(bundle)/'independent_inference_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='samples_detail'}),flush=True)

def benchmark(bundle,output,samples=256):
    fn,m=make_policy(bundle);dm=json.loads((Path(m['dataset'])/'manifest.json').read_text());source=Path(dm['source'])
    with np.load(Path(m['dataset'])/'data.npz') as f:
        selected=np.random.default_rng(906).choice(np.flatnonzero(f['split']==2),16,replace=False);groups=f['group_id'][selected];ticks=f['tick'][selected]
    entries={e['group_id']:e for e in json.loads((source/'index.json').read_text())};args=[]
    for group,tick in zip(groups,ticks):
        e=entries[str(group)];path=source/e['file']
        if sha256(path)!=e['sha256']:raise ValueError('Changed benchmark context')
        with np.load(path) as f:args.append(tuple(jnp.asarray(v,bool if i==3 else jnp.float64) for i,v in enumerate((f['observed_state'][tick],f['route_target'][tick],f['observed_obstacles'][tick],f['obstacle_mask']))))
    start=time.perf_counter();execute=fn.lower(*args[0]).compile();jax.block_until_ready(execute(*args[0]));cold=time.perf_counter()-start;times=[]
    for i in range(samples):
        start=time.perf_counter();jax.device_get(execute(*args[i%16]));times.append(time.perf_counter()-start)
    report=dict(device=str(jax.devices()[0]),samples=samples,observations=16,cold_seconds=cold,milliseconds=(1000*np.quantile(times,[.5,.95,.99,1])).tolist(),deadline50ms_misses=int(np.sum(np.asarray(times)>.05)),runtime_compilations=0,
        weights_sha256=m['weights_sha256'],scope='AOT synchronized full native flight policy including feature selection, MLP, hard QP, clip/status;16 real development observations. Whole physical episode throughput still required for final hardware selection.')
    write_json(output,report);print(json.dumps(report),flush=True)



from .quad2d_static_inputs import validate_parent, numpy_arrived

from .quad2d_rollout import NAMES

from .io import sanitize

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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','benchmark']);p.add_argument('--bundle',required=True);p.add_argument('--output');a=p.parse_args()
    audit(a.bundle) if a.action=='audit' else benchmark(a.bundle,a.output)
