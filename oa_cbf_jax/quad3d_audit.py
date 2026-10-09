"""Quad3d audit functions and shared contracts."""

import numpy as np

from numpy.polynomial import Polynomial

from .quad3d_control import Quad3DControlConfig

def coefficients(x, u, config=Quad3DControlConfig()):
    """Explicit component equations, independent of the JAX matrix/flow code."""
    r=config.robot; p=np.zeros((5,12),np.float64); p[0]=x
    az=np.sum(u)/r.mass
    qdot=r.arm/r.inertia_y*(u[1]-u[3]); pdot=r.arm/r.inertia_x*(u[0]-u[2])
    rdot=r.yaw_coefficient/r.inertia_z*(u[0]-u[1]+u[2]-u[3])
    p[1]=np.r_[x[6:9],x[9:12],r.gravity*x[3],-r.gravity*x[4],az,qdot,pdot,rdot]
    p[2,:3]=[r.gravity*x[3]/2,-r.gravity*x[4]/2,az/2]
    p[2,3:6]=[qdot/2,pdot/2,rdot/2]
    p[2,6:8]=[r.gravity*x[9]/2,-r.gravity*x[10]/2]
    p[3,:2]=[r.gravity*x[9]/6,-r.gravity*x[10]/6]
    p[3,6:8]=[r.gravity*qdot/6,-r.gravity*pdot/6]
    p[4,:2]=[r.gravity*qdot/24,-r.gravity*pdot/24]
    return p

def extrema(poly, duration):
    roots=poly.deriv().trim().roots()
    # Include the real part of nearly real roots. Additional points can only
    # tighten the sampled bound; do not discard tangent/double roots by sign.
    ts=[0.,duration,*[float(z.real) for z in roots if abs(z.imag)<1e-6 and 0<z.real<duration]]
    ts.extend(np.linspace(0,duration,17))
    values=poly(np.asarray(ts))
    if not np.isfinite(values).all():raise ValueError('Nonfinite polynomial physical audit')
    return float(np.min(values)),float(np.max(values))

def audit_hold(x,u,next_state,obstacles,mask,config=Quad3DControlConfig()):
    c=config; p=coefficients(np.asarray(x),np.asarray(u),c); dt=c.robot.dt
    expected=np.polynomial.polynomial.polyval(dt,p)
    error=float(np.max(abs(expected-next_state)))
    if error>2e-10:raise AssertionError(f'Full-state held-input replay mismatch: {error}')
    input_violation=float(max(np.max(u-c.robot.input_max),np.max(c.robot.input_min-u)))
    if input_violation>c.qp_tolerance:raise AssertionError('Applied actuator bounds violated')
    clearance=float('inf')
    for o in obstacles[mask]:
        px=Polynomial(p[:,0])-Polynomial([o[0],o[3]])
        py=Polynomial(p[:,1])-Polynomial([o[1],o[4]])
        minimum,_=extrema(px*px+py*py,dt)
        clearance=min(clearance,np.sqrt(max(minimum,0.))-c.robot.radius-o[2])
    violation=-float('inf')
    for i,limit in ((3,c.tilt_limit),(4,c.tilt_limit),(5,c.yaw_limit),
                    *[(k,c.velocity_limit) for k in range(6,9)],*[(k,c.rate_limit) for k in range(9,12)]):
        lo,hi=extrema(Polynomial(p[:,i]),dt);violation=max(violation,hi-limit,-lo-limit)
    lo,hi=extrema(Polynomial(p[:,2]),dt)
    violation=max(violation,hi-c.altitude_max,c.altitude_min-lo)
    return dict(replay_error=error,minimum_clearance=clearance,envelope_violation=violation,input_violation=input_violation)

def independent_obstacle_values(x,u,obstacles,mask,gains,config=Quad3DControlConfig()):
    """Vectorized t=0 derivatives of the independent held-motion polynomial.

    Keep polynomial_obstacle_values as the separate object-based reference.
    Only these t=0 derivatives are batched; continuous extrema/root checks in
    audit_hold and independent_held_cascade_minimum are unchanged.
    """
    p=coefficients(x,u,config);obs=np.asarray(obstacles)[mask]
    if len(obs)==0:return float('inf'),float('inf')
    d0=p[0,:2]-obs[:,:2];d1=p[1,:2]-obs[:,3:5]
    d2,d3,d4=p[2,:2],p[3,:2],p[4,:2]
    dot=lambda a,b:np.sum(a*b,axis=-1)
    radius=config.robot.radius+config.clearance_buffer+obs[:,2]
    values=np.stack((dot(d0,d0)-radius**2,2*dot(d0,d1),
        2*(dot(d1,d1)+2*dot(d0,d2)),12*(dot(d0,d3)+dot(d1,d2)),
        24*(dot(d2,d2)+2*dot(d1,d3)+2*dot(d0,d4))),axis=-1)
    cascade=float(np.min(values[:,0]))
    for stage,gain in enumerate(gains):
        values=values[:,1:]+gain*values[:,:-1]
        if stage<3:cascade=min(cascade,float(np.min(values[:,0])))
    return cascade,float(np.min(values[:,0]))

def independent_envelope_values(x,u,config=Quad3DControlConfig()):
    """Batch the independent linear barrier derivatives by relative degree."""
    c=config;p=coefficients(x,u,c)
    indices=np.repeat(np.array([3,4,5,6,7,8,9,10,11,2]),2)
    signs=np.tile(np.array([1.,-1.]),10)
    limits=np.r_[np.repeat([c.tilt_limit,c.tilt_limit,c.yaw_limit,
        c.velocity_limit,c.velocity_limit,c.velocity_limit,c.rate_limit,c.rate_limit,c.rate_limit],2),
        c.altitude_max,-c.altitude_min]
    orders=np.repeat(np.array([2,2,2,3,3,1,1,1,1,2]),2)
    values=-signs[:,None]*p[:4,indices].T*np.array([1.,1.,2.,6.])
    values[:,0]+=limits;cascade=residual=float('inf')
    for stage in range(3):
        cascade=min(cascade,float(np.min(values[orders>stage,0])))
        values=values[:,1:]+c.envelope_gain*values[:,:-1]
        residual=min(residual,float(np.min(values[orders==stage+1,0])))
    return cascade,residual

def independent_held_cascade_minimum(x,u,obstacles,mask,gains,config=Quad3DControlConfig()):
    """Extrema of the ACTUAL final cascades, not the controller's tangent rows."""
    c=config;p=coefficients(x,u,c);polys=[]
    for o in obstacles[mask]:
        px=Polynomial(p[:,0])-Polynomial([o[0],o[3]])
        py=Polynomial(p[:,1])-Polynomial([o[1],o[4]])
        h=px*px+py*py-(c.robot.radius+c.clearance_buffer+o[2])**2
        for gain in gains[:3]:h=h.deriv()+gain*h
        polys.append(h)
    for idx,limit,order in ((3,c.tilt_limit,2),(4,c.tilt_limit,2),(5,c.yaw_limit,2),
        (6,c.velocity_limit,3),(7,c.velocity_limit,3),(8,c.velocity_limit,1),
        (9,c.rate_limit,1),(10,c.rate_limit,1),(11,c.rate_limit,1)):
        for sign in (1,-1):
            h=limit-sign*Polynomial(p[:,idx])
            for _ in range(order-1):h=h.deriv()+c.envelope_gain*h
            polys.append(h)
    for h in (c.altitude_max-Polynomial(p[:,2]),Polynomial(p[:,2])-c.altitude_min):
        polys.append(h.deriv()+c.envelope_gain*h)
    return min(extrema(h,c.robot.dt)[0] for h in polys)


import jax

from .quad3d_observed_rollout import observed_control

_CACHE={}

def verify_held(x,goal,o,mask,gain,points,rm,cursor,noise,bias,c,trace,k):
    """Separate single-state AOT replay of the *one* previously held gain.

    This checks the controller failure signal, not a mathematical infeasibility
    theorem. Independent polynomial/physical audits still check applied control.
    No new observation, future noise, model prediction or alternate gain enters.
    """
    from .quad3d_data import to_device
    args=to_device(tuple(np.asarray(a) for a in (x,goal,o,mask,gain,points,rm,cursor,noise,bias)))
    key=(c,len(mask))
    if key not in _CACHE:
        fn=jax.jit(lambda x,g,o,m,a,p,rm,t,n,b:observed_control(x,g,o,m,a,p,rm,t,n,c,True,b)[:6])
        _CACHE[key]=(fn,fn.lower(*args).compile())
    fn,exe=_CACHE[key];actual=jax.device_get(exe(*args))
    for name,value in zip(('held_proposed','held_feasible','held_psi','held_domain','held_residual','held_iterations'),actual,strict=True):
        if name=='held_feasible':assert bool(trace[name][k])==bool(value)
        elif name!='held_iterations':np.testing.assert_allclose(trace[name][k],value,atol=1e-8,rtol=1e-9,equal_nan=True,err_msg=name)
    assert fn._cache_size()==0


from pathlib import Path

from statistics import NormalDist

import math


from scipy.special import expit

from .quad3d_control import control_config

from .quad3d_features import numpy_history_graph

from .quad3d_observation import numpy_update_bias

from .quad3d_observation import unit_tape, numpy_observe, numpy_obstacles, numpy_arrived

from .quad2d_trajectory_gate import numpy_cs

from .io import sha256

from .io import sanitize

def check_statistics(q,previous_gain,m,fit):
    bank=np.asarray(fit['candidates'],np.float64)
    assert fit['candidates']==fit['quad3d_contract']['gain_bank'] and bank.shape==(16,4)
    p=m['policy_config'];mu=q['prediction_mean'].astype(float);var=q['prediction_variance'].astype(float);logits=q['prediction_event_logits'].astype(float)
    n=len(q['query_tick'])
    if mu.shape!=(n,4,16,2) or var.shape!=mu.shape or logits.shape!=mu.shape:raise ValueError('Wrong Quad3D prediction shape')
    cs=numpy_cs(np.moveaxis(mu[...,0],1,0),np.moveaxis(var[...,0],1,0))
    z=NormalDist().inv_cdf(1-p['tail_mass']);coefficient=math.exp(-z*z/2)/math.sqrt(2*math.pi)/p['tail_mass']
    risk=np.max(mu[...,0]+np.sqrt(var[...,0])*coefficient,axis=1)
    e=fit['event_calibration'][1];adverse=expit(logits[...,1]/e['temperature']+e['bias']).max(axis=1);progress=mu[...,1].mean(axis=1)
    score=progress-p['gain_switch_penalty']*np.mean(np.abs(np.log(bank[None]/previous_gain[:,None])),axis=-1)
    for name,value in [('cs_score',cs),('finite_member_cvar',risk),('adverse_probability',adverse),('predicted_progress',progress),('ranking_score',score)]:
        np.testing.assert_allclose(q[name],value,atol=3e-5,rtol=3e-5,err_msg=name)
    finite=np.all(np.isfinite(mu)&np.isfinite(var)&(var>0)&np.isfinite(logits),axis=(1,3))&np.isfinite(q['cs_score'])&np.isfinite(q['finite_member_cvar'])&np.isfinite(q['adverse_probability'])
    threshold=np.float32(np.inf if m['phase']=='gate_calibration' else m['cs_threshold'])
    screened=finite&(q['cs_score']<=threshold)&(q['finite_member_cvar']<=np.float32(p['conditional_risk_limit']))&(q['adverse_probability']<=np.float32(p['adverse_probability_limit']))
    admissible=(q['candidate_psi']>=-m['config']['qp_tolerance'])&(q['candidate_domain'][:,None]>=-m['config']['qp_tolerance'])
    if m.get('input_box_admission',False):
        support=np.isfinite(q['candidate_input_margin'])&(q['candidate_input_margin']>=-m['config']['qp_tolerance'])
        np.testing.assert_array_equal(q['input_admissible'],support)
        admissible&=support
    accepted=screened&admissible
    for key,value in [('screened',screened),('admissible',admissible),('accepted',accepted)]:np.testing.assert_array_equal(q[key],value)
    any_=accepted.any(-1);index=np.argmax(np.where(accepted,q['ranking_score'],-np.inf),axis=-1)
    gain=np.where(any_[:,None],bank[index],previous_gain);index=np.where(any_,index,-1)
    uncertainty=~screened.any(-1);admission=screened.any(-1)&~any_
    if m['phase']=='gate_calibration':gain=np.full((n,4),p['initial_gain']);index=np.full(n,-2);uncertainty=np.zeros(n,bool);admission=np.zeros(n,bool)
    for key,value in [('network_gain',gain),('selected_index',index),('uncertainty_fallback',uncertainty),('admission_fallback',admission)]:np.testing.assert_array_equal(q[key],value)

def audit_one(task):
    from .quad3d_observation_audit import audit_parent
    from .quad3d_learning_contract import read
    parent,row,directory,m=task;root=Path(directory);c=control_config(m['config']);fit=read(m['prediction_fit']);bank=np.asarray(fit['candidates'],np.float64)
    if 'initial_gains_by_encoder' in parent:
        assert parent['gains']==parent['initial_gains_by_encoder'][m['encoder']]==[m['policy_config']['initial_gain']]*4
        assert row['initial_gains']==parent['gains'] and row['motion_stratum']==parent['motion_stratum']
    assert row['sha256']==sha256(root/row['file']) and row['query_sha256']==sha256(root/row['query_file'])
    with np.load(root/row['file']) as z:d=dict(z)
    with np.load(root/row['query_file']) as z:q=dict(z)
    physical=audit_parent((parent,row,directory,m))
    mask=np.asarray(parent['mask']);obs=np.asarray(parent['obstacles']);goal=np.asarray(parent['goal']);noise=np.asarray(parent['noise'])
    points=np.asarray(parent['route']['points']);rm=np.asarray(parent['route']['mask']);bx,bo,ix,io=unit_tape(parent['sensor_seed'],m['steps'],len(mask))
    ordered=m.get('ordered_mission',False);leg=0
    if ordered:
        goals=np.asarray(parent['waypoint_goals']);routes=np.asarray(parent['waypoint_routes']['points']);route_masks=np.asarray(parent['waypoint_routes']['mask'])
    query_ticks=np.flatnonzero(d['requery']);np.testing.assert_array_equal(q['query_tick'],query_ticks)
    previous_u=np.zeros(4);previous_gain=np.asarray(parent['gains']);j=0
    estimating=c.nominal_bias_observer=='innovation_ema_v97';bias_estimate=np.zeros(12);previous_seen=np.zeros(12)
    for k in range(len(d['active'])):
        np.testing.assert_array_equal(d['previous_control'][k],previous_u);np.testing.assert_array_equal(d['previous_gain'][k],previous_gain)
        true_o=obs.copy();true_o[:,:2]+=k*c.robot.dt*obs[:,3:5]
        seen,so=numpy_observe(d['state'][k],true_o,mask,bx,bo,noise,ix[k],io[k])
        running=k==0 or d['status'][k-1]==0;handoff=False
        if ordered:
            handoff=running and leg<parent['waypoint_count']-1 and numpy_arrived(seen,goals[leg],noise,c)
            leg+=int(handoff);goal=goals[leg];points=routes[leg];rm=route_masks[leg]
            assert d['waypoint_index'][k]==leg and d['waypoint_handoff'][k]==handoff
        scheduled=running and (k%m['policy_config']['query_every_ticks']==0 or handoff)
        extra=False
        if m.get('failure_requery',False):
            extra=running and not scheduled and not numpy_arrived(seen,goal,noise,c) and not bool(d['held_feasible'][k])
            assert bool(d['failure_requery'][k])==extra
        assert d['requery'][k]==(scheduled or extra)
        if estimating:
            if k>0:bias_estimate=numpy_update_bias(bias_estimate,previous_seen,seen,previous_u,noise,c)
            np.testing.assert_allclose(d['nominal_bias_estimate'][k],bias_estimate,atol=2e-10,rtol=1e-10)
        if m.get('failure_requery',False):
            if extra or (running and d['qp_switch_fallback'][k]):
                from .quad3d_audit import verify_held
                verify_held(seen,goal,so,mask,previous_gain,points,rm,d['route_cursor_before'][k],noise,bias_estimate,c,d,k)
            if not d['requery'][k] or d['qp_switch_fallback'][k]:
                for held,actual in (('held_proposed','proposed'),('held_feasible','feasible'),('held_psi','psi'),('held_domain','domain'),('held_residual','residual')):
                    np.testing.assert_allclose(d[held][k],d[actual][k],atol=1e-10,rtol=1e-10,equal_nan=True)
        if d['requery'][k]:
            feature,nm=numpy_history_graph(seen,goal,so,mask,points,rm,d['route_cursor_before'][k],previous_u,previous_gain,noise,
                config=c,nominal_bias=bias_estimate if estimating else None)
            np.testing.assert_allclose(q['features'][j],feature,atol=2e-7,rtol=2e-7);np.testing.assert_array_equal(q['node_mask'][j],nm)
            controlled=numpy_obstacles(so,mask,noise)
            psi=[independent_obstacle_values(seen,np.zeros(4),controlled,mask,g,c)[0] for g in bank]
            domain=independent_envelope_values(seen,np.zeros(4),c)[0]
            np.testing.assert_allclose(q['candidate_psi'][j],psi,atol=1e-8,rtol=1e-10)
            np.testing.assert_allclose(q['candidate_domain'][j],domain,atol=1e-8,rtol=1e-10)
            if m.get('input_box_admission',False):
                from .quad3d_qp import audit_margin
                margin=audit_margin(seen,goal,so,mask,points,rm,d['route_cursor_before'][k],noise,
                    bias_estimate,bank,c)
                np.testing.assert_allclose(q['candidate_input_margin'][j],margin,atol=1e-8,rtol=1e-10)
            np.testing.assert_array_equal(d['network_gain'][k],q['network_gain'][j]);j+=1
        else:np.testing.assert_array_equal(d['network_gain'][k],previous_gain)
        psi=independent_obstacle_values(seen,np.zeros(4),numpy_obstacles(so,mask,noise),mask,d['network_gain'][k],c)[0]
        domain=independent_envelope_values(seen,np.zeros(4),c)[0]
        np.testing.assert_allclose(d['attempted_psi'][k],psi,atol=1e-8,rtol=1e-10);np.testing.assert_allclose(d['attempted_domain'][k],domain,atol=1e-8,rtol=1e-10)
        valid=bool(d['attempted_feasible'][k] and d['attempted_psi'][k]>=-c.qp_tolerance and d['attempted_domain'][k]>=-c.qp_tolerance)
        retry=not valid and np.any(d['network_gain'][k]!=previous_gain)
        assert retry==d['qp_switch_fallback'][k]
        np.testing.assert_array_equal(d['controller_gain'][k],previous_gain if retry else d['network_gain'][k])
        assert np.any(np.all(bank==d['controller_gain'][k],axis=-1))
        if not retry:np.testing.assert_array_equal(d['proposed'][k],d['attempted_control'][k])
        if d['active'][k]:previous_u=d['control'][k];previous_gain=d['controller_gain'][k]
        previous_seen=seen
    check_statistics(q,d['previous_gain'][query_ticks],m,fit)
    expected=dict(queries=len(query_ticks),maximum_cs=float(np.max(q['cs_score'])),
        learned_queries=int(np.sum(q['selected_index']>=0)),uncertainty_fallback_queries=int(q['uncertainty_fallback'].sum()),
        admission_fallback_queries=int(q['admission_fallback'].sum()),qp_switch_fallbacks=int(d['qp_switch_fallback'].sum()),
        applied_gain_changes=int(np.sum(d['active']&np.any(d['controller_gain']!=d['previous_gain'],axis=-1))))
    if m.get('failure_requery',False):expected['failure_requeries']=int(d['failure_requery'].sum())
    for k,v in expected.items():assert row[k]==v
    return sanitize(dict(physical,policy_features_decisions_memories_verified=True))
