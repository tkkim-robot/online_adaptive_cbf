"""Bicycle audit functions and shared contracts."""

import numpy as np

def reference_flow(_,state,control,config):
    theta,v=state[2:];a,b=control
    return np.array([v*np.cos(theta)-v*np.sin(theta)*b,
        v*np.sin(theta)+v*np.cos(theta)*b,v*b/config.robot.rear_axle_distance,a])

def reference_barrier(state,obstacles,config):
    """Complex-step-compatible direct scalar definition, without Lie formulas."""
    direction=np.stack((np.cos(state[...,2]),np.sin(state[...,2])),axis=-1)
    p=obstacles[...,:2]-state[...,:2];v=obstacles[...,3:5]-state[...,3,None]*direction
    distance=np.sqrt(np.sum(p*p,axis=-1));unit=p/distance[...,None]
    radial=np.sum(unit*v,axis=-1);lateral=unit[...,0]*v[...,1]-unit[...,1]*v[...,0]
    radius=(config.robot.radius+config.clearance_buffer+obstacles[...,2])*config.barrier_inflation
    root=np.sqrt(np.sum(p*p,axis=-1)-radius**2)
    speed=np.sqrt(np.sum(v*v,axis=-1)+config.relative_speed_epsilon**2)
    shape=np.sqrt(config.barrier_inflation**2-1)/radius
    return radial+.5*shape*root*lateral*lateral/speed+shape*root

def reference_rows(state,obstacles,mask,alpha,config):
    x=np.asarray(state,float);o=np.asarray(obstacles,float);mask=np.asarray(mask,bool)
    # Inactive disks may lie anywhere, including at the ego position. Put them
    # at a benign location only for derivative calculation; their rows are0<=1.
    o=o.copy();o[~mask,:2]=x[:2]+np.array([10.,10.])
    radius=(config.robot.radius+config.clearance_buffer+o[:,2])*config.barrier_inflation
    domain=np.sum((o[:,:2]-x[:2])**2,axis=1)-radius**2
    if np.any(domain[mask]<=0):raise ValueError('Barrier derivative outside its geometric domain')
    n=len(o);variables=np.concatenate((np.broadcast_to(x,(n,4)),o[:,:2]),axis=1)
    complex_variables=variables[:,None,:].astype(complex)+1j*1e-25*np.eye(6)[None]
    expanded=np.broadcast_to(o[:,None,:],(n,6,5)).astype(complex).copy()
    expanded[...,:2]=complex_variables[...,4:]
    grad=reference_barrier(complex_variables[...,:4],expanded,config).imag/1e-25
    drift=reference_flow(0,x,np.zeros(2),config)
    g=np.column_stack((reference_flow(0,x,np.array([1.,0.]),config)-drift,
        reference_flow(0,x,np.array([0.,1.]),config)-drift))
    h=reference_barrier(np.broadcast_to(x,(n,4)),o,config)
    a=-grad[:,:4]@g;b=grad[:,:4]@drift+np.sum(grad[:,4:]*o[:,3:5],axis=1)+alpha*h
    a=np.where(mask[:,None],a,0.);b=np.where(mask,b,1.)
    c=config.robot
    bounds=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    rhs=np.array([min(c.acceleration_max,(c.speed_max-x[3])/c.dt),
        -max(-c.acceleration_max,(c.speed_min-x[3])/c.dt),c.slip_max,c.slip_max])
    return np.vstack((a,bounds)),np.r_[b,rhs],h,domain

def polygon_qp(reference,a,b,weights,lower,upper):
    """Independent 2D polygon clipping, then projection onto every boundary edge."""
    reference=np.asarray(reference,float);a=np.asarray(a,float);b=np.asarray(b,float);weights=np.asarray(weights,float)
    x0,y0=lower;x1,y1=upper
    polygon=[np.array([x0,y0]),np.array([x1,y0]),np.array([x1,y1]),np.array([x0,y1])]
    for row,rhs in zip(a,b):
        clipped=[]
        for start,end in zip(polygon,polygon[1:]+polygon[:1]):
            fs,fe=row@start-rhs,row@end-rhs
            if fs<=1e-12:clipped.append(start)
            if (fs<0<fe) or (fe<0<fs):clipped.append(start+(end-start)*fs/(fs-fe))
        polygon=clipped
        if not polygon:return None
    if np.max(a@reference-b)<=1e-12:return reference.copy()
    candidates=[]
    for start,end in zip(polygon,polygon[1:]+polygon[:1]):
        delta=end-start;t=np.clip(np.dot(weights*(reference-start),delta)/max(np.dot(weights*delta,delta),1e-30),0.,1.)
        candidates.append(start+t*delta)
    objective=[.5*np.sum(weights*(u-reference)**2) for u in candidates]
    return candidates[int(np.argmin(objective))]


CONTRACT = dict(intervals=[1, 10], physical_every_ticks=1, event_trigger=False,
                diagnostic_only=True, refit_required_before_promotion=True)

def validate_cadence(phase, interval, source):
    if type(interval) is not int or interval < 1:
        raise ValueError('Cadence must be a positive integer')
    if phase == 'cadence_diagnosis':
        if source.get('cadence_development') != CONTRACT or interval not in CONTRACT['intervals']:
            raise ValueError('Explicit frozen cadence development contract required')
    elif interval != 1:
        raise ValueError('Changed cadence is only authorized for cadence_diagnosis')

def audit_schedule(data, manifest):
    """Reject gain changes or invented neural scores between scheduled calls."""
    n = len(data['active'])
    interval = manifest.get('query_every_ticks', 1)
    validate_cadence(manifest['phase'], interval, manifest)
    if manifest['phase'] != 'cadence_diagnosis':
        if 'neural_query' in data:
            raise ValueError('Unrecognized query-mask schema')
        return np.ones(n, bool)
    query = np.asarray(data['neural_query'])
    if query.dtype != np.dtype(bool):
        raise ValueError('Query marker must be boolean')
    np.testing.assert_array_equal(query, np.arange(n) % interval == 0)
    held = ~query
    np.testing.assert_array_equal(data['controller_gain'][held], data['previous_gain'][held])
    np.testing.assert_array_equal(data['selected_index'][held], np.full(held.sum(), -4))
    absent = ('uncertainty_fallback', 'accepted', 'cs_score', 'finite_member_cvar',
              'adverse_probability', 'predicted_progress', 'ranking_score',
              'features', 'node_mask', 'prediction_mean', 'prediction_variance',
              'prediction_event_logits')
    absent += tuple(k for k in data if k.startswith('incumbent_') or k == 'selection_eligible')
    for key in absent:
        if np.any(data[key][held] != 0):
            raise ValueError('A held tick claimed a neural prediction: '+key)
    return query


from statistics import NormalDist

import math


from scipy.special import expit

def check_selection(d,m,fit):
    """Recompute statistics in NumPy; replay decisions on checked live FP32 scores."""
    if 'neural_query' in d:
        from .bicycle_audit import audit_schedule
        query=audit_schedule(d,m);n=len(query)
        d={k:(v[query] if np.ndim(v)>0 and np.shape(v)[0]==n else v)
           for k,v in d.items() if k!='neural_query'}
    cfg=m['policy_config'];mu=d['prediction_mean'].astype(float);var=d['prediction_variance'].astype(float);logit=d['prediction_event_logits'].astype(float)
    n=len(d['active'])
    from .bicycle_gain_contract import validate_bank
    candidates=validate_bank(fit['candidates'])
    if mu.shape!=(n,4,len(candidates),2) or var.shape!=mu.shape or logit.shape!=mu.shape:raise ValueError('Wrong predictor shape')
    mean=mu[...,0].transpose(0,2,1);variance=var[...,0].transpose(0,2,1)
    pair=-.5*(np.log(2*np.pi*(variance[:,:,:,None]+variance[:,:,None,:]))+(mean[:,:,:,None]-mean[:,:,None,:])**2/(variance[:,:,:,None]+variance[:,:,None,:]))
    own=-.5*np.log(4*np.pi*variance)
    cs=np.maximum(np.mean(.5*(own[:,:,:,None]+own[:,:,None,:])-pair,axis=(-1,-2)),0.)
    z=NormalDist().inv_cdf(1-cfg['tail_mass']);coefficient=math.exp(-z*z/2)/math.sqrt(2*math.pi)/cfg['tail_mass']
    risk=np.max(mu[...,0]+np.sqrt(var[...,0])*coefficient,axis=1)
    e=fit['event_calibration'][1];prob=np.max(expit(logit[...,1]/e['temperature']+e['bias']),axis=1);progress=mu[...,1].mean(axis=1)
    bank=np.asarray(fit['candidates'],np.float32)[:,0]
    score=progress-cfg['gain_switch_penalty']*np.abs(np.log(bank[None,:]/d['previous_gain'][:,None]))
    if cfg.get('adverse_progress_penalty',0.):
        score=score-cfg['adverse_progress_penalty']*prob
    for name,value in [('cs_score',cs),('finite_member_cvar',risk),('adverse_probability',prob),('predicted_progress',progress),('ranking_score',score)]:
        np.testing.assert_allclose(d[name],value,atol=3e-5,rtol=3e-5,equal_nan=True,err_msg=name)
    finite=np.all(np.isfinite(mu)&np.isfinite(var)&(var>0)&np.isfinite(logit),axis=(1,3))&np.isfinite(d['cs_score'])&np.isfinite(d['finite_member_cvar'])&np.isfinite(d['adverse_probability'])
    limit=np.float32(np.inf if m['phase']=='gate_calibration' else m['cs_threshold'])
    accept=finite&(d['cs_score']<=limit)&(d['finite_member_cvar']<=np.float32(cfg['conditional_risk_limit']))&(d['adverse_probability']<=np.float32(cfg['adverse_probability_limit']))
    np.testing.assert_array_equal(d['accepted'],accept)
    any_=accept.any(axis=1);index=np.argmax(np.where(accept,d['ranking_score'],-np.inf),axis=1)
    gain=np.where(any_,bank[index],d['previous_gain']);index=np.where(any_,index,-1);fallback=~any_
    if cfg.get('incumbent_progress',False) and m['phase']!='gate_calibration':
        matches=bank[None,:]==d['previous_gain'][:,None]
        if np.any(matches.sum(1)!=1):raise ValueError('Unrecognized recorded incumbent')
        rows=np.arange(n);previous_score=d['ranking_score'][rows,matches.argmax(1)]
        selected_score=d['ranking_score'][rows,np.maximum(index,0)]
        needed=any_&(gain!=d['previous_gain'])&np.isfinite(previous_score)&np.isfinite(selected_score)&(selected_score<=previous_score)
        np.testing.assert_array_equal(d['incumbent_comparison_needed'],needed)
        np.testing.assert_array_equal(d['incumbent_previous_score'],previous_score)
        np.testing.assert_array_equal(d['incumbent_proposed_score'],selected_score)
        held=needed&d['incumbent_witness_checked']&d['incumbent_witness_feasible']
        np.testing.assert_array_equal(d['incumbent_progress_hold'],held)
        np.testing.assert_array_equal(d['selection_eligible'],accept&~held[:,None])
        gain=np.where(held,d['previous_gain'],gain);index=np.where(held,-3,index);fallback=np.where(held,False,fallback)
    if m['phase']=='gate_calibration':gain=np.full(n,cfg['initial_gain'],np.float32);index=np.full(n,-2);fallback=np.zeros(n,bool)
    np.testing.assert_array_equal(d['controller_gain'],gain);np.testing.assert_array_equal(d['selected_index'],index);np.testing.assert_array_equal(d['uncertainty_fallback'],fallback)
