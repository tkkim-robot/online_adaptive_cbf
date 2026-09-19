"""Independent graph, selection, history and physical audit for bicycle V70."""
from pathlib import Path
from statistics import NormalDist
import math
import numpy as np
from scipy.special import expit
from .bicycle_experiment import read,control_config
from .bicycle_observed_audit import audit_trace,check_graph
from .dataset import sha256
from .cli import sanitize


def check_selection(d,m,fit):
    """Recompute statistics in NumPy; replay decisions on checked live FP32 scores."""
    cfg=m['policy_config'];mu=d['prediction_mean'].astype(float);var=d['prediction_variance'].astype(float);logit=d['prediction_event_logits'].astype(float)
    n=len(d['active'])
    if mu.shape!=(n,4,8,2) or var.shape!=mu.shape or logit.shape!=mu.shape:raise ValueError('Wrong predictor shape')
    mean=mu[...,0].transpose(0,2,1);variance=var[...,0].transpose(0,2,1)
    pair=-.5*(np.log(2*np.pi*(variance[:,:,:,None]+variance[:,:,None,:]))+(mean[:,:,:,None]-mean[:,:,None,:])**2/(variance[:,:,:,None]+variance[:,:,None,:]))
    own=-.5*np.log(4*np.pi*variance)
    cs=np.maximum(np.mean(.5*(own[:,:,:,None]+own[:,:,None,:])-pair,axis=(-1,-2)),0.)
    z=NormalDist().inv_cdf(1-cfg['tail_mass']);coefficient=math.exp(-z*z/2)/math.sqrt(2*math.pi)/cfg['tail_mass']
    risk=np.max(mu[...,0]+np.sqrt(var[...,0])*coefficient,axis=1)
    e=fit['event_calibration'][1];prob=np.max(expit(logit[...,1]/e['temperature']+e['bias']),axis=1);progress=mu[...,1].mean(axis=1)
    bank=np.asarray(fit['candidates'],np.float32)[:,0]
    score=progress-cfg['gain_switch_penalty']*np.abs(np.log(bank[None,:]/d['previous_gain'][:,None]))
    for name,value in [('cs_score',cs),('finite_member_cvar',risk),('adverse_probability',prob),('predicted_progress',progress),('ranking_score',score)]:
        np.testing.assert_allclose(d[name],value,atol=3e-5,rtol=3e-5,equal_nan=True,err_msg=name)
    finite=np.all(np.isfinite(mu)&np.isfinite(var)&(var>0)&np.isfinite(logit),axis=(1,3))&np.isfinite(d['cs_score'])&np.isfinite(d['finite_member_cvar'])&np.isfinite(d['adverse_probability'])
    limit=np.float32(np.inf if m['phase']=='gate_calibration' else m['cs_threshold'])
    accept=finite&(d['cs_score']<=limit)&(d['finite_member_cvar']<=np.float32(cfg['conditional_risk_limit']))&(d['adverse_probability']<=np.float32(cfg['adverse_probability_limit']))
    np.testing.assert_array_equal(d['accepted'],accept)
    any_=accept.any(axis=1);index=np.argmax(np.where(accept,d['ranking_score'],-np.inf),axis=1)
    gain=np.where(any_,bank[index],d['previous_gain']);index=np.where(any_,index,-1);fallback=~any_
    if m['phase']=='gate_calibration':gain=np.full(n,cfg['initial_gain'],np.float32);index=np.full(n,-2);fallback=np.zeros(n,bool)
    np.testing.assert_array_equal(d['controller_gain'],gain);np.testing.assert_array_equal(d['selected_index'],index);np.testing.assert_array_equal(d['uncertainty_fallback'],fallback)


def audit_one(task):
    directory,entry,m,parent=task;root=Path(directory);path=root/entry['file'];fit=read(m['prediction_fit']);c=control_config(m['config'])
    if sha256(path)!=entry['sha256'] or parent['group_id']!=entry['group_id'] or parent['family']!=entry['family']:raise ValueError('Changed trajectory/parent binding')
    if parent['partition']!=m['phase'] or fit['weights_sha256']!=m['weights_sha256']:raise ValueError('Wrong policy phase/weights')
    with np.load(path) as z:d={k:z[k] for k in z.files}
    for k in ('initial','goal','obstacles','mask','first_x','first_o','bias_x','bias_o','noise'):
        np.testing.assert_array_equal(d[k],np.asarray(parent[k],dtype=d[k].dtype))
    for field,key in [('points','points'),('route_mask','mask')]:np.testing.assert_array_equal(d[field],np.asarray(parent['route'][key],dtype=d[field].dtype))
    if bool(d['ready'])!=(parent['route']['status']=='ready') or float(d['cursor'])!=0 or float(d['alpha'])!=m['policy_config']['initial_gain']:raise ValueError('Changed initial policy memory')
    np.testing.assert_array_equal(d['key'],[0,(parent['seed']+4)%(2**32)])
    n=len(d['active']);np.testing.assert_array_equal(d['global_tick'],np.arange(n));np.testing.assert_array_equal(d['innovation_x'][0],0);np.testing.assert_array_equal(d['innovation_o'][0],0)
    prevu=np.zeros(2);prevg=m['policy_config']['initial_gain'];cursor=0.
    for k in range(n):
        np.testing.assert_array_equal(d['previous_control'][k],prevu);np.testing.assert_array_equal(d['previous_gain'][k],prevg);np.testing.assert_array_equal(d['cursor_before'][k],cursor)
        check_graph(d['features'][k],d['node_mask'][k],(d['observed_state'][k],d['goal'],d['observed_obstacles'][k],d['mask'],d['points'],d['route_mask'],d['noise'],c,cursor,np.asarray(prevu,np.float32),prevg))
        if d['active'][k]:prevu=d['control'][k];prevg=d['controller_gain'][k];cursor=d['route_progress'][k]
    check_selection(d,m,fit);physical=audit_trace(d,c)
    from .bicycle_guidance import guidance_from_controller
    guidance=guidance_from_controller(fit['controller'])
    if m.get('controller',fit['controller'])!=fit['controller']:raise ValueError('Policy/label controller contract mismatch')
    if guidance.observation_margin:
        from .bicycle_margin_audit import audit_margin_trace
        if m.get('controller')!=fit['controller']:raise ValueError('Explicit margin controller binding required')
        physical['margin']=audit_margin_trace(d,c)
    if physical['feasible_qp_rejected']:raise ValueError('Independently feasible QP rejected')
    if entry['steps']!=physical['steps'] or entry['queries']!=n or entry['status_code']!=int(d['final_status']) or int(d['horizon'])!=m['steps']:raise ValueError('Changed index counts/status/horizon')
    expected=dict(maximum_cs=float(np.max(d['cs_score'])),uncertainty_fallback_queries=int(d['uncertainty_fallback'].sum()),learned_queries=int(np.sum(d['selected_index']>=0)),applied_gain_changes=int(np.sum(d['active']&(d['controller_gain']!=d['previous_gain']))))
    if any(entry[k]!=v for k,v in expected.items()):raise ValueError('Changed policy summary')
    return sanitize(dict(group_id=entry['group_id'],trace_sha256=entry['sha256'],**physical,policy_queries=n))
