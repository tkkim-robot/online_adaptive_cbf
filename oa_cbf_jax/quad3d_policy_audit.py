"""Independent observed features, learned decision, memory and physical checks."""
from pathlib import Path
from statistics import NormalDist
import math
import numpy as np
from scipy.special import expit
from .quad3d_learning_contract import read
from .quad3d_control import control_config
from .quad3d_features import numpy_history_graph
from .quad3d_observer import numpy_update_bias
from .quad3d_observation import unit_tape,numpy_observe,numpy_obstacles,numpy_arrived
from .quad3d_audit import independent_obstacle_values,independent_envelope_values
from .quad3d_observation_audit import audit_parent
from .quad2d_trajectory_gate import numpy_cs
from .dataset import sha256
from .cli import sanitize


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
                from .quad3d_failure_requery import verify_held
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
                from .quad3d_input_support import audit_margin
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
