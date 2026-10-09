"""Shared quad2d policy implementation."""

from dataclasses import dataclass, asdict

import json

from pathlib import Path

import time

import numpy as np

import jax

import jax.numpy as jnp

from .quad2d import integrate_quad2d

from .quad2d_control import FlightConfig, flight_control, flight_arrived, physical_envelope_violation

from .quad2d_features import flight_graph
from .quad2d_rollout import flight_branch, flight_sensor_model, INADMISSIBLE, PLANNER_FAILURE, STATE_BOUND
from .simulation import RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT

from .routing import physical_route_coordinate

from .dynamics import signed_clearance, swept_disk_clearance

from .models import predict_ensemble

from .inference import ResearchPredictor

from .uncertainty import cs_disagreement, worst_member_cvar

from .io import sha256

from .quad2d_guidance import predictive_flight_control

LEARNED, BACKUP, FIXED, REJECTED, HELD = range(5)

@dataclass(frozen=True)
class FlightPolicyConfig:
    mode: str = 'learned'
    interval: int = 4
    validation_horizon: int = 32
    shortlist: int = 4
    tail_mass: float = .01
    risk_threshold: float = 0.
    collision_probability_limit: float = .01
    failure_probability_limit: float = .25
    fixed_gain: tuple = (4., 4.)
    backup_gains: tuple = ((1., 1.), (2., 2.), (4., 4.), (8., 8.))
    stop_finished: bool = True
    def __post_init__(self):
        if self.mode not in ('learned', 'fixed','backup'): raise ValueError('Unknown flight policy')
        if not 1 <= self.interval <= self.validation_horizon or self.shortlist < 1: raise ValueError('Invalid predictive budget')
        if not 0 < self.tail_mass < 1: raise ValueError('Invalid tail mass')
        if not all(np.isfinite(g).all() and np.min(g)>0 for g in (self.fixed_gain,*self.backup_gains)): raise ValueError('Invalid gain')

@dataclass(frozen=True)
class ProgressAdmissionPolicyConfig(FlightPolicyConfig):
    progress_admission: str = 'quad2d_paired_progress_admission_v1'

    def __post_init__(self):
        super().__post_init__()
        if self.progress_admission != 'quad2d_paired_progress_admission_v1' or self.backup_gains:
            raise ValueError('Paired progress admission requires previous-gain-only fallback')

@dataclass(frozen=True)
class IncumbentProgressPolicyConfig(FlightPolicyConfig):
    incumbent_progress: str = 'feasible_previous_score_v1'

    def __post_init__(self):
        super().__post_init__()
        if self.incumbent_progress!='feasible_previous_score_v1' or self.backup_gains or self.mode=='fixed':
            raise ValueError('Incumbent comparison requires guided previous-gain-only control')

@dataclass(frozen=True)
class EnsembleRiskPolicyConfig(IncumbentProgressPolicyConfig):
    """Explicit predictive-mixture risk ablation, not finite-member ambiguity."""
    risk_aggregation: str = 'equal_gaussian_ensemble_mixture_v1'

    def __post_init__(self):
        super().__post_init__()
        if (self.risk_aggregation!='equal_gaussian_ensemble_mixture_v1' or self.mode!='learned'
                or self.tail_mass!=.01 or self.risk_threshold!=0.):
            raise ValueError('Explicit learned Gaussian ensemble risk contract required')

def ensemble_risk_contract():
    return dict(schema='equal_gaussian_ensemble_mixture_v1',weights='uniform_four_members',
        risk='upper_tail_CVaR_of_predictive_mixture',tail_mass=.01,threshold=0.,
        disagreement='original_member_Gaussian_CS',
        interpretation='Alternative predictive risk statistic; neither it nor maximum-member CVaR universally bounds the other. No physical safety guarantee.')

def flight_risk(means,variances,policy):
    if isinstance(policy,EnsembleRiskPolicyConfig):
        if means.shape[-1]!=4:raise ValueError('Exactly four predictive members required')
        from .uncertainty import gaussian_mixture_cvar
        return gaussian_mixture_cvar(means,variances,policy.tail_mass)['cvar']
    return worst_member_cvar(means,variances,policy.tail_mass)

@dataclass(frozen=True)
class FeasibilityTriggeredPolicyConfig(FlightPolicyConfig):
    query_trigger: str = 'previous_gain_infeasible_v1'

    def __post_init__(self):
        super().__post_init__()
        if self.query_trigger != 'previous_gain_infeasible_v1' or self.backup_gains or self.mode=='fixed':
            raise ValueError('Feasibility trigger requires guided previous-gain-only adaptation')

def flight_policy_from_contract(contract):
    if 'risk_aggregation' in contract:
        return EnsembleRiskPolicyConfig(**contract)
    if 'incumbent_progress' in contract:
        return IncumbentProgressPolicyConfig(**contract)
    if 'query_trigger' in contract:
        return FeasibilityTriggeredPolicyConfig(**contract)
    cls = ProgressAdmissionPolicyConfig if 'progress_admission' in contract else FlightPolicyConfig
    return cls(**contract)

def make_selector(model, norm, config, policy):
    if policy.mode == 'learned':
        scale=jnp.asarray(norm['target_scale'],jnp.float32);mean=jnp.asarray(norm['target_mean'],jnp.float32)
    def select(params,calibration,x,goal,obs,mask,points,rm,cursor,previous_u,previous_gain,noise,candidates):
        if policy.mode=='fixed':
            return jnp.asarray(policy.fixed_gain,jnp.float32),jnp.asarray(True),jnp.int32(FIXED),jnp.zeros(5,jnp.int32)
        pool=jnp.concatenate((candidates,previous_gain[None]))
        features,node_mask=flight_graph(x,goal,obs,mask,points,rm,cursor,previous_u,previous_gain,noise,config)
        out=predict_ensemble(model,params,features[None],node_mask[None],pool[None])
        mu=out['mean'][:,0]*scale+mean
        variance=jnp.exp(out['log_variance'][:,0])*scale**2*calibration['variance_scale']
        if getattr(model,'risk_components',1)==2:
            from .clipped_inference import statistics
            mixture=statistics(out,norm,calibration['variance_scale'],calibration['temperature'],calibration['bias'],policy.tail_mass)
            mu,variance=mixture['mean'][:,0],mixture['variance'][:,0]
            cs,risk=mixture['disagreement'][0],mixture['finite_member_cvar'][0]
        else:
            cs=cs_disagreement(mu[:,:,0].T[...,None],variance[:,:,0].T[...,None])
            risk=flight_risk(mu[:,:,0].T,variance[:,:,0].T,policy)
        event=jnp.max(jax.nn.sigmoid(out['event_logits'][:,0]/calibration['temperature']+calibration['bias']),axis=0)
        finite=jnp.all(jnp.isfinite(mu)&jnp.isfinite(variance),axis=(0,2))&jnp.all(jnp.isfinite(event),axis=-1)&jnp.isfinite(cs)&jnp.isfinite(risk)
        epistemic=finite&(cs<=calibration['cs_threshold'])
        risk_ok=epistemic&(risk<=policy.risk_threshold)
        admitted=risk_ok&(event[:,0]<=policy.collision_probability_limit)&(event[:,1]<=policy.failure_probability_limit)
        ranking=jnp.mean(mu[:,:,1],axis=0)-.01*jnp.sum(jnp.log(pool/previous_gain)**2,axis=-1)
        _,indices=jax.lax.top_k(jnp.where(admitted,ranking,-jnp.inf),policy.shortlist)
        primary=pool[indices]
        # Prediction starts at the observed state, with NO true offsets or future
        # innovation access. Zero noise means deterministic nominal prediction.
        def validate(gains):
            result=jax.vmap(lambda gain:flight_branch(x,goal,obs,mask,gain,points,rm,cursor,jnp.zeros(7,x.dtype),
                jax.random.PRNGKey(0),True,config,policy.validation_horizon)[0])(gains)
            valid=((result['status']==GOAL)|(result['status']==TIMEOUT))&(result['min_clearance']>0)
            score=result['route_progress']/(policy.validation_horizon*config.robot.dt*config.cruise_speed)
            score-=.01*jnp.sum(jnp.log(gains/previous_gain)**2,axis=-1)
            return valid,score
        valid,_=validate(primary);valid &= admitted[indices]
        index=jnp.argmax(jnp.where(valid,ranking[indices],-jnp.inf));primary_ok=jnp.any(valid)
        backup_pool=jnp.concatenate((previous_gain[None],jnp.asarray(policy.backup_gains,x.dtype).reshape(-1,2)))
        def backup(_):
            ok,score=validate(backup_pool);index=jnp.argmax(jnp.where(ok,score,-jnp.inf))
            return backup_pool[index],jnp.any(ok),jnp.sum(ok).astype(jnp.int32)
        bg,bok,bcount=jax.lax.cond(primary_ok,lambda _:(previous_gain,jnp.asarray(False),jnp.int32(0)),backup,None)
        accepted=primary_ok|bok
        gain=jnp.where(primary_ok,primary[index],jnp.where(bok,bg,previous_gain))
        source=jnp.where(primary_ok,LEARNED,jnp.where(bok,BACKUP,REJECTED)).astype(jnp.int32)
        stages=jnp.stack((jnp.sum(finite),jnp.sum(epistemic),jnp.sum(risk_ok),jnp.sum(admitted),bcount)).astype(jnp.int32)
        return gain,accepted,source,stages
    return select

def empty_query_statistics(params, candidates, risk_components=1):
    """Static placeholders for held ticks; only actual query records are scored."""
    members=jax.tree.leaves(params)[0].shape[0]; count=candidates.shape[0]+1
    result=dict(query_mean=jnp.zeros((members,count,2),jnp.float32),
        query_variance=jnp.zeros((members,count,2),jnp.float32),
        query_cs=jnp.zeros(count,jnp.float32),query_risk=jnp.zeros(count,jnp.float32),
        query_event=jnp.zeros((count,2),jnp.float32))
    if risk_components==2:
        for key in ('mean','variance','probability'):
            result['query_risk_component_'+key]=jnp.zeros((members,count,2),jnp.float32)
    return result

def make_guided_selector(model,norm,config,policy,guidance,record_query_statistics=False):
    """Neural shortlist with the already computed held-profile witness.

    Each candidate predicts the same 28 guidance profiles once. The selected
    current QP and its profile record are returned for immediate application;
    there is no outer rollout that recursively replans 28 profiles every tick.
    Fixed-set fallback use remains separate from learned acceptance.
    """
    if policy.mode=='learned':
        scale=jnp.asarray(norm['target_scale'],jnp.float32);mean=jnp.asarray(norm['target_mean'],jnp.float32)
    def select(params,calibration,x,goal,obs,mask,points,rm,cursor,previous_u,previous_gain,noise,candidates,forecast_obstacles=None,observer_memory=None):
        def validate(gains):
            results,infos=jax.vmap(lambda gain:predictive_flight_control(x,goal,obs,mask,gain,points,rm,cursor,config,guidance,noise,forecast_obstacles))(gains)
            qp,h,psi,domain,_,_,_=results
            ok=infos['approved']&qp.feasible&(h>=-config.robot.qp_tolerance)&(psi>=-config.robot.qp_tolerance)&(domain>=-config.robot.qp_tolerance)
            return results,infos,ok
        compare_previous=isinstance(policy,IncumbentProgressPolicyConfig) and policy.mode=='learned'
        if compare_previous:
            previous_result,previous_info,previous_ok=validate(previous_gain[None])
        def backup(_):
            if compare_previous:
                return (previous_gain,previous_ok[0],jnp.int32(BACKUP),previous_ok[0].astype(jnp.int32),
                    jax.tree.map(lambda a:a[0],previous_result),jax.tree.map(lambda a:a[0],previous_info))
            # An explicitly empty alternative bank retains only the last
            # accepted gain. It still needs the same current-observation witness;
            # an invalid held gain stops the episode, never bypasses safety.
            gains=jnp.concatenate((previous_gain[None],jnp.asarray(policy.backup_gains,x.dtype).reshape(-1,2)))
            result,info,ok=validate(gains)
            score=info['score']-.01*jnp.sum(jnp.log(gains/previous_gain)**2,axis=-1)
            index=jnp.argmax(jnp.where(ok,score,-jnp.inf))
            return gains[index],jnp.any(ok),jnp.int32(BACKUP),jnp.sum(ok).astype(jnp.int32),jax.tree.map(lambda a:a[index],result),jax.tree.map(lambda a:a[index],info)
        if policy.mode=='backup':
            gain,accepted,source,count,result,info=backup(None)
            return gain,accepted,jnp.where(accepted,source,REJECTED).astype(jnp.int32),jnp.array([0,0,0,0,count],jnp.int32),result,info
        pool=jnp.concatenate((candidates,previous_gain[None]))
        from .quad2d_guidance import ObservedMotionGuidanceConfig
        if isinstance(guidance,ObservedMotionGuidanceConfig):
            if observer_memory is None:raise ValueError('Pre-query observed history required for graph50')
            from .quad2d_history import graph
            features,node_mask=graph(x,goal,obs,mask,points,rm,cursor,previous_u,previous_gain,noise,observer_memory,config,guidance)
        else:
            features,node_mask=flight_graph(x,goal,obs,mask,points,rm,cursor,previous_u,previous_gain,noise,config)
        out=predict_ensemble(model,params,features[None],node_mask[None],pool[None])
        mu=out['mean'][:,0]*scale+mean
        variance=jnp.exp(out['log_variance'][:,0])*scale**2*calibration['variance_scale']
        if getattr(model,'risk_components',1)==2:
            from .clipped_inference import statistics
            mixture=statistics(out,norm,calibration['variance_scale'],calibration['temperature'],calibration['bias'],policy.tail_mass)
            mu,variance=mixture['mean'][:,0],mixture['variance'][:,0]
            cs,risk=mixture['disagreement'][0],mixture['finite_member_cvar'][0]
        else:
            cs=cs_disagreement(mu[:,:,0].T[...,None],variance[:,:,0].T[...,None])
            risk=flight_risk(mu[:,:,0].T,variance[:,:,0].T,policy)
        event=jnp.max(jax.nn.sigmoid(out['event_logits'][:,0]/calibration['temperature']+calibration['bias']),axis=0)
        finite=jnp.all(jnp.isfinite(mu)&jnp.isfinite(variance),axis=(0,2))&jnp.all(jnp.isfinite(event),axis=-1)&jnp.isfinite(cs)&jnp.isfinite(risk)
        epistemic=finite&(cs<=calibration['cs_threshold']);risk_ok=epistemic&(risk<=policy.risk_threshold)
        admitted=risk_ok&(event[:,0]<=policy.collision_probability_limit)&(event[:,1]<=policy.failure_probability_limit)
        if isinstance(policy, ProgressAdmissionPolicyConfig):
            from .quad2d_training import admission
            admitted &= admission(mu[...,1],variance[...,1],calibration['progress_delta_quantile'])[0]
        ranking=jnp.mean(mu[:,:,1],axis=0)-.01*jnp.sum(jnp.log(pool/previous_gain)**2,axis=-1)
        _,indices=jax.lax.top_k(jnp.where(admitted,ranking,-jnp.inf),policy.shortlist)
        primary=pool[indices]
        results,infos,valid=validate(primary);valid &= admitted[indices]
        if compare_previous:
            # Original neural safety gates remain necessary. A physically valid
            # incumbent also stays in the utility comparison when those gates
            # exclude it. Failed incumbents still allow lower-scoring recovery.
            valid &= (~previous_ok[0]) | (ranking[indices]>ranking[-1])
        index=jnp.argmax(jnp.where(valid,ranking[indices],-jnp.inf));primary_ok=jnp.any(valid)
        primary_result=jax.tree.map(lambda a:a[index],results);primary_info=jax.tree.map(lambda a:a[index],infos)
        gain,accepted,source,bcount,result,info=jax.lax.cond(primary_ok,
            lambda _:(primary[index],jnp.asarray(True),jnp.int32(LEARNED),jnp.int32(0),primary_result,primary_info),backup,None)
        source=jnp.where(accepted,source,REJECTED).astype(jnp.int32)
        if compare_previous:
            info=dict(info,incumbent_previous_feasible=previous_ok[0],
                incumbent_previous_score=ranking[-1],
                incumbent_selected_score=jnp.where(primary_ok,ranking[indices[index]],ranking[-1]))
        stages=jnp.stack((jnp.sum(finite),jnp.sum(epistemic),jnp.sum(risk_ok),jnp.sum(admitted),bcount)).astype(jnp.int32)
        if record_query_statistics:
            info=dict(info,query_mean=mu,query_variance=variance,query_cs=cs,query_risk=risk,query_event=event)
            if getattr(model,'risk_components',1)==2:
                for key in ('risk_component_mean','risk_component_variance','risk_component_probability'):
                    info['query_'+key]=mixture[key][:,0]
        return gain,accepted,source,stages,result,info
    return select

def guided_contract(metadata,calibration,config,guidance):
    expected=None if guidance is None else json.loads(json.dumps(asdict(guidance)))
    controller=metadata['controller']
    if controller.get('predictive_guidance')!=expected:
        raise ValueError('Predictive guidance requires matched new labels and calibration')
    if guidance is not None:
        from .quad2d_guidance import ObservedMotionGuidanceConfig, ForecastMotionGuidanceConfig
        if isinstance(guidance,ObservedMotionGuidanceConfig):
            from .quad2d_history import SCHEMA, GRAPH_SCHEMA
            if (metadata.get('dataset_schema')!=SCHEMA or calibration.get('dataset_schema')!=SCHEMA
                    or metadata.get('graph_features')!=50 or calibration.get('graph_features')!=50
                    or controller.get('graph_schema')!=GRAPH_SCHEMA
                    or controller.get('observation_context')!='raw sensor plus two-anchor causal velocity history; no true state/bias in graph'
                    or controller.get('initial_gain')!=[4.,4.]):
                raise ValueError('Matched history-aware graph50 observation/initial-gain contract required')
        elif isinstance(guidance,ForecastMotionGuidanceConfig):
            raise ValueError('Forecast-only learned mode has no matched history-aware label contract')
        elif metadata.get('dataset_schema')!='oa_cbf_quad2d_guided_hurdle_v1' or calibration.get('dataset_schema')!=metadata['dataset_schema'] or controller.get('graph_schema')!='quad2d_route_observed_40_v1' or controller.get('observation_context')!='initial_or_visited_40' or controller.get('initial_gain')!=[4.,4.]:
            raise ValueError('Matched guided observation/initial-gain contract required')
        from .quad2d_guidance import TerminalGuidanceConfig
        if isinstance(guidance,TerminalGuidanceConfig):
            from .quad2d_control import contract
            target=controller.get('performance_target',{})
            if target.get('kind') not in ('route','terminal_task') or target!=contract(target['kind']) or metadata.get('targets',[None,None])[1]!=target['target']:
                raise ValueError('Explicit matched terminal performance target required')

def observed_waypoint_arrival(x,goal,noise,config=FlightConfig()):
    """Conservative pre-action arrival using only sensed state/error ranges."""
    return ((jnp.linalg.norm(x[:2]-goal)+jnp.sqrt(2.)*1.15*noise[0]<=config.goal_tolerance)
        &(jnp.linalg.norm(x[3:5])+jnp.sqrt(2.)*1.15*noise[2]<=config.terminal_speed)
        &(jnp.abs(x[2])+1.15*noise[1]<=config.terminal_pitch)
        &(jnp.abs(x[5])+1.15*noise[3]<=config.terminal_pitch_rate))

def make_episode(model,norm,config,policy,steps,guidance=None,ordered_waypoints=False,record_query_statistics=False,diagnostic_nearest_obstacles=None,return_stepper=False):
    if diagnostic_nearest_obstacles is not None:
        from .quad2d_static_inputs import neighborhood_contract, neighborhood_mask
        neighborhood_contract(diagnostic_nearest_obstacles)
        from .quad2d_guidance import TerminalGuidanceConfig
        if type(guidance) is not TerminalGuidanceConfig or ordered_waypoints:
            raise ValueError("Neighborhood diagnostic requires unchanged single-goal terminal guidance")
    if isinstance(policy,IncumbentProgressPolicyConfig):
        from .quad2d_guidance import TerminalGuidanceConfig
        if type(guidance) is not TerminalGuidanceConfig or ordered_waypoints:
            raise ValueError('Incumbent comparison requires raw single-goal terminal guidance')
    triggered=isinstance(policy,FeasibilityTriggeredPolicyConfig)
    if triggered:
        from .quad2d_guidance import TerminalGuidanceConfig
        if type(guidance) is not TerminalGuidanceConfig or ordered_waypoints:
            raise ValueError('Feasibility trigger requires single-goal raw terminal guidance')
    select=make_selector(model,norm,config,policy);c=config.robot
    from .quad2d_guidance import ForecastMotionGuidanceConfig
    from . import motion_observer, quad2d_guidance as quad2d_motion
    tracked=isinstance(guidance,ForecastMotionGuidanceConfig)
    guided_select=make_guided_selector(model,norm,config,policy,guidance,record_query_statistics) if guidance is not None and policy.mode in ('learned','backup') else None
    def episode(params,calibration,candidates,observed,goal,obstacles,mask,points,route_mask,noise,key,ready,waypoint_count=None):
        initial,truth_obs,xb,ob,xs,os,innovations=flight_sensor_model(observed,obstacles,mask,noise,key,steps,config.stationary_obstacles)
        minimum=jnp.min(signed_clearance(initial[:2],truth_obs,mask,c.radius))
        initial_goal=goal[0] if ordered_waypoints else goal
        initial_arrival=flight_arrived(initial,initial_goal,config)
        if ordered_waypoints:initial_arrival &= waypoint_count==1
        status=jnp.where(initial_arrival,GOAL,RUNNING)
        status=jnp.where(physical_envelope_violation(initial,config)>c.qp_tolerance,STATE_BOUND,status)
        status=jnp.where(minimum<=0,COLLISION,status);status=jnp.where(ready[0] if ordered_waypoints else ready,status,PLANNER_FAILURE)
        def tick(carry,inputs):
            x,status,count,minimum,cursor,worst,previous_u,gain=carry[:8];k,innovation=inputs;active=status==RUNNING
            sensed=x-xb+.15*xs*innovation[:6]
            seen=truth_obs.at[:,:2].set(truth_obs[:,:2]+k*c.dt*truth_obs[:,3:5])-ob+.15*os*innovation[6:].reshape(obstacles.shape)
            controller_mask=mask if diagnostic_nearest_obstacles is None else neighborhood_mask(sensed[:2],seen,mask,diagnostic_nearest_obstacles)
            forecast=None;forecast_info={}
            if tracked:
                memory,forecast,velocity_bound,lag,inconsistent=quad2d_motion.update(carry[-1],seen,mask,noise,c.dt,guidance.motion_window)
                forecast_info=dict(forecast_obstacles=forecast,forecast_velocity_bound=velocity_bound,
                    forecast_lag=lag,forecast_inconsistent=inconsistent)
            if ordered_waypoints:
                leg=carry[8];handoff=active&observed_waypoint_arrival(sensed,goal[leg],noise,config)&(leg<waypoint_count-1)
                leg+=handoff.astype(jnp.int32);cursor=jnp.where(handoff,jnp.float32(0),cursor)
                task_goal=goal[leg];task_points=points[leg];task_mask=route_mask[leg]
                status=jnp.where(active&~ready[leg],PLANNER_FAILURE,status);active=status==RUNNING
                mission_info=dict(waypoint_index=leg,waypoint_handoff=handoff,mission_goal=task_goal,
                    mission_previous_control=previous_u,mission_previous_gain=gain,mission_route_cursor_before=cursor)
            else:task_goal=goal;task_points=points;task_mask=route_mask;handoff=jnp.asarray(False)
            if guided_select is not None:
                def query(_):
                    return (*guided_select(params,calibration,sensed,task_goal,seen,controller_mask,task_points,task_mask,cursor,previous_u,gain,noise,candidates,forecast,carry[-1] if tracked else None),jnp.asarray(True))
                def hold(_):
                    result,info=predictive_flight_control(sensed,task_goal,seen,controller_mask,gain,task_points,task_mask,cursor,config,guidance,noise,forecast)
                    if isinstance(policy,IncumbentProgressPolicyConfig) and policy.mode=='learned':
                        info=dict(info,incumbent_previous_feasible=jnp.asarray(False),
                            incumbent_previous_score=jnp.float32(0),incumbent_selected_score=jnp.float32(0))
                    if record_query_statistics:
                        info=dict(info,**empty_query_statistics(params,candidates,getattr(model,'risk_components',1)))
                    qp,h,psi,domain,_,_,_=result
                    ok=info['approved']&qp.feasible&(h>=-c.qp_tolerance)&(psi>=-c.qp_tolerance)&(domain>=-c.qp_tolerance)
                    chosen=jax.lax.cond(ok,lambda _:(gain,jnp.asarray(True),jnp.int32(HELD),jnp.zeros(5,jnp.int32),result,info,jnp.asarray(False)),query,None)
                    if triggered:
                        chosen=(*chosen[:5],dict(chosen[5],trigger_previous_feasible=ok),chosen[6])
                    return chosen
                if triggered:
                    selected,selection_ok,source,stages,result,info,due=hold(None)
                else:
                    selected,selection_ok,source,stages,result,info,due=jax.lax.cond((k%policy.interval==0)|handoff,query,hold,None)
                due &= active
                qp,h,psi,domain,proposed,remaining,target=result
                guidance_info={'guidance_'+key:value for key,value in info.items()}
            else:
                held,h,psi,domain,_,_,_=flight_control(sensed,task_goal,seen,controller_mask,gain,task_points,task_mask,cursor,config)
                due=active&((k%policy.interval==0)|handoff|~held.feasible|(h<-c.qp_tolerance)|(psi<-c.qp_tolerance)|(domain<-c.qp_tolerance))
                selected,selection_ok,source,stages=jax.lax.cond(due,
                    lambda _:select(params,calibration,sensed,task_goal,seen,controller_mask,task_points,task_mask,cursor,previous_u,gain,noise,candidates),
                    lambda _:(gain,jnp.asarray(True),jnp.int32(HELD),jnp.zeros(5,jnp.int32)),None)
                if guidance is None:
                    qp,h,psi,domain,proposed,remaining,target=flight_control(sensed,task_goal,seen,controller_mask,selected,task_points,task_mask,cursor,config)
                    guidance_info={}
                else:
                    (qp,h,psi,domain,proposed,remaining,target),info=predictive_flight_control(sensed,task_goal,seen,controller_mask,selected,task_points,task_mask,cursor,config,guidance,noise,forecast)
                    selection_ok &= info['approved']
                    source=jnp.where(active&~info['approved'],REJECTED,source)
                    due |= active&~info['approved']
                    guidance_info={'guidance_'+key:value for key,value in info.items()}
            admissible=(h>=-c.qp_tolerance)&(psi>=-c.qp_tolerance)&(domain>=-c.qp_tolerance)
            accepted=active&selection_ok&qp.feasible&admissible
            u=jnp.where(accepted,qp.control,jnp.zeros(2,observed.dtype))
            y,sub=integrate_quad2d(x,u,c);starts=jnp.concatenate((x[None],sub[:-1]))
            times=k*c.dt+jnp.arange(c.integration_substeps)*c.dt/c.integration_substeps
            clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,truth_obs,mask,c.radius,t,t+c.dt/c.integration_substeps))(starts,sub,times))
            bound=jnp.max(jax.vmap(lambda state:physical_envelope_violation(state,config))(jnp.concatenate((x[None],sub))))
            status=jnp.where(active&~qp.feasible,INFEASIBLE,status)
            status=jnp.where(active&~admissible,INADMISSIBLE,status)
            status=jnp.where(active&~selection_ok,6,status)  # predictive policy rejection, not proven QP infeasibility
            arrived=flight_arrived(y,task_goal,config)
            if ordered_waypoints:arrived &= leg==waypoint_count-1
            status=jnp.where(accepted&arrived,GOAL,status)
            status=jnp.where(accepted&(bound>c.qp_tolerance),STATE_BOUND,status)
            status=jnp.where(accepted&(clear<=0),COLLISION,status)
            x=jnp.where(accepted,y,x);cursor=jnp.where(accepted,proposed,cursor);count+=accepted.astype(jnp.int32)
            minimum=jnp.minimum(minimum,jnp.where(accepted,clear,jnp.inf));worst=jnp.maximum(worst,jnp.where(accepted,qp.max_violation,-jnp.inf))
            gain=jnp.where(accepted,selected,gain);previous_u=jnp.where(accepted,u,previous_u)
            trace=dict(state=x,control=u,active=accepted,status=status,observed_state=sensed,observed_obstacles=seen,
                clearance=jnp.where(accepted,clear,jnp.nan),state_bound_violation=jnp.where(accepted,bound,jnp.nan),
                qp_violation=jnp.where(accepted,qp.max_violation,jnp.nan),h=h,psi1=psi,envelope_domain=domain,
                route_progress=cursor,route_target=target,route_remaining=remaining,gain=selected,source=source,requery=due,stages=stages,**guidance_info,**forecast_info)
            if diagnostic_nearest_obstacles is not None:
                trace["controller_obstacle_mask"]=controller_mask
            next_carry=(x,status,count,minimum,cursor,worst,previous_u,gain)
            if ordered_waypoints:
                trace.update(**mission_info,waypoints_visited=leg+(status==GOAL).astype(jnp.int32))
                next_carry+=(leg,)
            if tracked:next_carry+=(memory,)
            return next_carry,trace
        carry=(initial,status,jnp.int32(0),minimum,jnp.float32(0),jnp.float32(-jnp.inf),
            jnp.full(2,c.mass*c.gravity/2,jnp.float32),jnp.asarray((4.,4.) if guided_select is not None else (2.,2.) if policy.mode=='learned' else policy.fixed_gain,jnp.float32))
        if ordered_waypoints:carry+=(jnp.int32(0),)
        if tracked:carry+=(motion_observer.initialize(obstacles),)
        if return_stepper:
            return tick,carry,innovations,dict(initial_state=initial,obstacles=truth_obs)
        if policy.stop_finished:
            # Batched while uses a shared "any scene still running" condition.
            # Finished scenes keep their exact prefix; no horizon padding work
            # is required after the last scene stops. At least one tick records
            # initial-terminal observations without applying a command.
            template=jax.eval_shape(tick,carry,(jnp.int32(0),innovations[0]))[1]
            storage=jax.tree.map(lambda a:jnp.zeros((steps,*a.shape),a.dtype),template)
            def condition(state):
                k,c,_=state
                return (k<steps)&((k==0)|(c[1]==RUNNING))
            def body(state):
                k,c,history=state;c,record=tick(c,(k,innovations[k]))
                history=jax.tree.map(lambda a,v:jax.lax.dynamic_update_index_in_dim(a,v,k,0),history,record)
                return k+1,c,history
            _,last,trace=jax.lax.while_loop(condition,body,(jnp.int32(0),carry,storage))
        else:
            last,trace=jax.lax.scan(tick,carry,(jnp.arange(steps),innovations))
        final,status,count,minimum,cursor,worst,_,_=last[:8]
        status=jnp.where(status==RUNNING,TIMEOUT,status)
        if ordered_waypoints:
            leg=last[8];progress=physical_route_coordinate(final[:2],points[leg],route_mask[leg],cursor)
        else:progress=physical_route_coordinate(final[:2],points,route_mask,cursor)-physical_route_coordinate(initial[:2],points,route_mask,jnp.float32(0))
        summary=dict(final_state=final,status=status,steps=count,min_clearance=minimum,route_progress=progress,worst_qp_violation=worst)
        if ordered_waypoints:summary.update(waypoint_index=leg,waypoints_visited=leg+(status==GOAL).astype(jnp.int32))
        return summary,trace,dict(initial_state=initial,obstacles=truth_obs)
    return episode

class FlightPolicy:
    """Strict model/calibration/config identity, explicit AOT batch signatures."""
    def __init__(self,bundle=None,calibration=None,config=FlightConfig(),policy=FlightPolicyConfig(),guidance=None,record_query_statistics=False,clipped_mixture_pilot=False,diagnostic_nearest_obstacles=None,diagnostic_cruise_speed=None):
        if diagnostic_cruise_speed is not None and (diagnostic_nearest_obstacles is not None or clipped_mixture_pilot):
            raise ValueError('Speed diagnosis cannot combine another diagnostic treatment')
        self.diagnostic_nearest_obstacles=diagnostic_nearest_obstacles
        self.calibration_coverage_valid=diagnostic_nearest_obstacles is None and diagnostic_cruise_speed is None
        if diagnostic_nearest_obstacles is not None:
            from .quad2d_static_inputs import neighborhood_contract
            self.neighborhood_contract=neighborhood_contract(diagnostic_nearest_obstacles)
        self.config=config;self.policy=policy;self.metadata={};self.params={};self.calibration={};self.model=None;self.norm=None;self.compiled={}
        self.guidance=guidance;self.fresh_calibration_candidates=None
        self.record_query_statistics=record_query_statistics
        if isinstance(policy,IncumbentProgressPolicyConfig) and policy.mode=='learned' and not record_query_statistics:
            raise ValueError('Incumbent comparison requires recorded live query statistics')
        if isinstance(policy, ProgressAdmissionPolicyConfig) and guidance is None:
            raise ValueError('Paired progress admission requires matched guided control')
        if record_query_statistics and (guidance is None or policy.mode!='learned'):
            raise ValueError('Query statistics require a learned guided policy')
        from .quad2d_guidance import ForecastMotionGuidanceConfig, ObservedMotionGuidanceConfig
        if isinstance(guidance,ForecastMotionGuidanceConfig) and policy.mode=='learned' and (not isinstance(guidance,ObservedMotionGuidanceConfig) or bundle is None or calibration is None):
            raise ValueError('Forecast pilot requires new history-aware labels and calibration before learned use')
        if policy.mode=='backup' and guidance is None:raise ValueError('Flight backup-only ablation requires guided controller')
        if guidance is not None and policy.mode=='learned' and (bundle is None or calibration is None):
            raise ValueError('Predictive guidance requires matched new labels and calibration before learned use')
        if policy.mode=='learned':
            info=json.loads(Path(calibration).read_text())
            if isinstance(policy,EnsembleRiskPolicyConfig):
                if (clipped_mixture_pilot or diagnostic_nearest_obstacles is not None or diagnostic_cruise_speed is not None
                        or guidance is None or not record_query_statistics
                        or info.get('ensemble_risk_contract')!=ensemble_risk_contract()
                        or policy.tail_mass!=.01 or policy.risk_threshold!=0.):
                    raise ValueError('Matched explicit mixture-risk calibration and unchanged thresholds required')
            elif 'ensemble_risk_contract' in info:
                raise ValueError('Predictive mixture calibration requires its explicit policy')
            if clipped_mixture_pilot:
                from .clipped_inference import ClippedRiskPredictor
                from .clipped_inference import validate_calibration
                if not record_query_statistics or guidance is None:
                    raise ValueError('Clipped-mixture pilot requires guided live component statistics')
                predictor=ClippedRiskPredictor(bundle,allow_uncalibrated=True)
                validate_calibration(info,predictor.metadata,bundle)
            else:
                predictor=ResearchPredictor(bundle,allow_uncalibrated=True)
            metadata=predictor.metadata
            if info.get('bundle_manifest_sha256') and info['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json'):
                raise ValueError('Flight predictive calibration belongs to a different numerical bundle')
            if isinstance(policy,EnsembleRiskPolicyConfig) and getattr(predictor.model,'risk_components',1)!=1:
                raise ValueError('Ordinary Gaussian members required for ensemble mixture risk')
            if predictor.model.config.encoder=='nearest_fc':
                from .nearest_fc import validate_fit
                validate_fit(info,bundle)
            features=50 if isinstance(guidance,ObservedMotionGuidanceConfig) else 40
            if info.get('dynamics')!='Quad2D' or metadata.get('graph_features')!=features or info['weights_sha256']!=metadata['weights_sha256']:
                raise ValueError('Matched flight calibration/bundle required')
            from .quad2d_control import normalize_flight_contract
            if normalize_flight_contract(info['robot'])!=asdict(config) or normalize_flight_contract(metadata['controller']['config'])!=asdict(config) or info['controller']!=metadata['controller']:
                raise ValueError('Flight controller changed since labels/calibration')
            if info['dataset_manifest_sha256']!=metadata['dataset_manifest_sha256'] or info['targets']!=metadata['targets']:
                raise ValueError('Flight target/data mismatch')
            guided_contract(metadata,info,config,guidance)
            if (info.get('schema')=='oa_cbf_fresh_flight_predictive_calibration_v61'
                    or info.get('predictive_calibration_schema')=='oa_cbf_fresh_flight_predictive_calibration_v61'):
                if info.get('event_statistic')!='maximum_member_probability' or info.get('delayed_calibration_valid') is not False:
                    raise ValueError('Invalid fresh instantaneous calibration semantics')
                self.fresh_calibration_candidates=np.asarray(info['candidates'],np.float32)
                if self.fresh_calibration_candidates.shape!=(32,2) or not np.isfinite(self.fresh_calibration_candidates).all():
                    raise ValueError('Invalid fresh calibration candidate bank')
            if guidance is not None and policy.validation_horizon!=guidance.horizon:
                raise ValueError('Guided validation must record the actual guidance witness horizon')
            from .quad2d_trajectory_gate import SCHEMA as TRAJECTORY_GATE_SCHEMA, validate_runtime
            self.trajectory_gate_contract = validate_runtime(info,policy,guidance) if info.get('schema')==TRAJECTORY_GATE_SCHEMA else None
            self.model=predictor.model;self.params=predictor.params;self.norm=metadata['normalization'];self.metadata=metadata
            self.calibration={k:jnp.asarray(v,jnp.float32) for k,v in dict(variance_scale=info['variance_scale'],
                temperature=[e['temperature'] for e in info['event_calibration']],bias=[e['bias'] for e in info['event_calibration']],cs_threshold=info['cs_gate']['threshold']).items()}
            if bool(info.get('gain_improvement')) != isinstance(policy, ProgressAdmissionPolicyConfig):
                raise ValueError('Progress calibration and admission policy must match')
            if isinstance(policy, ProgressAdmissionPolicyConfig):
                from .quad2d_training import validate
                switch = validate(info)
                self.calibration['progress_delta_quantile'] = jnp.asarray(switch['threshold'],jnp.float32)
            self.calibration_sha256=sha256(calibration)
        if diagnostic_cruise_speed is not None:
            from .quad2d_guidance import TerminalGuidanceConfig
            from .quad2d_control import speed_contract
            if type(guidance) is not TerminalGuidanceConfig or not isinstance(policy,IncumbentProgressPolicyConfig):
                raise ValueError('Speed diagnosis requires the original guided incumbent policy')
            # All model/calibration identity checks above use the original
            # contract. The opt-in runtime change carries no coverage claim.
            self.config,self.nominal_speed_contract=speed_contract(config,diagnostic_cruise_speed)
    def warm(self,candidates,batch=16,capacity=64,route_capacity=64,steps=1600,waypoint_capacity=None):
        candidates=np.asarray(candidates,np.float32)
        if self.fresh_calibration_candidates is not None and not np.array_equal(candidates,self.fresh_calibration_candidates):
            raise ValueError('Fresh calibration candidate bank changed')
        if candidates.ndim!=2 or candidates.shape[1]!=2 or len(candidates)<self.policy.shortlist or not np.isfinite(candidates).all():raise ValueError('Invalid candidate bank')
        if self.metadata:
            domain=self.metadata['gain_domain']
            if candidates.min()<domain['lower'] or candidates.max()>domain['upper']:raise ValueError('Out-of-domain gain')
        ordered=waypoint_capacity is not None
        if getattr(self,'trajectory_gate_contract',None) is not None:
            contract=self.trajectory_gate_contract
            if ordered or capacity!=64 or route_capacity!=64 or steps!=contract['steps'] or not np.array_equal(candidates,np.asarray(contract['candidates'],np.float32)):
                raise ValueError('Trajectory gate horizon, mission or candidate pool changed')
        if ordered and (isinstance(waypoint_capacity,bool) or not isinstance(waypoint_capacity,int) or waypoint_capacity<1):raise ValueError('Invalid waypoint capacity')
        prefix=(batch,waypoint_capacity) if ordered else (batch,)
        fn=jax.jit(jax.vmap(make_episode(self.model,self.norm,self.config,self.policy,steps,self.guidance,ordered,self.record_query_statistics,self.diagnostic_nearest_obstacles),in_axes=(None,None,None,0,0,0,0,0,0,0,0,0)+((0,) if ordered else ())))
        args=(self.params,self.calibration,jnp.asarray(candidates),jnp.zeros((batch,6),jnp.float32),jnp.ones((*prefix,2),jnp.float32),
            jnp.zeros((batch,capacity,5),jnp.float32),jnp.zeros((batch,capacity),bool),jnp.zeros((*prefix,route_capacity,2),jnp.float32),
            jnp.ones((*prefix,route_capacity),bool),jnp.zeros((batch,7),jnp.float32),jax.random.split(jax.random.PRNGKey(0),batch),jnp.zeros(prefix,bool))
        if ordered:args+=(jnp.ones(batch,jnp.int32),)
        tick=time.perf_counter();executable=fn.lower(*args).compile();jax.block_until_ready(executable(*args))
        self.compiled[(batch,capacity,route_capacity,steps)+((waypoint_capacity,) if ordered else ())]=(executable,jnp.asarray(candidates))
        return time.perf_counter()-tick
    def run(self,x,goal,obs,mask,points,rm,noise,keys,ready,steps=1600,waypoint_count=None):
        key=(len(x),obs.shape[1],points.shape[-2],steps)+((points.shape[1],) if waypoint_count is not None else ())
        if key not in self.compiled:raise ValueError('Unwarmed flight episode; runtime compilation forbidden')
        fn,candidates=self.compiled[key]
        args=(x,goal,obs,mask,points,rm,noise,keys,ready)+((waypoint_count,) if waypoint_count is not None else ())
        return fn(self.params,self.calibration,candidates,*[jnp.asarray(a) for a in args])
