"""Shared continuation implementation."""

from functools import partial

from typing import NamedTuple

import jax

import jax.numpy as jnp

from .config import UnicycleConfig

from .motion_observer import MotionState, update, effective_noise

from .dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance

from .route_control import route_control, INADMISSIBLE

from .routing import physical_route_coordinate

from .simulation import Summary, RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT

from .stochastic import STATE_BOUND_VIOLATION

from .guidance import clearance_inflation

from .motion_observer import PositionState, position_observer_update as position_update

class Snapshot(NamedTuple):
    physical_state:jax.Array
    physical_obstacles:jax.Array
    state_bias:jax.Array
    obstacle_bias:jax.Array
    raw_state:jax.Array
    raw_obstacles:jax.Array
    observer:MotionState
    position:PositionState|None=None

@partial(jax.jit,static_argnames=('config','steps','sensor_margin_scale','margin_guidance','shared_clearance_budget','motion_observer_window','filter_obstacle_position'))
def rollout(snapshot,goal,mask,gains,points,route_mask,cursor,noise,key,config=UnicycleConfig(),steps=80,
            sensor_margin_scale=0.,margin_guidance=False,shared_clearance_budget=False,motion_observer_window=0,filter_obstacle_position=False):
    if filter_obstacle_position and (not motion_observer_window or snapshot.position is None):
        raise ValueError('Position-filtered continuation requires its actual pre-measurement memory')
    x0=snapshot.physical_state;obstacles=snapshot.physical_obstacles
    x_scale=jnp.array([noise[0],noise[0],noise[1],noise[2]])
    obs_scale=jnp.array([noise[3],noise[3],noise[5],noise[4],noise[4]])
    innovations=jax.random.uniform(key,(steps,4+obstacles.size),minval=-1.,maxval=1.)
    clearance=jnp.min(signed_clearance(x0[:2],obstacles,mask,config.radius))
    at_goal=(jnp.linalg.norm(x0[:2]-goal)<=config.goal_tolerance)&(jnp.abs(x0[3])<=.2)
    status=jnp.where(clearance<=0,COLLISION,jnp.where(at_goal,GOAL,RUNNING))
    def tick(carry,values):
        x,status,count,clear,psi_min,violation_max,progress,observer,position=carry
        k,innovation=values;active=status==RUNNING;now=k*config.dt
        physical_obs=obstacles.at[:,:2].set(obstacles[:,:2]+now*obstacles[:,3:5])
        raw_x=jnp.where(k==0,snapshot.raw_state,x-snapshot.state_bias+.15*x_scale*innovation[:4])
        raw_obs=jnp.where(k==0,snapshot.raw_obstacles,physical_obs-snapshot.obstacle_bias+.15*obs_scale*innovation[4:].reshape(obstacles.shape))
        sensed_obs=raw_obs;feature_noise=noise;bad=jnp.zeros_like(mask)
        if motion_observer_window:
            observer,sensed_obs,bounds,lag,bad=update(observer,raw_obs,mask,noise,config.dt,motion_observer_window)
            feature_noise=effective_noise(noise,bounds,mask)
        if filter_obstacle_position:
            position,sensed_obs,position_bad=position_update(position,sensed_obs,sensed_obs[:,3:5],
                1.15*feature_noise[4],mask,noise,config.dt)
            bad=bad|jnp.any(position_bad,axis=-1)
        qp,h,psi,proposed,remaining,target=route_control(raw_x,goal,sensed_obs,mask,gains,points,route_mask,progress,config,
            1.15*noise[2],clearance_inflation(noise,sensor_margin_scale),margin_guidance,shared_clearance_budget)
        admissible=(h>=-config.qp_tolerance)&(psi>=-config.qp_tolerance)
        can_step=active&qp.feasible&admissible;u=jnp.where(can_step,qp.control,jnp.zeros(2,x.dtype))
        y,sub=integrate_unicycle(x,u,config.dt,config.integration_substeps)
        starts=jnp.concatenate((x[None],sub[:-1]));times=now+jnp.arange(config.integration_substeps)*config.dt/config.integration_substeps
        step_clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,obstacles,mask,config.radius,t,t+config.dt/config.integration_substeps))(starts,sub,times))
        bound=jnp.maximum(-y[3],y[3]-config.v_max)
        reached=can_step&(jnp.linalg.norm(y[:2]-goal)<=config.goal_tolerance)&(jnp.abs(y[3])<=.2)
        status=jnp.where(active&~qp.feasible,INFEASIBLE,status);status=jnp.where(active&~admissible,INADMISSIBLE,status)
        status=jnp.where(reached,GOAL,status);status=jnp.where(can_step&(bound>config.qp_tolerance),STATE_BOUND_VIOLATION,status)
        status=jnp.where(can_step&(step_clear<=0),COLLISION,status)
        x=jnp.where(can_step,y,x);progress=jnp.where(can_step,proposed,progress);count+=can_step.astype(jnp.int32)
        clear=jnp.minimum(clear,jnp.where(can_step,step_clear,jnp.inf));psi_min=jnp.minimum(psi_min,jnp.where(active,psi,jnp.inf))
        violation_max=jnp.maximum(violation_max,jnp.where(can_step,qp.max_violation,-jnp.inf))
        trace=dict(state=x,control=u,active=can_step,status=status,observed_state=raw_x,observed_obstacles=sensed_obs,
            raw_observed_obstacles=raw_obs,effective_noise=feature_noise,observer_inconsistent=bad,
            clearance=jnp.where(can_step,step_clear,jnp.nan),state_bound_violation=jnp.where(can_step,bound,jnp.nan),
            route_progress=progress,route_remaining=remaining,route_target=target,
            qp_violation=jnp.where(can_step,qp.max_violation,jnp.nan),psi1=jnp.where(active,psi,jnp.nan))
        return (x,status,count,clear,psi_min,violation_max,progress,observer,position),trace
    initial=(x0,status,jnp.int32(0),clearance,jnp.asarray(jnp.inf,x0.dtype),jnp.asarray(-jnp.inf,x0.dtype),jnp.asarray(cursor,x0.dtype),snapshot.observer,snapshot.position)
    (x,status,count,clear,psi,violation,progress,_,_),trace=jax.lax.scan(tick,initial,(jnp.arange(steps),innovations))
    status=jnp.where(status==RUNNING,TIMEOUT,status)
    summary=Summary(x,status,count,clear,psi,jnp.linalg.norm(x0[:2]-goal)-jnp.linalg.norm(x[:2]-goal),violation)
    delta=physical_route_coordinate(x[:2],points,route_mask,progress)-physical_route_coordinate(x0[:2],points,route_mask,cursor)
    return summary,trace,dict(initial_state=x0,obstacles=obstacles,route_progress_delta=delta)

def snapshot_payload(snapshot):
    result={**{f'snapshot_{name}':getattr(snapshot,name) for name in Snapshot._fields if name not in ('observer','position')},
            **{f'observer_{name}':getattr(snapshot.observer,name) for name in MotionState._fields}}
    if snapshot.position is not None:
        result.update({f'position_{name}':getattr(snapshot.position,name) for name in PositionState._fields})
    return result
