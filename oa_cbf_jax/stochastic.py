"""Shared stochastic implementation."""

from functools import partial

import jax

import jax.numpy as jnp

from .config import UnicycleConfig

from .dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance

from .route_control import route_control, INADMISSIBLE

from .routing import physical_route_coordinate

from .simulation import Summary, RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT

from .guidance import clearance_inflation

STATE_BOUND_VIOLATION=8

def conditioned_sensor_model(observed_x,observed_obstacles,mask,noise,key,config,steps):
    """Shared physical prior and future sensor stream, independent of policy."""
    x_key,obs_key,innovation_key=jax.random.split(key,3)
    x_scale=jnp.array([noise[0],noise[0],noise[1],noise[2]])
    obs_scale=jnp.array([noise[3],noise[3],noise[5],noise[4],noise[4]])
    physical_x=observed_x+jax.random.uniform(x_key,(4,),minval=-1.,maxval=1.)*x_scale
    physical_x=physical_x.at[3].set(jnp.clip(physical_x[3],0.,config.v_max))
    x_bias=physical_x-observed_x
    obs_bias=jax.random.uniform(obs_key,observed_obstacles.shape,minval=-1.,maxval=1.)*obs_scale
    obs_bias=jnp.where(mask[:,None],obs_bias,0.)
    physical_obs=observed_obstacles+obs_bias
    physical_obs=physical_obs.at[:,2].set(jnp.maximum(physical_obs[:,2],1e-4))
    if config.stationary_obstacles:
        # Initial measured velocities may be noisy. They remain observations;
        # stationary physical obstacles have exactly zero velocity throughout.
        physical_obs=physical_obs.at[:,3:5].set(0.)
    obs_bias=physical_obs-observed_obstacles
    innovations=jax.random.uniform(innovation_key,(steps,4+observed_obstacles.size),minval=-1.,maxval=1.).at[0].set(0.)
    return physical_x,physical_obs,x_bias,obs_bias,x_scale,obs_scale,innovations

@partial(jax.jit, static_argnames=('config','steps','sensor_margin_scale','margin_guidance','shared_clearance_budget'))
def stochastic_branch(observed_x, goal, observed_obstacles, mask, alpha, points, route_mask,
                       route_progress, noise, key, config=UnicycleConfig(), steps=80, sensor_margin_scale=0.,margin_guidance=False,shared_clearance_budget=False):
    """noise = max absolute [xy, heading, speed, obs_xy, obs_velocity, radius].

    Latent offsets are uniform in the indicated ranges. Subsequent independent
    sensor innovations have 15% of those ranges (zero at step zero). Initial
    physical speed is clipped by the *generative prior* to [0,vmax]; plant states
    are never clipped after integration. Obstacles move at their latent constant
    velocities. All gains sharing a replica key receive common random numbers.
    """
    physical_x,physical_obs,x_bias,obs_bias,x_scale,obs_scale,innovations=conditioned_sensor_model(
        observed_x,observed_obstacles,mask,noise,key,config,steps)
    initial_clearance=jnp.min(signed_clearance(physical_x[:2],physical_obs,mask,config.radius))
    at_goal=(jnp.linalg.norm(physical_x[:2]-goal)<=config.goal_tolerance)&(jnp.abs(physical_x[3])<=.2)
    initial_status=jnp.where(initial_clearance<=0,COLLISION,jnp.where(at_goal,GOAL,RUNNING))
    def tick(carry, inputs):
        x,status,count,clearance,psi_min,violation_max,progress=carry
        k,innovation=inputs;active=status==RUNNING;now=k*config.dt
        sensed_x=x-x_bias+.15*x_scale*innovation[:4]
        truth_obs=physical_obs.at[:,:2].set(physical_obs[:,:2]+now*physical_obs[:,3:5])
        sensed_obs=truth_obs-obs_bias+.15*obs_scale*innovation[4:].reshape(observed_obstacles.shape)
        qp,h,psi,proposed,remaining,target=route_control(sensed_x,goal,sensed_obs,mask,alpha,points,route_mask,progress,config,
                                                       speed_uncertainty=1.15*noise[2],clearance_uncertainty=clearance_inflation(noise,sensor_margin_scale),margin_guidance=margin_guidance,shared_clearance_budget=shared_clearance_budget)
        admissible=(h>=-config.qp_tolerance)&(psi>=-config.qp_tolerance)
        can_step=active&qp.feasible&admissible
        u=jnp.where(can_step,qp.control,jnp.zeros(2,x.dtype))
        y,sub=integrate_unicycle(x,u,config.dt,config.integration_substeps)
        starts=jnp.concatenate((x[None],sub[:-1]));times=now+jnp.arange(config.integration_substeps)*config.dt/config.integration_substeps
        clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,physical_obs,mask,config.radius,t,
                        t+config.dt/config.integration_substeps))(starts,sub,times))
        reached=can_step&(jnp.linalg.norm(y[:2]-goal)<=config.goal_tolerance)&(jnp.abs(y[3])<=.2)
        bound_violation=jnp.maximum(-y[3],y[3]-config.v_max)
        status=jnp.where(active&~qp.feasible,INFEASIBLE,status)
        status=jnp.where(active&~admissible,INADMISSIBLE,status)
        status=jnp.where(reached,GOAL,status)
        status=jnp.where(can_step&(bound_violation>config.qp_tolerance),STATE_BOUND_VIOLATION,status)
        status=jnp.where(can_step&(clear<=0),COLLISION,status)
        new_x=jnp.where(can_step,y,x);progress=jnp.where(can_step,proposed,progress)
        count+=can_step.astype(jnp.int32);clearance=jnp.minimum(clearance,jnp.where(can_step,clear,jnp.inf))
        psi_min=jnp.minimum(psi_min,jnp.where(active,psi,jnp.inf))
        violation_max=jnp.maximum(violation_max,jnp.where(can_step,qp.max_violation,-jnp.inf))
        trace=dict(state=new_x,control=u,active=can_step,status=status,observed_state=sensed_x,
                   observed_obstacles=sensed_obs,clearance=jnp.where(can_step,clear,jnp.nan),
                   state_bound_violation=jnp.where(can_step,bound_violation,jnp.nan),
                   route_progress=progress,route_remaining=remaining,route_target=target,
                   qp_violation=jnp.where(can_step,qp.max_violation,jnp.nan),psi1=jnp.where(active,psi,jnp.nan))
        return (new_x,status,count,clearance,psi_min,violation_max,progress),trace
    carry=(physical_x,initial_status,jnp.int32(0),initial_clearance,jnp.asarray(jnp.inf,observed_x.dtype),
           jnp.asarray(-jnp.inf,observed_x.dtype),jnp.asarray(route_progress,observed_x.dtype))
    (x,status,count,clearance,psi,violation,final_cursor),trace=jax.lax.scan(tick,carry,(jnp.arange(steps),innovations))
    status=jnp.where(status==RUNNING,TIMEOUT,status)
    summary=Summary(x,status,count,clearance,psi,jnp.linalg.norm(physical_x[:2]-goal)-jnp.linalg.norm(x[:2]-goal),violation)
    route_delta=(physical_route_coordinate(x[:2],points,route_mask,final_cursor)
                 -physical_route_coordinate(physical_x[:2],points,route_mask,route_progress))
    truth=dict(initial_state=physical_x,obstacles=physical_obs,route_progress_delta=route_delta)
    return summary,trace,truth
