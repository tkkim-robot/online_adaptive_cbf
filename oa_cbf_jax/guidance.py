"""Experimental shared nominal guidance for observed moving obstacles.

A small bank of acceleration/turn-limited kinematic previews suggests a desired
velocity before the ordinary hard CBF-QP. These approximate previews are guidance,
not safety certificates. They use only current observed obstacle velocities.
Guidance is part of the declared method contract. An isolated gain/model
comparison uses identical guidance; an end-to-end nominal ablation must explicitly
report the differences and keep physical/sensor/scene contracts matched.
"""

import jax
import jax.numpy as jnp


def _preview_guidance(x, reference, obstacles, mask, config, horizon, vectorized, speed_limit=None,consider_static=False,clearance_credit=0.,return_diagnostics=False):
    offsets=jnp.linspace(-1.4,1.4,9,dtype=x.dtype)
    if config.guidance_wide_turns:
        # Retain every original target and add headings behind the robot's
        # route bearing. The same bounded acceleration/yaw recurrence applies;
        # these are candidate nominal motions, not instantaneous heading jumps.
        offsets=jnp.concatenate((jnp.asarray([-jnp.pi,-2.4,-1.9],x.dtype),offsets,
                                 jnp.asarray([1.9,2.4,jnp.pi],x.dtype)))
    speed_fractions=jnp.asarray([1.,.6,.25,0.],x.dtype)
    heading_offsets=jnp.tile(offsets,len(speed_fractions))
    fractions=jnp.repeat(speed_fractions,len(offsets))
    nominal_speed=jnp.clip(x[3]+reference[0]/2,0.,config.v_max)
    nominal_bearing=x[2]+reference[1]/2
    headings=nominal_bearing+heading_offsets
    cruise=jnp.maximum(nominal_speed,config.guidance_min_speed)
    if speed_limit is not None:
        cruise=jnp.minimum(cruise,speed_limit)
    speeds=cruise*fractions
    directions=jnp.stack((jnp.cos(nominal_bearing),jnp.sin(nominal_bearing)))
    lateral=jnp.stack((-directions[1],directions[0]))
    dt=horizon/24
    radius=obstacles[:,2]+config.radius+config.clearance_buffer
    if vectorized:
        # Exact solution of the *discrete* saturated proportional recurrence
        # used below. This removes the nested 24-step sequential loop from each
        # controller tick; it does not substitute a continuous-time trajectory.
        k=jnp.arange(24,dtype=x.dtype)[:,None]
        heading_error=jnp.arctan2(jnp.sin(headings-x[2]),jnp.cos(headings-x[2]))
        angle_errors=_discrete_errors(heading_error,config.w_max,dt,k)
        speed_errors=_discrete_errors(speeds-x[3],config.a_max,dt,k)
        angles=x[2]+heading_error-angle_errors
        velocities=speeds-speed_errors
        mid_angles=angles+.5*dt*jnp.clip(2*angle_errors,-config.w_max,config.w_max)
        mid_velocities=velocities+.5*dt*jnp.clip(2*speed_errors,-config.a_max,config.a_max)
        increments=dt*mid_velocities[...,None]*jnp.stack((jnp.cos(mid_angles),jnp.sin(mid_angles)),axis=-1)
        positions=x[:2]+jnp.cumsum(increments,axis=0)
        obs_positions=obstacles[None,:,:2]+(k+1)[:,:,None]*dt*obstacles[None,:,3:5]
        distances=jnp.linalg.norm(positions[:,:,None,:]-obs_positions[:,None,:,:],axis=-1)-radius
        clearances=jnp.where(mask,distances,jnp.inf)
        final_positions=positions[-1]
    else:
        initial=jnp.broadcast_to(x,(len(headings),4))
        def step(state,k):
            error=jnp.arctan2(jnp.sin(headings-state[:,2]),jnp.cos(headings-state[:,2]))
            turn=jnp.clip(2*error,-config.w_max,config.w_max)
            acceleration=jnp.clip(2*(speeds-state[:,3]),-config.a_max,config.a_max)
            mid_heading=state[:,2]+.5*dt*turn
            mid_speed=state[:,3]+.5*dt*acceleration
            new=state.at[:,:2].add(dt*mid_speed[:,None]*jnp.stack((jnp.cos(mid_heading),jnp.sin(mid_heading)),axis=-1))
            new=new.at[:,2].add(dt*turn).at[:,3].add(dt*acceleration)
            obs_position=obstacles[:,:2]+(k+1)*dt*obstacles[:,3:5]
            clearance=jnp.linalg.norm(new[:,:2,None]-obs_position.T[None],axis=1)-radius
            return new,jnp.where(mask,clearance,jnp.inf)
        final,clearances=jax.lax.scan(step,initial,jnp.arange(24))
        final_positions=final[:,:2]
    nominal_index=len(offsets)//2
    relevant=mask if consider_static else mask&(jnp.linalg.norm(obstacles[:,3:5],axis=-1)>1e-5)
    dynamic_clearance=jnp.min(jnp.where(relevant,clearances[:,nominal_index],jnp.inf))
    if config.guidance_turn_return:
        from .turn_return import preview_turn_return
        extra_positions,extra_headings,extra_speeds,extra_offsets,_=preview_turn_return(x,nominal_bearing,cruise,config,horizon)
        obs_positions=obstacles[None,:,:2]+jnp.arange(1,25,dtype=x.dtype)[:,None,None]*dt*obstacles[None,:,3:5]
        extra_clearances=jnp.linalg.norm(extra_positions[:,:,None,:]-obs_positions[:,None,:,:],axis=-1)-radius
        clearances=jnp.concatenate((clearances,jnp.where(mask,extra_clearances,jnp.inf)),axis=1)
        final_positions=jnp.concatenate((final_positions,extra_positions[-1]),axis=0)
        headings=jnp.concatenate((headings,extra_headings))
        speeds=jnp.concatenate((speeds,extra_speeds))
        heading_offsets=jnp.concatenate((heading_offsets,extra_offsets))
    minimum=jnp.min(clearances,axis=(0,2))
    moving=mask&(jnp.linalg.norm(obstacles[:,3:5],axis=-1)>1e-5)
    # Clearance padding is only a preference in the approximate nominal preview.
    # Exact CBF constraints and physical radii are unchanged.
    endpoint_slack=jnp.min(jnp.where(mask,jnp.linalg.norm(x[:2]-obstacles[:,:2],axis=-1)-radius,jnp.inf))
    # The caller may already include an uncertainty allowance in the preview
    # radii. Credit that allowance against the preferred extra .2m gap so the
    # two preferences share a budget. Inflated radii and hard CBF rows remain
    # unchanged; this is solely nominal trajectory ranking.
    desired_slack=jnp.minimum(jnp.maximum(.2-clearance_credit,0.),jnp.maximum(endpoint_slack-.02,0.))
    # A soft variant reserves eligibility for nonnegative inflated clearance;
    # additional clearance remains in the score below. The original preference
    # could exclude every forward option while accepting a stationary preview.
    # Neither approximate threshold is a certificate for the applied CBF action.
    safe=minimum>=(0. if config.guidance_soft_clearance else desired_slack)
    displacement=final_positions-x[:2]
    progress=jnp.sum(displacement*directions,axis=-1)/horizon
    cross_track=jnp.abs(jnp.sum(displacement*lateral,axis=-1))/horizon
    score=progress-.2*cross_track-.025*heading_offsets**2+.05*jnp.minimum(minimum,.5)
    # If no preview is clear, prefer the largest approximate margin, while the
    # downstream QP retains authority to reject an infeasible actual action.
    score=jnp.where(jnp.any(safe),jnp.where(safe,score,-jnp.inf),minimum+.02*score)
    chosen=jnp.argmax(score)
    angle=jnp.arctan2(jnp.sin(headings[chosen]-x[2]),jnp.cos(headings[chosen]-x[2]))
    suggested=jnp.stack((2*(speeds[chosen]-x[3]),2*angle))
    active=jnp.any(relevant)&(dynamic_clearance<.4)&(cruise>.02)
    result=jnp.where(active,suggested,reference)
    if return_diagnostics:
        return result,dict(active=active,chosen=chosen,final_positions=final_positions,
            minimum_clearance=minimum,safe=safe,score=score,desired_slack=desired_slack,
            headings=headings,speeds=speeds,nominal_bearing=nominal_bearing)
    return result


def _discrete_errors(initial,limit,dt,k):
    magnitude=jnp.abs(initial)
    saturated_steps=jnp.ceil(jnp.maximum(magnitude-limit/2,0)/(limit*dt))
    residual=magnitude-saturated_steps*limit*dt
    decay=jnp.power(jnp.maximum(1-2*dt,0),jnp.maximum(k-saturated_steps,0))
    values=jnp.where(k<saturated_steps,magnitude-k*limit*dt,residual*decay)
    return jnp.sign(initial)*values


def preview_guidance_scan(x, reference, obstacles, mask, config, horizon, speed_limit=None,consider_static=False,clearance_credit=0.):
    """Original sequential midpoint recurrence, retained as a numerical oracle."""
    return _preview_guidance(x,reference,obstacles,mask,config,horizon,False,speed_limit,consider_static,clearance_credit)


def preview_guidance_vectorized(x, reference, obstacles, mask, config, horizon, speed_limit=None,consider_static=False,clearance_credit=0.):
    """Same preview and selection with a parallel discrete recurrence."""
    return _preview_guidance(x,reference,obstacles,mask,config,horizon,True,speed_limit,consider_static,clearance_credit)


def preview_guidance(x, reference, obstacles, mask, config, horizon, speed_limit=None,consider_static=False,clearance_credit=0.):
    # Rounding at discrete selection boundaries can change the chosen command.
    # The implementation is therefore part of the data/controller contract;
    # historical trained/calibrated bundles retain their original scan default.
    return _preview_guidance(x,reference,obstacles,mask,config,horizon,
                             config.guidance_kernel=='vectorized',speed_limit,consider_static,clearance_credit)
