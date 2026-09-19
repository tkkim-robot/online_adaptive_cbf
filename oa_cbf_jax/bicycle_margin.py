"""Conservative next-observation barrier margin for OA nominal-profile guidance.

Only present observations, an actual proposed input and declared noise ranges
enter the envelope. This is a one-step mathematical model bound with explicit
FP32 rounding allowance, not a universal or recursive physical safety claim.
"""
import jax.numpy as jnp
import numpy as np
from .bicycle_control import BicycleControlConfig,constant


def next_observation_errors(state,control,predicted,obstacles,predicted_obstacles,noise,config=BicycleControlConfig()):
    """Disk/scalar envelopes under persistent bias + independent .15 innovations.

    Integrating two observed/true affine-slip states with the SAME input gives
    the position error by bounding the velocity integral and rotation chord.
    Persistent position/heading/speed biases cancel at the next observation;
    their effect on the predicted dynamics remains explicitly bounded.
    """
    dtype=jnp.float64;x=state.astype(dtype);u=control.astype(dtype);y=predicted.astype(dtype);n=noise.astype(dtype);c=config.robot
    dt=constant(c.dt,dtype);lr=constant(c.rear_axle_distance,dtype)
    magnitude=jnp.max(jnp.concatenate((jnp.abs(x),jnp.abs(y),jnp.abs(obstacles).reshape(-1),jnp.abs(predicted_obstacles).reshape(-1))))
    rounding=constant(8*np.finfo(np.float32).eps,dtype)*(1+magnitude)
    current_speed=constant(1.15,dtype)*n[2]+rounding
    current_heading=constant(1.15,dtype)*n[1]+rounding
    angle=current_heading+jnp.abs(u[1])*dt/lr*current_speed
    motion=dt*jnp.sqrt(1+u[1]**2)*(current_speed+(jnp.abs(x[3])+jnp.abs(u[0])*dt)*jnp.minimum(angle,2.))
    ego_position=constant(.3,dtype)*n[0]+motion+rounding
    ego_heading=constant(.3,dtype)*n[1]+jnp.abs(u[1])*dt/lr*current_speed+rounding
    ego_speed=constant(.3,dtype)*n[2]+rounding
    obstacle_position=constant(.3,dtype)*n[3]+dt*(constant(1.15,dtype)*n[4]+rounding)+rounding
    obstacle_velocity=constant(.3,dtype)*n[4]+rounding
    obstacle_radius=constant(.3,dtype)*n[5]+rounding
    return dict(position_error=ego_position+obstacle_position,velocity_error=obstacle_velocity+ego_speed+jnp.abs(y[3])*jnp.minimum(ego_heading,2.),radius_error=obstacle_radius,
        ego_position_error=ego_position,ego_heading_error=ego_heading,ego_speed_error=ego_speed,obstacle_position_error=obstacle_position,obstacle_velocity_error=obstacle_velocity,rounding_allowance=rounding)


def lower_barrier(state,obstacles,mask,position_error,velocity_error,radius_error,config=BicycleControlConfig()):
    """Lower h over relative-position/velocity disks and radius interval.

    Each positive parabolic term is bounded from below. Unsupported geometric
    domains return -inf; square-root guards never turn them into valid margins.
    """
    dtype=jnp.float64;x=state.astype(dtype);o=obstacles.astype(dtype)
    p=o[:,:2]-x[:2];d=jnp.sqrt(jnp.maximum(jnp.sum(p*p,axis=1),constant(1e-24,dtype)));unit=p/d[:,None]
    v=o[:,3:5]-x[3]*jnp.stack((jnp.cos(x[2]),jnp.sin(x[2])))
    speed=jnp.sqrt(jnp.sum(v*v,axis=1));radial=jnp.sum(unit*v,axis=1);lateral=unit[:,0]*v[:,1]-unit[:,1]*v[:,0]
    eta=2*jnp.sin(jnp.arcsin(jnp.clip(position_error/d,0.,1.))/2)
    component_error=speed*eta+velocity_error
    radius=(constant(config.robot.radius+config.clearance_buffer,dtype)+o[:,2]+radius_error)*constant(config.barrier_inflation,dtype)
    domain=(d-position_error)**2-radius**2;supported=(d>position_error)&(radius>0)&(d-position_error>radius)
    root=jnp.sqrt(jnp.maximum(domain,constant(1e-24,dtype)));shape=jnp.sqrt(constant(config.barrier_inflation**2-1,dtype))/jnp.maximum(radius,constant(1e-12,dtype))
    q=jnp.sqrt((speed+velocity_error)**2+constant(config.relative_speed_epsilon**2,dtype))
    low=radial-component_error+.5*shape*root*jnp.maximum(jnp.abs(lateral)-component_error,0.)**2/q+shape*root
    finite=jnp.isfinite(low)&jnp.isfinite(domain)&(position_error>=0)&(velocity_error>=0)&(radius_error>=0)
    per_obstacle=jnp.where(mask,jnp.where(supported&finite,low,-jnp.inf),jnp.inf)
    return dict(lower=jnp.min(per_obstacle),domain_supported=jnp.all(jnp.where(mask,supported&finite,True)),per_obstacle=per_obstacle)


def next_margin(state,control,predicted,obstacles,predicted_obstacles,mask,noise,config=BicycleControlConfig()):
    errors=next_observation_errors(state,control,predicted,obstacles,predicted_obstacles,noise,config)
    bound=lower_barrier(predicted,predicted_obstacles,mask,errors['position_error'],errors['velocity_error'],errors['radius_error'],config)
    return {**bound,**errors}
