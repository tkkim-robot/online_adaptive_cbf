"""Observed-state DPCBF-QP for the affine-slip bicycle; no learned policy yet.

The obstacle barrier follows the existing continuous parabolic shape with an
explicit smooth relative-speed regularizer. Geometry inside the inflated disk
is rejected separately; numerical square-root guards never make it admissible.
"""
from dataclasses import dataclass,field,asdict
import math
import numpy as np
import jax
import jax.numpy as jnp
from .bicycle import BicycleConfig,steering_to_slip
from .controllers import QPResult,solve_qp2
from .qp2_interval import solve_qp2_intervals
from .routing import route_target_from_position


def constant(value,dtype):
    # With explicit-x64 allowed but global defaults32, JAX0.8.2 converts a
    # Python scalar/list through FP32 even with an explicit requested64 dtype.
    # A typed NumPy constant preserves the declared physical parameter.
    return jnp.asarray(np.asarray(value,np.float64),dtype)


def observed32(value):
    # XLA may elide a bare64->32->64 conversion in a fused consumer while
    # writing a rounded32 trace. Explicit reduce_precision makes the rounding
    # part of the computation, including observed-state and actuator rechecks.
    return jax.lax.reduce_precision(value,exponent_bits=8,mantissa_bits=23).astype(jnp.float32)


@dataclass(frozen=True)
class BicycleControlConfig:
    robot:BicycleConfig=field(default_factory=BicycleConfig)
    clearance_buffer:float=.05
    barrier_inflation:float=1.05
    relative_speed_epsilon:float=.02
    cruise_speed:float=1.5
    steering_feedback:float=.8
    speed_feedback:float=2.
    goal_tolerance:float=.25
    terminal_speed:float=.35
    qp_tolerance:float=1e-5
    solver:str='interval'

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for k,v in asdict(self).items() if k not in ('robot','solver')):
            raise ValueError('Invalid bicycle control settings')
        if (self.barrier_inflation<=1 or not self.robot.speed_min<=self.terminal_speed<=self.cruise_speed<=self.robot.speed_max
                or self.solver not in ('interval','enumeration')):
            raise ValueError('Invalid bicycle task/solver contract')

    @property
    def radius(self):return self.robot.radius


def parabolic_terms(state,obstacles,config=BicycleControlConfig()):
    """h, geometric domain, position-gradient and relative-velocity gradient."""
    c=config.robot;direction=jnp.stack((jnp.cos(state[2]),jnp.sin(state[2])))
    position=obstacles[:,:2]-state[:2];distance2=jnp.sum(position**2,axis=1)
    distance=jnp.sqrt(jnp.maximum(distance2,1e-24));unit=position/distance[:,None]
    normal=jnp.stack((-unit[:,1],unit[:,0]),axis=1)
    relative=obstacles[:,3:5]-state[3]*direction
    radial=jnp.sum(unit*relative,axis=1);lateral=jnp.sum(normal*relative,axis=1)
    radius=(constant(c.radius+config.clearance_buffer,state.dtype)+obstacles[:,2])*constant(config.barrier_inflation,state.dtype)
    domain=distance2-radius**2;root=jnp.sqrt(jnp.maximum(domain,1e-12))
    speed=jnp.sqrt(jnp.sum(relative**2,axis=1)+constant(config.relative_speed_epsilon**2,state.dtype))
    factor=jnp.sqrt(constant(config.barrier_inflation**2-1,state.dtype))/radius
    lam=.5*factor;mu=factor
    h=radial+lam*root*lateral**2/speed+mu*root
    dz=position/root[:,None]
    drad=lateral[:,None]*normal/distance[:,None]
    dlat=-radial[:,None]*normal/distance[:,None]
    dp=drad+(lam*lateral**2/speed)[:,None]*dz+(2*lam*root*lateral/speed)[:,None]*dlat+mu[:,None]*dz
    dv=unit+(lam*root)[:,None]*(2*lateral[:,None]*normal/speed[:,None]-lateral[:,None]**2*relative/speed[:,None]**3)
    return h,domain,dp,dv


def bicycle_rows(state,obstacles,mask,alpha,config=BicycleControlConfig(),speed_error=0.):
    # FP64 row construction and solving; recheck after casting the actuator
    # command to the physical state dtype. No global dtype default is changed.
    x=state.astype(jnp.float64);obs=obstacles.astype(jnp.float64);c=config.robot
    h,domain,dp,dv=parabolic_terms(x,obs,config)
    direction=jnp.stack((jnp.cos(x[2]),jnp.sin(x[2])));normal=jnp.stack((-direction[1],direction[0]))
    relative=obs[:,3:5]-x[3]*direction
    drift=jnp.sum(dp*relative,axis=1)
    authority=jnp.stack((-dv@direction,-x[3]*(dp@normal)-x[3]**2/constant(c.rear_axle_distance,x.dtype)*(dv@normal)),axis=1)
    obstacle_a=jnp.where(mask[:,None],-authority,0.)
    gain=constant(alpha,x.dtype) if isinstance(alpha,(float,int,np.generic)) else alpha.astype(x.dtype)
    obstacle_b=jnp.where(mask,drift+gain*h,1.)
    low_speed=x[3];high_speed=x[3]
    if not isinstance(speed_error,(float,int)) or speed_error!=0:
        error=jnp.asarray(speed_error,x.dtype);low_speed-=error;high_speed+=error
    lower=jnp.maximum(constant(-c.acceleration_max,x.dtype),(constant(c.speed_min,x.dtype)-low_speed)/constant(c.dt,x.dtype))
    upper=jnp.minimum(constant(c.acceleration_max,x.dtype),(constant(c.speed_max,x.dtype)-high_speed)/constant(c.dt,x.dtype))
    bounds=jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],jnp.float64)
    rhs=jnp.stack((upper,-lower,constant(c.slip_max,x.dtype),constant(c.slip_max,x.dtype)))
    return jnp.concatenate((obstacle_a,bounds)),jnp.concatenate((obstacle_b,rhs)),h,domain


def nominal_bicycle(state,goal,target,config=BicycleControlConfig()):
    delta=target-state[:2];bearing=jnp.arctan2(delta[1],delta[0])-state[2]
    error=jnp.arctan2(jnp.sin(bearing),jnp.cos(bearing))
    steering=jnp.clip(config.steering_feedback*error,-config.robot.steering_max,config.robot.steering_max)
    distance=jnp.maximum(jnp.linalg.norm(goal-state[:2])-.1,0.)
    speed=jnp.maximum(config.robot.speed_min,jnp.minimum(config.cruise_speed,1.2*distance)*jnp.maximum(jnp.cos(error),0.))
    return jnp.stack((config.speed_feedback*(speed-state[3]),steering_to_slip(steering,config.robot)))


def project_bicycle_reference(reference,a,b,dtype,config=BicycleControlConfig()):
    """Project a nominal input, then check every row on the rounded actuator."""
    reference=reference.astype(jnp.float64)
    weights=constant([1/config.robot.acceleration_max**2,1/config.robot.slip_max**2],jnp.float64)
    solver=solve_qp2_intervals if config.solver=='interval' else solve_qp2
    result=solver(reference,a,b,weights,config.qp_tolerance)
    u=observed32(result.control) if dtype==jnp.float32 else result.control.astype(dtype)
    violation=jnp.max(jnp.matmul(a,u.astype(a.dtype),precision='highest')-b)
    feasible=result.feasible&jnp.isfinite(violation)&(violation<=config.qp_tolerance)
    return QPResult(jnp.where(feasible,u,jnp.full_like(u,jnp.nan)),feasible,violation.astype(dtype),result.objective.astype(dtype))


def bicycle_control(state,goal,obstacles,mask,alpha,points,route_mask,cursor,config=BicycleControlConfig(),speed_error=0.):
    target,proposed,remaining=route_target_from_position(state[:2],state[3],points,route_mask,cursor)
    reference=nominal_bicycle(state,goal,target,config)
    a,b,h,domain=bicycle_rows(state,obstacles,mask,alpha,config,speed_error)
    result=project_bicycle_reference(reference,a,b,state.dtype,config)
    return result,jnp.min(jnp.where(mask,h,jnp.inf)).astype(state.dtype),jnp.min(jnp.where(mask,domain,jnp.inf)).astype(state.dtype),proposed,remaining,target


def bicycle_arrived(state,goal,config=BicycleControlConfig()):
    # Rolling arrival, consistent with the positive declared minimum speed.
    return (jnp.linalg.norm(state[:2]-goal)<=config.goal_tolerance)&(state[3]<=config.terminal_speed)
