"""Bicycle control functions and shared contracts."""

import jax

import jax.numpy as jnp

from .controllers import QPResult

@jax.jit
def solve_qp2_intervals(reference, a, b, weights, tolerance=1e-5):
    invw=1/weights
    length=jnp.sqrt(jnp.sum(a*a,axis=1));nonzero=length>0
    scale=jnp.where(nonzero,length,1.)
    rows=a/scale[:,None];rhs=b/scale
    denominator=jnp.sum(rows*rows*invw,axis=1)
    residual=jnp.matmul(rows,reference,precision='highest')-rhs
    origins=reference-(residual/jnp.maximum(denominator,1e-30))[:,None]*rows*invw
    tangent=jnp.stack((-rows[:,1],rows[:,0]),axis=1)
    coefficient=jnp.matmul(tangent,rows.T,precision='highest')
    available=rhs[None]-jnp.matmul(origins,rows.T,precision='highest')
    parallel=jnp.abs(coefficient)<=1e-12
    bound=available/jnp.where(parallel,1.,coefficient)
    low=jnp.max(jnp.where(coefficient < -1e-12,bound,-jnp.inf),axis=1)
    high=jnp.min(jnp.where(coefficient > 1e-12,bound,jnp.inf),axis=1)
    compatible=jnp.all(~parallel | (available>=-1e-12*(1+jnp.abs(rhs[None]))),axis=1)
    parameter=jnp.maximum(low,jnp.minimum(jnp.zeros_like(high),high))
    face=origins+parameter[:,None]*tangent
    candidates=jnp.concatenate((reference[None],face))
    geometry=jnp.concatenate((jnp.ones(1,bool),nonzero&compatible&(low<=high+1e-12)))
    violation=jnp.max(jnp.matmul(candidates,a.T,precision='highest')-b[None],axis=1)
    valid=geometry&jnp.all(jnp.isfinite(candidates),axis=1)&(violation<=tolerance)
    cost=.5*jnp.sum(weights*(candidates-reference)**2,axis=1)
    candidate_cost=jnp.where(valid,cost,jnp.inf)
    index=jnp.argmax(candidate_cost==jnp.min(candidate_cost));feasible=jnp.any(valid)
    return QPResult(jnp.where(feasible,candidates[index],jnp.full_like(reference,jnp.nan)),feasible,
        jnp.where(feasible,violation[index],jnp.inf),jnp.where(feasible,cost[index],jnp.inf))


from dataclasses import dataclass, field, asdict

import math

import numpy as np


from .bicycle import BicycleConfig, steering_to_slip

from .controllers import solve_qp2

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


import json

from pathlib import Path


def read(path):return json.loads(Path(path).read_text())

def control_config(value):
    """Restore the complete versioned physical/control contract, no defaults substitution."""
    value=dict(value);value['robot']=BicycleConfig(**value['robot'])
    config=BicycleControlConfig(**value)
    if asdict(config)!=dict(value,robot=asdict(value['robot'])):raise ValueError('Noncanonical bicycle configuration')
    return config


def contract():
    return dict(schema='bicycle_log_quadratic_affine_gain',gain_dimension=1,
        fields=['log(alpha/2)/log(4)','(log(alpha/2)/log(4))^2','alpha/2-1'],
        unchanged_scene_encoding=True,controller_solve=False,training_labels_used=False,
        explanation='Fixed alpha=2 reference. Last coordinate exposes the exact linear gain dependence in drift+alpha*h.')

def coordinates(gains):
    if gains.shape[-1]!=1:raise ValueError('Bicycle affine gain requires one scalar gain')
    value=(jnp.log(jnp.maximum(gains,1e-6))-math.log(2.))/math.log(4.)
    return jnp.concatenate((value,value**2,gains/2.-1.),axis=-1)
