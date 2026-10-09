"""Shared bicycle implementation."""

from dataclasses import dataclass

import math

import numpy as np

import jax

import jax.numpy as jnp

@dataclass(frozen=True)
class BicycleConfig:
    dt: float = .05
    wheel_base: float = .4
    rear_axle_distance: float = .2
    radius: float = .3
    acceleration_max: float = 5.
    steering_max: float = math.radians(32)
    speed_min: float = .2
    speed_max: float = 3.5
    integration_substeps: int = 8

    def __post_init__(self):
        if (not all(math.isfinite(x) and x>0 for x in
                (self.dt,self.wheel_base,self.rear_axle_distance,self.radius,self.acceleration_max,self.steering_max,self.speed_max))
                or self.rear_axle_distance>=self.wheel_base or self.steering_max>=math.pi/2
                or not math.isfinite(self.speed_min) or not 0<=self.speed_min<self.speed_max
                or type(self.integration_substeps) is not int or self.integration_substeps<1):
            raise ValueError('Invalid affine bicycle configuration')

    @property
    def slip_max(self):
        return math.atan(self.rear_axle_distance/self.wheel_base*math.tan(self.steering_max))

def steering_to_slip(steering, config=BicycleConfig()):
    return jnp.arctan(config.rear_axle_distance/config.wheel_base*jnp.tan(steering))

def held_bicycle_state(state, control, duration, config=BicycleConfig()):
    """Exact held-input flow, including straight motion and speed reversals.

    With path parameter s=v0*t+a*t²/2, heading=heading0+slip*s/Lr.
    Integrating the rotated longitudinal/lateral vector gives a stable sinc
    expression. The planar speed is |v|*sqrt(1+slip²) for this affine model.
    """
    control=control.astype(state.dtype)
    duration=jnp.asarray(np.asarray(duration,np.float64),state.dtype) if isinstance(duration,(float,int,np.generic)) else duration.astype(state.dtype)
    acceleration,slip=control
    distance=state[3]*duration+.5*acceleration*duration**2
    angle=slip*distance/jnp.asarray(np.asarray(config.rear_axle_distance,np.float64),state.dtype)
    mid=state[2]+.5*angle
    length=distance*jnp.sinc(.5*angle/jnp.asarray(np.asarray(np.pi,np.float64),state.dtype))
    offset=length*jnp.stack((jnp.cos(mid)-slip*jnp.sin(mid),jnp.sin(mid)+slip*jnp.cos(mid)))
    return jnp.concatenate((state[:2]+offset,jnp.stack((state[2]+angle,state[3]+acceleration*duration))))

def integrate_bicycle(state, control, config=BicycleConfig()):
    times=jnp.arange(1,config.integration_substeps+1,dtype=state.dtype)*jnp.asarray(np.asarray(config.dt/config.integration_substeps,np.float64),state.dtype)
    substeps=jax.vmap(lambda t:held_bicycle_state(state,control,t,config))(times)
    return substeps[-1],substeps

def bicycle_state_violation(state, config=BicycleConfig()):
    return jnp.maximum(config.speed_min-state[3],state[3]-config.speed_max)
