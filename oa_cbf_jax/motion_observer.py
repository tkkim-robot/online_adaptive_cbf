"""Causal bounded-velocity estimation from identified obstacle tracks.

Two position anchors retain at most two windows of history in constant memory.
For constant physical obstacle velocity and constant position bias, position
increments cancel the bias. Their bounded innovations give a velocity interval,
intersected with the current sensor velocity interval. No latent/future data enter.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class MotionState(NamedTuple):
    current_anchor:jax.Array
    previous_anchor:jax.Array
    ticks:jax.Array


def initialize(obstacles):
    return MotionState(obstacles[:,:2],obstacles[:,:2],jnp.int32(0))


def update(state,obstacles,mask,noise,dt,window=32):
    """Return (new state, estimated obstacles, bounds [M,2], lag, inconsistency).

    Obstacles have stable row identities throughout the episode. The measurement
    model is position=truth-constant_bias+bounded innovation, with innovation
    size .15*noise[3]. Velocity measurement error is at most1.15*noise[4].
    On an empty interval, use the raw reading and its original bound and flag
    that track; do not manufacture a narrow uncertainty interval. No position,
    radius, robot state or physical object is changed.
    """
    if isinstance(window,bool) or not isinstance(window,int) or window<1:
        raise ValueError('Observer window must be a positive integer')
    phase=state.ticks%window
    rotate=(state.ticks>0)&(phase==0)
    previous=jnp.where(rotate,state.current_anchor,state.previous_anchor)
    current=jnp.where(rotate,obstacles[:,:2],state.current_anchor)
    anchor=jnp.where(state.ticks>=window,previous,current)
    lag=jnp.where(state.ticks>=window,window+phase,state.ticks)
    elapsed=jnp.maximum(lag,1)*dt
    slope=(obstacles[:,:2]-anchor)/elapsed
    # FP32 subtraction/division error allowance scales with coordinates. It
    # bounds arithmetic roundoff, not unmodelled obstacle acceleration.
    rounding=16*jnp.finfo(obstacles.dtype).eps*(1+jnp.abs(obstacles[:,:2])+jnp.abs(anchor))/elapsed
    increment_bound=.3*noise[3]/elapsed+rounding
    sensor_bound=jnp.broadcast_to(1.15*noise[4],slope.shape)
    low=jnp.maximum(slope-increment_bound,obstacles[:,3:5]-sensor_bound)
    high=jnp.minimum(slope+increment_bound,obstacles[:,3:5]+sensor_bound)
    inconsistent=mask&(lag>0)&jnp.any(low>high,axis=-1)
    use=mask&(lag>0)&~inconsistent
    estimate=jnp.where(use[:,None],(low+high)/2,obstacles[:,3:5])
    bound=jnp.where(use[:,None],(high-low)/2,sensor_bound)
    estimated=obstacles.at[:,3:5].set(jnp.where(mask[:,None],estimate,0.))
    # Inactive entries have no object or uncertainty and retain neutral padding.
    estimated=jnp.where(mask[:,None],estimated,0.)
    return MotionState(current,previous,state.ticks+1),estimated,jnp.where(mask[:,None],bound,0.),lag,inconsistent


def replay(obstacles,mask,noise,dt,window=32):
    """Offline replay of the exact causal update, not bidirectional smoothing."""
    def step(state,value):
        state,estimate,bound,lag,bad=update(state,value,mask,noise,dt,window)
        return state,dict(obstacles=estimate,bound=bound,lag=lag,inconsistent=bad,
                          current_anchor=state.current_anchor,previous_anchor=state.previous_anchor,ticks=state.ticks)
    return jax.lax.scan(step,initialize(obstacles[0]),obstacles)


def effective_noise(raw_noise,bounds,mask):
    """Conservative scalar velocity-error feature; all other bounds unchanged."""
    maximum=jnp.max(jnp.where(mask[:,None],bounds,0.))
    return raw_noise.at[4].set(maximum/1.15)
