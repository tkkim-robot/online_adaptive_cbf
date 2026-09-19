"""Causal innovation filtering for fixed-bias, constant-velocity obstacle tracks.

The interval encloses physical position PLUS its unknown constant sensor bias.
It does not estimate or remove that bias. The caller must retain the original
geometric uncertainty allowance. Velocity intervals must enclose the constant
physical velocity; fixed track identities and bounded innovations are assumed.
Use requires an explicit matching controller/data contract and copied memory.
The original geometric uncertainty allowance must remain in the safety layer.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class PositionState(NamedTuple):
    center: jax.Array
    halfwidth: jax.Array
    ready: jax.Array


def initialize(obstacles):
    return PositionState(obstacles[:,:2],jnp.zeros_like(obstacles[:,:2]),
                         jnp.zeros(obstacles.shape[0],bool))


def update(memory,raw,velocity,velocity_error,mask,noise,dt):
    """Intersect a propagated biased-position interval with the latest reading.

    Empty intersections reset only the contradictory coordinate and raise a
    flag. First readings receive the full innovation width even if a synthetic
    generator happens to emit no initial innovation. No physical state is reset.
    """
    reading=raw[:,:2]
    prediction=memory.center+dt*velocity
    rounding=8*jnp.finfo(raw.dtype).eps*(1+jnp.abs(prediction)+jnp.abs(reading))
    half=memory.halfwidth+dt*velocity_error+rounding
    innovation=.15*noise[3]+rounding
    lower=jnp.maximum(prediction-half,reading-innovation)
    upper=jnp.minimum(prediction+half,reading+innovation)
    inconsistent=(lower>upper)&memory.ready[:,None]&mask[:,None]
    usable=memory.ready[:,None]&~inconsistent
    center=jnp.where(usable,(lower+upper)/2,reading)
    width=jnp.where(usable,(upper-lower)/2,innovation)
    # Fixed identity/mask tracks are required; absent rows do not affect output.
    center=jnp.where(mask[:,None],center,memory.center)
    width=jnp.where(mask[:,None],width,memory.halfwidth)
    memory=PositionState(center,width,memory.ready|mask)
    filtered=raw.at[:,:2].set(jnp.where(mask[:,None],center,reading))
    return memory,filtered,inconsistent


def replay(raw,velocity,velocity_error,mask,noise,dt):
    def step(memory,values):
        memory,filtered,inconsistent=update(memory,*values,mask,noise,dt)
        return memory,(filtered,memory.halfwidth,inconsistent)
    return jax.lax.scan(step,initialize(raw[0]),(raw,velocity,velocity_error))
