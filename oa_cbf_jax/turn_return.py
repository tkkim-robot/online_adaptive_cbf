"""Bounded two-stage nominal previews; approximate geometry, not safety checks.

First turn toward an offset bearing, then return toward the original bearing.
Speed tracks one of the same four cruise fractions for both stages. Actual CBF
actions remain recomputed by the ordinary controller and its short validation.
"""
import jax.numpy as jnp


def preview_turn_return(x,bearing,cruise,config,horizon):
    # No straight candidate is needed: the original bank already includes it.
    offsets=jnp.asarray([-2.4,-1.4,-.7,.7,1.4,2.4],x.dtype)
    offsets=jnp.tile(offsets,4)
    speeds=cruise*jnp.repeat(jnp.asarray([1.,.6,.25,0.],x.dtype),6)
    initial_headings=bearing+offsets
    from .guidance import _discrete_errors
    dt=horizon/24
    k=jnp.arange(24,dtype=x.dtype)[:,None]
    first_error=jnp.arctan2(jnp.sin(initial_headings-x[2]),jnp.cos(initial_headings-x[2]))
    first_heading=x[2]+first_error-_discrete_errors(first_error,config.w_max,dt,jnp.minimum(k,12))
    switch_heading=x[2]+first_error-_discrete_errors(first_error,config.w_max,dt,jnp.asarray(12,x.dtype))
    second_error=jnp.arctan2(jnp.sin(bearing-switch_heading),jnp.cos(bearing-switch_heading))
    second_residual=_discrete_errors(second_error,config.w_max,dt,jnp.maximum(k-12,0))
    second_heading=switch_heading+second_error-second_residual
    headings=jnp.where(k<12,first_heading,second_heading)
    errors=jnp.where(k<12,_discrete_errors(first_error,config.w_max,dt,k),second_residual)
    turn=jnp.clip(2*errors,-config.w_max,config.w_max)
    speed_error=_discrete_errors(speeds-x[3],config.a_max,dt,k)
    velocity=speeds-speed_error
    acceleration=jnp.clip(2*speed_error,-config.a_max,config.a_max)
    midpoint_heading=headings+.5*dt*turn
    midpoint_speed=velocity+.5*dt*acceleration
    increments=dt*midpoint_speed[...,None]*jnp.stack((jnp.cos(midpoint_heading),jnp.sin(midpoint_heading)),axis=-1)
    positions=x[:2]+jnp.cumsum(increments,axis=0)
    return positions,initial_headings,speeds,offsets,jnp.stack((acceleration,turn),axis=-1)
