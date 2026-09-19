"""Observed, route-following nominal preview; never a safety certificate.

Unlike the constant-heading bank, this preview updates the route target and
terminal braking at each sample. It contains no CBF solve, gain search, hidden
physical state, sensor oracle, or recursive call to preview guidance. The actual
controller still applies its learned gates, CBF validation and hard input bounds.
"""

import jax
import jax.numpy as jnp
from .routing import route_nominal


def preview_route(x,goal,obstacles,mask,points,route_mask,progress,config,horizon):
    dt=horizon/24
    progress=jnp.asarray(progress,x.dtype)
    radius=obstacles[:,2]+config.radius+config.clearance_buffer
    initial_gap=jnp.min(jnp.where(mask,jnp.linalg.norm(x[:2]-obstacles[:,:2],axis=-1)-radius,jnp.inf))

    def step(carry,k):
        state,cursor,minimum=carry
        reference,updated,_,_=route_nominal(state,goal,points,route_mask,cursor,config)
        # Bound the approximate action, not the resulting physical plant state.
        acceleration=jnp.clip(reference[0],jnp.maximum(-config.a_max,-state[3]/dt),
                              jnp.minimum(config.a_max,(config.v_max-state[3])/dt))
        turn=jnp.clip(reference[1],-config.w_max,config.w_max)
        angle=state[2]+.5*dt*turn;speed=state[3]+.5*dt*acceleration
        new=state.at[:2].add(dt*speed*jnp.stack((jnp.cos(angle),jnp.sin(angle))))
        new=new.at[2].add(dt*turn).at[3].add(dt*acceleration)
        # Exact relative line-segment clearance for this approximate midpoint
        # trajectory. Checking only sample endpoints misses fast crossing disks.
        start=state[:2]-obstacles[:,:2]-k*dt*obstacles[:,3:5]
        end=new[:2]-obstacles[:,:2]-(k+1)*dt*obstacles[:,3:5]
        displacement=end-start
        fraction=jnp.clip(-jnp.sum(start*displacement,axis=-1)/jnp.maximum(jnp.sum(displacement**2,axis=-1),1e-12),0.,1.)
        gap=jnp.linalg.norm(start+fraction[:,None]*displacement,axis=-1)-radius
        minimum=jnp.minimum(minimum,jnp.min(jnp.where(mask,gap,jnp.inf)))
        return (new,updated,minimum),(new,jnp.stack((acceleration,turn)))

    (final,cursor,minimum),trace=jax.lax.scan(step,(x,progress,initial_gap),jnp.arange(24))
    return final,cursor,minimum,trace


def route_preview_reference(x,goal,obstacles,mask,points,route_mask,progress,config,
                            reference,bank_reference,clearance_credit=0.):
    final,cursor,minimum,_=preview_route(x,goal,obstacles,mask,points,route_mask,progress,config,config.guidance_horizon)
    radius=obstacles[:,2]+config.radius+config.clearance_buffer
    gap=jnp.min(jnp.where(mask,jnp.linalg.norm(x[:2]-obstacles[:,:2],axis=-1)-radius,jnp.inf))
    preferred=jnp.minimum(jnp.maximum(.2-clearance_credit,0.),jnp.maximum(gap-.02,0.))
    reached=(jnp.linalg.norm(final[:2]-goal)<=config.goal_tolerance)&(jnp.abs(final[3])<=.2)
    usable=(minimum>=preferred)&((cursor>progress+.01)|reached)
    return jnp.where(usable,reference,bank_reference)
