"""Shared quad3d routing implementation."""

from dataclasses import dataclass

import numpy as np

import jax.numpy as jnp

from .routing import plan_route, route_geometry, route_target_from_position

from .quad3d_control import Quad3DControlConfig

@dataclass(frozen=True)
class RouteBody:
    radius: float
    clearance_buffer: float

def observed_route(x, goal, obstacles, mask, config=Quad3DControlConfig()):
    # Explicit adapter: never interpret pitch as speed or altitude as heading.
    return plan_route(np.asarray(x)[:2], np.asarray(goal)[:2], obstacles, mask,
        RouteBody(config.robot.radius, config.clearance_buffer), capacity=64,
        vertices=16, margin=.25, visibility_batch_nodes=32)

def flight_target(x, goal, obstacles, mask, points, route_mask, cursor, config=Quad3DControlConfig()):
    _, proposed, remaining = route_target_from_position(x[:2], jnp.linalg.norm(x[6:8]), points, route_mask, cursor)
    vectors, valid, lengths, cumulative = route_geometry(points, route_mask)
    # Four-pole feedback has steady speed w*position_lookahead/4. The long
    # lookahead is reduced by *observed* static-disk visibility, not by future
    # success/failure. This avoids the slow .45m lookahead intended for a
    # different, second-order controller. All constants have explicit precision.
    distances = jnp.asarray(np.linspace(.2, 4*config.cruise_speed/config.horizontal_frequency, 16), x.dtype)
    desired = jnp.minimum(proposed+distances, cumulative[-1])
    index = jnp.argmax(valid[None, :] & (cumulative[None, 1:] >= desired[:, None]-1e-6), axis=1)
    fraction = jnp.clip((desired-cumulative[index])/jnp.maximum(lengths[index], 1e-12), 0., 1.)
    targets = points[index]+fraction[:, None]*vectors[index]
    delta = targets-x[:2]
    relative = obstacles[None, :, :2]-x[:2]
    t = jnp.clip(jnp.sum(relative*delta[:, None], axis=-1)/jnp.maximum(jnp.sum(delta**2, axis=-1)[:, None], 1e-24), 0., 1.)
    closest = x[:2]+t[..., None]*delta[:, None]
    radius = obstacles[:, 2]+jnp.asarray(np.asarray(config.robot.radius+config.clearance_buffer, np.float64), x.dtype)
    static = mask & (jnp.linalg.norm(obstacles[:, 3:5], axis=-1)<1e-10)
    visible = jnp.all(jnp.where(static[None], jnp.linalg.norm(closest-obstacles[None, :, :2], axis=-1)-radius >= -1e-9, True), axis=-1)
    chosen = jnp.max(jnp.where(visible, jnp.arange(16), 0))
    # If no chord is visible, request only the nearest route target and let the
    # hard CBF reject/modify it. Never invent a successful route or move state.
    target = jnp.concatenate((targets[chosen], goal[2:3]))
    return target, proposed, remaining, jnp.any(visible)

def numpy_flight_target(x, goal, obstacles, mask, points, route_mask, cursor, config):
    """Independent scalar NumPy reconstruction for trace auditing."""
    x, goal, o, points = map(lambda v: np.asarray(v, np.float64), (x, goal, obstacles, points))
    valid=np.asarray(route_mask, bool)[:-1]&np.asarray(route_mask, bool)[1:]
    vectors=np.diff(points, axis=0);lengths=np.where(valid,np.linalg.norm(vectors,axis=1),0.)
    cumulative=np.r_[0.,np.cumsum(lengths)]
    fractions=np.clip(np.sum((x[:2]-points[:-1])*vectors,axis=1)/np.maximum(lengths**2,1e-12),0.,1.)
    projections=cumulative[:-1]+fractions*lengths
    eligible=valid&(projections>=cursor-.05)&(projections<=cursor+1.)
    nearest=points[:-1]+fractions[:,None]*vectors
    chosen=int(np.argmin(np.where(eligible,np.sum((nearest-x[:2])**2,axis=1),np.inf)))
    proposed=max(cursor,projections[chosen]) if eligible.any() else cursor
    target_list=[];visibility=[]
    static=np.asarray(mask,bool)&(np.linalg.norm(o[:,3:5],axis=1)<1e-10)
    for offset in np.linspace(.2,4*config.cruise_speed/config.horizontal_frequency,16):
        arc=min(proposed+offset,cumulative[-1]);segment=int(np.argmax(valid&(cumulative[1:]>=arc-1e-6)))
        t=np.clip((arc-cumulative[segment])/max(lengths[segment],1e-12),0.,1.)
        target=points[segment]+t*vectors[segment];target_list.append(target)
        direction=target-x[:2];obstacles=o[static]
        t=np.clip((obstacles[:,:2]-x[:2])@direction/max(np.dot(direction,direction),1e-24),0.,1.)
        clearance=np.linalg.norm(x[:2]+t[:,None]*direction-obstacles[:,:2],axis=1)-obstacles[:,2]-config.robot.radius-config.clearance_buffer
        ok=bool(np.all(clearance>=-1e-9))
        visibility.append(ok)
    selected=max([i for i,ok in enumerate(visibility) if ok],default=0)
    return np.r_[target_list[selected],goal[2]],proposed,max(cumulative[-1]-proposed,0.),any(visibility)
