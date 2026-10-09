"""Shared routing implementation."""

from dataclasses import dataclass

from concurrent.futures import ThreadPoolExecutor

import heapq

import time

import numpy as np

import jax.numpy as jnp

from .config import UnicycleConfig

@dataclass
class Route:
    points: np.ndarray
    mask: np.ndarray
    length: float
    planning_margin: float
    status: str

def plan_routes(scenes,robot=UnicycleConfig(),*,workers=1,**options):
    """Plan independent scenes in input order; propagate every planner failure.

    Threads share no route state. Timings describe individual calls, including
    contention, and must not be summed as a parallel batch's wall time.
    """
    if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
        raise ValueError('Route workers must be a positive integer')
    def one(scene):
        start=time.perf_counter()
        route=plan_route(scene.initial_state[:2],scene.goal,scene.obstacles,scene.obstacle_mask,robot,**options)
        return route,time.perf_counter()-start
    if workers==1:result=[one(scene) for scene in scenes]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:result=list(pool.map(one,scenes))
    return [r for r,_ in result],[t for _,t in result]

def segment_clearances(starts, ends, centers, radii):
    """NumPy exact line-segment/disk clearance with arbitrary leading axes."""
    delta = ends - starts
    relative = centers - starts[..., None, :]
    denominator = np.sum(delta * delta, axis=-1)
    fraction = np.clip(np.sum(relative * delta[..., None, :], axis=-1) /
                       np.maximum(denominator[..., None], 1e-24), 0., 1.)
    closest = starts[..., None, :] + fraction[..., None] * delta[..., None, :]
    return np.linalg.norm(closest - centers, axis=-1) - radii

def visibility_weights(nodes, centers, radii, batch_nodes=32):
    """Same complete visibility graph with bounded temporary geometry arrays.

    Every edge is checked against every disk; batching changes only memory use.
    ``None`` retains the dense expression for independent equivalence/timing.
    """
    if batch_nodes is not None and (isinstance(batch_nodes,bool) or not isinstance(batch_nodes,int) or batch_nodes<1):
        raise ValueError('Visibility batch must be a positive integer or None')
    weights=np.linalg.norm(nodes[:,None]-nodes[None],axis=-1)
    batch=len(nodes) if batch_nodes is None else batch_nodes
    for start in range(0,len(nodes),batch):
        stop=min(start+batch,len(nodes))
        visible=np.all(segment_clearances(nodes[start:stop,None],nodes[None],centers,radii)>=-1e-9,axis=-1)
        weights[start:stop]=np.where(visible,weights[start:stop],np.inf)
    np.fill_diagonal(weights,np.inf)
    return weights

def plan_route(start, goal, obstacles, mask, robot=UnicycleConfig(), *,
               capacity=32, vertices=16, margin=.25, visibility_batch_nodes=None):
    start, goal, obs = np.asarray(start, float), np.asarray(goal, float), np.asarray(obstacles, float)
    mask = np.asarray(mask, bool)
    if (start.shape != (2,) or goal.shape != (2,) or obs.ndim != 2 or obs.shape[1] != 5
            or mask.shape != obs.shape[:1] or capacity < 2 or vertices < 8 or margin < 0
            or not np.isfinite(np.r_[start, goal, obs.ravel(), margin]).all() or np.any(obs[mask, 2] < 0)):
        raise ValueError('Invalid route inputs')
    static = mask & (np.linalg.norm(obs[:, 3:5], axis=1) < 1e-10)
    centers = obs[static, :2]
    base = obs[static, 2] + robot.radius + robot.clearance_buffer
    effective_margin = float(margin)
    def package(points, status):
        if len(points) > capacity:
            raise ValueError(f'Route capacity exceeded: {len(points)} > {capacity}; truncation forbidden')
        padded = np.broadcast_to(goal, (capacity, 2)).copy()
        padded[:len(points)] = points
        return Route(padded, np.arange(capacity) < len(points),
                     float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), effective_margin, status)
    if len(base) == 0:
        return package(np.stack((start, goal)), 'ready')
    endpoint_slack = np.min(np.linalg.norm(np.stack((start, goal))[:, None] - centers, axis=-1) - base)
    if endpoint_slack <= 0:
        return package(np.stack((start, goal)), 'endpoint_inside_planning_obstacle')
    # Reduce only optional route padding if an endpoint is in that extra band.
    # The physical radius and common CBF buffer are never reduced.
    effective_margin = min(effective_margin, max(0., float(endpoint_slack) - 1e-4))
    radii = base + effective_margin
    if np.all(segment_clearances(start, goal, centers, radii) >= 0):
        return package(np.stack((start, goal)), 'ready')
    angles = np.arange(vertices) * (2 * np.pi / vertices)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    # Circumscribed polygon edges also clear the disk (inscribed vertices would
    # produce colliding chords). Overlapping disks share the same visibility test.
    nodes = (centers[:, None] + (radii[:, None, None] / np.cos(np.pi / vertices) + 1e-5)
             * directions).reshape(-1, 2)
    valid = np.all(np.linalg.norm(nodes[:, None] - centers, axis=-1) >= radii - 1e-9, axis=1)
    nodes = np.concatenate((np.stack((start, goal)), nodes[valid]))
    weights = visibility_weights(nodes,centers,radii,visibility_batch_nodes)
    costs = np.full(len(nodes), np.inf); costs[0] = 0.
    parents = np.full(len(nodes), -1, dtype=int)
    queue = [(0., 0)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost > costs[i]:
            continue
        if i == 1:
            break
        for j in np.flatnonzero(np.isfinite(weights[i])):
            next_cost = cost + weights[i, j]
            if next_cost < costs[j]:
                costs[j], parents[j] = next_cost, i
                heapq.heappush(queue, (next_cost, int(j)))
    if not np.isfinite(costs[1]):
        return package(np.stack((start, goal)), 'no_visibility_path')
    indices = [1]
    while indices[-1] != 0:
        indices.append(int(parents[indices[-1]]))
    return package(nodes[indices[::-1]], 'ready')

def route_geometry(points, mask):
    vectors = points[1:] - points[:-1]
    valid = mask[1:] & mask[:-1]
    lengths = jnp.where(valid, jnp.linalg.norm(vectors, axis=-1), 0.)
    cumulative = jnp.concatenate((jnp.zeros(1, points.dtype), jnp.cumsum(lengths)))
    return vectors, valid, lengths, cumulative

def physical_route_coordinate(position, points, mask, cursor_hint):
    """Ground-truth arclength metric near the committed route branch.

    Unlike the monotone controller cursor, this metric can decrease when the
    physical robot moves backward. The cursor identifies the branch at nearby
    parallel route segments; latent sensor errors do not become model features.
    """
    vectors,valid,lengths,cumulative=route_geometry(points,mask)
    low=jnp.maximum(cumulative[:-1],cursor_hint-1.)
    high=jnp.minimum(cumulative[1:],cursor_hint+1.)
    eligible=valid&(low<=high)
    fraction=jnp.sum((position-points[:-1])*vectors,axis=-1)/jnp.maximum(lengths**2,1e-12)
    coordinate=jnp.clip(cumulative[:-1]+fraction*lengths,low,jnp.maximum(low,high))
    projected=points[:-1]+((coordinate-cumulative[:-1])/jnp.maximum(lengths,1e-12))[:,None]*vectors
    index=jnp.argmin(jnp.where(eligible,jnp.sum((projected-position)**2,axis=-1),jnp.inf))
    return coordinate[index]

def route_target(x, points, mask, progress, config=UnicycleConfig()):
    return route_target_from_position(x[:2],x[3],points,mask,progress)

def route_target_from_position(position,speed,points,mask,progress):
    """Track continuous arclength without jumping to a distant route branch.

    Progress is controller memory. Candidate rollouts receive a copy; only an
    applied action may commit its updated progress. Padded nodes never count.
    """
    vectors, valid, lengths, cumulative = route_geometry(points, mask)
    fractions = jnp.clip(jnp.sum((position - points[:-1]) * vectors, axis=-1) /
                         jnp.maximum(lengths ** 2, 1e-12), 0., 1.)
    projected = cumulative[:-1] + fractions * lengths
    # A one-metre search window handles short polygon segments but prevents
    # global nearest-point jumps across a U-shaped route.
    eligible = valid & (projected >= progress - .05) & (projected <= progress + 1.)
    nearest = points[:-1] + fractions[:, None] * vectors
    distances = jnp.where(eligible, jnp.sum((nearest - position)**2, axis=-1), jnp.inf)
    # JAX0.8.2's vmapped argmin initializes the value reduction in default32
    # under explicit-only64 mode. Boolean first-min selection preserves the
    # same tie rule and supports an explicitly64 reference/feature path.
    index = jnp.argmax(distances == jnp.min(distances)) if distances.dtype == jnp.float64 else jnp.argmin(distances)
    updated = jnp.where(jnp.any(eligible), jnp.maximum(progress, projected[index]), progress)
    lookahead = jnp.clip(.45 + .65 * speed, .35, 1.1)
    desired = jnp.minimum(updated + lookahead, cumulative[-1])
    target_index = jnp.argmax(valid & (cumulative[1:] >= desired - 1e-6))
    fraction = jnp.clip((desired - cumulative[target_index]) / jnp.maximum(lengths[target_index], 1e-12), 0., 1.)
    target = points[target_index] + fraction * vectors[target_index]
    return target, updated, jnp.maximum(cumulative[-1] - updated, 0.)

def route_nominal(x, goal, points, mask, progress, config=UnicycleConfig()):
    target, updated, remaining = route_target(x, points, mask, progress, config)
    delta = target - x[:2]
    bearing = jnp.arctan2(delta[1], delta[0]) - x[2]
    error = jnp.arctan2(jnp.sin(bearing), jnp.cos(bearing))
    curvature = 2 * jnp.sin(error) / jnp.maximum(jnp.linalg.norm(delta), .15)
    # Pure-pursuit curvature gives a speed cap compatible with turn authority.
    # Heading correction at zero speed permits turns without synthetic movement.
    corner_speed = .8 * config.w_max / jnp.maximum(jnp.abs(curvature), .01)
    stopping_distance = jnp.maximum(jnp.linalg.norm(goal - x[:2]) - .1, 0.)
    desired_speed = jnp.minimum(config.v_max, jnp.minimum(corner_speed,
                      jnp.minimum(1.2 * stopping_distance, jnp.sqrt(2 * config.a_max * stopping_distance))))
    desired_speed *= jnp.maximum(jnp.cos(error), 0.)
    reference = jnp.stack((2 * (desired_speed - x[3]), 2 * error))
    return reference, updated, remaining, target
