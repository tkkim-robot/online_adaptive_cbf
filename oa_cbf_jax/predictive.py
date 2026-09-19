"""Nonlearned gain-search baseline with the same route/plant/CBF as fixed gains.

This is a computational reference for the learned shortlist, not OA-CBF itself.
Candidates branch from one copied controller state. No candidate can mutate the
applied gain, route cursor, plant, or another candidate's solver state.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp

from .config import UnicycleConfig
from .dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance
from .route_control import rollout_route, route_control, INADMISSIBLE
from .routing import route_target
from .simulation import Summary, RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT

PREDICTIVE_REJECTED = 6


@dataclass(frozen=True)
class SearchConfig:
    horizon: int = 40
    interval: int = 4
    gain_change_penalty: float = .01
    clearance_reward: float = .05

    def __post_init__(self):
        if self.horizon < 1 or self.interval < 1 or self.interval > self.horizon:
            raise ValueError('Invalid predictive horizon or interval')


class Selection(NamedTuple):
    gains: jax.Array
    accepted: jax.Array
    index: jax.Array
    scores: jax.Array
    eligible: jax.Array
    statuses: jax.Array
    survival_steps: jax.Array


def select_gain(x, goal, obstacles, mask, candidates, points, route_mask, progress, previous_gain,
                config=UnicycleConfig(), search=SearchConfig()):
    # The previously committed gain is explicitly retained in the candidate set.
    gains = jnp.concatenate((candidates, previous_gain[None]))
    summaries, traces = jax.vmap(lambda g: rollout_route(x, goal, obstacles, mask, g, points, route_mask,
                                 progress, config=config, steps=search.horizon))(gains)
    _, _, initial_remaining = route_target(x, points, route_mask, progress, config)
    def final_remaining(state, cursor):
        return route_target(state, points, route_mask, cursor, config)[2]
    remaining = jax.vmap(final_remaining)(summaries.final_state, traces['route_progress'][:, -1])
    distance_progress = initial_remaining - remaining
    # Completion beats equal path progress without a stop. For already completed
    # branches, count time to goal; never reward their frozen trailing states.
    completion = (summaries.status == GOAL).astype(x.dtype)
    speed = distance_progress / (search.horizon * config.dt)
    clearance = jnp.minimum(summaries.min_clearance, .5) / .5
    change = jnp.sum((jnp.log(gains) - jnp.log(previous_gain))**2, axis=-1)
    scores = speed + .1 * completion + search.clearance_reward * clearance - search.gain_change_penalty * change
    eligible = ((summaries.status == TIMEOUT) | (summaries.status == GOAL)) & (summaries.min_clearance > 0)
    index = jnp.argmax(jnp.where(eligible, scores, -jnp.inf))
    accepted = jnp.any(eligible)
    return Selection(jnp.where(accepted, gains[index], previous_gain), accepted, index,
                     scores, eligible, summaries.status, summaries.steps)


@partial(jax.jit, static_argnames=('config', 'search', 'steps'))
def rollout_search(x0, goal, obstacles, mask, candidates, points, route_mask,
                   initial_gain=jnp.array([2., 2.]), route_progress=0., config=UnicycleConfig(),
                   search=SearchConfig(), steps=800):
    initial_clearance = jnp.min(signed_clearance(x0[:2], obstacles, mask, config.radius))
    at_goal = (jnp.linalg.norm(x0[:2] - goal) <= config.goal_tolerance) & (jnp.abs(x0[3]) <= .2)
    initial_status = jnp.where(initial_clearance <= 0, COLLISION, jnp.where(at_goal, GOAL, RUNNING))
    def tick(carry, k):
        x, status, count, clearance, psi_min, violation_max, progress, gain = carry
        active = status == RUNNING
        now = k * config.dt
        current_obs = obstacles.at[:, :2].set(obstacles[:, :2] + now * obstacles[:, 3:5])
        def propose(_):
            result = select_gain(x, goal, current_obs, mask, candidates, points, route_mask, progress, gain, config, search)
            return result.gains, result.accepted
        # Keep this condition scalar across a vmapped batch. Including `active`
        # makes it batch-dependent, causing vmap to evaluate both branches on
        # every tick and accidentally spending 4x the intended search compute.
        proposed_gain, accepted = jax.lax.cond(k % search.interval == 0, propose,
                                               lambda _: (gain, jnp.asarray(True)), operand=None)
        qp, h, psi, proposed_progress, remaining, target = route_control(
            x, goal, current_obs, mask, proposed_gain, points, route_mask, progress, config)
        admissible = (h >= -config.qp_tolerance) & (psi >= -config.qp_tolerance)
        can_step = active & accepted & qp.feasible & admissible
        u = jnp.where(can_step, qp.control, jnp.zeros(2, x.dtype))
        y, sub = integrate_unicycle(x, u, config.dt, config.integration_substeps)
        starts = jnp.concatenate((x[None], sub[:-1]))
        times = now + jnp.arange(config.integration_substeps) * config.dt / config.integration_substeps
        swept = jax.vmap(lambda a, b, t: swept_disk_clearance(a, b, obstacles, mask, config.radius,
                          t, t + config.dt/config.integration_substeps))(starts, sub, times)
        step_clearance = jnp.min(swept)
        reached = can_step & (jnp.linalg.norm(y[:2] - goal) <= config.goal_tolerance) & (jnp.abs(y[3]) <= .2)
        status = jnp.where(active & ~accepted, PREDICTIVE_REJECTED, status)
        status = jnp.where(active & accepted & ~qp.feasible, INFEASIBLE, status)
        status = jnp.where(active & accepted & ~admissible, INADMISSIBLE, status)
        status = jnp.where(reached, GOAL, status)
        status = jnp.where(can_step & (step_clearance <= 0), COLLISION, status)
        new_x = jnp.where(can_step, y, x)
        progress = jnp.where(can_step, proposed_progress, progress)
        gain = jnp.where(can_step, proposed_gain, gain)
        count += can_step.astype(jnp.int32)
        clearance = jnp.minimum(clearance, jnp.where(can_step, step_clearance, jnp.inf))
        psi_min = jnp.minimum(psi_min, jnp.where(active & accepted, psi, jnp.inf))
        violation_max = jnp.maximum(violation_max, jnp.where(can_step, qp.max_violation, -jnp.inf))
        trace = dict(state=new_x, control=u, active=can_step, status=status, gains=gain,
                     clearance=jnp.where(can_step, step_clearance, jnp.nan),
                     qp_violation=jnp.where(can_step, qp.max_violation, jnp.nan),
                     psi1=jnp.where(active & accepted, psi, jnp.nan), route_progress=progress,
                     route_remaining=remaining, route_target=target, selection_accepted=accepted)
        return (new_x, status, count, clearance, psi_min, violation_max, progress, gain), trace
    initial = (x0, initial_status, jnp.int32(0), initial_clearance, jnp.asarray(jnp.inf, x0.dtype),
               jnp.asarray(-jnp.inf, x0.dtype), jnp.asarray(route_progress, x0.dtype), jnp.asarray(initial_gain, x0.dtype))
    (x, status, count, clearance, psi, violation, _, _), trace = jax.lax.scan(tick, initial, jnp.arange(steps))
    status = jnp.where(status == RUNNING, TIMEOUT, status)
    progress = jnp.linalg.norm(x0[:2]-goal)-jnp.linalg.norm(x[:2]-goal)
    return Summary(x, status, count, clearance, psi, progress, violation), trace
