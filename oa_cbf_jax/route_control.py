"""Shared route control implementation."""

from functools import partial

import jax

import jax.numpy as jnp

from .config import UnicycleConfig

from .controllers import unicycle_cbf_qp, solve_qp2

from .dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance

from .routing import route_nominal

from .guidance import preview_guidance

from .guidance import route_preview_reference

from .simulation import Summary, RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT

INADMISSIBLE = 5

def route_problem(x, goal, obstacles, mask, alpha, points, route_mask, progress, config=UnicycleConfig(), speed_uncertainty=0., clearance_uncertainty=0.,margin_guidance=False,shared_clearance_budget=False):
    """Shared nominal command, joint constraints and route update before solving."""
    reference, updated, remaining, target = route_nominal(x, goal, points, route_mask, progress, config)
    if config.guidance_detour:
        from .detour import detour_reference
        guide_obs=obstacles.at[:,2].add(clearance_uncertainty) if margin_guidance else obstacles
        reference,target=detour_reference(x,goal,reference,target,points,route_mask,updated,guide_obs,mask,config,
            clearance_uncertainty if shared_clearance_budget else 0.)
    if config.guidance_horizon>0:
        speed_limit=None
        if config.guidance_goal_braking:
            # Same terminal-speed envelope as route_nominal, now also applied
            # to the guidance cruise floor. This is a shared nominal preference;
            # it neither alters hard CBF constraints nor clips the plant state.
            distance=jnp.maximum(jnp.linalg.norm(goal-x[:2])-.1,0.)
            speed_limit=jnp.minimum(1.2*distance,jnp.sqrt(2*config.a_max*distance))
        guide_obs=obstacles.at[:,2].add(clearance_uncertainty) if margin_guidance else obstacles
        bank_reference=preview_guidance(x,reference,guide_obs,mask,config,config.guidance_horizon,speed_limit,margin_guidance,
            clearance_uncertainty if shared_clearance_budget else 0.)
        reference=route_preview_reference(x,goal,guide_obs,mask,points,route_mask,progress,config,
            reference,bank_reference,clearance_uncertainty if shared_clearance_budget else 0.) if config.guidance_route_preview else bank_reference
    _, a, b, h, psi = unicycle_cbf_qp(x, goal, obstacles, mask, alpha, config, clearance_uncertainty)
    # If speed is observed with a known bounded error, enforce next-step speed
    # bounds for every physical speed consistent with that observation. This
    # tightens input bounds; it never clips or projects the simulated plant.
    possible_low=jnp.maximum(0.,x[3]-speed_uncertainty)
    possible_high=jnp.minimum(config.v_max,x[3]+speed_uncertainty)
    b=b.at[-4].set(jnp.minimum(b[-4],(config.v_max-possible_high)/config.dt))
    b=b.at[-3].set(jnp.minimum(b[-3],possible_low/config.dt))
    return reference, a, b, h, psi, updated, remaining, target

def route_control(x, goal, obstacles, mask, alpha, points, route_mask, progress, config=UnicycleConfig(), speed_uncertainty=0., clearance_uncertainty=0.,margin_guidance=False,shared_clearance_budget=False):
    reference, a, b, h, psi, updated, remaining, target = route_problem(
        x, goal, obstacles, mask, alpha, points, route_mask, progress, config, speed_uncertainty, clearance_uncertainty,margin_guidance,shared_clearance_budget)
    qp = solve_qp2(reference, a, b, jnp.ones(2, x.dtype), config.qp_tolerance)
    min_h = jnp.min(jnp.where(mask, h, jnp.inf))
    min_psi = jnp.min(jnp.where(mask, psi, jnp.inf))
    return qp, min_h, min_psi, updated, remaining, target

@partial(jax.jit, static_argnames=('config', 'steps', 'require_admissible','margin_guidance','shared_clearance_budget'))
def rollout_route(x0, goal, obstacles, mask, alpha, points, route_mask,
                  route_progress=0., config=UnicycleConfig(), steps=400, require_admissible=True, speed_uncertainty=0., clearance_uncertainty=0.,margin_guidance=False,shared_clearance_budget=False):
    initial_clearance = jnp.min(signed_clearance(x0[:2], obstacles, mask, config.radius))
    at_goal = (jnp.linalg.norm(x0[:2] - goal) <= config.goal_tolerance) & (jnp.abs(x0[3]) <= .2)
    initial_status = jnp.where(initial_clearance <= 0, COLLISION, jnp.where(at_goal, GOAL, RUNNING))
    def tick(carry, k):
        x, status, count, clearance, psi_min, violation_max, progress = carry
        active = status == RUNNING
        now = k * config.dt
        current_obs = obstacles.at[:, :2].set(obstacles[:, :2] + now * obstacles[:, 3:5])
        qp, h, psi, proposed_progress, remaining, target = route_control(
            x, goal, current_obs, mask, alpha, points, route_mask, progress, config, speed_uncertainty, clearance_uncertainty,margin_guidance,shared_clearance_budget)
        admissible = ((h >= -config.qp_tolerance) & (psi >= -config.qp_tolerance)) | (not require_admissible)
        can_step = active & qp.feasible & admissible
        u = jnp.where(can_step, qp.control, jnp.zeros(2, x.dtype))
        y, sub = integrate_unicycle(x, u, config.dt, config.integration_substeps)
        starts = jnp.concatenate((x[None, :], sub[:-1]))
        times = now + jnp.arange(config.integration_substeps) * config.dt / config.integration_substeps
        swept = jax.vmap(lambda a, b, t: swept_disk_clearance(a, b, obstacles, mask, config.radius,
                          t, t + config.dt / config.integration_substeps))(starts, sub, times)
        step_clearance = jnp.min(swept)
        collided = can_step & (step_clearance <= 0)
        reached = can_step & (jnp.linalg.norm(y[:2] - goal) <= config.goal_tolerance) & (jnp.abs(y[3]) <= .2)
        status = jnp.where(active & ~qp.feasible, INFEASIBLE, status)
        status = jnp.where(active & ~admissible, INADMISSIBLE, status)
        status = jnp.where(reached, GOAL, status)
        status = jnp.where(collided, COLLISION, status)
        new_x = jnp.where(can_step, y, x)
        progress = jnp.where(can_step, proposed_progress, progress)
        count += can_step.astype(jnp.int32)
        clearance = jnp.minimum(clearance, jnp.where(can_step, step_clearance, jnp.inf))
        psi_min = jnp.minimum(psi_min, jnp.where(active, psi, jnp.inf))
        violation_max = jnp.maximum(violation_max, jnp.where(can_step, qp.max_violation, -jnp.inf))
        trace = dict(state=new_x, control=u, active=can_step, status=status,
                     clearance=jnp.where(can_step, step_clearance, jnp.nan),
                     qp_violation=jnp.where(can_step, qp.max_violation, jnp.nan),
                     psi1=jnp.where(active, psi, jnp.nan), route_progress=progress,
                     route_remaining=remaining, route_target=target)
        return (new_x, status, count, clearance, psi_min, violation_max, progress), trace
    initial = (x0, initial_status, jnp.int32(0), initial_clearance, jnp.asarray(jnp.inf, x0.dtype),
               jnp.asarray(-jnp.inf, x0.dtype), jnp.asarray(route_progress, x0.dtype))
    (x, status, count, clearance, psi, violation, _), trace = jax.lax.scan(tick, initial, jnp.arange(steps))
    status = jnp.where(status == RUNNING, TIMEOUT, status)
    progress = jnp.linalg.norm(x0[:2] - goal) - jnp.linalg.norm(x[:2] - goal)
    return Summary(x, status, count, clearance, psi, progress, violation), trace
