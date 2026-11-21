"""Dynamic unicycle benchmark controllers with observed-motion stall and recovery policies."""

from dataclasses import replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .. import motion_observer
from ..adaptive import BACKUP, LEARNED, REJECTED, PREDICTIVE_REJECTED
from ..models import predict_ensemble, route_graph
from ..uncertainty import cs_disagreement, worst_member_cvar
from ..routing import route_geometry, physical_route_coordinate
from ..guidance import preview_route, preview_guidance
from ..config import UnicycleConfig
from ..simulation import Summary
from ..closed_loop import PLANNER_FAILURE
from ..controllers import solve_qp2
from ..dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance
from ..guidance import clearance_inflation
from ..model_release import selector
from ..route_control import INADMISSIBLE, route_problem
from ..simulation import COLLISION, GOAL, INFEASIBLE, RUNNING, TIMEOUT
from ..stochastic import STATE_BOUND_VIOLATION
from .sensors import arguments, unicycle_sensor_model
from .protocol import STEPS


class Memory(NamedTuple):
    anchor: object
    ticks: object
    delta_speed: object
    delta_arc: object
    angle: object
    low_angle: object
    high_angle: object
    low_speed: object
    high_speed: object


def initial(dtype):
    zero = jnp.asarray(0.0, dtype=dtype)
    return Memory(
        jnp.zeros(2, dtype), jnp.int32(0), zero, zero, zero, zero, zero, zero, jnp.asarray(1.0, dtype)
    )


def estimate(m, measured, noise, dt):
    # A bounded acceleration updates the speed interval exactly. With a constant
    # sensor position bias, differences cancel that bias. For a nonnegative-speed
    # trajectory, chord <= path length <= chord/cos(half heading spread).
    # The spread over a window of at most two seconds is < pi for this robot.
    elapsed = jnp.maximum(1, m.ticks) * dt
    chord = jnp.linalg.norm(measured[:2] - m.anchor)
    # The position innovation term is bounded componentwise by 0.15 sigma.
    # Include coordinate-scale FP32 state storage drift over the entire window.
    arithmetic = (1 + jnp.max(jnp.abs(measured[:2]))) * jnp.finfo(measured.dtype).eps * 8 * (m.ticks + 1)
    position_error = 0.3 * jnp.sqrt(2.0) * noise[0] + arithmetic
    cos_half = jnp.cos((m.high_angle - m.low_angle) / 2)
    lower = (jnp.maximum(0.0, chord - position_error) - m.delta_arc) / elapsed + m.delta_speed
    upper = ((chord + position_error) / jnp.maximum(0.1, cos_half) - m.delta_arc) / elapsed + m.delta_speed
    raw_error = 1.15 * noise[2]
    raw_min = jnp.maximum(0.0, measured[3] - raw_error)
    raw_max = jnp.minimum(1.0, measured[3] + raw_error)
    keep = (m.ticks >= 4) & (cos_half > 0.1)
    lo = jnp.maximum(jnp.maximum(m.low_speed, raw_min), jnp.where(keep, lower - 1.0e-4, 0.0))
    hi = jnp.minimum(jnp.minimum(m.high_speed, raw_max), jnp.where(keep, upper + 1.0e-4, 1.0))
    consistent = lo <= hi
    lo = jnp.where(consistent, lo, raw_min)
    hi = jnp.where(consistent, hi, raw_max)
    center = 0.5 * (hi + lo)
    return measured.at[3].set(center), 0.5 * (hi - lo), lo, hi


def update(m, k, measured_xy, command, lo, hi, dt):
    reset = k % 40 == 0
    anchor = jnp.where(reset, measured_xy, m.anchor)
    dv = jnp.where(reset, 0.0, m.delta_speed)
    arc = jnp.where(reset, 0.0, m.delta_arc) + dt * (dv + command[0] * dt / 2)
    angle = jnp.where(reset, 0.0, m.angle) + command[1] * dt
    return Memory(
        anchor,
        jnp.where(reset, 0, m.ticks) + 1,
        dv + command[0] * dt,
        arc,
        angle,
        jnp.minimum(jnp.where(reset, 0.0, m.low_angle), angle),
        jnp.maximum(jnp.where(reset, 0.0, m.high_angle), angle),
        jnp.maximum(0.0, lo + command[0] * dt - 2.0e-6),
        jnp.minimum(1.0, hi + command[0] * dt + 2.0e-6),
    )


def _recovery_qp(reference, a, b, weights, tolerance=1.0e-5):
    # Each actuator's bounds are <=0.5 here; the four original box rows always
    # remain. Include roundoff and the original feasibility tolerance.
    rows = a[:-4]
    may_bind = jnp.sum(jnp.abs(rows), axis=1) * (0.5 + 2.0 * tolerance) >= b[:-4] - 2.0 * tolerance
    size = 6
    ids = jnp.nonzero(may_bind, size=size, fill_value=0)[0]
    count = jnp.sum(may_bind)
    mask = jnp.arange(size) < count
    reduced_a = jnp.concatenate((jnp.where(mask[:, None], rows[ids], 0.0), a[-4:]))
    reduced_b = jnp.concatenate((jnp.where(mask, b[:-4][ids], 1.0), b[-4:]))
    candidate = solve_qp2(reference, reduced_a, reduced_b, weights, tolerance)
    # Recompute the reported violation against every original constraint.
    actual = jnp.max(jnp.matmul(a, candidate.control, precision="highest") - b)
    result = candidate._replace(max_violation=jnp.where(candidate.feasible, actual, candidate.max_violation))
    overflow = (count > size) | (candidate.feasible & (actual > tolerance))
    _, result = jax.lax.while_loop(
        lambda pair: pair[0],
        lambda pair: (jnp.asarray(False), solve_qp2(reference, a, b, weights, tolerance)),
        (overflow, result),
    )
    return result


class StallMemory(NamedTuple):
    latched: object
    anchor_cursor: object
    anchor_tick: object
    pending: object
    window_valid: object
    motion_sum: object
    motion_count: object


def _stall_gate(k, active, cursor, goal_distance, position_bound, speed, speed_noise, memory):
    """Latch soft clearance after two consecutive low-motion four-second windows."""
    latched, anchor_cursor, anchor_tick, pending, window_valid, motion_sum, motion_count = memory
    original_finite = (
        (k >= 0)
        & (k < STEPS)
        & (anchor_tick >= 0)
        & (anchor_tick <= k)
        & jnp.isfinite(cursor)
        & jnp.isfinite(anchor_cursor)
        & jnp.isfinite(goal_distance)
        & jnp.isfinite(position_bound)
        & (position_bound >= 0)
    )
    motion_finite = (
        jnp.isfinite(speed)
        & jnp.isfinite(speed_noise)
        & (speed_noise >= 0)
        & jnp.isfinite(motion_sum)
        & (motion_sum >= 0)
        & (motion_count >= 0)
        & (motion_count <= 80)
    )
    domain = original_finite & motion_finite
    robust = jnp.abs(speed) + jnp.asarray(1.15, cursor.dtype) * speed_noise
    sample = active & domain & (k > anchor_tick)
    total = jnp.where(sample, motion_sum + robust, motion_sum)
    count = motion_count + sample.astype(jnp.int32)
    finite = domain & jnp.isfinite(robust) & jnp.isfinite(total)
    due = active & finite & (k - anchor_tick >= 80)
    interval_valid = window_valid & finite & active
    mean = jnp.where((count == 80) & finite, total / jnp.asarray(80, cursor.dtype), jnp.nan)
    stalled = (
        due
        & (cursor - anchor_cursor < 0.25)
        & (goal_distance - position_bound > 0.25)
        & interval_valid
        & (count == 80)
        & finite
        & (mean <= jnp.asarray(0.2, cursor.dtype))
    )
    return StallMemory(
        latched | (pending & stalled),
        jnp.where(due, cursor, anchor_cursor),
        jnp.where(due, k, anchor_tick),
        jnp.where(~finite | ~active, False, jnp.where(due, stalled, pending)),
        jnp.where(due, True, interval_valid),
        jnp.where(due | ~finite, 0.0, total),
        jnp.where(due | ~finite, 0, count),
    )


def _route_preview_usable(
    x,
    goal,
    obstacles,
    mask,
    points,
    route_mask,
    progress,
    config,
    clearance_credit=0.0,
):
    final, cursor, minimum, _ = preview_route(
        x, goal, obstacles, mask, points, route_mask, progress, config, config.guidance_horizon
    )
    radius = obstacles[:, 2] + config.radius + config.clearance_buffer
    gap = jnp.min(jnp.where(mask, jnp.linalg.norm(x[:2] - obstacles[:, :2], axis=-1) - radius, jnp.inf))
    preferred = jnp.minimum(jnp.maximum(0.2 - clearance_credit, 0.0), jnp.maximum(gap - 0.02, 0.0))
    reached = (jnp.linalg.norm(final[:2] - goal) <= config.goal_tolerance) & (jnp.abs(final[3]) <= 0.2)
    terminal = (progress >= route_geometry(points, route_mask)[3][-1] - 0.01) & (
        jnp.linalg.norm(final[:2] - goal) < jnp.linalg.norm(x[:2] - goal) - 0.01
    )
    return (minimum >= preferred) & ((cursor > progress + 0.01) | reached | terminal)


def _control(
    x,
    goal,
    obstacles,
    mask,
    gains,
    points,
    route_mask,
    cursor,
    config,
    speed_uncertainty,
    clearance_uncertainty,
    margin_guidance,
    shared_clearance_budget,
    *,
    recovery=False,
):
    # Obtain the original CBF, speed and actuator rows and the common routed nominal.
    # The guidance overrides affect the nominal command presented to the QP.
    unpreviewed = replace(config, guidance_horizon=0.0)
    reference, a, b, h, psi, updated, remaining, target = route_problem(
        x,
        goal,
        obstacles,
        mask,
        gains,
        points,
        route_mask,
        cursor,
        unpreviewed,
        speed_uncertainty,
        clearance_uncertainty,
        margin_guidance,
        shared_clearance_budget,
    )
    speed_limit = None
    if config.guidance_goal_braking:
        distance = jnp.maximum(jnp.linalg.norm(goal - x[:2]) - 0.1, 0.0)
        speed_limit = jnp.minimum(1.2 * distance, jnp.sqrt(2 * config.a_max * distance))
    guide_obs = obstacles.at[:, 2].add(clearance_uncertainty) if margin_guidance else obstacles
    if recovery:
        guide_obs = guide_obs.at[:, 2].add(0.08)
    credit = clearance_uncertainty if shared_clearance_budget else 0.0

    def bank():
        return preview_guidance(
            x, reference, guide_obs, mask, config, config.guidance_horizon, speed_limit, margin_guidance, credit
        )

    if config.guidance_route_preview:
        use_route = _route_preview_usable(x, goal, guide_obs, mask, points, route_mask, cursor, config, credit)
        # Preserve the route-preview decision while avoiding a second, unused preview.
        # Under vmap the loop skips the bank when all validating candidates use the route.
        _, reference = jax.lax.while_loop(
            lambda value: value[0], lambda _: (jnp.asarray(False), bank()), (~use_route, reference)
        )
    else:
        reference = bank()
    if recovery:
        reference = reference.at[0].set(jnp.minimum(reference[0], 2 * (0.9 - x[3])))
    solve = _recovery_qp if recovery else solve_qp2
    qp = solve(reference, a, b, jnp.ones(2, x.dtype), config.qp_tolerance)
    return (
        qp,
        jnp.min(jnp.where(mask, h, jnp.inf)),
        jnp.min(jnp.where(mask, psi, jnp.inf)),
        updated,
        remaining,
        target,
    )


@partial(
    jax.jit,
    static_argnames=(
        "config",
        "steps",
        "require_admissible",
        "margin_guidance",
        "shared_clearance_budget",
        "recovery",
    ),
)
def _rollout(
    x0,
    goal,
    obstacles,
    mask,
    alpha,
    points,
    route_mask,
    route_progress=0.0,
    config=UnicycleConfig(),
    steps=400,
    require_admissible=True,
    speed_uncertainty=0.0,
    clearance_uncertainty=0.0,
    margin_guidance=False,
    shared_clearance_budget=False,
    recovery=False,
):
    initial_clearance = jnp.min(signed_clearance(x0[:2], obstacles, mask, config.radius))
    at_goal = (jnp.linalg.norm(x0[:2] - goal) <= config.goal_tolerance) & (jnp.abs(x0[3]) <= 0.2)
    initial_status = jnp.where(initial_clearance <= 0, COLLISION, jnp.where(at_goal, GOAL, RUNNING))

    def tick(carry, k):
        x, status, count, clearance, psi_min, violation_max, progress = carry
        active = status == RUNNING
        now = k * config.dt
        current_obs = obstacles.at[:, :2].set(obstacles[:, :2] + now * obstacles[:, 3:5])
        qp, h, psi, proposed_progress, remaining, target = _control(
            x,
            goal,
            current_obs,
            mask,
            alpha,
            points,
            route_mask,
            progress,
            config,
            speed_uncertainty,
            clearance_uncertainty,
            margin_guidance,
            shared_clearance_budget,
            recovery=recovery,
        )
        admissible = ((h >= -config.qp_tolerance) & (psi >= -config.qp_tolerance)) | (not require_admissible)
        can_step = active & qp.feasible & admissible
        u = jnp.where(can_step, qp.control, jnp.zeros(2, x.dtype))
        y, sub = integrate_unicycle(x, u, config.dt, config.integration_substeps)
        starts = jnp.concatenate((x[None, :], sub[:-1]))
        times = now + jnp.arange(config.integration_substeps) * config.dt / config.integration_substeps
        swept = jax.vmap(
            lambda a, b, t: swept_disk_clearance(
                a, b, obstacles, mask, config.radius, t, t + config.dt / config.integration_substeps
            )
        )(starts, sub, times)
        step_clearance = jnp.min(swept)
        collided = can_step & (step_clearance <= 0)
        reached = can_step & (jnp.linalg.norm(y[:2] - goal) <= config.goal_tolerance) & (jnp.abs(y[3]) <= 0.2)
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
        trace = dict(
            state=new_x,
            control=u,
            active=can_step,
            status=status,
            clearance=jnp.where(can_step, step_clearance, jnp.nan),
            qp_violation=jnp.where(can_step, qp.max_violation, jnp.nan),
            psi1=jnp.where(active, psi, jnp.nan),
            route_progress=progress,
            route_remaining=remaining,
            route_target=target,
        )
        return (new_x, status, count, clearance, psi_min, violation_max, progress), trace

    initial = (
        x0,
        initial_status,
        jnp.int32(0),
        initial_clearance,
        jnp.asarray(jnp.inf, x0.dtype),
        jnp.asarray(-jnp.inf, x0.dtype),
        jnp.asarray(route_progress, x0.dtype),
    )
    (x, status, count, clearance, psi, violation, _), trace = jax.lax.scan(tick, initial, jnp.arange(steps))
    status = jnp.where(status == RUNNING, TIMEOUT, status)
    progress = jnp.linalg.norm(x0[:2] - goal) - jnp.linalg.norm(x[:2] - goal)
    return Summary(x, status, count, clearance, psi, progress, violation), trace


def _selector(model, normalization, robot, config, *, recovery=False):
    mean = jnp.asarray(normalization["target_mean"], jnp.float32)
    scale = jnp.asarray(normalization["target_scale"], jnp.float32)

    def validate(x, goal, obs, mask, gains, points, route_mask, cursor, previous_gain, noise):
        result, trace = jax.vmap(
            lambda g: _rollout(
                x,
                goal,
                obs,
                mask,
                g,
                points,
                route_mask,
                cursor,
                config=robot,
                steps=config.validation_horizon,
                speed_uncertainty=1.15 * noise[2],
                clearance_uncertainty=clearance_inflation(noise, config.sensor_margin_scale),
                margin_guidance=config.margin_guidance,
                shared_clearance_budget=config.shared_clearance_budget,
                recovery=recovery,
            )
        )(gains)
        first = physical_route_coordinate(x[:2], points, route_mask, cursor)
        last = jax.vmap(lambda state, c: physical_route_coordinate(state[:2], points, route_mask, c))(
            result.final_state, trace["route_progress"][:, -1]
        )
        score = (last - first) / (config.validation_horizon * robot.dt * robot.v_max)
        score -= config.gain_change_penalty * jnp.sum((jnp.log(gains) - jnp.log(previous_gain)) ** 2, axis=-1)
        score += 0.05 * jnp.minimum(result.min_clearance, 0.5) / 0.5
        valid = ((result.status == GOAL) | (result.status == TIMEOUT)) & (result.min_clearance > 0)
        return valid, score

    def select(
        params,
        calibration,
        x,
        goal,
        obs,
        mask,
        candidates,
        points,
        route_mask,
        cursor,
        previous_gain,
        previous_control,
        noise,
    ):
        pool = jnp.concatenate((candidates, previous_gain[None]))
        same = jnp.max(jnp.abs(jnp.log(pool[:, None]) - jnp.log(pool[None, :])), axis=-1) < 1e-6
        unique = ~jnp.any(same & (jnp.arange(len(pool))[None, :] < jnp.arange(len(pool))[:, None]), axis=1)
        features, node_mask = route_graph(
            x, goal, obs, mask, points, route_mask, cursor, previous_control, previous_gain, noise, robot
        )
        output = predict_ensemble(model, params, features[None], node_mask[None], pool[None])
        mu = output["mean"][:, 0] * scale + mean
        variance = jnp.exp(output["log_variance"][:, 0]) * scale**2 * calibration["variance_scale"]
        means = mu[:, :, 0].T
        variances = variance[:, :, 0].T
        cs = cs_disagreement(means[..., None], variances[..., None])
        risk = worst_member_cvar(means, variances, config.tail_mass)
        event = jnp.max(
            jax.nn.sigmoid(output["event_logits"][:, 0] / calibration["temperature"] + calibration["bias"]),
            axis=0,
        )
        finite = (
            jnp.all(jnp.isfinite(mu) & jnp.isfinite(variance), axis=(0, 2))
            & jnp.all(jnp.isfinite(event), axis=-1)
            & jnp.isfinite(cs)
            & jnp.isfinite(risk)
        )
        qp, h, psi, _, _, _ = jax.vmap(
            lambda g: _control(
                x,
                goal,
                obs,
                mask,
                g,
                points,
                route_mask,
                cursor,
                robot,
                1.15 * noise[2],
                clearance_inflation(noise, config.sensor_margin_scale),
                config.margin_guidance,
                config.shared_clearance_budget,
                recovery=recovery,
            )
        )(pool)
        current = finite & unique & qp.feasible & (h >= -robot.qp_tolerance) & (psi >= -robot.qp_tolerance)
        epistemic = current & (cs <= calibration["cs_threshold"])
        risk_ok = epistemic & (risk <= config.risk_threshold)
        event_ok = (
            risk_ok
            & (event[:, 0] <= config.collision_probability_limit)
            & (event[:, 1] <= config.failure_probability_limit)
        )
        admitted = event_ok
        score = jnp.mean(mu[:, :, 1], axis=0) - config.gain_change_penalty * jnp.sum(
            (jnp.log(pool) - jnp.log(previous_gain)) ** 2, axis=-1
        )
        _, indices = jax.lax.top_k(jnp.where(admitted, score, -jnp.inf), config.shortlist)
        primary = pool[indices]
        screen = admitted[indices]
        ranking = score[indices]
        source = LEARNED
        valid, actual_score = validate(
            x, goal, obs, mask, primary, points, route_mask, cursor, previous_gain, noise
        )
        valid &= screen
        chosen = jnp.argmax(jnp.where(valid, ranking, -jnp.inf))
        primary_ok = jnp.any(valid)
        # A documented common backup, evaluated from the same copied state.
        # Under batched vmap this conditional may execute both sides; single-
        # observation online timing must be measured separately from throughput.
        backup_pool = jnp.concatenate(
            (previous_gain[None], jnp.asarray(config.backup_gains, x.dtype).reshape(-1, 2))
        )

        def backup(_):
            ok, value = validate(
                x, goal, obs, mask, backup_pool, points, route_mask, cursor, previous_gain, noise
            )
            index = jnp.argmax(jnp.where(ok, value, -jnp.inf))
            return backup_pool[index], jnp.any(ok), jnp.sum(ok).astype(jnp.int32)

        backup_gain, backup_ok, backup_count = jax.lax.cond(
            primary_ok, lambda _: (previous_gain, jnp.asarray(False), jnp.int32(0)), backup, operand=None
        )
        accepted = primary_ok | backup_ok
        gain = jnp.where(primary_ok, primary[chosen], jnp.where(backup_ok, backup_gain, previous_gain))
        selected_source = jnp.where(primary_ok, source, jnp.where(backup_ok, BACKUP, REJECTED)).astype(
            jnp.int32
        )
        return gain, accepted, selected_source

    return select


class Carry(NamedTuple):
    state: object
    status: object
    count: object
    cursor: object
    gain: object
    previous_control: object
    observer: object
    position: object
    speed_memory: object
    stall: object


def _make_stage(p, candidates, robot, cfg, recovery=False, dual=False):
    """Compile the physical loop for the primary controller or its one-tick recovery."""
    norm = p.predictor.metadata["normalization"]
    model = p.predictor.model

    def make_select(physical):
        preferred = _selector(model, norm, physical, cfg, recovery=recovery)
        if not dual:
            return preferred
        fallback = _selector(model, norm, physical, replace(cfg, validation_horizon=4))

        def select(*inputs):
            decision = preferred(*inputs)
            return jax.lax.cond(decision[1], lambda _: decision, lambda _: fallback(*inputs), None)

        return select

    hard = make_select(robot)
    soft_robot = replace(robot, guidance_soft_clearance=True)
    soft = hard if recovery else make_select(soft_robot)

    def execute(params, a, resume=None):
        goal, obstacles, mask, noise = a["goal"], a["obstacles"], a["mask"], a["noise"]
        points, route_mask = a["points"], a["route_mask"]
        x0, world, x_bias, o_bias, xs, os, innovations = unicycle_sensor_model(
            a["initial"],
            obstacles,
            mask,
            noise,
            a["key"],
            robot,
            STEPS,
            unit_x=a["unit_x"],
            unit_o=a["unit_o"],
            unit_i=a["unit_i"],
        )
        initial_gap = jnp.min(signed_clearance(x0[:2], world, mask, robot.radius))
        arrived = (jnp.linalg.norm(x0[:2] - goal) <= robot.goal_tolerance) & (jnp.abs(x0[3]) <= 0.2)
        status = jnp.where(
            ~a["ready"],
            PLANNER_FAILURE,
            jnp.where(initial_gap <= 0, COLLISION, jnp.where(arrived, GOAL, RUNNING)),
        )
        uncertainty = clearance_inflation(noise, cfg.sensor_margin_scale)
        zero = jnp.asarray(0.0, x0.dtype)
        carry = Carry(
            x0,
            status,
            jnp.int32(0),
            zero,
            jnp.asarray(cfg.fixed_gain, x0.dtype),
            jnp.zeros(2, x0.dtype),
            motion_observer.initialize(obstacles),
            motion_observer.position_observer_initialize(obstacles) if cfg.filter_obstacle_position else None,
            initial(x0.dtype) if recovery else None,
            None if recovery else StallMemory(False, zero, jnp.int32(0), False, True, zero, jnp.int32(0)),
        )
        start = jnp.int32(0)
        if recovery:
            start, history = resume

            def replay(i, memory):
                ob, xy, speed = memory
                raw_obs = history["raw_observed_obstacles"][i]
                ob, estimated, bounds, _, _ = motion_observer.update(
                    ob, raw_obs, mask, noise, robot.dt, cfg.motion_observer_window
                )
                gap = 1.15 * motion_observer.effective_noise(noise, bounds, mask)[4]
                xy, _, _ = motion_observer.position_observer_update(
                    xy, estimated, estimated[:, 3:5], gap, mask, noise, robot.dt
                )
                state = history["observed_state"][i]
                _, _, lo, hi = estimate(speed, state, noise, robot.dt)
                speed = update(speed, i, state[:2], history["control"][i], lo, hi, robot.dt)
                return ob, xy, speed

            ob, xy, speed = jax.lax.fori_loop(
                0, start, replay, (carry.observer, carry.position, carry.speed_memory)
            )
            prev = jnp.maximum(start - 1, 0)
            carry = carry._replace(
                state=jnp.where(start > 0, history["state"][prev], x0),
                status=jnp.int32(RUNNING),
                count=start,
                cursor=jnp.where(start > 0, history["route_progress"][prev], carry.cursor),
                gain=jnp.where(start > 0, history["gains"][prev], carry.gain),
                previous_control=jnp.where(start > 0, history["control"][prev], carry.previous_control),
                observer=ob,
                position=xy,
                speed_memory=speed,
            )

        def tick(c, k):
            active = c.status == RUNNING
            now = k * robot.dt
            innovation = innovations[k]
            sensed = c.state - x_bias + 0.15 * xs * innovation[:4]
            speed_error = 1.15 * noise[2]
            if recovery:
                sensed, speed_error, low, high = estimate(c.speed_memory, sensed, noise, robot.dt)
            moving = world.at[:, :2].set(world[:, :2] + now * world[:, 3:5])
            raw_obs = moving - o_bias + 0.15 * os * innovation[4:].reshape(obstacles.shape)
            ob, estimated, bounds, _, _ = motion_observer.update(
                c.observer, raw_obs, mask, noise, robot.dt, cfg.motion_observer_window
            )
            feature_noise = motion_observer.effective_noise(noise, bounds, mask)
            position = c.position
            if cfg.filter_obstacle_position:
                position, estimated, _ = motion_observer.position_observer_update(
                    position, estimated, estimated[:, 3:5], 1.15 * feature_noise[4], mask, noise, robot.dt
                )
            stall = c.stall
            if recovery:
                feature_noise = feature_noise.at[2].set(speed_error / 1.15)
            else:
                bound = jnp.sqrt(jnp.asarray(2.0, sensed.dtype)) * 1.15 * noise[0]
                stall = _stall_gate(
                    k,
                    active,
                    c.cursor,
                    jnp.linalg.norm(sensed[:2] - goal),
                    bound,
                    sensed[3],
                    feature_noise[2],
                    stall,
                )
            inputs = (
                params,
                p.calibration,
                sensed,
                goal,
                estimated,
                mask,
                candidates,
                points,
                route_mask,
                c.cursor,
                c.gain,
                c.previous_control,
                feature_noise,
            )
            if recovery:
                selected, accepted, source = hard(*inputs)
            else:
                selected, accepted, source = jax.lax.cond(
                    stall.latched, lambda _: soft(*inputs), lambda _: hard(*inputs), None
                )

            def control(physical):
                return _control(
                    sensed,
                    goal,
                    estimated,
                    mask,
                    selected,
                    points,
                    route_mask,
                    c.cursor,
                    physical,
                    speed_error,
                    uncertainty,
                    cfg.margin_guidance,
                    cfg.shared_clearance_budget,
                    recovery=recovery,
                )

            if recovery:
                qp, h, psi, cursor, _, _ = control(robot)
            else:
                qp, h, psi, cursor, _, _ = jax.lax.cond(
                    stall.latched, lambda _: control(soft_robot), lambda _: control(robot), None
                )
            admissible = (h >= -robot.qp_tolerance) & (psi >= -robot.qp_tolerance)
            applied = active & accepted & qp.feasible & admissible
            u = jnp.where(applied, qp.control, jnp.zeros(2, x0.dtype))
            y, sub = integrate_unicycle(c.state, u, robot.dt, robot.integration_substeps)
            starts = jnp.concatenate((c.state[None], sub[:-1]))
            times = now + jnp.arange(robot.integration_substeps) * robot.dt / robot.integration_substeps
            clear = jnp.min(
                jax.vmap(
                    lambda a, b, t: swept_disk_clearance(
                        a, b, world, mask, robot.radius, t, t + robot.dt / robot.integration_substeps
                    )
                )(starts, sub, times)
            )
            bound = jnp.maximum(-y[3], y[3] - robot.v_max)
            reached = applied & (jnp.linalg.norm(y[:2] - goal) <= robot.goal_tolerance) & (jnp.abs(y[3]) <= 0.2)
            status = jnp.where(active & ~accepted, PREDICTIVE_REJECTED, c.status)
            status = jnp.where(active & accepted & ~qp.feasible, INFEASIBLE, status)
            status = jnp.where(active & accepted & ~admissible, INADMISSIBLE, status)
            status = jnp.where(reached, GOAL, status)
            status = jnp.where(applied & (bound > robot.qp_tolerance), STATE_BOUND_VIOLATION, status)
            status = jnp.where(applied & (clear <= 0), COLLISION, status)
            speed = update(c.speed_memory, k, sensed[:2], u, low, high, robot.dt) if recovery else None
            nxt = Carry(
                jnp.where(applied, y, c.state),
                status,
                c.count + applied.astype(jnp.int32),
                jnp.where(applied, cursor, c.cursor),
                jnp.where(applied, selected, c.gain),
                jnp.where(applied, u, c.previous_control),
                ob,
                position,
                speed,
                stall,
            )
            record = dict(
                state=nxt.state,
                status=status,
                active=applied,
                control=u,
                gains=nxt.gain,
                route_progress=nxt.cursor,
                clearance=jnp.where(applied, clear, jnp.nan),
                qp_violation=jnp.where(applied, qp.max_violation, jnp.nan),
                state_bound_violation=jnp.where(applied, bound, jnp.nan),
                raw_observed_obstacles=raw_obs,
                observed_state=sensed,
                query=active,
                source=jnp.where(applied, source, REJECTED),
                selection_accepted=accepted,
            )
            return nxt, record

        template = jax.eval_shape(tick, carry, jnp.int32(0))[1]
        blank = jax.tree.map(lambda v: jnp.zeros((STEPS, *v.shape), v.dtype), template)

        def advance(state):
            k, c, hist = state
            nxt, record = tick(c, k)
            hist = jax.tree.map(lambda h, v: jax.lax.dynamic_update_index_in_dim(h, v, k, 0), hist, record)
            return k + 1, nxt, hist

        k, last, hist = jax.lax.while_loop(
            lambda v: (v[0] < STEPS) & ((v[0] == 0) | (v[1].status == RUNNING)), advance, (start, carry, blank)
        )
        return jnp.where(last.status == RUNNING, TIMEOUT, last.status), last.count, k, x0, world, hist

    return jax.jit(execute)


def build(models, method):
    """Build the fixed FC or the GAT with recovery after an actual first rejection."""
    import time
    from .protocol import configure_precision

    gat = method in ("ours_gat", "gat")
    if not gat and method not in ("ours_fc", "nearest_fc", "fc"):
        raise ValueError(f"Unknown learned unicycle method {method!r}")
    configure_precision("unicycle")
    p, spec = selector(models, "unicycle", "gat" if gat else "nearest_fc")
    bank = jnp.asarray(spec["candidates"], jnp.float32)
    robot = replace(p.robot, stationary_obstacles=False, guidance_route_preview=True, guidance_turn_return=True)
    cfg = replace(p.config, interval=1, validation_horizon=40 if gat else 2, gain_change_penalty=0.05)
    primary = _make_stage(p, bank, robot, cfg, dual=gat)
    recovery = None
    if gat:
        extra = tuple((a, b) for a in (1.0, 4.0, 8.0, 16.0) for b in (0.3, 2.0, 6.0, 16.0) if a != b)
        cautious = replace(
            p.robot,
            stationary_obstacles=False,
            guidance_soft_clearance=True,
            guidance_horizon=4.0,
            guidance_min_speed=0.9,
        )
        cfg = replace(
            p.config,
            interval=1,
            validation_horizon=1,
            gain_change_penalty=0.05,
            backup_gains=p.config.backup_gains + extra,
            filter_obstacle_position=True,
        )
        recovery = _make_stage(p, bank, cautious, cfg, recovery=True)
    compiled = {}

    def execute(stage, name, *inputs):
        elapsed_compile = 0.0
        if name not in compiled:
            mark = time.perf_counter()
            compiled[name] = stage.lower(p.params, *inputs).compile()
            elapsed_compile = time.perf_counter() - mark
        mark = time.perf_counter()
        result = jax.device_get(compiled[name](p.params, *inputs))
        return result, elapsed_compile, time.perf_counter() - mark

    def run(case):
        configure_precision("unicycle")
        a = arguments(case)
        result, setup_seconds, execution_seconds = execute(primary, "primary", a)
        status, count, end, x0, world, history = result
        end = int(end)
        trace = {name: values[:end] for name, values in history.items()}
        if gat and int(status) == PREDICTIVE_REJECTED:
            first = int(count)
            kept = {
                name: history[name]
                for name in (
                    "state",
                    "gains",
                    "route_progress",
                    "control",
                    "observed_state",
                    "raw_observed_obstacles",
                )
            }
            result, setup, elapsed = execute(recovery, "recovery", a, (np.int32(first), kept))
            setup_seconds += setup
            execution_seconds += elapsed
            status, count, end, _, _, suffix = result
            trace = {
                name: np.concatenate((values[:first], suffix[name][first : int(end)]))
                for name, values in trace.items()
            }
        # At an initial planner stop the original empty decision is accepted, with no
        # selection query or applied action. Keep that telemetry on the single stop frame.
        trace["selection_accepted"] = ~trace["query"] | trace["selection_accepted"]
        trace.update(
            states=np.vstack((x0, trace["state"])),
            true_initial_state=x0,
            true_obstacles=world,
            obstacles=world,
            mask=a["mask"],
            goal=a["goal"],
            dt=np.asarray(robot.dt),
            radius=np.asarray(robot.radius),
        )
        initial_distance = np.min(
            np.where(
                a["mask"], np.linalg.norm(x0[:2] - world[:, :2], axis=1) - robot.radius - world[:, 2], np.inf
            )
        )
        applied = trace["active"]
        minimum = float(np.minimum(initial_distance, np.min(trace["clearance"][applied], initial=np.inf)))
        residual = float(np.max(trace["qp_violation"][applied], initial=-np.inf))
        report = dict(
            dynamics="unicycle",
            method="ours_gat" if gat else "ours_fc",
            status=int(status),
            success=int(status) == GOAL,
            steps=int(count),
            ticks=int(end),
            physical_navigation_seconds=int(count) * robot.dt,
            min_clearance=minimum if np.isfinite(minimum) else None,
            max_qp_violation=residual if np.isfinite(residual) else None,
            execution_seconds=execution_seconds,
            setup_compile_seconds=setup_seconds,
        )
        return report, trace

    return run
