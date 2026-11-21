"""Observed bicycle benchmark with the released GAT or nearest-obstacle FC weights."""

import time
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .. import bicycle_guidance as guidance
from ..bicycle import bicycle_state_violation, integrate_bicycle
from ..bicycle_control import (
    bicycle_arrived,
    bicycle_control,
    bicycle_rows,
    constant,
    observed32,
    parabolic_terms,
    project_bicycle_reference,
)
from ..bicycle_features import bicycle_graph, bicycle_inference_graph
from ..bicycle_observation import next_margin, observe, speed_error_bound, unit_errors
from ..bicycle_observed_rollout import make_observed_episode
from ..bicycle_policy import filter_proposal, select_gain
from ..bicycle_rollout import (
    COLLISION,
    GOAL,
    INADMISSIBLE,
    INFEASIBLE,
    PLANNER_FAILURE,
    RUNNING,
    STATE_BOUND,
    TIMEOUT,
)
from ..controllers import QPResult
from ..dynamics import signed_clearance, swept_disk_clearance
from ..model_release import selector
from ..models import predict_ensemble
from ..routing import physical_route_coordinate, route_geometry, route_target_from_position
from .protocol import configure_precision
from .sensors import arguments


def _make_predictor(policy):
    """Rank the released gain bank using the calibrated ensemble and gain-32 prior."""
    model = policy.predictor.model
    normalization = policy.metadata["normalization"]
    mean = jnp.asarray(normalization["target_mean"], jnp.float32)
    scale = jnp.asarray(normalization["target_scale"], jnp.float32)
    variance_scale = jnp.asarray(policy.fit["variance_scale"], jnp.float32)
    temperature = jnp.asarray([e["temperature"] for e in policy.fit["event_calibration"]], jnp.float32)
    bias = jnp.asarray([e["bias"] for e in policy.fit["event_calibration"]], jnp.float32)
    bank = jnp.asarray(policy.bank)
    graph = (
        bicycle_graph
        if model.config.compute_dtype == "float32"
        else partial(bicycle_inference_graph, compute_dtype=model.config.compute_dtype)
    )

    @jax.jit
    def predict(
        params, state, goal, obstacles, mask, points, route_mask, cursor, previous_u, previous_gain, noise
    ):
        features, node_mask = jax.vmap(partial(graph, config=policy.robot))(
            state, goal, obstacles, mask, points, route_mask, cursor, previous_u, previous_gain, noise
        )
        gains = jnp.broadcast_to(bank, (len(state), len(bank), 1))
        outputs = predict_ensemble(model, params, features, node_mask, gains)
        means = outputs["mean"] * scale + mean
        variances = jnp.exp(outputs["log_variance"]) * scale**2 * variance_scale
        logits = outputs["event_logits"]
        decision = select_gain(
            means,
            variances,
            logits,
            previous_gain,
            bank,
            temperature,
            bias,
            jnp.asarray(policy.threshold),
            policy.config,
        )
        decision["model_finite"] = (
            jnp.all(
                jnp.isfinite(means) & jnp.isfinite(variances) & (variances > 0) & jnp.isfinite(logits),
                axis=(0, 3),
            )
            & jnp.isfinite(decision["cs_score"])
            & jnp.isfinite(decision["finite_member_cvar"])
            & jnp.isfinite(decision["adverse_probability"])
        )
        score = decision["ranking_score"] - 0.25 * jnp.abs(jnp.log(bank[None, :, 0] / 32.0))
        accepted = decision["accepted"]
        best = jnp.argmax(jnp.where(accepted, score, -jnp.inf), axis=1)
        any_accepted = jnp.any(accepted, axis=1)
        decision.update(
            ranking_score=score,
            controller_gain=jnp.where(any_accepted, bank[best, 0], previous_gain),
            selected_index=jnp.where(any_accepted, best, -1),
        )
        return filter_proposal(
            decision,
            state,
            goal,
            obstacles,
            mask,
            points,
            route_mask,
            cursor,
            previous_gain,
            noise,
            bank,
            policy.robot,
            policy.guidance,
        )

    return predict


def _advisory(selection, state, goal, obstacles, mask, points, route_mask, cursor, noise, policy):
    """On an empty neural acceptance set, check each supported gain against the current QP."""
    bank = jnp.asarray(policy.bank, jnp.float32)[:, 0]
    empty = ~jnp.any(selection["accepted"][0])
    supported = selection["model_finite"][0] & (selection["cs_score"][0] <= jnp.asarray(policy.threshold))
    original = selection["controller_gain"][0]
    error = speed_error_bound(noise)

    def check(gain):
        qp, h, domain, *_ = bicycle_control(
            state, goal, obstacles, mask, gain, points, route_mask, cursor, policy.robot, error
        )
        return (
            qp.feasible & (h >= -policy.robot.qp_tolerance) & (domain > 0),
            qp.control,
            h,
            domain,
            qp.max_violation,
        )

    unused = (
        jnp.zeros(16, bool),
        jnp.zeros((16, 2), jnp.float32),
        jnp.zeros(16, jnp.float32),
        jnp.zeros(16, jnp.float32),
        jnp.zeros(16, jnp.float32),
    )
    physical, controls, h, domain, residual = jax.lax.cond(
        empty, lambda _: jax.vmap(check)(bank), lambda _: unused, None
    )
    adverse = selection["adverse_probability"][0]
    eligible = supported & physical & jnp.isfinite(adverse)
    rank = jnp.where(eligible, adverse, jnp.inf)
    index = jnp.argmax(rank == jnp.min(rank))
    used = empty & jnp.any(eligible)
    return jnp.where(used, bank[index], original), {
        "observed_advisory_called": empty,
        "observed_advisory_used": used,
        "observed_advisory_model_supported": supported,
        "observed_advisory_model_finite": selection["model_finite"][0],
        "observed_advisory_physical": physical,
        "observed_advisory_controls": controls,
        "observed_advisory_h": h,
        "observed_advisory_domain": domain,
        "observed_advisory_residual": residual,
        "observed_advisory_index": jnp.where(used, index, -1),
        "observed_advisory_original_gain": original,
        "observed_advisory_QP_calls": jnp.where(empty, jnp.int32(16), jnp.int32(0)),
    }


def _arc_reference(state, goal, fraction, config):
    """Track the shortest feasible maximum-slip arc and tangent line.

    The centre of the held-slip arc is behind the body reference point by
    one rear axle offset. Both tangent solutions use the original slip bound.
    """
    robot = config.robot
    longitudinal = robot.rear_axle_distance
    lateral = longitudinal / robot.slip_max
    turn_radius = (longitudinal**2 + lateral**2) ** 0.5
    turn_sign = jnp.array([-1.0, 1.0], dtype=state.dtype)
    forward = jnp.stack((jnp.cos(state[2]), jnp.sin(state[2])))
    normal = jnp.stack((-forward[1], forward[0]))
    centres = state[:2] - longitudinal * forward + turn_sign[:, None] * lateral * normal
    delta = goal - centres
    distance = jnp.linalg.norm(delta, axis=1)
    straight = jnp.sqrt(jnp.maximum(0.0, distance**2 - lateral**2)) - longitudinal
    tangent = jnp.arctan2(delta[:, 1], delta[:, 0]) + turn_sign * jnp.arcsin(
        jnp.clip(lateral / jnp.maximum(distance, 1e-6), 0.0, 1.0)
    )
    turn_angle = jnp.mod(turn_sign * (tangent - state[2]), 2 * jnp.pi)
    length = jnp.where(straight >= 0, turn_radius * turn_angle + straight, 1e5)
    choice = jnp.argmin(length)
    goal_offset = goal - state[:2]
    direct_angle = jnp.arctan2(goal_offset[1], goal_offset[0]) - state[2]
    direct_angle = jnp.arctan2(jnp.sin(direct_angle), jnp.cos(direct_angle))
    slip = turn_sign[choice] * robot.slip_max
    towards = jnp.clip(direct_angle * 1.1, -robot.slip_max, robot.slip_max)
    slip = jnp.where((jnp.abs(direct_angle) < 0.16) | (turn_angle[choice] < 0.16), towards, slip)
    # A clear turn can run faster. Reduce speed progressively for the rolling
    # arrival using the actual range and existing actuator constraints.
    cap = 0.45 + 0.55 * fraction
    speed = jnp.minimum(cap, 0.20 + 0.60 * jnp.maximum(jnp.linalg.norm(goal_offset) - 0.20, 0.0))
    return jnp.array((config.speed_feedback * (speed - state[3]), slip), dtype=state.dtype)


def _turn_straight_cost(states, goal, robot):
    """Minimum length of a maximum-steer arc followed by a straight approach."""
    d = jnp.array([-1.0, 1.0], dtype=states.dtype)
    heading = states[:, 2, None]
    direction = jnp.concatenate((jnp.cos(heading), jnp.sin(heading)), axis=1)
    left = jnp.stack((-direction[:, 1], direction[:, 0]), axis=1)
    longitudinal = robot.rear_axle_distance
    lateral = longitudinal / robot.slip_max
    radius = (longitudinal**2 + lateral**2) ** 0.5
    centres = (
        states[:, None, :2]
        - longitudinal * direction[:, None, :]
        + d[None, :, None] * lateral * left[:, None, :]
    )
    rel = goal - centres
    mag = jnp.linalg.norm(rel, axis=2)
    straight = jnp.sqrt(jnp.maximum(0.0, mag**2 - lateral**2)) - longitudinal
    angle = jnp.arctan2(rel[:, :, 1], rel[:, :, 0]) + d * jnp.arcsin(
        jnp.clip(lateral / jnp.maximum(mag, 1e-6), 0, 1)
    )
    turn = jnp.mod(d[None, :] * (angle - heading), 2 * jnp.pi)
    length = radius * turn + jnp.maximum(straight, 0.0)
    # Near the point the car can already qualify for rolling arrival, even
    # when the exact point lies inside both non-reversing turning circles.
    near = jnp.linalg.norm(states[:, :2] - goal, axis=1)
    travel = jnp.min(jnp.where(straight >= 0, length, 1e4), axis=1)
    return jnp.where(near < 0.16, near, jnp.minimum(travel, near + 2 * jnp.pi * radius))


def _terminal_control(
    state, goal, obstacles, mask, gain, points, route_mask, cursor, config, settings, speed_error, noise
):
    """Preview 34 admissible approaches and rank the earliest robust rolling arrival."""
    offsets, fractions = guidance.profile_bank()
    count = len(offsets)
    robot = config.robot
    dt = constant(robot.dt, jnp.float64)
    radius = constant(robot.radius, jnp.float64)
    original = jnp.arange(count) == 0
    obs0 = obstacles.astype(jnp.float64)
    x = jnp.broadcast_to(state.astype(jnp.float64), (count, 4))
    initial = (
        x,
        jnp.full((count,), cursor, jnp.float32),
        jnp.ones(count, bool),
        jnp.zeros(count, jnp.int32),
        jnp.full(count, jnp.inf, jnp.float64),
        jnp.full(count, jnp.inf, jnp.float64),
        jnp.zeros(count, jnp.float32),
        jnp.full(count, -jnp.inf, jnp.float64),
    )

    def preview(carry, k):
        x, cursors, alive, steps, minimum, min_h, energy, first_margin = carry
        observed = observed32(x)
        seen = observed32(obs0.at[:, :2].set(obs0[:, :2] + k.astype(jnp.float64) * dt * obs0[:, 3:5]))
        target, updated, _ = jax.vmap(
            lambda s, p: route_target_from_position(s[:2], s[3], points, route_mask, p)
        )(observed, cursors)
        decay = jnp.clip(2.0 - 2.0 * k / settings.horizon, 0.0, 1.0)
        references = jax.vmap(
            lambda s, t, o, f, z: guidance.profile_reference(
                s, goal, jnp.where(z, t, goal), o * decay, f, z, config
            )
        )(observed, target, offsets, fractions, original)
        arcs = jax.vmap(lambda s, f: _arc_reference(s, goal, f, config))(observed, fractions)
        references = jnp.where((jnp.abs(offsets) > 3.1)[:, None], arcs, references)
        a, b, h, domain = jax.vmap(lambda s: bicycle_rows(s, seen, mask, gain, config, speed_error))(
            observed
        )
        qp = jax.vmap(lambda r, a, b: project_bicycle_reference(r, a, b, jnp.float32, config))(
            references, a, b
        )
        hmin = jnp.min(jnp.where(mask, h, jnp.inf), axis=1)
        dmin = jnp.min(jnp.where(mask, domain, jnp.inf), axis=1)
        accepted = alive & qp.feasible & (hmin >= -config.qp_tolerance) & (dmin > 0)
        u = jnp.where(accepted[:, None], qp.control, jnp.zeros((count, 2), jnp.float32))
        y, sub = jax.vmap(lambda s, v: integrate_bicycle(s, v, robot))(x, u)
        starts = jnp.concatenate((x[:, None], sub[:, :-1]), axis=1)
        times = (
            k.astype(jnp.float64) * dt
            + jnp.arange(robot.integration_substeps, dtype=jnp.float64) * dt / robot.integration_substeps
        )

        def clearance_at_tick(start, end):
            gaps = jax.vmap(
                lambda a, b, t: swept_disk_clearance(
                    a, b, obs0, mask, radius, t, t + dt / robot.integration_substeps
                )
            )(start, end, times)
            return jnp.min(gaps)

        clearance = jax.vmap(clearance_at_tick)(starts, sub)
        violation = jnp.max(jax.vmap(jax.vmap(lambda s: bicycle_state_violation(s, robot)))(sub), axis=1)
        after_seen = observed32(
            obs0.at[:, :2].set(obs0[:, :2] + (k.astype(jnp.float64) + 1) * dt * obs0[:, 3:5])
        ).astype(jnp.float64)
        after_h, after_domain, _, _ = jax.vmap(
            lambda s: parabolic_terms(observed32(s).astype(jnp.float64), after_seen, config)
        )(y)
        next_h = jnp.min(jnp.where(mask, after_h, jnp.inf), axis=1)
        next_domain = jnp.min(jnp.where(mask, after_domain, jnp.inf), axis=1)

        def first_observation_margin(_):
            bound = jax.vmap(
                lambda s, u, y: next_margin(s, u, observed32(y), seen, after_seen, mask, noise, config)[
                    "lower"
                ]
            )(observed, u, y)
            return jnp.where(accepted, bound, -jnp.inf)

        first_margin = jax.lax.cond(k == 0, first_observation_margin, lambda _: first_margin, None)
        next_alive = (
            accepted
            & (clearance > 0)
            & (violation <= config.qp_tolerance)
            & (next_h >= -config.qp_tolerance)
            & (next_domain > 0)
        )
        x = jnp.where(accepted[:, None], y, x)
        cursors = jnp.where(accepted, updated, cursors)
        steps += accepted.astype(jnp.int32)
        minimum = jnp.minimum(minimum, jnp.where(accepted, clearance, jnp.inf))
        min_h = jnp.minimum(
            min_h, jnp.where(alive, jnp.where(accepted, jnp.minimum(hmin, next_h), hmin), jnp.inf)
        )
        energy += jnp.where(
            accepted,
            jnp.sum((u / constant([robot.acceleration_max, robot.slip_max], jnp.float32)) ** 2, axis=1),
            0.0,
        )
        radius_goal = jnp.maximum(0.11, config.goal_tolerance - 0.01 - 1.15 * jnp.sqrt(2.0) * noise[0])
        speed_goal = config.terminal_speed - 0.01 - 1.15 * noise[2]
        distance = jnp.linalg.norm(y[:, :2] - goal, axis=1)
        detail = {
            "control": qp.control,
            "feasible": qp.feasible,
            "violation": qp.max_violation,
            "objective": qp.objective,
            "arriving": accepted & (distance <= radius_goal) & (y[:, 3] <= speed_goal),
            "miss": jnp.where(accepted, distance + 1.15 * jnp.maximum(y[:, 3] - speed_goal, 0.0), 1e4),
        }
        return (x, cursors, next_alive, steps, minimum, min_h, energy, first_margin), detail

    (final, cursors, alive, steps, minimum, min_h, energy, first_margin), history = jax.lax.scan(
        preview, initial, jnp.arange(settings.horizon)
    )
    coordinate = jax.vmap(lambda x, p: physical_route_coordinate(observed32(x)[:2], points, route_mask, p))(
        final, cursors
    )
    progress = coordinate - physical_route_coordinate(state[:2], points, route_mask, cursor)
    vectors, valid, lengths, cumulative = route_geometry(points, route_mask)
    segment = jax.vmap(lambda s: jnp.argmax(valid & (cumulative[1:] >= s - 1e-6)))(coordinate)
    position = (
        points[segment]
        + ((coordinate - cumulative[segment]) / jnp.maximum(lengths[segment], 1e-12))[:, None]
        * vectors[segment]
    )
    cross_track = jnp.linalg.norm(final[:, :2] - position, axis=1)
    earliest = jnp.argmax(history["arriving"], axis=0)
    minimum_miss = jnp.min(history["miss"], axis=0)
    travel = _turn_straight_cost(final, goal, robot)
    score = jnp.where(
        jnp.any(history["arriving"], axis=0),
        20.0 - 0.05 * earliest - 0.02 * travel,
        -travel - 0.5 * minimum_miss - 0.03 * energy / settings.horizon,
    )
    rank = jnp.where(alive, score, -1e6 + steps * 100.0 + score)
    witness = first_margin >= 0.0
    prefer = jnp.any(noise > 0) & jnp.any(witness)
    rank = jnp.where(prefer & ~witness, -jnp.inf, rank)
    original_index = jnp.argmax(rank == jnp.max(rank))
    eligible = (steps >= steps[original_index]) & (~alive[original_index] | alive)
    index, recovered = guidance.margin_recovery_index(
        original_index, first_margin, jnp.any(noise > 0), eligible
    )
    metrics = {
        "selected": index,
        "complete": alive[index],
        "complete_profiles": jnp.sum(alive),
        "steps": steps[index],
        "clearance": minimum[index],
        "barrier": min_h[index],
        "route_progress": progress[index],
        "cross_track": cross_track[index],
        "score": score[index],
        "next_observation_lower": first_margin[index],
        "margin_profiles": jnp.sum(witness),
        "margin_preferred": prefer,
        "recovery_applied": recovered,
        "recovery_original_index": original_index,
        "recovery_profile_lower": first_margin,
        "recovery_first_controls": history["control"][0],
        "recovery_profile_steps": steps,
        "recovery_profile_complete": alive,
    }
    result = QPResult(
        history["control"][0, index],
        history["feasible"][0, index],
        history["violation"][0, index],
        history["objective"][0, index],
    )
    _, _, h, domain = bicycle_rows(state, obstacles, mask, gain, config, speed_error)
    target, updated, remaining = route_target_from_position(state[:2], state[3], points, route_mask, cursor)
    return (
        result,
        jnp.min(jnp.where(mask, h, jnp.inf)).astype(state.dtype),
        jnp.min(jnp.where(mask, domain, jnp.inf)).astype(state.dtype),
        updated,
        remaining,
        target,
        metrics,
    )


def _terminal_advance(state, sensed, seen, current, args, cursor, gain, policy):
    """Apply exactly one held input to the shared plant, with eight swept-clearance checks."""
    config = policy.robot
    robot = config.robot
    goal, mask, noise = args["goal"], args["mask"], args["noise"]
    dt, radius = constant(robot.dt, jnp.float64), constant(robot.radius, jnp.float64)
    status = jnp.where(args["ready"], RUNNING, PLANNER_FAILURE)
    status = jnp.where(bicycle_arrived(state, goal, config), GOAL, status)
    status = jnp.where(bicycle_state_violation(state, robot) > config.qp_tolerance, STATE_BOUND, status)
    status = jnp.where(jnp.min(signed_clearance(state[:2], current, mask, radius)) <= 0, COLLISION, status)
    error = speed_error_bound(noise)
    qp, h, domain, proposed, _, _, metrics = _terminal_control(
        sensed,
        goal,
        seen,
        mask,
        gain,
        args["points"],
        args["route_mask"],
        cursor,
        config,
        policy.guidance,
        error,
        noise=noise,
    )
    active = status == RUNNING
    admissible = (h >= -config.qp_tolerance) & (domain > 0)
    accepted = active & qp.feasible & admissible
    u = jnp.where(accepted, qp.control, jnp.zeros(2, jnp.float32))
    y, sub = integrate_bicycle(state, u, robot)
    starts = jnp.concatenate((state[None], sub[:-1]))
    times = jnp.arange(robot.integration_substeps, dtype=jnp.float64) * dt / robot.integration_substeps
    clearance = jnp.min(
        jax.vmap(
            lambda a, b, t: swept_disk_clearance(
                a, b, current, mask, radius, t, t + dt / robot.integration_substeps
            )
        )(starts, sub, times)
    )
    violation = jnp.max(
        jax.vmap(lambda s: bicycle_state_violation(s, robot))(jnp.concatenate((state[None], sub)))
    )
    status = jnp.where(active & ~qp.feasible, INFEASIBLE, status)
    status = jnp.where(active & ~admissible, INADMISSIBLE, status)
    status = jnp.where(accepted & bicycle_arrived(y, goal, config), GOAL, status)
    status = jnp.where(accepted & (violation > config.qp_tolerance), STATE_BOUND, status)
    status = jnp.where(accepted & (clearance <= 0), COLLISION, status)
    return dict(
        state=jnp.where(accepted, y, state),
        route_progress=jnp.where(accepted, proposed, cursor),
        control=u,
        active=accepted,
        status=status,
        h=h,
        domain=domain,
        qp_violation=qp.max_violation,
        clearance=jnp.where(accepted, clearance, jnp.nan),
        state_bound_violation=jnp.where(accepted, violation, jnp.nan),
        **{"guidance_" + name: value for name, value in metrics.items()},
    )


def _make_episode(policy, use_terminal):
    predict = _make_predictor(policy)
    robot = policy.robot.robot
    normal_plant = make_observed_episode(policy.robot, steps=1, guidance=policy.guidance)

    def episode(params, args):
        initial = (
            args["initial"],
            jnp.int32(TIMEOUT),
            jnp.int32(0),
            jnp.float32(0),
            jnp.zeros(2, jnp.float32),
            jnp.float32(policy.config.initial_gain),
            jnp.bool_(False),
        )

        def tick(carry, k):
            state, _, steps, cursor, previous_u, previous_gain, latched = carry
            current = (
                args["obstacles"]
                .at[:, :2]
                .add(k.astype(jnp.float64) * constant(robot.dt, jnp.float64) * args["obstacles"][:, 3:5])
            )
            ix, io = unit_errors(jax.random.fold_in(args["key"], k), len(args["mask"]))
            ix = jnp.where(k == 0, 0.0, ix)
            io = jnp.where(k == 0, 0.0, io)
            sx, so = observe(state, current, args["mask"], args["bx"], args["bo"], args["noise"], ix, io)
            sx = jnp.where(k == 0, args["first_x"], sx)
            so = jnp.where(k == 0, args["first_o"], so)
            values = (
                sx,
                args["goal"],
                so,
                args["mask"],
                args["points"],
                args["route_mask"],
                cursor,
                previous_u,
                previous_gain,
                args["noise"],
            )
            decision = predict(params, *[v[None] for v in values])
            gain, advice = _advisory(
                decision,
                sx,
                args["goal"],
                so,
                args["mask"],
                args["points"],
                args["route_mask"],
                cursor,
                args["noise"],
                policy,
            )

            def ordinary(_):
                _, history = normal_plant(
                    state,
                    args["goal"],
                    current,
                    args["mask"],
                    gain,
                    args["points"],
                    args["route_mask"],
                    args["ready"],
                    cursor,
                    sx,
                    so,
                    args["bx"],
                    args["bo"],
                    args["noise"],
                    args["key"],
                )
                fields = (
                    "state",
                    "route_progress",
                    "control",
                    "active",
                    "status",
                    "h",
                    "domain",
                    "qp_violation",
                    "clearance",
                )
                result = {key: history[key][0] for key in fields}
                result["state_bound_violation"] = history["state_violation"][0]
                result.update({k: v[0] for k, v in history.items() if k.startswith("guidance_")})
                return result

            if use_terminal:
                latched = latched | ((k >= 1200) & (jnp.linalg.norm(sx[:2] - args["goal"]) <= 3.0))
                result = jax.lax.cond(
                    latched,
                    lambda _: _terminal_advance(state, sx, so, current, args, cursor, gain, policy),
                    ordinary,
                    None,
                )
            else:
                result = ordinary(None)
            active = result["active"]
            next_status = jnp.where(result["status"] == RUNNING, TIMEOUT, result["status"]).astype(
                jnp.int32
            )
            next_state = (
                result["state"],
                next_status,
                steps + active.astype(jnp.int32),
                result["route_progress"],
                jnp.where(active, result["control"], previous_u),
                jnp.where(active, gain, previous_gain),
                latched,
            )
            speed_error = speed_error_bound(args["noise"])
            low = sx[3].astype(jnp.float64) - speed_error
            high = sx[3].astype(jnp.float64) + speed_error
            result.update(
                gains=gain,
                query=jnp.bool_(True),
                innovation_x=ix,
                innovation_o=io,
                observed_state=sx,
                observed_obstacles=so,
                observed_h_min=result["h"],
                observed_domain_min=result["domain"],
                observed_speed=sx[3],
                observed_speed_error_bound=speed_error,
                robust_speed_lower=low,
                robust_speed_upper=high,
                acceleration_row_lower=jnp.maximum(
                    constant(-robot.acceleration_max, jnp.float64),
                    (constant(robot.speed_min, jnp.float64) - low) / constant(robot.dt, jnp.float64),
                ),
                acceleration_row_upper=jnp.minimum(
                    constant(robot.acceleration_max, jnp.float64),
                    (constant(robot.speed_max, jnp.float64) - high) / constant(robot.dt, jnp.float64),
                ),
                neural_accepted_count=jnp.sum(decision["accepted"][0]),
                neural_previous_gain=previous_gain,
                neural_incumbent_diagnostics_available=jnp.bool_(True),
                **advice,
            )
            for name in (
                "accepted",
                "selected_index",
                "uncertainty_fallback",
                "incumbent_progress_hold",
                "incumbent_comparison_needed",
                "incumbent_witness_checked",
                "incumbent_witness_feasible",
                "cs_score",
                "finite_member_cvar",
                "adverse_probability",
                "predicted_progress",
                "ranking_score",
            ):
                result["neural_" + name] = decision[name][0]
            return next_state, result

        horizon = 1600
        template = jax.eval_shape(lambda c, k: tick(c, k)[1], initial, jnp.int32(0))
        storage = jax.tree.map(lambda t: jnp.zeros((horizon, *t.shape), t.dtype), template)

        def condition(loop):
            k, carry, _ = loop
            return (k < horizon) & ((k == 0) | (carry[1] == TIMEOUT))

        def advance(loop):
            k, carry, history = loop
            carry, record = tick(carry, k)
            history = jax.tree.map(
                lambda h, v: jax.lax.dynamic_update_index_in_dim(h, v, k, 0), history, record
            )
            return k + 1, carry, history

        ticks, last, history = jax.lax.while_loop(condition, advance, (jnp.int32(0), initial, storage))
        initial_clearance = jnp.min(
            jnp.where(
                args["mask"],
                jnp.linalg.norm(args["initial"][:2] - args["obstacles"][:, :2], axis=-1)
                - robot.radius
                - args["obstacles"][:, 2],
                jnp.inf,
            )
        )
        minimum = jnp.minimum(
            initial_clearance, jnp.min(jnp.where(history["active"], history["clearance"], jnp.inf))
        )
        worst = jnp.max(jnp.where(history["active"], history["qp_violation"], -jnp.inf))
        return ticks, last[1], last[2], minimum, worst, history

    return jax.jit(episode)


def build(models, method):
    """Load the CPU-qualified bicycle controller and return a simulator for complete cases."""
    encoders = {
        "ours_gat": "gat",
        "ours_fc": "nearest_fc",
        "fc": "nearest_fc",
        "gat": "gat",
        "nearest_fc": "nearest_fc",
    }
    if method not in encoders:
        raise ValueError(f"Unknown bicycle learned method {method!r}")
    configure_precision("bicycle")
    encoder = encoders[method]
    policy, _ = selector(models, "bicycle", encoder)
    policy.config = replace(policy.config, adverse_progress_penalty=0.5)
    policy.guidance = replace(policy.guidance, horizon=60)
    simulate = _make_episode(policy, encoder == "gat")
    robot = policy.robot.robot

    def run(case):
        configure_precision("bicycle")
        args = arguments(case)
        start = time.perf_counter()
        ticks, status, steps, minimum, worst, history = jax.device_get(
            simulate(policy.predictor.params, args)
        )
        seconds = time.perf_counter() - start
        trace = {k: np.asarray(v)[: int(ticks)] for k, v in history.items()}
        trace.update(
            states=np.vstack((args["initial"], trace["state"])),
            true_initial_state=args["initial"],
            true_obstacles=args["obstacles"],
            obstacles=args["obstacles"],
            mask=args["mask"],
            goal=args["goal"],
            dt=np.asarray(robot.dt),
            radius=np.asarray(robot.radius),
        )
        raw = {
            "dynamics": "bicycle",
            "method": encoder,
            "status": int(status),
            "steps": int(steps),
            "ticks": int(ticks),
            "success": int(status) == GOAL,
            "min_clearance": float(minimum),
            "max_qp_violation": float(worst) if np.isfinite(worst) else None,
            "physical_navigation_seconds": int(steps) * robot.dt,
            "execution_seconds": seconds,
        }
        return raw, trace

    return run
