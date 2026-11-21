"""Observed Quad3D reference selection, held-input control and stop recovery."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ..model_release import selector
from ..quad3d import integrate_quad3d
from ..quad3d_control import quad3d_rows, solve_qp4
from ..quad3d_held import hold_rows
from ..quad3d_observation import controller_obstacles, observe
from ..quad3d_observed_rollout import observed_problem
from ..quad3d_policy import Quad3DSelector
from ..quad3d_policy_rollout import make_policy_rollout
from ..quad3d_qp import InputSupportSelector
from ..quad3d_transition import transition_rows
from .protocol import configure_precision
from .quad2d import STEPS, model_name, run_ticks, runner


def _advisory(prediction, previous, bank, switch_penalty, consensus):
    """Compare the neural nomination to the admissible (4,4,4,4) reference."""
    ref_idx = 5
    reference = bank[ref_idx].astype(previous.dtype)
    accepted, scores = prediction["accepted"], prediction["ranking_score"]
    usable = prediction["admissible"][ref_idx]
    finite = accepted & jnp.isfinite(scores)
    best = jnp.max(jnp.where(finite, scores, -jnp.inf))
    ref_score = scores[ref_idx]
    nonempty = jnp.any(accepted)
    comparable = nonempty & usable & jnp.any(finite) & jnp.isfinite(ref_score)
    dominance = comparable & ~accepted[ref_idx] & (ref_score > best)
    material = comparable & accepted[ref_idx] & jnp.isfinite(best) & (best - ref_score <= jnp.float32(0.05))
    request = (~nonempty & usable) | dominance | material
    chosen = jnp.where(request, reference, prediction["network_gain"])
    if consensus:
        # Compare member-by-member progress after the same gain-switch cost.
        progress = prediction["prediction_mean"][..., 1]
        cost = switch_penalty * jnp.mean(jnp.abs(jnp.log(bank / previous.astype(jnp.float32))), axis=-1)
        k = prediction["selected_index"]
        index = jnp.clip(k, 0, len(bank) - 1)
        member_score = progress - cost[None, :]
        advantage = member_score[:, index] - member_score[:, ref_idx]
        ref_ok = (
            usable
            & jnp.isfinite(ref_score)
            & jnp.all(jnp.isfinite(progress[:, ref_idx]))
            & jnp.isfinite(cost[ref_idx])
        )
        candidate = (
            (k >= 0)
            & (k < len(bank))
            & accepted[index]
            & jnp.isfinite(scores[index])
            & jnp.all(jnp.isfinite(progress[:, index]))
            & jnp.isfinite(cost[index])
        )
        votes = jnp.isfinite(advantage) & (advantage > jnp.float32(0.0))
        mean = jnp.mean(advantage)
        sd = jnp.sqrt(jnp.mean((advantage - mean) ** 2))
        ratio = sd / mean
        absolute = jnp.all(jnp.isfinite(advantage) & (advantage > jnp.float32(0.05)))
        agreement = (
            jnp.all(votes)
            & jnp.isfinite(mean)
            & (mean > 0)
            & jnp.isfinite(sd)
            & jnp.isfinite(ratio)
            & (ratio <= jnp.float32(0.15))
            & ((ratio <= jnp.float32(0.10)) | absolute)
        )
        allow = candidate & jnp.all(chosen == bank[index].astype(previous.dtype)) & agreement
        chosen = jnp.where(ref_ok & jnp.any(chosen != reference) & ~allow, reference, chosen)
    return chosen


def _recover(carry, k, old_carry, old, args, config, bank):
    """On a genuine unapplied stop, solve the original observed QP across its gain bank."""
    c = config
    x, _, count, cursor = carry[:4]
    seen = old["observed"]
    obs, goal, mask, points, rm, noise = (
        args[n] for n in ("obstacles", "goal", "mask", "points", "route_mask", "noise")
    )
    dt = jnp.asarray(np.asarray(c.robot.dt, np.float64), x.dtype)
    true_obs = obs.at[:, :2].add(k.astype(x.dtype) * dt * obs[:, 3:5])
    _, sensed_obs = observe(x, true_obs, mask, args["bx"], args["bo"], noise, args["ix"][k], args["io"][k])
    reference, a0, b0, p0, d0, target, progress, remaining, visible = observed_problem(
        seen, goal, sensed_obs, mask, bank[0], points, rm, cursor, noise, c, True, nominal_bias=old_carry[6]
    )
    inflated = controller_obstacles(sensed_obs, mask, noise)

    def candidate(_, index):
        gain = bank[index]

        def rows(_):
            a, b, psi, domain = quad3d_rows(seen, inflated, mask, gain, c)
            ha, hb = hold_rows(seen, reference, inflated, mask, gain, c)
            ta, tb = transition_rows(seen, reference, inflated, mask, gain, noise, c)
            return jnp.concatenate((a, ha, ta)), jnp.concatenate((b, hb, tb)), psi, domain

        a, b, psi, domain = jax.lax.cond(index == 0, lambda _: (a0, b0, p0, d0), rows, None)
        checked = jnp.all(gain == old["controller_gain"])
        primary = jnp.all(gain == old["network_gain"])
        held = jnp.all(gain == old["previous_gain"])
        reused = checked | primary | held
        checked_result = (old["proposed"], old["feasible"], old["residual"], old["iterations"])
        primary_result = (
            old["attempted_control"],
            old["attempted_feasible"],
            old["attempted_residual"],
            old["attempted_iterations"],
        )
        held_result = (old["held_proposed"], old["held_feasible"], old["held_residual"], old["held_iterations"])
        cached = jax.lax.cond(
            checked,
            lambda _: checked_result,
            lambda _: jax.lax.cond(primary, lambda _: primary_result, lambda _: held_result, None),
            None,
        )
        admitted = (psi >= -c.qp_tolerance) & (domain >= -c.qp_tolerance)

        def optimize(_):
            return solve_qp4(
                reference, a, b, c.qp_tolerance, (c.robot.input_min, c.robot.input_max), refinement=c.qp_refinement
            )

        rejected = (jnp.zeros(4, x.dtype), jnp.bool_(False), jnp.asarray(np.inf, x.dtype), jnp.int32(0))
        u, feasible, residual, iterations = jax.lax.cond(
            reused, lambda _: cached, lambda _: jax.lax.cond(admitted, optimize, lambda _: rejected, None), None
        )
        actual = jnp.max(a @ u - b)
        box = jnp.maximum(jnp.max(c.robot.input_min - u), jnp.max(u - c.robot.input_max))
        eligible = (
            admitted & feasible & jnp.all(jnp.isfinite(u)) & (actual <= c.qp_tolerance) & (box <= c.qp_tolerance)
        )
        objective = jnp.where(eligible, jnp.sum((u - reference) ** 2), jnp.asarray(np.inf, x.dtype))
        return None, (u, feasible, psi, domain, residual, iterations, eligible, objective)

    _, data = jax.lax.scan(candidate, None, jnp.arange(len(bank), dtype=jnp.int32))
    controls, feasible, psi, domain, residual, iterations, eligible, objective = data
    index = jnp.argmin(objective)
    u, gain = controls[index], bank[index]

    def apply(_):
        yy, sub = integrate_quad3d(x, u, c.robot)
        times = (
            k + jnp.asarray(np.arange(1, c.robot.integration_substeps + 1) / c.robot.integration_substeps, x.dtype)
        ) * dt
        centers = obs[None, :, :2] + times[:, None, None] * obs[None, :, 3:5]
        distance = jnp.linalg.norm(sub[:, None, :2] - centers, axis=-1) - c.robot.radius - obs[None, :, 2]
        clearance = jnp.min(jnp.where(mask[None, :], distance, jnp.inf))
        limits = jnp.asarray(
            np.array([c.tilt_limit, c.tilt_limit, c.yaw_limit] + [c.velocity_limit] * 3 + [c.rate_limit] * 3),
            x.dtype,
        )
        violation = jnp.maximum(
            jnp.max(jnp.abs(sub[:, 3:]) - limits),
            jnp.maximum(jnp.max(sub[:, 2] - c.altitude_max), jnp.max(c.altitude_min - sub[:, 2])),
        )
        status = jnp.where(clearance <= 0, 4, jnp.where(violation > c.qp_tolerance, 5, 0)).astype(jnp.int32)
        updated = (yy, status, count + jnp.int32(1), progress, u, gain, old_carry[6], old_carry[7], old_carry[8])
        record = dict(
            old,
            next_state=yy,
            control=u,
            proposed=u,
            active=jnp.bool_(True),
            status=status,
            feasible=feasible[index],
            psi=psi[index],
            domain=domain[index],
            residual=residual[index],
            iterations=iterations[index],
            clearance=clearance,
            envelope=violation,
            route_target=target,
            route_progress=progress,
            route_remaining=remaining,
            route_visible=visible,
            controller_gain=gain,
        )
        return updated, record

    return jax.lax.cond(jnp.any(eligible), apply, lambda _: (old_carry, old), None)


def build(models, method):
    """Build a calibrated bank selector and its physical recovery controller."""
    method = model_name(method)
    gat = method == "gat"
    configure_precision("quad3d")
    p, spec = selector(models, "quad3d", method)
    p.config = replace(p.config, adverse_probability_limit=0.02)
    if gat:
        p.robot = replace(p.robot, cruise_speed=0.60, horizontal_frequency=1.20)
    Quad3DSelector._build_predictor(p)
    p = InputSupportSelector(p)
    predict = p.predict
    bank = jnp.asarray(p.bank, jnp.float32)
    if not np.array_equal(p.bank[5], np.full(4, 4.0)):
        raise ValueError("The published gain bank requires reference (4,4,4,4) at index five")

    def predict_gain(
        params, x, goal, obs, mask, points, rm, cursor, previous_u, previous_gain, noise, nominal_bias=None
    ):
        pred = predict(
            params, x, goal, obs, mask, points, rm, cursor, previous_u, previous_gain, noise, nominal_bias
        )
        chosen = _advisory(pred, previous_gain, bank, p.config.gain_switch_penalty, gat)
        return dict(pred, network_gain=chosen)

    p.predict = predict_gain
    setup = make_policy_rollout(p, STEPS, failure_requery=spec["failure_requery"], return_stepper=True)

    def episode(params, args):
        base_tick, initial, finish, truth = setup(
            params,
            args["initial"],
            args["goal"],
            args["obstacles"],
            args["mask"],
            jnp.full(4, p.config.initial_gain, jnp.float64),
            args["points"],
            args["route_mask"],
            args["noise"],
            args["bx"],
            args["bo"],
            args["ix"],
            args["io"],
        )

        def tick(carry, k):
            old_carry, (old, _) = base_tick(carry, k)
            stop = (carry[1] == 0) & ~old["active"] & ((old["status"] == 2) | (old["status"] == 3))
            updated, data = jax.lax.cond(
                stop,
                lambda _: _recover(carry, k, old_carry, old, args, p.robot, bank.astype(jnp.float64)),
                lambda _: (old_carry, old),
                None,
            )
            trace = {
                "state": data["next_state"],
                "control": data["control"],
                "active": data["active"],
                "status": data["status"],
                "clearance": data["clearance"],
                "state_bound_violation": data["envelope"],
                "qp_violation": data["residual"],
                "psi": data["psi"],
                "domain": data["domain"],
                "gains": data["controller_gain"],
                "query": data["requery"],
                "route_progress": data["route_progress"],
                "observed_state": data["observed"],
                "network_gain": data["network_gain"],
                "failure_requery": data["failure_requery"],
                "qp_switch_fallback": data["qp_switch_fallback"],
                "stop_recovery_called": stop,
            }
            return updated, trace

        elapsed, last, trace = run_ticks(tick, initial)
        return finish(last, elapsed), last[2], elapsed, truth["initial_state"], truth["obstacles"], trace

    return runner(episode, p.predictor.params, p.robot.robot, method, "quad3d")
