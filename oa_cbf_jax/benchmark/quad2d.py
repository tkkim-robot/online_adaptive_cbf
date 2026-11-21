"""Paired moving-obstacle Quad2D benchmark with explicit sensor primitives."""

import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ..dynamics import signed_clearance, swept_disk_clearance
from ..model_release import selector
from ..models import predict_ensemble
from ..quad2d import integrate_quad2d
from ..quad2d_control import flight_arrived, physical_envelope_violation
from ..quad2d_features import flight_graph
from ..quad2d_guidance import predictive_flight_control
from ..quad2d_policy import BACKUP, HELD, LEARNED, REJECTED, flight_risk
from ..quad2d_rollout import INADMISSIBLE, PLANNER_FAILURE, STATE_BOUND
from ..simulation import COLLISION, GOAL, INFEASIBLE, RUNNING, TIMEOUT
from ..uncertainty import cs_disagreement
from .protocol import configure_precision
from .sensors import arguments, quad2d_sensor_model

STEPS = 1600
BACKUP_GAINS = ((8, 8), (8, 4), (4, 8), (4, 4), (12, 12), (12, 4))
WIDE_ANGLES = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, -120.0, 120.0, -150.0, 150.0, 180.0)


def model_name(method):
    if method in ("ours_gat", "gat"):
        return "gat"
    if method in ("ours_fc", "fc", "nearest_fc"):
        return "nearest_fc"
    raise ValueError(f"Unknown learned method {method!r}")


def _at(values, index):
    return jax.tree.map(lambda value: value[index], values)


def run_ticks(tick, initial):
    """Stop on the first terminal event, counting its observation as one tick."""
    template = jax.eval_shape(tick, initial, jnp.int32(0))[1]
    storage = jax.tree.map(lambda value: jnp.zeros((STEPS, *value.shape), value.dtype), template)

    def running(state):
        k, carry, _ = state
        return (k < STEPS) & ((k == 0) | (carry[1] == 0))

    def advance(state):
        k, carry, history = state
        carry, record = tick(carry, k)
        history = jax.tree.map(lambda h, v: jax.lax.dynamic_update_index_in_dim(h, v, k, 0), history, record)
        return k + 1, carry, history

    return jax.lax.while_loop(running, advance, (jnp.int32(0), initial, storage))


def runner(episode, params, config, method, dynamics):
    prepare = jax.jit(episode)
    compiled = {}

    def run(case):
        configure_precision(dynamics)
        args = arguments(case)
        signature = tuple((key, value.shape, value.dtype) for key, value in sorted(args.items()))
        setup_seconds = 0.0
        if signature not in compiled:
            started = time.perf_counter()
            compiled[signature] = prepare.lower(params, args).compile()
            setup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        output = jax.block_until_ready(compiled[signature](params, args))
        execution_seconds = time.perf_counter() - started
        status, count, elapsed, initial, obstacles, history = jax.device_get(output)
        elapsed = int(elapsed)
        trace = {name: values[:elapsed] for name, values in history.items()}
        trace.update(
            states=np.vstack((initial, trace["state"])),
            true_initial_state=initial,
            true_obstacles=obstacles,
            obstacles=obstacles,
            mask=np.asarray(args["mask"]),
            goal=np.asarray(args["goal"]),
            dt=np.asarray(config.dt),
            radius=np.asarray(config.radius),
        )
        mask = trace["mask"]
        initial_distance = np.min(
            np.where(
                mask,
                np.linalg.norm(initial[:2] - obstacles[:, :2], axis=-1) - config.radius - obstacles[:, 2],
                np.inf,
            )
        )
        active = trace["active"]
        minimum = np.minimum(initial_distance, np.min(trace["clearance"][active], initial=np.inf))
        worst = np.max(trace["qp_violation"][active], initial=-np.inf)
        report = {
            "dynamics": dynamics,
            "method": method,
            "status": int(status),
            "success": int(status) == 1,
            "steps": int(count),
            "ticks": elapsed,
            "min_clearance": float(minimum) if np.isfinite(minimum) else None,
            "max_qp_violation": float(worst) if np.isfinite(worst) else None,
            "physical_navigation_seconds": int(count) * config.dt,
            "execution_seconds": execution_seconds,
            "setup_compile_seconds": setup_seconds,
        }
        return report, trace

    return run


def build(models, method):
    """Build the selected GAT or four-query-confirmation FC for moving obstacles."""
    method = model_name(method)
    gat = method == "gat"
    configure_precision("quad2d")
    p, spec = selector(models, "quad2d", method)
    config = replace(p.config, stationary_obstacles=False)
    flight = config.robot
    policy = p.policy
    guidance = replace(p.guidance, clearance_reward=0.45, angles_degrees=WIDE_ANGLES) if gat else p.guidance
    predictive = replace(config, cruise_speed=1.0) if gat else config
    short_guidance = replace(guidance, horizon=16)
    candidates = jnp.asarray(spec["candidates"], jnp.float32)
    scale = jnp.asarray(p.norm["target_scale"], jnp.float32)
    mean = jnp.asarray(p.norm["target_mean"], jnp.float32)
    cal = p.calibration
    penalty = 0.015 if gat else 0.01

    def episode(params, a):
        x0, goal, obs, mask = (a[k] for k in ("initial", "goal", "obstacles", "mask"))
        points, route_mask, noise = (a[k] for k in ("points", "route_mask", "noise"))
        initial, truth, xb, ob, xs, os, innovations = quad2d_sensor_model(
            x0,
            obs,
            mask,
            noise,
            a["key"],
            STEPS,
            config.stationary_obstacles,
            unit_x=a["unit_x"],
            unit_o=a["unit_o"],
            unit_i=a["unit_i"],
        )
        clearance = jnp.min(signed_clearance(initial[:2], truth, mask, flight.radius))
        status = jnp.where(flight_arrived(initial, goal, config), GOAL, RUNNING)
        status = jnp.where(physical_envelope_violation(initial, config) > flight.qp_tolerance, STATE_BOUND, status)
        status = jnp.where(clearance <= 0, COLLISION, status)
        status = jnp.where(a["ready"], status, PLANNER_FAILURE)
        initial_carry = (
            initial,
            status,
            jnp.int32(0),
            jnp.float32(0),
            jnp.full(2, flight.mass * flight.gravity / 2, jnp.float32),
            jnp.full(2, 8.0, jnp.float32),
            jnp.int32(-1),
            jnp.int32(0),
            jnp.zeros(2, jnp.float32),
        )

        def tick(carry, k):
            x, status, count, cursor, previous_u, gain, pending_index, pending_count, pending_gain = carry
            active = status == RUNNING
            innovation = innovations[k]
            sensed = x - xb + 0.15 * xs * innovation[:6]
            seen = (
                truth.at[:, :2].set(truth[:, :2] + k * flight.dt * truth[:, 3:5])
                - ob
                + 0.15 * os * innovation[6:].reshape(obs.shape)
            )

            def calculate(g, guide=guidance):
                return predictive_flight_control(
                    sensed, goal, seen, mask, g, points, route_mask, cursor, predictive, guide, noise, None
                )

            def valid(results, infos):
                qp, h, psi, domain, *_ = results
                return (
                    infos["approved"]
                    & qp.feasible
                    & (h >= -flight.qp_tolerance)
                    & (psi >= -flight.qp_tolerance)
                    & (domain >= -flight.qp_tolerance)
                )

            def query(failed_hold):
                previous_result, previous_info = jax.vmap(calculate)(gain[None])
                previous_ok = valid(previous_result, previous_info)[0]
                incumbent = (
                    gain,
                    previous_ok,
                    jnp.int32(BACKUP),
                    previous_ok.astype(jnp.int32),
                    _at(previous_result, 0),
                    _at(previous_info, 0),
                )
                pool = jnp.concatenate((candidates, gain[None]))
                features, node_mask = flight_graph(
                    sensed, goal, seen, mask, points, route_mask, cursor, previous_u, gain, noise, config
                )
                out = predict_ensemble(p.model, params, features[None], node_mask[None], pool[None])
                mu = out["mean"][:, 0] * scale + mean
                variance = jnp.exp(out["log_variance"][:, 0]) * scale**2 * cal["variance_scale"]
                cs = cs_disagreement(mu[:, :, 0].T[..., None], variance[:, :, 0].T[..., None])
                risk = flight_risk(mu[:, :, 0].T, variance[:, :, 0].T, policy)
                event = jnp.max(
                    jax.nn.sigmoid(out["event_logits"][:, 0] / cal["temperature"] + cal["bias"]), axis=0
                )
                finite = (
                    jnp.all(jnp.isfinite(mu) & jnp.isfinite(variance), axis=(0, 2))
                    & jnp.all(jnp.isfinite(event), axis=-1)
                    & jnp.isfinite(cs)
                    & jnp.isfinite(risk)
                )
                epistemic = finite & (cs <= cal["cs_threshold"])
                risk_ok = epistemic & (risk <= policy.risk_threshold)
                admitted = (
                    risk_ok
                    & (event[:, 0] <= policy.collision_probability_limit)
                    & (event[:, 1] <= policy.failure_probability_limit)
                )
                ranking = jnp.mean(mu[:, :, 1], axis=0) - penalty * jnp.sum(jnp.log(pool / gain) ** 2, axis=-1)
                _, indices = jax.lax.top_k(jnp.where(admitted, ranking, -jnp.inf), policy.shortlist)
                gains = pool[indices]
                results, infos = jax.vmap(calculate)(gains)
                ok = (
                    valid(results, infos) & admitted[indices] & ((~previous_ok) | (ranking[indices] > ranking[-1]))
                )
                index = jnp.argmax(jnp.where(ok, ranking[indices], -jnp.inf))
                primary_ok = jnp.any(ok)
                primary = (
                    gains[index],
                    jnp.bool_(True),
                    jnp.int32(LEARNED),
                    jnp.int32(0),
                    _at(results, index),
                    _at(infos, index),
                )

                def recovery(_):
                    backups = jnp.asarray(BACKUP_GAINS, sensed.dtype)

                    def search(bank, guide):
                        results, infos = jax.vmap(lambda g: calculate(g, guide))(bank)
                        ok = valid(results, infos)
                        scores = infos["score"] - 0.01 * jnp.sum(jnp.log(bank / gain) ** 2, axis=-1)
                        i = jnp.argmax(jnp.where(ok, scores, -jnp.inf))
                        return (
                            bank[i],
                            jnp.any(ok),
                            jnp.int32(BACKUP),
                            jnp.sum(ok).astype(jnp.int32),
                            _at(results, i),
                            _at(infos, i),
                        )

                    full = search(backups, guidance)
                    return jax.lax.cond(
                        full[1],
                        lambda _: full,
                        lambda _: search(jnp.concatenate((gain[None], backups)), short_guidance),
                        None,
                    )

                def backup(_):
                    return jax.lax.cond(previous_ok, lambda _: incumbent, recovery, None) if gat else incumbent

                chosen = jax.lax.cond(primary_ok, lambda _: primary, backup, None)
                next_index, next_count, next_gain = pending_index, pending_count, pending_gain
                if not gat:
                    # Require the same in-bank nomination on four consecutive queries.
                    # A failed physical witness bypasses the optional deferral.
                    proposal_index = jnp.where(primary_ok, indices[index], -1).astype(jnp.int32)
                    eligible = (
                        previous_ok
                        & ~failed_hold
                        & primary_ok
                        & (proposal_index >= 0)
                        & (proposal_index < len(candidates))
                        & jnp.any(chosen[0] != gain)
                    )
                    match = (pending_index == proposal_index) & jnp.all(pending_gain == chosen[0])
                    n = jnp.where(eligible, jnp.where(match, pending_count + 1, 1), 0).astype(jnp.int32)
                    deferred = eligible & (n < 4)
                    next_count = jnp.where(deferred, n, 0).astype(jnp.int32)
                    next_index = jnp.where(deferred, proposal_index, -1).astype(jnp.int32)
                    next_gain = jnp.where(deferred, chosen[0], jnp.zeros_like(gain))
                    chosen = jax.lax.cond(deferred, lambda _: incumbent, lambda _: chosen, None)
                selected, accepted, source, n_backup, result, info = chosen
                stages = jnp.stack(
                    (finite.sum(), epistemic.sum(), risk_ok.sum(), admitted.sum(), n_backup)
                ).astype(jnp.int32)
                return (
                    selected,
                    accepted,
                    jnp.where(accepted, source, REJECTED).astype(jnp.int32),
                    stages,
                    result,
                    info,
                    jnp.bool_(True),
                    next_index,
                    next_count,
                    next_gain,
                )

            def hold(_):
                result, info = calculate(gain)
                ok = valid(result, info)
                held = (
                    gain,
                    jnp.bool_(True),
                    jnp.int32(HELD),
                    jnp.zeros(5, jnp.int32),
                    result,
                    info,
                    jnp.bool_(False),
                    pending_index,
                    pending_count,
                    pending_gain,
                )
                return jax.lax.cond(ok, lambda _: held, query, jnp.bool_(True))

            selected, ok, source, stages, result, info, due, pending_index, pending_count, pending_gain = (
                jax.lax.cond(k % policy.interval == 0, query, hold, jnp.bool_(False))
            )
            qp, h, psi, domain, proposed, _, _ = result
            admissible = (
                (h >= -flight.qp_tolerance) & (psi >= -flight.qp_tolerance) & (domain >= -flight.qp_tolerance)
            )
            accepted = active & ok & qp.feasible & admissible
            u = jnp.where(accepted, qp.control, jnp.zeros(2, x.dtype))
            y, sub = integrate_quad2d(x, u, flight)
            starts = jnp.concatenate((x[None], sub[:-1]))
            dt = flight.dt / flight.integration_substeps
            times = (
                k * flight.dt + jnp.arange(flight.integration_substeps) * flight.dt / flight.integration_substeps
            )
            clear = jnp.min(
                jax.vmap(
                    lambda start, end, t: swept_disk_clearance(start, end, truth, mask, flight.radius, t, t + dt)
                )(starts, sub, times)
            )
            bound = jnp.max(
                jax.vmap(lambda state: physical_envelope_violation(state, config))(jnp.concatenate((x[None], sub)))
            )
            status = jnp.where(active & ~qp.feasible, INFEASIBLE, status)
            status = jnp.where(active & ~admissible, INADMISSIBLE, status)
            status = jnp.where(active & ~ok, 6, status)
            status = jnp.where(accepted & flight_arrived(y, goal, config), GOAL, status)
            status = jnp.where(accepted & (bound > flight.qp_tolerance), STATE_BOUND, status)
            status = jnp.where(accepted & (clear <= 0), COLLISION, status)
            x = jnp.where(accepted, y, x)
            cursor = jnp.where(accepted, proposed, cursor)
            carry = (
                x,
                status,
                count + accepted.astype(jnp.int32),
                cursor,
                jnp.where(accepted, u, previous_u),
                jnp.where(accepted, selected, gain),
                pending_index,
                pending_count,
                pending_gain,
            )
            record = {
                "state": x,
                "control": u,
                "active": accepted,
                "status": status,
                "gains": selected,
                "source": source,
                "query": active & due,
                "stages": stages,
                "h": h,
                "psi": psi,
                "domain": domain,
                "clearance": jnp.where(accepted, clear, jnp.nan),
                "state_bound_violation": jnp.where(accepted, bound, jnp.nan),
                "qp_violation": jnp.where(accepted, qp.max_violation, jnp.nan),
                "observed_state": sensed,
                "observed_obstacles": seen,
                "route_progress": cursor,
                "guidance_score": info["score"],
                "guidance_valid_candidates": info["valid_candidates"],
            }
            return carry, record

        elapsed, last, history = run_ticks(tick, initial_carry)
        return jnp.where(last[1] == RUNNING, TIMEOUT, last[1]), last[2], elapsed, initial, truth, history

    return runner(episode, p.params, flight, method, "quad2d")
