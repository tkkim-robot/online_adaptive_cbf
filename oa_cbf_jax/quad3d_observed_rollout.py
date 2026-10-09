"""Shared quad3d observed rollout implementation."""

import numpy as np

import jax

import jax.numpy as jnp

from .quad3d import integrate_quad3d

from .quad3d_control import Quad3DControlConfig, quad3d_problem, solve_qp4

from .quad3d_routing import flight_target

from .quad3d_observation import observe, controller_obstacles, guidance_obstacles, observed_arrived

def observed_problem(x, goal, obstacles, mask, gains, points, route_mask, cursor, noise, config, transition_guard=False, nominal_bias=None):
    """No physical state, sensor bias, innovations, or scene identity accepted."""
    target, proposed, remaining, visible = flight_target(x, goal,
        guidance_obstacles(obstacles, mask, noise), mask, points, route_mask, cursor, config)
    o = controller_obstacles(obstacles, mask, noise)
    reference, a, b, psi, domain = quad3d_problem(x, target, o, mask, gains, config,
        nominal_state=None if nominal_bias is None else x-nominal_bias)
    if transition_guard:
        from .quad3d_transition import transition_rows
        aa, bb = transition_rows(x, reference, o, mask, gains, noise, config)
        a = jnp.concatenate((a, aa)); b = jnp.concatenate((b, bb))
    return reference, a, b, psi, domain, target, proposed, remaining, visible

def observed_control(x, goal, obstacles, mask, gains, points, route_mask, cursor, noise, config, transition_guard=False, nominal_bias=None):
    reference, a, b, psi, domain, target, proposed, remaining, visible = observed_problem(
        x, goal, obstacles, mask, gains, points, route_mask, cursor, noise, config, transition_guard, nominal_bias)
    u, feasible, residual, iterations = solve_qp4(reference, a, b, config.qp_tolerance,
                                                 (config.robot.input_min, config.robot.input_max),refinement=config.qp_refinement)
    return u, feasible, psi, domain, residual, iterations, target, proposed, remaining, visible, observed_arrived(x, goal, noise, config)

def make_observed_rollout(steps=1600, config=Quad3DControlConfig(hold_guard='bernstein_v87'), transition_guard=False):
    c = config
    def rollout(x, goal, obs, mask, gains, points, route_mask, noise, bx, bo, ix, io, initial_cursor=None, initial_nominal_bias=None, gain_schedule=None):
        dt = jnp.asarray(np.asarray(c.robot.dt, np.float64), x.dtype)
        estimating=c.nominal_bias_observer=='innovation_ema_v97'
        if initial_nominal_bias is not None and not estimating:
            raise ValueError('Initial observer memory requires the observer controller')
        if gain_schedule is not None and (not estimating or not transition_guard):
            raise ValueError('Exploration requires the observer and checked transition guard')
        if gain_schedule is not None and gain_schedule.shape!=((steps+199)//200,4):
            # Collection/probes may supply the complete 1600-tick schedule.
            if gain_schedule.shape!=(8,4) or steps>1600:raise ValueError('Incomplete exploration schedule')
        def current(state, k):
            true_o = obs.at[:, :2].add(k.astype(x.dtype)*dt*obs[:, 3:5])
            seen_x, seen_o = observe(state, true_o, mask, bx, bo, noise, ix[k], io[k])
            return seen_x, seen_o
        def tick(carry, k):
            state, status, count, cursor, bias_estimate, previous_seen, previous_u, previous_gain = carry
            seen_x, seen_o = current(state, k)
            if estimating:
                from .quad3d_observation import update_bias
                updated=update_bias(bias_estimate,previous_seen,seen_x,previous_u,noise,c)
                bias_estimate=jnp.where(k>0,updated,bias_estimate)
            if gain_schedule is None:
                result = observed_control(seen_x, goal, seen_o, mask, gains, points, route_mask, cursor, noise, c, transition_guard,
                    nominal_bias=bias_estimate if estimating else None)
                selected_gain=gains
            else:
                from .quad3d_policy_rollout import checked_control
                proposal=jnp.where(k%200==0,gain_schedule[k//200],previous_gain)
                result,selected_gain,retry,attempt=checked_control(seen_x,goal,seen_o,mask,proposal,previous_gain,
                    points,route_mask,cursor,noise,c,nominal_bias=bias_estimate)
            u, feasible, psi, domain, residual, iterations, target, proposed_cursor, remaining, visible, done = result
            status = jnp.where((status == 0) & done, 1, status)
            status = jnp.where((status == 0) & ((psi < -c.qp_tolerance) | (domain < -c.qp_tolerance)), 2, status)
            status = jnp.where((status == 0) & ~feasible, 3, status)
            active = status == 0
            applied = jnp.where(active, u, jnp.zeros(4, state.dtype))
            yy, sub = integrate_quad3d(state, applied, c.robot)
            next_state = jnp.where(active, yy, state)
            # Truth is used only for physical evolution and outcome accounting.
            times = (k+jnp.asarray(np.arange(1, c.robot.integration_substeps+1)/c.robot.integration_substeps, state.dtype))*dt
            centers = obs[None, :, :2]+times[:, None, None]*obs[None, :, 3:5]
            distances = jnp.linalg.norm(sub[:, None, :2]-centers, axis=-1)-c.robot.radius-obs[None, :, 2]
            clear = jnp.min(jnp.where(mask[None, :], distances, jnp.inf))
            limits = jnp.asarray(np.array([c.tilt_limit, c.tilt_limit, c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3), state.dtype)
            violation = jnp.maximum(jnp.max(jnp.abs(sub[:, 3:])-limits), jnp.maximum(jnp.max(sub[:, 2]-c.altitude_max), jnp.max(c.altitude_min-sub[:, 2])))
            status = jnp.where(active & (clear <= 0), 4, status)
            status = jnp.where(active & (status == 0) & (violation > c.qp_tolerance), 5, status)
            next_cursor = jnp.where(active, proposed_cursor, cursor)
            data = dict(state=state, next_state=next_state, observed=seen_x, control=applied, proposed=u,
                active=active, status=status, feasible=feasible, psi=psi, domain=domain, residual=residual,
                iterations=iterations, clearance=clear, envelope=violation, route_target=target,
                route_cursor_before=cursor, route_progress=next_cursor, route_remaining=remaining, route_visible=visible)
            if estimating:data.update(nominal_bias_estimate=bias_estimate,nominal_observation=seen_x-bias_estimate)
            if gain_schedule is not None:
                data.update(controller_gain=selected_gain,scheduled_gain=proposal,previous_gain=previous_gain,
                    previous_control=previous_u,qp_switch_fallback=retry,**attempt)
            return (next_state, status, count+active.astype(jnp.int32), next_cursor, bias_estimate,seen_x,
                    jnp.where(active,applied,previous_u),jnp.where(active,selected_gain,previous_gain)), data
        cursor0=jnp.zeros((),x.dtype) if initial_cursor is None else initial_cursor
        bias0=jnp.zeros(12,x.dtype) if initial_nominal_bias is None else initial_nominal_bias
        result, trace = jax.lax.scan(tick, (x, jnp.int32(0), jnp.int32(0), cursor0,bias0,jnp.zeros(12,x.dtype),jnp.zeros(4,x.dtype),gains), jnp.arange(steps, dtype=jnp.int32))
        state, status, count, cursor, *_ = result
        final_seen, _ = current(state, jnp.int32(steps))
        status = jnp.where((status == 0) & observed_arrived(final_seen, goal, noise, c), 1, status)
        status = jnp.where(status == 0, 6, status)
        return dict(final_state=state, status=status, steps=count), trace
    return rollout
