"""Compare a learned proposal with the current gain, without gain search."""
import jax
import jax.numpy as jnp
import numpy as np


def comparison(selection, previous, bank):
    matches = bank[None, :, 0] == previous[:, None]
    index = jnp.argmax(matches, axis=1)
    rows = jnp.arange(len(previous))
    current = selection['ranking_score'][rows, index]
    proposed = selection['ranking_score'][rows, jnp.maximum(selection['selected_index'], 0)]
    needed = ((matches.sum(axis=1) == 1) & jnp.isfinite(current) & jnp.isfinite(proposed)
              & (selection['selected_index'] >= 0)
              & (selection['controller_gain'] != previous) & (proposed <= current))
    return needed, current, proposed


def retain(selection, previous, bank, checked, feasible):
    needed, current, proposed = comparison(selection, previous, bank)
    held = needed & checked & feasible
    result = dict(selection)
    result.update(controller_gain=jnp.where(held, previous, selection['controller_gain']),
        selected_index=jnp.where(held, -3, selection['selected_index']),
        # A progress-based hold is distinct from an uncertainty fallback.
        uncertainty_fallback=jnp.where(held, False, selection['uncertainty_fallback']),
        incumbent_comparison_needed=needed, incumbent_progress_hold=held,
        incumbent_previous_score=current, incumbent_proposed_score=proposed,
        incumbent_witness_checked=checked, incumbent_witness_feasible=feasible,
        selection_eligible=selection['accepted'] & ~held[:, None])
    return result


def filter_proposal(selection, x, goal, obstacles, mask, points, route_mask, cursor,
                    previous, noise, bank, config, guidance):
    """Check only the current gain when a learned proposal would score worse.

    The current gain uses the unchanged predictive controller and original QP.
    No alternative gain is solved here, and no physical future is consulted.
    The ordinary final QP still checks whichever gain is actually applied.
    """
    from .bicycle_guidance import guided_bicycle_control
    from .bicycle_observation import speed_error_bound
    needed, _, _ = comparison(selection, previous, bank)

    def witness(a, g, o, m, p, rm, c, gain, n):
        qp, h, domain, *_ = guided_bicycle_control(a, g, o, m, gain, p, rm, c,
            config, guidance, speed_error_bound(n), noise=n)
        feasible = qp.feasible & (h >= -config.qp_tolerance) & (domain > 0)
        return feasible, qp.control, h, domain

    batch = len(previous)
    empty = (jnp.zeros(batch, bool), jnp.zeros((batch, 2), jnp.float32),
             jnp.zeros(batch, jnp.float32), jnp.zeros(batch, jnp.float32))
    feasible, control, h, domain = jax.lax.cond(jnp.any(needed),
        lambda _: jax.vmap(witness)(x, goal, obstacles, mask, points, route_mask, cursor, previous, noise),
        lambda _: empty, operand=None)
    result = retain(selection, previous, bank, needed, needed & feasible)
    result.update(incumbent_witness_control=jnp.where(needed[:, None], control, 0.),
                  incumbent_witness_h=jnp.where(needed, h, 0.),
                  incumbent_witness_domain=jnp.where(needed, domain, 0.))
    return result


def audit_witnesses(data, config):
    """Independent NumPy CBF rows and polygon feasibility, no JAX replay."""
    from .bicycle_audit import reference_rows, polygon_qp
    checked = np.asarray(data['incumbent_witness_checked'], bool)
    feasible = np.asarray(data['incumbent_witness_feasible'], bool)
    np.testing.assert_array_equal(checked, data['incumbent_comparison_needed'])
    if np.any(feasible & ~checked):
        raise ValueError('Uncomputed incumbent marked feasible')
    c = config.robot
    for k in np.flatnonzero(checked):
        x = data['observed_state'][k].astype(float)
        o = data['observed_obstacles'][k].astype(float)
        geometry = np.sum((o[:, :2]-x[:2])**2, axis=1)-((c.radius+config.clearance_buffer+o[:, 2])*config.barrier_inflation)**2
        if np.any(geometry[data['mask']] <= 0):
            if feasible[k]:raise ValueError('Incumbent witness outside barrier domain')
            np.testing.assert_allclose(data['incumbent_witness_domain'][k], np.min(geometry[data['mask']]), atol=3e-5, rtol=2e-6)
            continue  # Complex-step derivatives have no real value here.
        a, b, h, domain = reference_rows(x, o, data['mask'], float(data['previous_gain'][k]), config)
        error = 1.15*float(data['noise'][2])+(5e-7 if data['noise'][2] > 0 else 0.)
        b[-4] = min(c.acceleration_max, (c.speed_max-x[3]-error)/c.dt)
        b[-3] = -max(-c.acceleration_max, (c.speed_min-x[3]+error)/c.dt)
        hmin = np.min(h[data['mask']], initial=np.inf)
        dmin = np.min(domain[data['mask']], initial=np.inf)
        np.testing.assert_allclose(data['incumbent_witness_h'][k], hmin, atol=3e-5, rtol=2e-6)
        np.testing.assert_allclose(data['incumbent_witness_domain'][k], dmin, atol=3e-5, rtol=2e-6)
        admissible = hmin >= -config.qp_tolerance and dmin > 0
        if feasible[k]:
            u = np.asarray(data['incumbent_witness_control'][k], float)
            if not admissible or not np.isfinite(u).all() or np.max(a@u-b) > config.qp_tolerance:
                raise ValueError('Invalid incumbent feasibility witness')
        elif admissible and polygon_qp([0., 0.], a, b, [1., 1.],
                [-c.acceleration_max, -c.slip_max], [c.acceleration_max, c.slip_max]) is not None:
            raise ValueError('Feasible incumbent incorrectly rejected')
    return dict(incumbent_witnesses=int(checked.sum()),
                incumbent_progress_holds=int(np.asarray(data['incumbent_progress_hold']).sum()),
                incumbent_recovery_proposals=int(np.sum(checked & ~feasible)),
                incumbent_witness_audit_passed=True)
