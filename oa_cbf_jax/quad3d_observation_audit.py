"""Independent observation, route, CBF and continuous physical trace accounting."""
from pathlib import Path
import numpy as np
from .dataset import sha256
from .quad3d_control import control_config
from .quad3d_audit import (audit_hold, independent_obstacle_values,
    independent_envelope_values, independent_held_cascade_minimum)
from .quad3d_routing import numpy_flight_target
from .quad3d_observation import unit_tape, numpy_observe, numpy_obstacles, numpy_arrived


def audit_parent(arguments):
    p, row, directory, manifest = arguments
    path = Path(directory)/row['file']
    assert p['id'] == row['id'] and sha256(path) == row['sha256']
    c = control_config(manifest['config'])
    with np.load(path) as z:
        d = dict(z)
    count = row['steps']; length = min(manifest['steps'], count+1)
    assert len(d['active']) == length
    np.testing.assert_array_equal(np.flatnonzero(d['active']), np.arange(count))
    np.testing.assert_array_equal(d['control'][~d['active']], np.zeros((length-count, 4)))
    np.testing.assert_array_equal(d['state'][0], p['x'])
    np.testing.assert_array_equal(d['next_state'][-1], row['final_state'])
    x = np.asarray(p['x']); o = np.asarray(p['obstacles']); mask = np.asarray(p['mask'])
    noise = np.asarray(p['noise']); goal = np.asarray(p['goal'])
    points = np.asarray(p['route']['points']); rm = np.asarray(p['route']['mask'])
    ordered=manifest.get('ordered_mission',False);leg=0
    if ordered:
        goals=np.asarray(p['waypoint_goals']);total=p['waypoint_count'];goal=goals[0]
        routes=np.asarray(p['waypoint_routes']['points']);route_masks=np.asarray(p['waypoint_routes']['mask']);points=routes[0];rm=route_masks[0]
    bx, bo, ix, io = unit_tape(p['sensor_seed'], manifest['steps'], len(mask))
    if 'branch' in p:
        q=p['branch']['query_tick']
        bx,bo,qx,qo=unit_tape(p['sensor_seed'],q,len(mask))
        _,_,ix,io=unit_tape(p['branch']['future_seed'],manifest['steps'],len(mask))
        ix[0]=qx[q];io[0]=qo[q]
    min_clear = float(np.min(np.linalg.norm(x[:2]-o[mask, :2], axis=-1)-c.robot.radius-o[mask, 2], initial=np.inf))
    limits = np.array([c.tilt_limit, c.tilt_limit, c.yaw_limit]+[c.velocity_limit]*3+[c.rate_limit]*3)
    collision = min_clear <= 0
    envelope = bool(np.any(abs(x[3:])-limits > c.qp_tolerance) or x[2] > c.altitude_max+c.qp_tolerance or x[2] < c.altitude_min-c.qp_tolerance)
    first_stop=0 if collision or envelope else None
    first_status=4 if collision else 5 if envelope else None
    first_clear=min_clear if first_stop is not None else None
    max_error = 0.; min_held = np.inf; cursor = float(p.get('initial_cursor',0.)); status = 0
    true_cbf_negative = 0; min_true_psi = np.inf; min_transition = np.inf
    estimated_bias=np.asarray(p.get('initial_nominal_bias',np.zeros(12)),dtype=np.float64)
    for k in range(length):
        gains=d['controller_gain'][k] if manifest.get('adaptive_gain_trace',False) else p['gains']
        if k:
            np.testing.assert_array_equal(d['state'][k], d['next_state'][k-1])
        truth_o = o.copy(); truth_o[:, :2] += k*c.robot.dt*o[:, 3:5]
        seen_x, seen_o = numpy_observe(d['state'][k], truth_o, mask, bx, bo, noise, ix[k], io[k])
        np.testing.assert_allclose(d['observed'][k], seen_x, atol=2e-12, rtol=1e-12)
        if ordered:
            handoff=status==0 and leg<total-1 and numpy_arrived(seen_x,goals[leg],noise,c)
            if handoff:
                assert numpy_arrived(d['state'][k],goals[leg],np.zeros(7),c),'Waypoint handoff was not true full-state arrival'
                leg+=1;cursor=0.
            goal=goals[leg];points=routes[leg];rm=route_masks[leg]
            assert d['waypoint_index'][k]==leg and d['waypoint_handoff'][k]==handoff
            np.testing.assert_array_equal(d['mission_goal'][k],goal)
        if c.nominal_bias_observer=='innovation_ema_v97':
            from .quad3d_observer import numpy_update_bias
            if k:estimated_bias=numpy_update_bias(estimated_bias,d['observed'][k-1],seen_x,d['control'][k-1],noise,c)
            np.testing.assert_allclose(d['nominal_bias_estimate'][k],estimated_bias,atol=2e-10,rtol=1e-9)
            np.testing.assert_allclose(d['nominal_observation'][k],seen_x-estimated_bias,atol=2e-10,rtol=1e-9)
        controlled_o = numpy_obstacles(seen_o, mask, noise)
        target, progress, remaining, visible = numpy_flight_target(seen_x, goal,
            numpy_obstacles(seen_o, mask, noise, guidance=True), mask, points, rm, cursor, c)
        np.testing.assert_allclose(d['route_cursor_before'][k], cursor, atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(d['route_target'][k], target, atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(d['route_remaining'][k], remaining, atol=1e-8, rtol=1e-10)
        assert bool(d['route_visible'][k]) == visible
        # Lower cascades do not depend on u; use zero for rejecting/nonfinite QPs.
        psi, _ = independent_obstacle_values(seen_x, np.zeros(4), controlled_o, mask, gains, c)
        domain, _ = independent_envelope_values(seen_x, np.zeros(4), c)
        np.testing.assert_allclose(d['psi'][k], psi, atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(d['domain'][k], domain, atol=1e-8, rtol=1e-10)
        if status == 0:
            if (not ordered or leg==total-1) and numpy_arrived(seen_x, goal, noise, c): status = 1
            elif psi < -c.qp_tolerance or domain < -c.qp_tolerance: status = 2
            elif not d['feasible'][k]: status = 3
        assert bool(d['active'][k]) == (status == 0)
        if d['active'][k]:
            u = d['control'][k]
            np.testing.assert_array_equal(u, d['proposed'][k])
            assert d['residual'][k] <= c.qp_tolerance and np.isfinite(u).all()
            _, residual = independent_obstacle_values(seen_x, u, controlled_o, mask, gains, c)
            _, er = independent_envelope_values(seen_x, u, c)
            assert residual >= -c.qp_tolerance-1e-8 and er >= -c.qp_tolerance-1e-8
            held = independent_held_cascade_minimum(seen_x, u, controlled_o, mask, gains, c)
            assert held >= -c.qp_tolerance-1e-8, 'Observed held-cascade certificate failed'
            min_held = min(min_held, held)
            if manifest.get('transition_guard',False):
                from .quad3d_transition import numpy_transition_certificate
                certificate,_,_=numpy_transition_certificate(seen_x,u,controlled_o,mask,np.asarray(gains),noise,c)
                assert certificate >= -c.qp_tolerance-1e-8, 'Independent next-observation domain guard failed'
                min_transition=min(min_transition,certificate)
            actual = audit_hold(d['state'][k], u, d['next_state'][k], truth_o, mask, c)
            min_clear = min(min_clear, actual['minimum_clearance'])
            collision |= actual['minimum_clearance'] <= 0
            envelope |= actual['envelope_violation'] > c.qp_tolerance
            if first_stop is None and (actual['minimum_clearance']<=0 or actual['envelope_violation']>c.qp_tolerance):
                first_stop=k+1;first_status=4 if actual['minimum_clearance']<=0 else 5;first_clear=min_clear
            max_error = max(max_error, actual['replay_error'])
            tp, tr = independent_obstacle_values(d['state'][k], u, truth_o, mask, gains, c)
            min_true_psi = min(min_true_psi, tp)
            true_cbf_negative += int(tp < -c.qp_tolerance or tr < -c.qp_tolerance)
            # A physical audit miss overrides final outcome below. Keep runtime
            # status reconstruction separate from the stronger physical screen.
            if d['clearance'][k] <= 0: status = 4
            elif d['envelope'][k] > c.qp_tolerance: status = 5
            cursor = progress
        else:
            np.testing.assert_array_equal(d['state'][k], d['next_state'][k])
        assert int(d['status'][k]) == status
        np.testing.assert_allclose(d['route_progress'][k], cursor, atol=1e-8, rtol=1e-10)
    if status == 0:
        assert count == manifest['steps']
        truth_o = o.copy(); truth_o[:, :2] += count*c.robot.dt*o[:, 3:5]
        seen, _ = numpy_observe(np.asarray(row['final_state']), truth_o, mask, bx, bo, noise, ix[count], io[count])
        status = 1 if (not ordered or leg==total-1) and numpy_arrived(seen, goal, noise, c) else 6
    assert row['status'] == status
    audited_status = first_status if first_stop is not None else status
    if audited_status == 1:
        final = np.asarray(row['final_state'])
        assert numpy_arrived(final, goal, np.zeros(7), c), 'Observed termination was not a true full-state arrival'
    if ordered:
        assert row['waypoint_index']==leg and row['waypoints_visited']==leg+int(status==1)
        assert int(d['waypoint_handoff'].sum())==leg and (audited_status!=1 or leg==total-1)
    return dict(**row, audited_status=audited_status, physical_collision=bool(collision), envelope_exit=bool(envelope),
        first_physical_stop_step=first_stop,first_physical_stop_status=first_status,first_physical_stop_clearance=first_clear,
        minimum_clearance=min_clear, max_replay_error=max_error, minimum_observed_held_cascade=min_held,
        true_cbf_negative_steps=true_cbf_negative, minimum_true_initial_cascade=min_true_psi,
        minimum_transition_certificate=min_transition,
        nominal_bias_observer_verified=c.nominal_bias_observer=='innovation_ema_v97',
        observation_route_and_continuous_physics_verified=True,
        limitation='Observed CBF satisfaction is not a robust true-state CBF certificate.')
