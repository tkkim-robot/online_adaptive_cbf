"""Default flight MPC on the common noisy nonlinear plant, with full replay.

All parents and failures are retained. CPU NLP solves use default IPOPT settings;
fixed FP32 JAX executables provide the same sensor prior, route and physical step
as the learned flight experiments. No learned filter or rescue action is used.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .quad2d import integrate_quad2d
from .quad2d_control import FlightConfig, flight_arrived, physical_envelope_violation
from .quad2d_rollout import flight_sensor_model, NAMES, GOAL, COLLISION, TIMEOUT, STATE_BOUND, PLANNER_FAILURE
from .quad2d_mpc import Quad2DMPC, DEFAULTS, numpy_barrier, prediction_residual
from .quad2d_audit import check_trace
from .routing import route_target_from_position, physical_route_coordinate
from .dynamics import signed_clearance, swept_disk_clearance
from .dataset import sha256, source_fingerprint
from .io import write_json
from .cli import sanitize


class FlightPhysicalKernels:
    def __init__(self, capacity, route_capacity, steps, config=FlightConfig()):
        with jax.enable_x64(False): self._compile(capacity, route_capacity, steps, config)

    def _compile(self, capacity, route_capacity, steps, config):
        c = config.robot; x = jnp.zeros(6); obs = jnp.zeros((capacity, 5)); mask = jnp.zeros(capacity, bool)
        noise = jnp.zeros(7); points = jnp.zeros((route_capacity, 2)); rm = jnp.ones(route_capacity, bool)
        start = time.perf_counter()
        self.prepare = jax.jit(lambda x, o, m, n, key: flight_sensor_model(x, o, m, n, key, steps)).lower(x, obs, mask, noise, jax.random.PRNGKey(0)).compile()
        def sense(x, truth, xb, ob, xs, os, innovation, k):
            seen = truth.at[:, :2].set(truth[:, :2]+k*c.dt*truth[:, 3:5])-ob+.15*os*innovation[6:].reshape(obs.shape)
            return x-xb+.15*xs*innovation[:6], seen
        self.sense = jax.jit(sense).lower(x, obs, x, obs, x, jnp.zeros(5), jnp.zeros(6+capacity*5), jnp.int32(0)).compile()
        def advance(x, u, truth, mask, k):
            y, sub = integrate_quad2d(x, u, c); starts = jnp.concatenate((x[None], sub[:-1]))
            times = k*c.dt+jnp.arange(c.integration_substeps)*c.dt/c.integration_substeps
            clear = jnp.min(jax.vmap(lambda a, b, t: swept_disk_clearance(a, b, truth, mask, c.radius, t, t+c.dt/c.integration_substeps))(starts, sub, times))
            bound = jnp.max(jax.vmap(lambda s: physical_envelope_violation(s, config))(jnp.concatenate((x[None], sub))))
            return y, clear, bound
        self.advance = jax.jit(advance).lower(x, jnp.zeros(2), obs, mask, jnp.int32(0)).compile()
        self.target = jax.jit(lambda x, p, m, cursor: route_target_from_position(x[:2], jnp.linalg.norm(x[3:5]), p, m, cursor)).lower(x, points, rm, jnp.float32(0)).compile()
        self.coordinate = jax.jit(physical_route_coordinate).lower(x[:2], points, rm, jnp.float32(0)).compile()
        self.clearance = jax.jit(lambda x, o, m: jnp.min(signed_clearance(x[:2], o, m, c.radius))).lower(x, obs, mask).compile()
        self.arrived = jax.jit(lambda x, g: flight_arrived(x, g, config)).lower(x, jnp.zeros(2)).compile()
        self.bound = jax.jit(lambda x: physical_envelope_violation(x, config)).lower(x).compile()
        self.compile_seconds = time.perf_counter()-start


def command_residual(x, u, obs, mask, effective_gains, config):
    c = config.robot
    residual = numpy_barrier(x, u, obs, effective_gains, np.ones(2), c)[mask]
    return np.r_[residual, u-c.force_min, c.force_max-u], np.inf, np.inf


def episode(parent, solver, kernels, steps, ordered=False):
    config = solver.config; c = config.robot
    observed, goal, obs, mask, noise = (np.asarray(parent[k], bool if k == 'obstacle_mask' else np.float32)
        for k in ('initial_state', 'goal', 'obstacles', 'obstacle_mask', 'noise'))
    if ordered:
        from .quad2d_waypoints import validate_parent,numpy_arrived
        validate_parent(parent);goals=np.asarray(parent['waypoint_goals'],np.float32);total=parent['waypoint_count'];leg=0
        routes=np.asarray(parent['waypoint_routes']['points'],np.float32);route_masks=np.asarray(parent['waypoint_routes']['mask'],bool);route_ready=np.asarray(parent['waypoint_routes']['ready'],bool)
        goal=goals[0];points=routes[0];rm=route_masks[0]
    else:
        points = np.asarray(parent['route']['points'], np.float32); rm = np.asarray(parent['route']['mask'], bool)
    key = np.asarray(jax.random.PRNGKey(parent['seed']+7193))
    initial, truth, xb, ob, xs, os, innovations = kernels.prepare(observed, obs, mask, noise, key)
    x = initial; previous = np.zeros(2, np.float32); solver.last_omega = np.zeros(2)
    cursor = np.float32(0); count = 0; status = 0; minimum = float(kernels.clearance(x, truth, mask))
    if bool(kernels.arrived(x, goal)) and (not ordered or total==1): status = GOAL
    if float(kernels.bound(x)) > c.qp_tolerance: status = STATE_BOUND
    if minimum <= 0: status = COLLISION
    if not (bool(route_ready[0]) if ordered else parent['route']['status']=='ready'): status = PLANNER_FAILURE
    records = []; reason = NAMES[status]; start = time.perf_counter()
    for k in range(steps):
        sensed, seen = map(np.asarray, kernels.sense(x, truth, xb, ob, xs, os, innovations[k], np.int32(k)))
        mission_info={}
        if ordered:
            handoff=status==0 and leg<total-1 and numpy_arrived(sensed.astype(float),goals[leg].astype(float),config,noise)
            if handoff:leg+=1;cursor=np.float32(0)
            goal=goals[leg];points=routes[leg];rm=route_masks[leg]
            if status==0 and not route_ready[leg]:status=PLANNER_FAILURE;reason=NAMES[status]
            mission_info=dict(waypoint_index=leg,waypoint_handoff=handoff,mission_goal=goal,mission_previous_control=previous,mission_route_cursor_before=cursor)
        target, proposed, remaining = map(np.asarray, kernels.target(sensed, points, rm, cursor))
        attempted = status == 0; accepted = False; control = np.zeros(2, np.float32); clear = bound = np.nan
        result = dict(states=np.full((11, 6), np.nan), controls=np.full((10, 2), np.nan), omegas=np.full((10, 2), np.nan),
            omega=np.ones(2), solver_success=False, solver_status='not_attempted', iterations=0, solve_seconds=0.,
            max_equality_error=np.nan, max_constraint_violation=np.nan, feasible=False)
        if attempted:
            result = solver.solve(sensed, target, seen, mask, previous)
            candidate = result['control'].astype(np.float32)
            residual, _, _ = command_residual(sensed, candidate, seen, mask, solver.gains*result['omega'], config)
            accepted = result['feasible'] and np.isfinite(candidate).all() and np.isfinite(residual).all() and residual.min() >= -c.qp_tolerance
            if not accepted:
                status = 3
                reason = ('solver_reported_failure:'+result['solver_status'] if not result['solver_success'] else
                    'independent_prediction_rejected' if not result['feasible'] else 'stored_command_rejected')
            else:
                control = candidate; x, clear, bound = kernels.advance(x, control, truth, mask, np.int32(k))
                clear = float(clear); bound = float(bound); minimum = min(minimum, clear)
                count += 1; cursor = np.float32(proposed); previous = control; solver.last_omega = result['omega'].copy()
                if bool(kernels.arrived(x, goal)) and (not ordered or leg==total-1): status = GOAL
                if bound > c.qp_tolerance: status = STATE_BOUND
                if clear <= 0: status = COLLISION
                reason = NAMES[status]
        if ordered:mission_info['waypoints_visited']=leg+int(status==GOAL)
        records.append(dict(state=np.asarray(x), control=control, active=accepted, status=status, observed_state=sensed, observed_obstacles=seen,
            clearance=clear, state_bound_violation=bound, route_progress=cursor, route_target=target, route_remaining=remaining,
            solver_attempted=attempted, solver_success=result['solver_success'], solver_status=result['solver_status'], solver_feasible=result['feasible'],
            solver_iterations=result['iterations'], solve_seconds=result['solve_seconds'], prediction_equality=result['max_equality_error'],
            prediction_violation=result['max_constraint_violation'], predicted_states=result['states'], predicted_controls=result['controls'],
            predicted_omegas=result['omegas'], omega=result['omega'],**mission_info))
        if status != 0: break
    if status == 0: status = TIMEOUT; reason = NAMES[status]
    progress = float(kernels.coordinate(x[:2], points, rm, cursor)-kernels.coordinate(initial[:2], points, rm, np.float32(0)))
    data = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    data.update(true_initial_state=np.asarray(initial), true_obstacles=np.asarray(truth), initial_observation=observed,
        observed_obstacles_initial=obs, obstacle_mask=mask, noise=noise, goal=goal, points=points, route_mask=rm, key=key)
    row = dict(group_id=parent['group_id'], family=parent['family'], obstacles=int(mask.sum()), noise_scale=float(noise[0]/.015),
        status=NAMES[status], status_code=status, steps=count, min_clearance=minimum, route_progress=progress,
        final_state=np.asarray(x).tolist(), termination_reason=reason, execution_seconds=time.perf_counter()-start,
        solver_attempts=int(data['solver_attempted'].sum()), solver_seconds=float(data['solve_seconds'].sum()))
    if ordered:
        data['goal']=np.asarray(parent['goal'],np.float32)
        row.update(waypoint_index=leg,waypoints_visited=leg+int(status==GOAL),required_waypoints=total,waypoint_handoffs=int(data['waypoint_handoff'].sum()))
    return sanitize(row), data


def run(source, output, method, steps=1600, shard_index=0, shards=1):
    source = Path(source); root = Path(output); root.mkdir(parents=True, exist_ok=False); config = FlightConfig()
    sm = json.loads((source/'manifest.json').read_text()); parents = json.loads((source/'scenes.json').read_text())
    if any('waypoint_goals' in p for p in parents):raise ValueError('Use the ordered flight runner; final-goal bypass forbidden')
    if sm['scenes_sha256'] != sha256(source/'scenes.json') or sm['config'] != asdict(config): raise ValueError('Changed common flight source')
    if steps < 1 or not 0 <= shard_index < shards or not parents[shard_index::shards]: raise ValueError('Invalid episode/shard budget')
    selected = parents[shard_index::shards]
    solver = Quad2DMPC(len(parents[0]['obstacles']), method, config)
    kernels = FlightPhysicalKernels(solver.capacity, len(parents[0]['route']['points']), steps, config)
    manifest = dict(schema='oa_cbf_quad2d_default_mpc_development_v1', source=str(source.resolve()), source_manifest_sha256=sha256(source/'manifest.json'),
        source_fingerprint=source_fingerprint(), config=asdict(config), discrete_mpc=solver.contract(), steps=steps,
        shard_index=shard_index, shards=shards, selected_parents=[p['group_id'] for p in selected], final_test=False,
        reference='Identical shared route_target_from_position(sensed xy, sensed speed, saved common route, cursor), with the pinned zero-velocity state target. Commit cursor only on applied commands.',
        device=str(jax.devices()[0]), physical_compile_seconds=kernels.compile_seconds, solver_setup_seconds=solver.setup_seconds,
        scope='Default-method development comparator with disclosed common-task adapters. Censored stops are not collision-free completed trajectories.')
    write_json(root/'manifest.json', manifest); print(json.dumps(dict(stage='warmed', physical_compile_seconds=kernels.compile_seconds, solver_setup_seconds=solver.setup_seconds)), flush=True)
    start = time.perf_counter(); index = []
    for i, parent in enumerate(selected):
        row, data = episode(parent, solver, kernels, steps); path = root/f'episode_{shard_index+i*shards:05d}.npz'
        np.savez_compressed(path, **data); row.update(file=path.name, sha256=sha256(path)); index.append(row); write_json(root/'index.json', index)
        print(json.dumps(dict(completed=len(index), total=len(selected), applied_steps=row['steps'], status=row['status'], episode_seconds=row['execution_seconds'], elapsed_seconds=time.perf_counter()-start)), flush=True)
    summary = dict(complete=True, manifest_sha256=sha256(root/'manifest.json'), index_sha256=sha256(root/'index.json'),
        execution_seconds=time.perf_counter()-start, aggregate=counts(index), physical_audit_pending=True)
    write_json(root/'summary.json', summary)


def counts(rows):
    return dict(episodes=len(rows), applied_steps=sum(r['steps'] for r in rows),
        outcomes={name:sum(r['status']==name for r in rows) for name in sorted({r['status'] for r in rows})},
        solver_seconds=sum(r['solver_seconds'] for r in rows), solver_attempts=sum(r['solver_attempts'] for r in rows))


def numpy_target(x, points, mask, cursor, *, projection_index=None):
    vectors = points[1:]-points[:-1]; valid = mask[:-1]&mask[1:]
    lengths = np.where(valid, np.linalg.norm(vectors, axis=1), 0.); cumulative = np.r_[0., np.cumsum(lengths)]
    fractions = np.clip(np.sum((x[:2]-points[:-1])*vectors, axis=1)/np.maximum(lengths**2, 1e-12), 0., 1.)
    projected = cumulative[:-1]+fractions*lengths; eligible = valid&(projected>=cursor-.05)&(projected<=cursor+1.)
    index = np.argmin(np.where(eligible, np.sum((points[:-1]+fractions[:, None]*vectors-x[:2])**2, axis=1), np.inf))
    if projection_index is not None:
        if not eligible[projection_index]: raise ValueError('Ineligible route segment')
        index = projection_index
    updated = max(cursor, projected[index]) if eligible.any() else cursor
    desired = min(updated+np.clip(.45+.65*np.linalg.norm(x[3:5]), .35, 1.1), cumulative[-1])
    segment = np.argmax(valid&(cumulative[1:]>=desired-1e-6))
    target = points[segment]+np.clip((desired-cumulative[segment])/max(lengths[segment], 1e-12), 0., 1.)*vectors[segment]
    return target, updated


def audit_episode(data, row, config, gains, waypoint_parent=None):
    summary = dict(status=row['status_code'], steps=row['steps'], min_clearance=row['min_clearance'] if row['min_clearance'] is not None else np.inf)
    result = check_trace(data, summary, data['initial_observation'], data['observed_obstacles_initial'], data['obstacle_mask'],
        data['noise'], np.asarray(gains)*data['omega'], config, command_residual=command_residual)
    if not result['audit_passed']: raise ValueError('Independent flight physical audit failed: '+str(result))
    active = data['active']; attempted = data['solver_attempted']; cursor = np.float32(0); equality = violation = 0.; ambiguous = 0
    from .route_audit import check_transition
    if np.any(data['control'][~active] != 0): raise ValueError('Action applied after rejection')
    for k in range(len(active)):
        if waypoint_parent is None:
            ambiguous += int(check_transition(data['observed_state'][k], data['points'], data['route_mask'], cursor,
                data['route_target'][k], data['route_progress'][k], active[k]))
        if active[k]:
            if not (attempted[k] and data['solver_success'][k] and data['solver_feasible'][k]): raise ValueError('Unapproved MPC command')
            eq, bad = prediction_residual(data['predicted_states'][k], data['predicted_controls'][k], data['predicted_omegas'][k],
                data['observed_state'][k], data['observed_obstacles'][k], data['obstacle_mask'], gains, config)
            equality = max(equality, eq); violation = max(violation, bad)
            if max(eq, bad) > config.robot.qp_tolerance: raise ValueError('Independent MPC prediction audit failed')
            np.testing.assert_array_equal(data['control'][k], data['predicted_controls'][k, 0].astype(np.float32))
            np.testing.assert_array_equal(data['omega'][k], data['predicted_omegas'][k, 0])
        cursor = data['route_progress'][k]
    final = data['state'][-1]; np.testing.assert_array_equal(final, np.asarray(row['final_state'], np.float32))
    if row['status_code'] == GOAL:
        if not (np.linalg.norm(final[:2]-data['goal'])<=config.goal_tolerance+1e-6 and np.linalg.norm(final[3:5])<=config.terminal_speed+1e-6
            and abs(final[2])<=config.terminal_pitch+1e-6 and abs(final[5])<=config.terminal_pitch_rate+1e-6): raise ValueError('False flight goal')
    if row['status_code'] == 3 and (active[-1] or not attempted[-1]): raise ValueError('Missing failed solver decision')
    result.update(prediction_equality_error=equality, prediction_violation=violation, all_applied_predictions_checked=True,
        roundoff_ambiguous_route_decisions=ambiguous)
    if waypoint_parent is not None:
        from .quad2d_waypoints import check_episode
        result.update(check_episode(waypoint_parent,data,row,config,adaptive=False))
    return result


def audit(directory):
    root = Path(directory); manifest = json.loads((root/'manifest.json').read_text()); config = FlightConfig()
    if manifest['schema'] != 'oa_cbf_quad2d_default_mpc_development_v1' or manifest['config'] != asdict(config): raise ValueError('Changed MPC physical contract')
    source = Path(manifest['source']); sm = json.loads((source/'manifest.json').read_text())
    if sha256(source/'manifest.json') != manifest['source_manifest_sha256'] or sm['scenes_sha256'] != sha256(source/'scenes.json'): raise ValueError('Changed source')
    parents = json.loads((source/'scenes.json').read_text())[manifest['shard_index']::manifest['shards']]
    index = json.loads((root/'index.json').read_text()); reports = []
    if [r['group_id'] for r in index] != [p['group_id'] for p in parents]: raise ValueError('Lost or reordered parents')
    gains = np.asarray(DEFAULTS[manifest['discrete_mpc']['method']])
    np.testing.assert_array_equal(gains, manifest['discrete_mpc']['gains'])
    for row, parent in zip(index, parents):
        if sha256(root/row['file']) != row['sha256']: raise ValueError('Changed trace')
        with np.load(root/row['file']) as f: data = dict(f)
        for field, original in [('initial_observation', 'initial_state'), ('observed_obstacles_initial', 'obstacles'), ('obstacle_mask', 'obstacle_mask'), ('noise', 'noise'), ('goal', 'goal')]:
            np.testing.assert_array_equal(data[field], np.asarray(parent[original], data[field].dtype))
        np.testing.assert_array_equal(data['points'], np.asarray(parent['route']['points'], np.float32))
        np.testing.assert_array_equal(data['route_mask'], parent['route']['mask'])
        if row['status_code'] == TIMEOUT and row['steps'] != manifest['steps']: raise ValueError('Censoring mislabeled as timeout')
        reports.append(dict(group_id=row['group_id'], **audit_episode(data, row, config, gains)))
    report = dict(audit_passed=bool(reports) and all(r['audit_passed'] for r in reports), auditor_source_fingerprint=source_fingerprint(), manifest_sha256=sha256(root/'manifest.json'),
        index_sha256=sha256(root/'index.json'), audited_episodes=len(reports), replayed_steps=sum(r['steps'] for r in reports), rows=reports,
        scope='Every physical/sensor prefix with independent SciPy6state integration and swept obstacles; every applied full10step MPC prediction with NumPy discrete CBF/equalities/input/envelope checks; common route target/cursor binding.',
        limitation='Observed prefixes only; stopped future is censored. No continuous-time or unseen-scenario guarantee.')
    write_json(root/'independent_replay.json', sanitize(report)); print(json.dumps({k:v for k,v in report.items() if k!='rows'}), flush=True)


def merge(directories, output):
    """Merge only complete, audited, disjoint shards of the identical task."""
    import os
    roots = list(map(Path, directories)); manifests = [json.loads((r/'manifest.json').read_text()) for r in roots]
    first = manifests[0]; source = Path(first['source']); parents = json.loads((source/'scenes.json').read_text())
    if len(roots) != first['shards'] or sorted(m['shard_index'] for m in manifests) != list(range(len(roots))): raise ValueError('Missing or duplicate shards')
    combined = {}; replays = {}; excluded = {'shard_index', 'selected_parents', 'physical_compile_seconds', 'solver_setup_seconds'}
    for root, manifest in zip(roots, manifests):
        if {k:v for k,v in manifest.items() if k not in excluded} != {k:v for k,v in first.items() if k not in excluded}: raise ValueError('Incompatible flight MPC shards')
        audit_report = json.loads((root/'independent_replay.json').read_text())
        if not audit_report['audit_passed'] or audit_report['manifest_sha256'] != sha256(root/'manifest.json') or audit_report['index_sha256'] != sha256(root/'index.json'): raise ValueError('Exact shard audit required')
        for row in json.loads((root/'index.json').read_text()):
            group = row['group_id']
            if group in combined or sha256(root/row['file']) != row['sha256']: raise ValueError('Duplicated or altered episode')
            combined[group] = (row, root/row['file'])
        for row in audit_report['rows']: replays[row['group_id']] = row
    ids = [p['group_id'] for p in parents]
    if set(combined) != set(ids) or set(replays) != set(ids): raise ValueError('Missing parent or physical replay')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = dict(first, shards=1, shard_index=0, selected_parents=ids,
        physical_compile_seconds=sum(m['physical_compile_seconds'] for m in manifests), solver_setup_seconds=sum(m['solver_setup_seconds'] for m in manifests),
        audited_shards=[dict(directory=str(r.resolve()), manifest_sha256=sha256(r/'manifest.json'), index_sha256=sha256(r/'index.json'), audit_sha256=sha256(r/'independent_replay.json')) for r in roots])
    write_json(root/'manifest.json', manifest); index = []
    for group in ids:
        row, path = combined[group]; os.link(path, root/row['file']); index.append(row)
    write_json(root/'index.json', index)
    write_json(root/'summary.json', dict(complete=True, manifest_sha256=sha256(root/'manifest.json'), index_sha256=sha256(root/'index.json'),
        aggregate=counts(index), by_family={f:counts([r for r in index if r['family']==f]) for f in sorted({r['family'] for r in index})}, physical_audit_pending=False))
    write_json(root/'independent_replay.json', dict(audit_passed=True, manifest_sha256=sha256(root/'manifest.json'), index_sha256=sha256(root/'index.json'),
        audited_episodes=len(index), replayed_steps=sum(r['steps'] for r in index), rows=[replays[g] for g in ids],
        scope='Union of exact individually audited shard traces, all parents retained in source order; hard-linked bytes rechecked before merge.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run'); r.add_argument('--source', required=True); r.add_argument('--output', required=True); r.add_argument('--method', choices=list(DEFAULTS), required=True)
    r.add_argument('--steps', type=int, default=1600); r.add_argument('--shard-index', type=int, default=0); r.add_argument('--shards', type=int, default=1)
    a = sub.add_parser('audit'); a.add_argument('--directory', required=True)
    m = sub.add_parser('merge'); m.add_argument('--directories', nargs='+', required=True); m.add_argument('--output', required=True)
    args = vars(p.parse_args()); action = args.pop('action'); {'run':run, 'audit':audit, 'merge':merge}[action](**args)
