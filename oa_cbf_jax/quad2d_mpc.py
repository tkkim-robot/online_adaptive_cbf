"""Quad2d mpc functions and shared contracts."""

from dataclasses import asdict

import time

import numpy as np

from .quad2d_control import FlightConfig

DEFAULTS = {'fixed_low': (.01, .01), 'fixed_high': (.99, .99), 'optimal_decay': (.01, .01)}

Q = np.array([25., 25., 50., 10., 10., 50.])

def numpy_euler(x, u, robot):
    x = np.asarray(x, float); u = np.asarray(u, float)
    force = u.sum() / robot.mass
    return x + robot.dt * np.array([x[3], x[4], x[5], -np.sin(x[2])*force,
        np.cos(x[2])*force-robot.gravity, robot.arm/robot.inertia*(u[0]-u[1])])

def numpy_barrier(x, u, obs, gains, omega, robot, buffer=None):
    """Pinned two repeated-input Euler differences; centers remain observed/fixed.

    Legacy angle wrapping cannot affect the position at either of these two
    Euler steps, hence this expression also agrees outside the flight envelope.
    """
    obs = np.asarray(obs, float); a = np.asarray(gains)*np.asarray(omega)
    radius = robot.radius + (robot.clearance_buffer if buffer is None else buffer)
    x1 = numpy_euler(x, u, robot); x2 = numpy_euler(x1, u, robot)
    def h(y): return np.sum((y[:2]-obs[:, :2])**2, axis=-1)-1.01*(radius+obs[:, 2])**2
    h0, h1, h2 = h(x), h(x1), h(x2)
    return h2-2*h1+h0+a.sum()*(h1-h0)+a.prod()*h0

def prediction_residual(states, controls, omegas, x, obs, mask, gains, config, shared_contract=True):
    """Independent NumPy check of every equality, obstacle, actuator and bound."""
    c = config.robot
    arrays = (states, controls, omegas, x, obs, gains)
    if not all(np.isfinite(a).all() for a in arrays): return np.inf, np.inf
    equality = max(float(np.max(np.abs(states[0]-x))), max(float(np.max(np.abs(
        states[k+1]-numpy_euler(states[k], controls[k], c)))) for k in range(len(controls))))
    barriers = np.stack([numpy_barrier(s, u, obs, gains, w, c, buffer=None if shared_contract else 0.)
        for s, u, w in zip(states, controls, omegas)])
    violation = max(float(np.max(-barriers[:, mask], initial=-np.inf)),
        float(np.max(c.force_min-controls)), float(np.max(controls-c.force_max)))
    if shared_contract:
        limits = np.array([config.pitch_limit, config.velocity_limit, config.velocity_limit, config.pitch_rate_limit])
        violation = max(violation, float(np.max(np.abs(states[1:, 2:])-limits)))
    return equality, violation

class Quad2DMPC:
    def __init__(self, capacity, method='fixed_low', config=FlightConfig(), *, shared_contract=True):
        import casadi as ca
        if method not in DEFAULTS or isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('Invalid flight MPC method/capacity')
        self.capacity = capacity; self.method = method; self.config = config; self.horizon = 10
        self.gains = np.asarray(DEFAULTS[method]); self.decay = method == 'optimal_decay'
        self.shared_contract = shared_contract; self.last_omega = np.zeros(2)
        c = config.robot; H = self.horizon; C = capacity
        X = ca.SX.sym('x', 6, H+1); U = ca.SX.sym('u', 2, H)
        W = ca.SX.sym('omega', 2, H) if self.decay else ca.DM.ones(2, H)
        initial = ca.SX.sym('initial', 6); goal = ca.SX.sym('goal', 2)
        obs = ca.SX.sym('obs', C, 5); active = ca.SX.sym('active', C); previous = ca.SX.sym('previous', 2)
        parameters = ca.vertcat(initial, goal, ca.vec(obs), active, previous)
        variables = ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(W)) if self.decay else ca.vertcat(ca.vec(X), ca.vec(U))
        target = ca.vertcat(goal, 0, 0, 0, 0); weights = ca.DM(Q)
        def step(x, u):
            force = ca.sum1(u)/c.mass
            return x+c.dt*ca.vertcat(x[3], x[4], x[5], -ca.sin(x[2])*force,
                ca.cos(x[2])*force-c.gravity, c.arm/c.inertia*(u[0]-u[1]))
        def barrier(x, u, omega):
            radius = c.radius+(c.clearance_buffer if shared_contract else 0.)
            def h(y): return (obs[:, 0]-y[0])**2+(obs[:, 1]-y[1])**2-1.01*(obs[:, 2]+radius)**2
            x1 = step(x, u); x2 = step(x1, u); h0, h1, h2 = h(x), h(x1), h(x2)
            a1, a2 = self.gains[0]*omega[0], self.gains[1]*omega[1]
            return h2-2*h1+h0+(a1+a2)*(h1-h0)+a1*a2*h0
        equality = [X[:, 0]-initial]; inequality = []; cost = 0
        limits = ca.DM([config.pitch_limit, config.velocity_limit, config.velocity_limit, config.pitch_rate_limit])
        for k in range(H):
            cost += ca.dot(weights, (X[:, k]-target)**2)
            # Pinned do-mpc5.1.2 replaces the first custom rterm on the second
            # call in OptimalDecayMPCCBF: preserve its effective omega cost.
            cost += 10*ca.sumsqr(W[:, k]-1) if self.decay else .5*ca.sumsqr(U[:, k]-(previous if k == 0 else U[:, k-1]))
            equality.append(X[:, k+1]-step(X[:, k], U[:, k]))
            inequality.append(active*barrier(X[:, k], U[:, k], W[:, k])+(1-active))
            if shared_contract: inequality.append(ca.vertcat(limits-X[2:, k+1], limits+X[2:, k+1]))
        cost += ca.dot(weights, (X[:, H]-target)**2)
        constraints = ca.vertcat(*equality, *inequality); self.equalities = 6*(H+1)
        self.lbg = np.zeros(int(constraints.numel())); self.ubg = np.r_[np.zeros(self.equalities), np.full(len(self.lbg)-self.equalities, np.inf)]
        self.lbx = np.full(int(variables.numel()), -np.inf); self.ubx = -self.lbx.copy()
        self.lbx[6*(H+1):6*(H+1)+2*H] = c.force_min; self.ubx[6*(H+1):6*(H+1)+2*H] = c.force_max
        self.solver_options = {'ipopt.print_level': 0, 'print_time': False}
        start = time.perf_counter()
        self.solver = ca.nlpsol('quad2d_mpc', 'ipopt', dict(x=variables, p=parameters, f=cost, g=constraints), self.solver_options)
        self.setup_seconds = time.perf_counter()-start
        sx = ca.SX.sym('state', 6); su = ca.SX.sym('control', 2); sw = ca.SX.sym('omega', 2)
        self.barrier_expression = ca.Function('flight_discrete_barrier', [sx, su, obs, sw], [barrier(sx, su, sw)])

    def contract(self):
        return dict(method=self.method, horizon=self.horizon, gains=self.gains.tolist(), Q=Q.tolist(),
            input_cost='10*sum((omega-1)^2), effective pinned custom rterm' if self.decay else '.5*sum((u[k]-u[k-1])^2)',
            omega_bounds='unrestricted' if self.decay else 'fixed1', obstacle_capacity=self.capacity,
            predictor='Euler6state, fixed observed centers, two repeated-input Euler differences in each discrete HOCBF row',
            radius_beta=1.01, shared_contract=self.shared_contract, config=asdict(self.config), solver='CasADi/IPOPT', solver_options=self.solver_options,
            adaptation='All active observed obstacles; shared clearance buffer; future pitch/rate/component-velocity bounds. Initial sensed state is not projected. Shared route target supplied by experiment. Not a literal original five-obstacle simulation.',
            initialization='Repeated current sensed state and previous applied action; previous action and omega start at zero each episode. No multiplier warm-start or solver tuning.')

    def solve(self, x, goal, obs, mask, previous):
        x = np.asarray(x, float); goal = np.asarray(goal, float); obs = np.asarray(obs, float)
        mask = np.asarray(mask, bool); previous = np.asarray(previous, float); H = self.horizon
        if x.shape != (6,) or goal.shape != (2,) or obs.shape != (self.capacity, 5) or mask.shape != (self.capacity,) or previous.shape != (2,):
            raise ValueError('Flight MPC input shape mismatch')
        p = np.r_[x, goal, obs.ravel(order='F'), mask.astype(float), previous]
        if not np.isfinite(p).all(): raise ValueError('Nonfinite flight MPC input')
        guess = np.r_[np.tile(x, H+1), np.tile(previous, H)]
        if self.decay: guess = np.r_[guess, np.tile(self.last_omega, H)]
        start = time.perf_counter(); answer = self.solver(x0=guess, p=p, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        elapsed = time.perf_counter()-start; stats = self.solver.stats(); v = np.asarray(answer['x']).ravel()
        X = v[:6*(H+1)].reshape(H+1, 6); U = v[6*(H+1):6*(H+1)+2*H].reshape(H, 2)
        W = v[-2*H:].reshape(H, 2) if self.decay else np.ones((H, 2))
        eq, violation = prediction_residual(X, U, W, x, obs, mask, self.gains, self.config, self.shared_contract)
        feasible = bool(stats.get('success', False) and eq <= self.config.robot.qp_tolerance and violation <= self.config.robot.qp_tolerance)
        return dict(control=U[0], omega=W[0], states=X, controls=U, omegas=W, feasible=feasible,
            solver_success=bool(stats.get('success', False)), solver_status=str(stats.get('return_status')), iterations=int(stats.get('iter_count', -1)),
            solve_seconds=elapsed, max_equality_error=eq, max_constraint_violation=violation, objective=float(answer['f']))


import jax

import jax.numpy as jnp


from .quad2d import integrate_quad2d

from .quad2d_control import flight_arrived, physical_envelope_violation
from .quad2d_rollout import flight_sensor_model, NAMES, STATE_BOUND, PLANNER_FAILURE
from .simulation import GOAL, COLLISION, TIMEOUT

from .routing import route_target_from_position, physical_route_coordinate

from .dynamics import signed_clearance, swept_disk_clearance

from .io import sanitize

class FlightPhysicalKernels:
    def __init__(self, capacity, route_capacity, steps, config=FlightConfig()):
        with jax.enable_x64(False): self._compile(capacity, route_capacity, steps, config)

    def _compile(self, capacity, route_capacity, steps, config):
        c = config.robot; x = jnp.zeros(6); obs = jnp.zeros((capacity, 5)); mask = jnp.zeros(capacity, bool)
        noise = jnp.zeros(7); points = jnp.zeros((route_capacity, 2)); rm = jnp.ones(route_capacity, bool)
        start = time.perf_counter()
        self.prepare = jax.jit(lambda x, o, m, n, key: flight_sensor_model(x, o, m, n, key, steps,config.stationary_obstacles)).lower(x, obs, mask, noise, jax.random.PRNGKey(0)).compile()
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
        from .quad2d_static_inputs import validate_parent, numpy_arrived
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
