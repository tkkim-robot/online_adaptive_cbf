"""Pinned Quad2D discrete MPC baselines; direct CasADi/IPOPT transcription.

The shared-task adapter uses every sensed obstacle, the common clearance buffer,
and predicted flight-envelope bounds. Method gains, horizon, objective, decay
penalties and solver defaults are preserved. No OA safety filter is appended.
"""
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
