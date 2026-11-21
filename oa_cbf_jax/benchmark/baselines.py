"""Native comparison controllers with constant-velocity obstacle prediction."""

from dataclasses import replace
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from oa_cbf_jax import (
    discrete_mpc as umpc,
    optimal_decay as uod,
    barriernet_inference as ubn,
    quad2d_mpc as qmpc,
    quad2d_odqp as qod,
    quad2d_barriernet_inference as qbn,
    quad3d_mpc as tmpc,
    demo_barriernet as dbn,
    barriernet_variant_inference as vbn,
    barriernet_variants as variants,
)
from oa_cbf_jax.model_release import verify
from .protocol import STEPS
from .sensors import unicycle_sensor_model, quad2d_sensor_model


def _unicycle_barrier(x, u, obstacles, gains, omega, robot, buffer=None, *, stage=0):
    """Two repeated-input Euler differences, advancing centers at each time."""
    import numpy as np
    from oa_cbf_jax.discrete_mpc import numpy_euler

    obstacles = np.asarray(obstacles, float)
    effective = np.asarray(gains) * np.asarray(omega)
    radius = robot.radius + (robot.clearance_buffer if buffer is None else buffer)
    x1 = numpy_euler(x, u, robot.dt)
    x2 = numpy_euler(x1, u, robot.dt)

    def h(state, tick):
        center = obstacles[:, :2] + tick * robot.dt * obstacles[:, 3:5]
        return np.sum((state[:2] - center) ** 2, axis=-1) - 1.01 * (radius + obstacles[:, 2]) ** 2

    h0, h1, h2 = (h(x, stage), h(x1, stage + 1), h(x2, stage + 2))
    return h2 - 2 * h1 + h0 + effective.sum() * (h1 - h0) + effective.prod() * h0


class UnicycleMPC(umpc.DiscreteMPC):
    def __init__(
        self, capacity, method="fixed_low", robot=umpc.UnicycleConfig(), *, shared_contract=True
    ):
        import casadi as ca

        if (
            method not in umpc.DEFAULTS
            or isinstance(capacity, bool)
            or (not isinstance(capacity, int))
            or (capacity < 1)
        ):
            raise ValueError("Unknown discrete MPC method/capacity")
        self.capacity = capacity
        self.method = method
        self.robot = robot
        self.horizon = 10
        self.gains = np.asarray(umpc.DEFAULTS[method])
        self.decay = method == "optimal_decay"
        self.shared_contract = shared_contract
        self.last_omega = np.zeros(2)
        H = self.horizon
        C = capacity
        X = ca.SX.sym("x", 4, H + 1)
        U = ca.SX.sym("u", 2, H)
        W = ca.SX.sym("omega", 2, H) if self.decay else ca.DM.ones(2, H)
        initial = ca.SX.sym("initial", 4)
        goal = ca.SX.sym("goal", 2)
        obs = ca.SX.sym("obs", C, 5)
        active = ca.SX.sym("active", C)
        previous = ca.SX.sym("previous", 2)
        speed_error = ca.SX.sym("speed_error")
        parameters = ca.vertcat(initial, goal, ca.vec(obs), active, previous, speed_error)
        variables = (
            ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(W))
            if self.decay
            else ca.vertcat(ca.vec(X), ca.vec(U))
        )
        goal_state = ca.vertcat(goal, 0, 0)
        Q = ca.DM([50, 50, 0.01, 30])
        equality = [X[:, 0] - initial]
        inequality = []
        cost = 0

        def step(x, u):
            return x + robot.dt * ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), u[1], u[0])

        def barrier(x, u, omega, stage=0):
            radius = robot.radius + (robot.clearance_buffer if shared_contract else 0.0)
            h = lambda y, t: (
                (obs[:, 0] + t * robot.dt * obs[:, 3] - y[0]) ** 2
                + (obs[:, 1] + t * robot.dt * obs[:, 4] - y[1]) ** 2
                - 1.01 * (obs[:, 2] + radius) ** 2
            )
            x1 = step(x, u)
            x2 = step(x1, u)
            h0, h1, h2 = (h(x, stage), h(x1, stage + 1), h(x2, stage + 2))
            a1, a2 = (self.gains[0] * omega[0], self.gains[1] * omega[1])
            return h2 - 2 * h1 + h0 + (a1 + a2) * (h1 - h0) + a1 * a2 * h0

        for k in range(H):
            error = X[:, k] - goal_state
            cost += ca.dot(Q, error**2)
            if self.decay:
                cost += 10 * ca.sumsqr(W[:, k] - 1)
            else:
                last = previous if k == 0 else U[:, k - 1]
                cost += 0.5 * ca.sumsqr(U[:, k] - last)
            equality.append(X[:, k + 1] - step(X[:, k], U[:, k]))
            inequality.append(active * barrier(X[:, k], U[:, k], W[:, k], k) + (1 - active))
            lower = (
                initial[3] - ca.fmax(0.0, initial[3] - speed_error)
                if shared_contract
                else -robot.v_max
            )
            upper = (
                robot.v_max + initial[3] - ca.fmin(robot.v_max, initial[3] + speed_error)
                if shared_contract
                else robot.v_max
            )
            inequality.append(ca.vertcat(X[3, k + 1] - lower, upper - X[3, k + 1]))
        cost += ca.dot(Q, (X[:, H] - goal_state) ** 2)
        constraints = ca.vertcat(*equality, *inequality)
        self.equalities = 4 * (H + 1)
        self.lbg = np.r_[
            np.zeros(self.equalities), np.zeros(int(constraints.numel()) - self.equalities)
        ]
        self.ubg = np.r_[
            np.zeros(self.equalities), np.full(int(constraints.numel()) - self.equalities, np.inf)
        ]
        self.lbx = np.full(int(variables.numel()), -np.inf)
        self.ubx = -self.lbx.copy()
        lower_u = np.tile([-robot.a_max, -robot.w_max], H)
        self.lbx[4 * (H + 1) : 4 * (H + 1) + 2 * H] = lower_u
        self.ubx[4 * (H + 1) : 4 * (H + 1) + 2 * H] = -lower_u
        self.solver_options = {"ipopt.print_level": 0, "print_time": False}
        start = time.perf_counter()
        self.solver = ca.nlpsol(
            "discrete_mpc",
            "ipopt",
            dict(x=variables, p=parameters, f=cost, g=constraints),
            self.solver_options,
        )
        self.setup_seconds = time.perf_counter() - start
        self.expression = ca.Function(
            "discrete_mpc_expressions", [variables, parameters], [cost, constraints]
        )
        sx = ca.SX.sym("state", 4)
        su = ca.SX.sym("control", 2)
        sw = ca.SX.sym("omega", 2)
        self.barrier_expression = ca.Function(
            "discrete_barrier", [sx, su, obs, sw], [barrier(sx, su, sw)]
        )

    def solve(self, x, goal, obs, mask, previous, speed_error=0.0):
        H = self.horizon
        p = self.pack(x, goal, obs, mask, previous, speed_error)
        guess = np.r_[np.tile(np.asarray(x, float), H + 1), np.tile(previous, H)]
        if self.decay:
            guess = np.r_[guess, np.tile(self.last_omega, H)]
        start = time.perf_counter()
        answer = self.solver(x0=guess, p=p, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        elapsed = time.perf_counter() - start
        stats = self.solver.stats()
        v = np.asarray(answer["x"]).reshape(-1)
        X = v[: 4 * (H + 1)].reshape(H + 1, 4)
        U = v[4 * (H + 1) : 4 * (H + 1) + 2 * H].reshape(H, 2)
        W = v[-2 * H :].reshape(H, 2) if self.decay else np.ones((H, 2))
        finite = np.isfinite(v).all()
        eq = np.inf
        ineq = np.inf
        if finite:
            eq = max(
                float(np.max(np.abs(X[0] - x))),
                max(
                    (
                        float(
                            np.max(np.abs(X[k + 1] - umpc.numpy_euler(X[k], U[k], self.robot.dt)))
                        )
                        for k in range(H)
                    )
                ),
            )
            residuals = np.stack(
                [
                    _unicycle_barrier(
                        X[k],
                        U[k],
                        obs,
                        self.gains,
                        W[k],
                        self.robot,
                        buffer=None if self.shared_contract else 0.0,
                        stage=k,
                    )
                    for k in range(H)
                ]
            )
            cbf = float(np.max(-residuals[:, np.asarray(mask, bool)], initial=-np.inf))
            low = x[3] - max(0.0, x[3] - speed_error) if self.shared_contract else -self.robot.v_max
            high = (
                self.robot.v_max + x[3] - min(self.robot.v_max, x[3] + speed_error)
                if self.shared_contract
                else self.robot.v_max
            )
            ineq = max(
                cbf,
                float(np.max(low - X[1:, 3])),
                float(np.max(X[1:, 3] - high)),
                float(np.max(np.abs(U) - [self.robot.a_max, self.robot.w_max])),
            )
        feasible = bool(
            stats.get("success", False)
            and finite
            and (eq <= self.robot.qp_tolerance)
            and (ineq <= self.robot.qp_tolerance)
        )
        return dict(
            control=U[0],
            omega=W[0],
            states=X,
            controls=U,
            omegas=W,
            feasible=feasible,
            solver_success=bool(stats.get("success", False)),
            solver_status=str(stats.get("return_status")),
            iterations=int(stats.get("iter_count", -1)),
            solve_seconds=elapsed,
            max_equality_error=eq,
            max_constraint_violation=ineq,
            objective=float(answer["f"]),
        )


def _check_unicycle_command(sensed, control, obstacles, mask, omega, solver, speed_error):
    """Recheck the actual FP32 command; reject rather than clip a bad action."""
    robot = solver.robot
    residual = _unicycle_barrier(sensed, control, obstacles, solver.gains, omega, robot)
    low = max(0.0, float(sensed[3]) - speed_error) + robot.dt * float(control[0])
    high = min(robot.v_max, float(sensed[3]) + speed_error) + robot.dt * float(control[0])
    violation = max(
        float(np.max(-residual[np.asarray(mask, bool)], initial=-np.inf)),
        float(np.max(np.abs(control) - [robot.a_max, robot.w_max])),
        -low,
        high - robot.v_max,
    )
    return (
        bool(
            np.isfinite(control).all()
            and np.isfinite(omega).all()
            and (violation <= robot.qp_tolerance)
        ),
        violation,
    )


def _unicycle_mpc_episode(record, solver, kernels, steps, noise_scale, ordered=False):
    robot = solver.robot
    scene = record["scene"]
    f = lambda a: np.asarray(a, np.float32)
    mask = np.asarray(scene["obstacle_mask"], bool)
    observed = f(scene["initial_state"])
    obstacles = f(scene["obstacles"])
    if ordered:
        count = record["waypoint_count"]
        goals = f(record["waypoint_goals"])
        routes = record["waypoint_routes"]
        points = f(routes["points"])
        rmask = np.asarray(routes["mask"], bool)
        ready = np.asarray(routes["ready"], bool)
    else:
        count = 1
        goals = f([scene["goal"]])
        points = f([record["route"]["points"]])
        rmask = np.asarray([record["route"]["mask"]], bool)
        ready = np.asarray([record["route"]["status"] == "ready"])
    noise = noise_scale * np.array([0.02, 0.03, 0.02, 0.03, 0.025, 0.01])
    fn = f(noise)
    seed = int.from_bytes(umpc.hashlib.sha256(scene["scene_id"].encode()).digest()[:4], "little")
    key = np.asarray(jax.random.PRNGKey(seed))
    x, truth_obs, xb, ob, xs, os, innovations = kernels.prepare(observed, obstacles, mask, fn, key)
    initial = np.asarray(x)
    truth = np.asarray(truth_obs)
    minimum = float(kernels.clearance(x, truth_obs, mask))
    status = (
        umpc.PLANNER_FAILURE
        if not ready[0]
        else umpc.COLLISION
        if minimum <= 0
        else umpc.GOAL
        if count == 1 and bool(kernels.arrived(x, goals[0], np.zeros(6, np.float32)))
        else umpc.RUNNING
    )
    leg = 0
    cursor = np.float32(0)
    previous = np.zeros(2, np.float32)
    solver.last_omega = np.zeros(2)
    traces = []
    applied = 0
    solve_times = []
    step_times = []
    solver_statuses = {}
    start = time.perf_counter()
    for k in range(steps):
        tick = time.perf_counter()
        sx, so = kernels.sense(x, truth_obs, xb, ob, xs, os, innovations[k], np.int32(k))
        sx, so = (np.asarray(sx), np.asarray(so))
        handoff = (
            status == umpc.RUNNING and leg < count - 1 and bool(kernels.arrived(sx, goals[leg], fn))
        )
        if handoff:
            leg += 1
            cursor = np.float32(0)
        if status == umpc.RUNNING and (not ready[leg]):
            status = umpc.PLANNER_FAILURE
        attempt = status == umpc.RUNNING
        result = None
        u = np.zeros(2, np.float32)
        omega = np.ones(2)
        clear = bound = violation = np.nan
        target, proposal, remaining = kernels.target(sx, points[leg], rmask[leg], cursor)
        if attempt:
            result = solver.solve(
                sx, goals[leg], so, mask, previous, speed_error=1.15 * float(fn[2])
            )
            solve_times.append(result["solve_seconds"])
            label = result["solver_status"]
            solver_statuses[label] = solver_statuses.get(label, 0) + 1
            candidate = f(result["control"])
            omega = result["omega"]
            accepted, violation = _check_unicycle_command(
                sx, candidate, so, mask, omega, solver, 1.15 * float(fn[2])
            )
            accepted = accepted and result["feasible"]
            if accepted:
                u = candidate
                x, clear, bound = kernels.advance(x, u, truth_obs, mask, np.int32(k))
                clear, bound = (float(clear), float(bound))
                minimum = min(minimum, clear)
                cursor = np.float32(proposal)
                previous = u.copy()
                solver.last_omega = omega.copy()
                applied += 1
                if leg == count - 1 and bool(
                    kernels.arrived(x, goals[leg], np.zeros(6, np.float32))
                ):
                    status = umpc.GOAL
                if bound > robot.qp_tolerance:
                    status = umpc.STATE_BOUND_VIOLATION
                if clear <= 0:
                    status = umpc.COLLISION
            else:
                status = umpc.INFEASIBLE
        else:
            accepted = False
        traces.append(
            dict(
                state=np.asarray(x),
                control=u,
                active=accepted,
                status=status,
                gains=solver.gains,
                omega=omega,
                observed_state=sx,
                observed_obstacles=so,
                clearance=clear,
                state_bound_violation=bound,
                qp_violation=violation,
                route_target=np.asarray(target),
                mpc_reference=goals[leg],
                route_remaining=float(remaining),
                route_progress=cursor,
                selection_tick=attempt,
                selection_accepted=accepted,
                solver_status="" if result is None else result["solver_status"],
                solver_success=False if result is None else result["solver_success"],
                solver_iterations=-1 if result is None else result["iterations"],
                solver_seconds=np.nan if result is None else result["solve_seconds"],
                predicted_states=np.full((11, 4), np.nan) if result is None else result["states"],
                predicted_controls=np.full((10, 2), np.nan)
                if result is None
                else result["controls"],
                predicted_omegas=np.full((10, 2), np.nan) if result is None else result["omegas"],
                solver_equality_error=np.nan if result is None else result["max_equality_error"],
                solver_constraint_violation=np.nan
                if result is None
                else result["max_constraint_violation"],
                waypoint_index=leg,
                waypoint_handoff=handoff,
                waypoints_visited=leg + int(status == umpc.GOAL),
                mission_goal=goals[leg],
                collision_event=accepted and clear <= 0,
                state_bound_event=accepted and bound > robot.qp_tolerance,
            )
        )
        step_times.append(time.perf_counter() - tick)
        if status != umpc.RUNNING:
            break
    if status == umpc.RUNNING:
        status = umpc.TIMEOUT
    trace = {k: np.stack([t[k] for t in traces]) for k in traces[0]}
    row = dict(
        scene_id=scene["scene_id"],
        family=scene["family"],
        mode="discrete_mpc_" + solver.method,
        status=umpc.NAMES[status],
        steps=applied,
        min_clearance=minimum,
        goal_progress=float(
            np.linalg.norm(initial[:2] - goals[count - 1])
            - np.linalg.norm(np.asarray(x)[:2] - goals[count - 1])
        ),
        final_route_coordinate=float(kernels.coordinate(x[:2], points[leg], rmask[leg], cursor)),
        applied_source_counts={"default_discrete_mpc": applied},
        selection_attempts=len(solve_times),
        selection_rejections=int(status == umpc.INFEASIBLE),
        reactive_reselections=0,
        gate_stage_totals=[],
        gain_total_variation=0.0,
        max_physical_bound_violation=float(np.max(trace["state_bound_violation"][trace["active"]]))
        if applied
        else None,
        solver_status_counts=solver_statuses,
        elapsed_seconds=time.perf_counter() - start,
        solver_seconds=sum(solve_times),
        tick_seconds_p50_p99_max=np.quantile(step_times, [0.5, 0.99, 1]).tolist(),
        collision_event=bool(np.any(trace["collision_event"])) or minimum <= 0,
        state_bound_event=bool(np.any(trace["state_bound_event"])),
    )
    if status == umpc.INFEASIBLE:
        row["termination_reason"] = (
            "solver_reported_failure:" + result["solver_status"]
            if not result["solver_success"]
            else "independent_prediction_residual_rejection"
            if not result["feasible"]
            else "applied_command_residual_rejection"
        )
    if ordered:
        row.update(
            waypoint_index=leg,
            waypoints_visited=leg + int(status == umpc.GOAL),
            required_waypoints=count,
            waypoint_handoffs=int(trace["waypoint_handoff"].sum()),
        )
    return (
        row,
        trace,
        dict(
            true_initial_state=initial,
            true_obstacles=truth,
            scene_id=scene["scene_id"],
            noise=noise,
            key=key,
        ),
    )


def _unicycle_bn_rows(state, obstacles, parameters, radius=0.25):
    """BarrierNet total derivatives for constant-velocity obstacle centers."""
    import jax.numpy as jnp
    from oa_cbf_jax.barriernet import constraints as original

    rows, static_rhs = original(state, obstacles, parameters, radius)
    delta = state[:2] - obstacles[:, :2]
    direction = jnp.stack((jnp.cos(state[2]), jnp.sin(state[2])))
    relative = state[3] * direction - obstacles[:, 3:5]
    h = jnp.sum(delta**2, axis=-1) - 1.01 * (obstacles[:, 2] + radius) ** 2
    hd = 2 * jnp.sum(delta * relative, axis=-1)
    rhs = (
        2 * jnp.sum(relative**2, axis=-1)
        + jnp.sum(parameters, axis=-1) * hd
        + jnp.prod(parameters, axis=-1) * h
    )
    return (rows, jnp.where(jnp.all(obstacles[:, 3:5] == 0, axis=-1), static_rhs, rhs))


def _unicycle_bn_numpy(context, parameters, radius):
    """Independent scalar expansion; original five rows and input box."""
    import numpy as np
    from oa_cbf_jax.barriernet_inference import numpy_constraints as original

    rows, rhs = original(context, parameters, radius)
    x = np.asarray(context[:4], float)
    obstacles = np.asarray(context[6:], float).reshape(5, 7)
    for i, (obstacle, gain) in enumerate(zip(obstacles, parameters, strict=True)):
        if np.all(obstacle[3:5] == 0):
            continue
        dx, dy = x[:2] - obstacle[:2]
        vx = x[3] * np.cos(x[2]) - obstacle[3]
        vy = x[3] * np.sin(x[2]) - obstacle[4]
        h = dx * dx + dy * dy - 1.01 * (obstacle[2] + radius) ** 2
        rhs[i] = (
            2 * (vx * vx + vy * vy)
            + (gain[0] + gain[1]) * 2 * (dx * vx + dy * vy)
            + gain[0] * gain[1] * h
        )
    return (rows, rhs)


def _unicycle_bn_policy(bundle, radius=None):
    manifest, model, params, mean, std = ubn.load_bundle(bundle)
    radius = manifest["radius"] if radius is None else radius

    def policy(state, goal, obstacles, mask):
        z, ctx = ubn.features(state, goal, obstacles, mask, radius)
        reference = ubn.nominal(state, goal)
        u_nom, p = model.apply({"params": params}, (z - mean) / std, state, goal, reference)
        G, h = _unicycle_bn_rows(state, ctx[6:].reshape(5, 7), p, radius)
        control, valid, violation = ubn.hard_deployment_qp(u_nom, G, h)
        return (control, valid, violation, p, u_nom)

    return (jax.jit(policy), manifest)


class UnicycleBarrierNet(ubn.Controller):
    def __init__(self, bundle, capacity, robot=ubn.UnicycleConfig()):
        self.robot = robot
        fn, self.manifest = _unicycle_bn_policy(bundle, robot.radius)
        begin = time.perf_counter()
        self.execute = fn.lower(
            jnp.zeros(4, jnp.float64),
            jnp.zeros(2, jnp.float64),
            jnp.zeros((capacity, 5), jnp.float64),
            jnp.zeros(capacity, bool),
        ).compile()
        self.compile_seconds = time.perf_counter() - begin

    def solve(self, state, goal, obstacles, mask):
        begin = time.perf_counter()
        u, feasible, violation, p, reference = jax.device_get(
            self.execute(
                jnp.asarray(state, jnp.float64),
                jnp.asarray(goal, jnp.float64),
                jnp.asarray(obstacles, jnp.float64),
                jnp.asarray(mask),
            )
        )
        applied = np.asarray(u, np.float32)
        _, ctx = ubn.numpy_features(
            np.asarray(state, float),
            np.asarray(goal, float),
            np.asarray(obstacles, float),
            mask,
            self.robot.radius,
        )
        A, b = _unicycle_bn_numpy(ctx, p, self.robot.radius)
        applied_violation = float(np.max(A @ applied - b))
        accepted = bool(
            feasible
            and np.isfinite(applied).all()
            and (applied_violation <= self.robot.qp_tolerance)
        )
        return dict(
            control=applied,
            feasible=accepted,
            solver_feasible=bool(feasible),
            gains=p,
            objective_reference=reference,
            violation=applied_violation,
            solver_violation=float(violation),
            seconds=time.perf_counter() - begin,
            solver_status="solved"
            if accepted
            else "post_command_rejected"
            if feasible
            else "solver_rejected",
        )


def _unicycle_od_functions(robot, config):
    steps = STEPS

    def initialize(x, goal, obs, mask, noise, key, ready, unit_x, unit_o, unit_i):
        truth = unicycle_sensor_model(
            x, obs, mask, noise, key, robot, steps, unit_x=unit_x, unit_o=unit_o, unit_i=unit_i
        )
        initial, physical_obs, *_ = truth
        clearance = jnp.min(uod.signed_clearance(initial[:2], physical_obs, mask, robot.radius))
        at_goal = (jnp.linalg.norm(initial[:2] - goal) <= robot.goal_tolerance) & (
            jnp.abs(initial[3]) <= 0.2
        )
        status = jnp.where(
            ~ready,
            uod.PLANNER_FAILURE,
            jnp.where(clearance <= 0, uod.COLLISION, jnp.where(at_goal, uod.GOAL, uod.RUNNING)),
        )
        return (truth, clearance, status)

    _, assemble, advance = uod.make_kernels(robot, config, steps)
    return jax.jit(jax.vmap(initialize)), assemble, advance


class UnicycleDecayQP:
    def __init__(self, robot, spec):
        self.config = robot
        self.od_config = uod.OptimalDecayConfig(**spec["policy"])
        self.functions = _unicycle_od_functions(robot, self.od_config)
        self.compiled = {}
        self.backend = spec.get("backend", "osqp")
        self.steps = STEPS

    def _execute(self, name, fn, args):
        with jax.enable_x64(False):
            if name not in self.compiled:
                self.compiled[name] = fn.lower(*args).compile()
            return jax.device_get(self.compiled[name](*args))

    def run(self, scenario):
        """Same native QP and arithmetic as optimal_decay.run; one reusable lane."""
        from oa_cbf_jax.controllers import NativeOSQP, NativeProxQP, NativeClarabel

        robot, config = (self.config, self.od_config)
        p = scenario["parent"]
        scene = p["scene"]
        f = lambda value: np.asarray(value, np.float32)
        x0, goal, obs = [f([scene[k]]) for k in ("initial_state", "goal", "obstacles")]
        mask = np.asarray([scene["obstacle_mask"]], bool)
        points = f([p["route"]["points"]])
        rm = np.asarray([p["route"]["mask"]], bool)
        noise = f([scenario["noise_scale"] * np.array([0.02, 0.03, 0.02, 0.03, 0.025, 0.01])])
        binding = {
            "key": np.asarray(scenario["sensor_key"], np.uint32),
            "batched_units": tuple(
                (scenario["sensor_units"][name][None] for name in ("unit_x", "unit_o", "unit_i"))
            ),
        }
        keys = binding["key"][None]
        init_fn, assemble_fn, advance_fn = self.functions
        truth, minimum, status = self._execute(
            "init",
            init_fn,
            (
                x0,
                goal,
                obs,
                mask,
                noise,
                keys,
                np.asarray([p["route"]["status"] == "ready"]),
                binding["batched_units"][0],
                binding["batched_units"][1],
                binding["batched_units"][2],
            ),
        )
        x, physical, xb, ob, xs, os, innovations = truth
        initial = x[0].copy()
        physical0 = physical[0].copy()
        cursor = np.zeros(1, np.float32)
        count, rows, solver = (0, [], None)
        begin = time.perf_counter()
        solver_seconds = 0.0
        for k in range(self.steps):
            if status[0] != 0:
                break
            assembled = self._execute(
                "assemble",
                assemble_fn,
                (
                    np.int32(k),
                    x,
                    physical,
                    xb,
                    ob,
                    xs,
                    os,
                    innovations[:, k],
                    goal,
                    mask,
                    points,
                    rm,
                    cursor,
                    noise,
                ),
            )
            (ref, a, b, h, psi, proposed, remaining, target), sx, so = assembled
            if solver is None:
                factory = {"osqp": NativeOSQP, "proxqp": NativeProxQP, "clarabel": NativeClarabel}[
                    self.backend
                ]
                solver = factory(
                    b.shape[1],
                    config.weights,
                    **{"solver_defaults": True} if self.backend == "osqp" else {},
                )
            t = time.perf_counter()
            result = solver.solve(ref[0], a[0], b[0])
            solver_seconds += time.perf_counter() - t
            solution = np.zeros_like(ref)
            ok = np.zeros(1, bool)
            residual = np.nan
            reason = result["status"]
            if result["feasible"]:
                applied = np.asarray(result["control"], np.float32)
                residual = float(
                    np.max(np.asarray(a[0], float) @ applied - np.asarray(b[0], float))
                )
                if np.isfinite(applied).all() and residual <= robot.qp_tolerance:
                    solution[0] = applied
                    ok[0] = True
                else:
                    reason = "applied_precision_rejected"
            admissible = (
                np.all(~mask | (h >= -robot.qp_tolerance) & (psi >= -robot.qp_tolerance), axis=1)
                if config.formulation == "hocbf"
                else np.ones(1, bool)
            )
            x, status, active, clear, bound = self._execute(
                "advance",
                advance_fn,
                (np.int32(k), x, goal, physical, mask, solution[:, :2], status, ok, admissible),
            )
            cursor = np.where(active, proposed, cursor)
            count += int(active[0])
            minimum = np.minimum(minimum, np.where(active, clear, np.inf))
            rows.append(
                dict(
                    state=x[0].copy(),
                    control=np.where(active[:, None], solution[:, :2], 0.0)[0],
                    active=active[0],
                    status=status[0],
                    clearance=clear[0],
                    state_bound_violation=bound[0],
                    qp_violation=residual,
                    solver_status=reason,
                    solver_iterations=result["iterations"],
                    observed_state=sx[0],
                    observed_obstacles=so[0],
                    odqp_coefficients=solution[0, 2:].copy(),
                )
            )
        names = {
            0: "running",
            1: "goal_reached",
            2: "collision",
            3: "solver_rejected",
            4: "timeout",
            5: "hocbf_inadmissible",
            7: "planner_failure",
            8: "state_bound_violation",
        }
        if status[0] == 0:
            status = np.asarray([4])
        if not rows:
            rows = [
                dict(
                    state=x[0],
                    control=np.zeros(2, np.float32),
                    active=False,
                    status=status[0],
                    clearance=np.nan,
                    state_bound_violation=np.nan,
                    qp_violation=np.nan,
                    solver_status="not_attempted",
                    solver_iterations=0,
                    observed_state=x0[0],
                    observed_obstacles=obs[0],
                    odqp_coefficients=np.zeros(2, np.float32),
                )
            ]
        data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        data.update(true_initial_state=initial, true_obstacles=physical0)
        report = dict(
            status=names[int(status[0])],
            steps=count,
            min_clearance=float(minimum[0]),
            solver_seconds=solver_seconds,
            execution_seconds=time.perf_counter() - begin,
            termination_reason=str(data["solver_status"][-1]),
            solver_attempts=sum((r["solver_status"] != "not_attempted" for r in rows)),
        )
        return (report, data)


def _quad2d_barrier(x, u, obs, gains, omega, robot, buffer=None, *, stage=0):
    """Same repeated-input Euler formula, centers at stages k,k+1,k+2."""
    import numpy as np

    obs = np.asarray(obs, float)
    if not np.any(obs[:, 3:5]):
        return qmpc.numpy_barrier(x, u, obs, gains, omega, robot, buffer)
    a = np.asarray(gains) * np.asarray(omega)
    radius = robot.radius + (robot.clearance_buffer if buffer is None else buffer)
    x1 = qmpc.numpy_euler(x, u, robot)
    x2 = qmpc.numpy_euler(x1, u, robot)

    def h(y, k):
        centers = obs[:, :2] + k * robot.dt * obs[:, 3:5]
        return np.sum((y[:2] - centers) ** 2, axis=-1) - 1.01 * (radius + obs[:, 2]) ** 2

    h0, h1, h2 = (h(x, stage), h(x1, stage + 1), h(x2, stage + 2))
    return h2 - 2 * h1 + h0 + a.sum() * (h1 - h0) + a.prod() * h0


def _quad2d_prediction_residual(
    states, controls, omegas, x, obs, mask, gains, config, shared_contract=True
):
    """Independent NumPy check of every equality, obstacle, actuator and bound."""
    c = config.robot
    arrays = (states, controls, omegas, x, obs, gains)
    if not all((np.isfinite(a).all() for a in arrays)):
        return (np.inf, np.inf)
    equality = max(
        float(np.max(np.abs(states[0] - x))),
        max(
            (
                float(np.max(np.abs(states[k + 1] - qmpc.numpy_euler(states[k], controls[k], c))))
                for k in range(len(controls))
            )
        ),
    )
    barriers = np.stack(
        [
            _quad2d_barrier(
                s, u, obs, gains, w, c, buffer=None if shared_contract else 0.0, stage=k
            )
            for k, (s, u, w) in enumerate(zip(states, controls, omegas))
        ]
    )
    violation = max(
        float(np.max(-barriers[:, mask], initial=-np.inf)),
        float(np.max(c.force_min - controls)),
        float(np.max(controls - c.force_max)),
    )
    if shared_contract:
        limits = np.array(
            [
                config.pitch_limit,
                config.velocity_limit,
                config.velocity_limit,
                config.pitch_rate_limit,
            ]
        )
        violation = max(violation, float(np.max(np.abs(states[1:, 2:]) - limits)))
    return (equality, violation)


class Quad2DMPC(qmpc.Quad2DMPC):
    def __init__(
        self, capacity, method="fixed_low", config=qmpc.FlightConfig(), *, shared_contract=True
    ):
        import casadi as ca

        if (
            method not in qmpc.DEFAULTS
            or isinstance(capacity, bool)
            or (not isinstance(capacity, int))
            or (capacity < 1)
        ):
            raise ValueError("Invalid flight MPC method/capacity")
        self.capacity = capacity
        self.method = method
        self.config = config
        self.horizon = 10
        self.gains = np.asarray(qmpc.DEFAULTS[method])
        self.decay = method == "optimal_decay"
        self.shared_contract = shared_contract
        self.last_omega = np.zeros(2)
        c = config.robot
        H = self.horizon
        C = capacity
        X = ca.SX.sym("x", 6, H + 1)
        U = ca.SX.sym("u", 2, H)
        W = ca.SX.sym("omega", 2, H) if self.decay else ca.DM.ones(2, H)
        initial = ca.SX.sym("initial", 6)
        goal = ca.SX.sym("goal", 2)
        obs = ca.SX.sym("obs", C, 5)
        active = ca.SX.sym("active", C)
        previous = ca.SX.sym("previous", 2)
        parameters = ca.vertcat(initial, goal, ca.vec(obs), active, previous)
        variables = (
            ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(W))
            if self.decay
            else ca.vertcat(ca.vec(X), ca.vec(U))
        )
        target = ca.vertcat(goal, 0, 0, 0, 0)
        weights = ca.DM(qmpc.Q)

        def step(x, u):
            force = ca.sum1(u) / c.mass
            return x + c.dt * ca.vertcat(
                x[3],
                x[4],
                x[5],
                -ca.sin(x[2]) * force,
                ca.cos(x[2]) * force - c.gravity,
                c.arm / c.inertia * (u[0] - u[1]),
            )

        def barrier(x, u, omega, stage):
            radius = c.radius + (c.clearance_buffer if shared_contract else 0.0)

            def h(y):
                return (
                    (obs[:, 0] - y[0]) ** 2
                    + (obs[:, 1] - y[1]) ** 2
                    - 1.01 * (obs[:, 2] + radius) ** 2
                )

            x1 = step(x, u)
            x2 = step(x1, u)
            h0, h1, h2 = (h(x), h(x1), h(x2))
            a1, a2 = (self.gains[0] * omega[0], self.gains[1] * omega[1])
            static = h2 - 2 * h1 + h0 + (a1 + a2) * (h1 - h0) + a1 * a2 * h0

            def moving_h(y, k):
                centers = obs[:, :2] + k * c.dt * obs[:, 3:5]
                return (
                    (centers[:, 0] - y[0]) ** 2
                    + (centers[:, 1] - y[1]) ** 2
                    - 1.01 * (obs[:, 2] + radius) ** 2
                )

            m0, m1, m2 = (moving_h(x, stage), moving_h(x1, stage + 1), moving_h(x2, stage + 2))
            moving = m2 - 2 * m1 + m0 + (a1 + a2) * (m1 - m0) + a1 * a2 * m0
            return ca.if_else(ca.mmax(ca.fabs(obs[:, 3:5])) > 0, moving, static)

        equality = [X[:, 0] - initial]
        inequality = []
        cost = 0
        limits = ca.DM(
            [
                config.pitch_limit,
                config.velocity_limit,
                config.velocity_limit,
                config.pitch_rate_limit,
            ]
        )
        for k in range(H):
            cost += ca.dot(weights, (X[:, k] - target) ** 2)
            cost += (
                10 * ca.sumsqr(W[:, k] - 1)
                if self.decay
                else 0.5 * ca.sumsqr(U[:, k] - (previous if k == 0 else U[:, k - 1]))
            )
            equality.append(X[:, k + 1] - step(X[:, k], U[:, k]))
            inequality.append(active * barrier(X[:, k], U[:, k], W[:, k], k) + (1 - active))
            if shared_contract:
                inequality.append(ca.vertcat(limits - X[2:, k + 1], limits + X[2:, k + 1]))
        cost += ca.dot(weights, (X[:, H] - target) ** 2)
        constraints = ca.vertcat(*equality, *inequality)
        self.equalities = 6 * (H + 1)
        self.lbg = np.zeros(int(constraints.numel()))
        self.ubg = np.r_[
            np.zeros(self.equalities), np.full(len(self.lbg) - self.equalities, np.inf)
        ]
        self.lbx = np.full(int(variables.numel()), -np.inf)
        self.ubx = -self.lbx.copy()
        self.lbx[6 * (H + 1) : 6 * (H + 1) + 2 * H] = c.force_min
        self.ubx[6 * (H + 1) : 6 * (H + 1) + 2 * H] = c.force_max
        self.solver_options = {"ipopt.print_level": 0, "print_time": False}
        start = time.perf_counter()
        self.solver = ca.nlpsol(
            "quad2d_mpc",
            "ipopt",
            dict(x=variables, p=parameters, f=cost, g=constraints),
            self.solver_options,
        )
        self.setup_seconds = time.perf_counter() - start
        sx = ca.SX.sym("state", 6)
        su = ca.SX.sym("control", 2)
        sw = ca.SX.sym("omega", 2)
        self.barrier_expression = ca.Function(
            "flight_discrete_barrier", [sx, su, obs, sw], [barrier(sx, su, sw, 0)]
        )

    def solve(self, x, goal, obs, mask, previous):
        x = np.asarray(x, float)
        goal = np.asarray(goal, float)
        obs = np.asarray(obs, float)
        mask = np.asarray(mask, bool)
        previous = np.asarray(previous, float)
        H = self.horizon
        if (
            x.shape != (6,)
            or goal.shape != (2,)
            or obs.shape != (self.capacity, 5)
            or (mask.shape != (self.capacity,))
            or (previous.shape != (2,))
        ):
            raise ValueError("Flight MPC input shape mismatch")
        p = np.r_[x, goal, obs.ravel(order="F"), mask.astype(float), previous]
        if not np.isfinite(p).all():
            raise ValueError("Nonfinite flight MPC input")
        guess = np.r_[np.tile(x, H + 1), np.tile(previous, H)]
        if self.decay:
            guess = np.r_[guess, np.tile(self.last_omega, H)]
        start = time.perf_counter()
        answer = self.solver(x0=guess, p=p, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        elapsed = time.perf_counter() - start
        stats = self.solver.stats()
        v = np.asarray(answer["x"]).ravel()
        X = v[: 6 * (H + 1)].reshape(H + 1, 6)
        U = v[6 * (H + 1) : 6 * (H + 1) + 2 * H].reshape(H, 2)
        W = v[-2 * H :].reshape(H, 2) if self.decay else np.ones((H, 2))
        eq, violation = _quad2d_prediction_residual(
            X, U, W, x, obs, mask, self.gains, self.config, self.shared_contract
        )
        feasible = bool(
            stats.get("success", False)
            and eq <= self.config.robot.qp_tolerance
            and (violation <= self.config.robot.qp_tolerance)
        )
        return dict(
            control=U[0],
            omega=W[0],
            states=X,
            controls=U,
            omegas=W,
            feasible=feasible,
            solver_success=bool(stats.get("success", False)),
            solver_status=str(stats.get("return_status")),
            iterations=int(stats.get("iter_count", -1)),
            solve_seconds=elapsed,
            max_equality_error=eq,
            max_constraint_violation=violation,
            objective=float(answer["f"]),
        )


def _check_quad2d_command(x, u, obs, mask, effective_gains, config):
    c = config.robot
    residual = _quad2d_barrier(x, u, obs, effective_gains, np.ones(2), c)[mask]
    return (np.r_[residual, u - c.force_min, c.force_max - u], np.inf, np.inf)


def _quad2d_mpc_episode(parent, solver, kernels, steps, ordered=False):
    config = solver.config
    c = config.robot
    observed, goal, obs, mask, noise = (
        np.asarray(parent[k], bool if k == "obstacle_mask" else np.float32)
        for k in ("initial_state", "goal", "obstacles", "obstacle_mask", "noise")
    )
    if ordered:
        from oa_cbf_jax.quad2d_static_inputs import validate_parent, numpy_arrived

        validate_parent(parent)
        goals = np.asarray(parent["waypoint_goals"], np.float32)
        total = parent["waypoint_count"]
        leg = 0
        routes = np.asarray(parent["waypoint_routes"]["points"], np.float32)
        route_masks = np.asarray(parent["waypoint_routes"]["mask"], bool)
        route_ready = np.asarray(parent["waypoint_routes"]["ready"], bool)
        goal = goals[0]
        points = routes[0]
        rm = route_masks[0]
    else:
        points = np.asarray(parent["route"]["points"], np.float32)
        rm = np.asarray(parent["route"]["mask"], bool)
    key = np.asarray(jax.random.PRNGKey(parent["seed"] + 7193))
    initial, truth, xb, ob, xs, os, innovations = kernels.prepare(observed, obs, mask, noise, key)
    x = initial
    previous = np.zeros(2, np.float32)
    solver.last_omega = np.zeros(2)
    cursor = np.float32(0)
    count = 0
    status = 0
    minimum = float(kernels.clearance(x, truth, mask))
    if bool(kernels.arrived(x, goal)) and (not ordered or total == 1):
        status = qmpc.GOAL
    if float(kernels.bound(x)) > c.qp_tolerance:
        status = qmpc.STATE_BOUND
    if minimum <= 0:
        status = qmpc.COLLISION
    if not (bool(route_ready[0]) if ordered else parent["route"]["status"] == "ready"):
        status = qmpc.PLANNER_FAILURE
    records = []
    reason = qmpc.NAMES[status]
    start = time.perf_counter()
    for k in range(steps):
        sensed, seen = map(
            np.asarray, kernels.sense(x, truth, xb, ob, xs, os, innovations[k], np.int32(k))
        )
        mission_info = {}
        if ordered:
            handoff = (
                status == 0
                and leg < total - 1
                and numpy_arrived(sensed.astype(float), goals[leg].astype(float), config, noise)
            )
            if handoff:
                leg += 1
                cursor = np.float32(0)
            goal = goals[leg]
            points = routes[leg]
            rm = route_masks[leg]
            if status == 0 and (not route_ready[leg]):
                status = qmpc.PLANNER_FAILURE
                reason = qmpc.NAMES[status]
            mission_info = dict(
                waypoint_index=leg,
                waypoint_handoff=handoff,
                mission_goal=goal,
                mission_previous_control=previous,
                mission_route_cursor_before=cursor,
            )
        target, proposed, remaining = map(np.asarray, kernels.target(sensed, points, rm, cursor))
        attempted = status == 0
        accepted = False
        control = np.zeros(2, np.float32)
        clear = bound = np.nan
        result = dict(
            states=np.full((11, 6), np.nan),
            controls=np.full((10, 2), np.nan),
            omegas=np.full((10, 2), np.nan),
            omega=np.ones(2),
            solver_success=False,
            solver_status="not_attempted",
            iterations=0,
            solve_seconds=0.0,
            max_equality_error=np.nan,
            max_constraint_violation=np.nan,
            feasible=False,
        )
        if attempted:
            result = solver.solve(sensed, target, seen, mask, previous)
            candidate = result["control"].astype(np.float32)
            residual, _, _ = _check_quad2d_command(
                sensed, candidate, seen, mask, solver.gains * result["omega"], config
            )
            accepted = (
                result["feasible"]
                and np.isfinite(candidate).all()
                and np.isfinite(residual).all()
                and (residual.min() >= -c.qp_tolerance)
            )
            if not accepted:
                status = 3
                reason = (
                    "solver_reported_failure:" + result["solver_status"]
                    if not result["solver_success"]
                    else "independent_prediction_rejected"
                    if not result["feasible"]
                    else "stored_command_rejected"
                )
            else:
                control = candidate
                x, clear, bound = kernels.advance(x, control, truth, mask, np.int32(k))
                clear = float(clear)
                bound = float(bound)
                minimum = min(minimum, clear)
                count += 1
                cursor = np.float32(proposed)
                previous = control
                solver.last_omega = result["omega"].copy()
                if bool(kernels.arrived(x, goal)) and (not ordered or leg == total - 1):
                    status = qmpc.GOAL
                if bound > c.qp_tolerance:
                    status = qmpc.STATE_BOUND
                if clear <= 0:
                    status = qmpc.COLLISION
                reason = qmpc.NAMES[status]
        if ordered:
            mission_info["waypoints_visited"] = leg + int(status == qmpc.GOAL)
        records.append(
            dict(
                state=np.asarray(x),
                control=control,
                active=accepted,
                status=status,
                observed_state=sensed,
                observed_obstacles=seen,
                clearance=clear,
                state_bound_violation=bound,
                route_progress=cursor,
                route_target=target,
                route_remaining=remaining,
                solver_attempted=attempted,
                solver_success=result["solver_success"],
                solver_status=result["solver_status"],
                solver_feasible=result["feasible"],
                solver_iterations=result["iterations"],
                solve_seconds=result["solve_seconds"],
                prediction_equality=result["max_equality_error"],
                prediction_violation=result["max_constraint_violation"],
                predicted_states=result["states"],
                predicted_controls=result["controls"],
                predicted_omegas=result["omegas"],
                omega=result["omega"],
                **mission_info,
            )
        )
        if status != 0:
            break
    if status == 0:
        status = qmpc.TIMEOUT
        reason = qmpc.NAMES[status]
    progress = float(
        kernels.coordinate(x[:2], points, rm, cursor)
        - kernels.coordinate(initial[:2], points, rm, np.float32(0))
    )
    data = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    data.update(
        true_initial_state=np.asarray(initial),
        true_obstacles=np.asarray(truth),
        initial_observation=observed,
        observed_obstacles_initial=obs,
        obstacle_mask=mask,
        noise=noise,
        goal=goal,
        points=points,
        route_mask=rm,
        key=key,
    )
    row = dict(
        group_id=parent["group_id"],
        family=parent["family"],
        obstacles=int(mask.sum()),
        noise_scale=float(noise[0] / 0.015),
        status=qmpc.NAMES[status],
        status_code=status,
        steps=count,
        min_clearance=minimum,
        route_progress=progress,
        final_state=np.asarray(x).tolist(),
        termination_reason=reason,
        execution_seconds=time.perf_counter() - start,
        solver_attempts=int(data["solver_attempted"].sum()),
        solver_seconds=float(data["solve_seconds"].sum()),
    )
    if ordered:
        data["goal"] = np.asarray(parent["goal"], np.float32)
        row.update(
            waypoint_index=leg,
            waypoints_visited=leg + int(status == qmpc.GOAL),
            required_waypoints=total,
            waypoint_handoffs=int(data["waypoint_handoff"].sum()),
        )
    return (qmpc.sanitize(row), data)


def _quad2d_moving_problem(x, target, obs, mask, config=qod.FlightConfig(), shared_buffer=True):
    x = np.asarray(x, float)
    obs = np.asarray(obs, float)
    mask = np.asarray(mask, bool)
    c = config.robot
    selected = qod.nearest(x, obs, mask)
    A = np.zeros((5, 4))
    b = np.zeros(5)
    if selected >= 0:
        o = obs[selected]
        d = x[:2] - o[:2]
        v = x[3:5] - o[3:5]
        h = d @ d - 1.01 * (o[2] + c.radius + (c.clearance_buffer if shared_buffer else 0.0)) ** 2
        hd = 2 * d @ v
        drift = 2 * v @ v - 2 * c.gravity * d[1]
        collective = 2 * (-d[0] * np.sin(x[2]) + d[1] * np.cos(x[2])) / c.mass
        A[0] = [-collective, -collective, -hd, -0.25 * h]
        b[0] = drift
    A[1:, :2] = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b[1:] = [c.force_max, -c.force_min, c.force_max, -c.force_min]
    return (np.r_[qod.nominal(x, target, config), 1.0, 1.0], A, b, selected)


def _quad2d_od_problem(x, target, obstacles, mask, config, shared_buffer=True):
    if np.any(obstacles[:, 3:5]):
        return _quad2d_moving_problem(x, target, obstacles, mask, config, shared_buffer)
    return qod.problem(x, target, obstacles, mask, config, shared_buffer)


class Quad2DDecayQP(qod.Quad2DODQP):
    def solve(self, x, target, obs, mask):
        ref, A, b, selected = _quad2d_od_problem(
            x, target, obs, mask, self.config, self.shared_buffer
        )
        if not all((np.isfinite(v).all() for v in [ref, A, b])):
            raise ValueError("Nonfinite original OD-QP inputs")
        if not self.initialized:
            from scipy import sparse

            start = time.perf_counter()
            pattern = sparse.csc_matrix(
                (A.ravel(order="F"), np.tile(np.arange(5), 4), np.arange(0, 21, 5)), shape=(5, 4)
            )
            self.solver.setup(
                P=sparse.diags(2 * self.weights, format="csc"),
                q=-2 * self.weights * ref,
                A=pattern,
                l=np.full(5, -np.inf),
                u=b,
                verbose=False,
            )
            self.setup_seconds = time.perf_counter() - start
            self.initialized = True
        else:
            self.solver.update(q=-2 * self.weights * ref, Ax=A.ravel(order="F"), u=b)
        start = time.perf_counter()
        result = self.solver.solve(raise_error=False)
        solution = np.full(4, np.nan) if result.x is None else np.asarray(result.x).copy()
        success = bool(result.info.status_val in (1, 2) and np.isfinite(solution).all())
        raw_violation = float(np.max(A @ solution - b)) if np.isfinite(solution).all() else np.inf
        applied = solution.copy()
        applied[:2] = solution[:2].astype(np.float32)
        stored_violation = float(np.max(A @ applied - b)) if np.isfinite(applied).all() else np.inf
        return dict(
            solution=solution,
            reference=ref,
            selected_obstacle=selected,
            solver_success=success,
            solver_status=result.info.status,
            status_value=int(result.info.status_val),
            iterations=int(result.info.iter),
            solve_seconds=time.perf_counter() - start,
            primal_residual=float(result.info.prim_res),
            dual_residual=float(result.info.dual_res),
            raw_violation=raw_violation,
            stored_violation=stored_violation,
            feasible=bool(
                success and max(raw_violation, stored_violation) <= self.config.robot.qp_tolerance
            ),
        )


def _quad2d_od_episode(parent, kernels, steps, config=qod.FlightConfig()):
    c = config.robot
    solver = Quad2DDecayQP(config)
    ordered = "waypoint_count" in parent
    observed, final_goal, obs, mask, noise = (
        np.asarray(parent[k], bool if k == "obstacle_mask" else np.float32)
        for k in ["initial_state", "goal", "obstacles", "obstacle_mask", "noise"]
    )
    if ordered:
        qod.validate_parent(parent)
        goals = np.asarray(parent["waypoint_goals"], np.float32)
        total = parent["waypoint_count"]
        routes = np.asarray(parent["waypoint_routes"]["points"], np.float32)
        route_masks = np.asarray(parent["waypoint_routes"]["mask"], bool)
        ready = np.asarray(parent["waypoint_routes"]["ready"], bool)
    else:
        goals = final_goal[None]
        total = 1
        routes = np.asarray(parent["route"]["points"], np.float32)[None]
        route_masks = np.asarray(parent["route"]["mask"], bool)[None]
        ready = [parent["route"]["status"] == "ready"]
    leg = 0
    key = np.asarray(jax.random.PRNGKey(parent["seed"] + 7193))
    initial, truth, xb, ob, xs, os, innovations = kernels.prepare(observed, obs, mask, noise, key)
    x = initial
    previous = np.zeros(2, np.float32)
    cursor = np.float32(0)
    count = 0
    minimum = float(kernels.clearance(x, truth, mask))
    status = 1 if total == 1 and bool(kernels.arrived(x, goals[0])) else 0
    if float(kernels.bound(x)) > c.qp_tolerance:
        status = 8
    if minimum <= 0:
        status = 2
    if not ready[0]:
        status = 7
    records = []
    reason = qod.NAMES[status]
    start = time.perf_counter()
    for k in range(steps):
        sensed, seen = map(
            np.asarray, kernels.sense(x, truth, xb, ob, xs, os, innovations[k], np.int32(k))
        )
        handoff = bool(
            status == 0
            and leg < total - 1
            and qod.numpy_arrived(sensed.astype(float), goals[leg].astype(float), config, noise)
        )
        if handoff:
            leg += 1
            cursor = np.float32(0)
        goal = goals[leg]
        points = routes[leg]
        rm = route_masks[leg]
        if status == 0 and (not ready[leg]):
            status = 7
            reason = qod.NAMES[status]
        before_cursor = cursor
        before_control = previous.copy()
        target, proposed, remaining = map(np.asarray, kernels.target(sensed, points, rm, cursor))
        attempted = status == 0
        accepted = False
        u = np.zeros(2, np.float32)
        clear = bound = np.nan
        result = dict(
            solution=np.full(4, np.nan),
            reference=np.r_[qod.nominal(sensed, target, config), 1.0, 1.0],
            selected_obstacle=qod.nearest(sensed, seen, mask),
            solver_success=False,
            solver_status="not_attempted",
            status_value=0,
            iterations=0,
            solve_seconds=0.0,
            primal_residual=np.nan,
            dual_residual=np.nan,
            raw_violation=np.nan,
            stored_violation=np.nan,
            feasible=False,
        )
        if attempted:
            result = solver.solve(sensed, target, seen, mask)
            accepted = result["feasible"]
            if not accepted:
                status = 3
                reason = (
                    "solver_reported_failure:" + result["solver_status"]
                    if not result["solver_success"]
                    else "independent_stored_qp_rejected"
                )
            else:
                u = result["solution"][:2].astype(np.float32)
                x, clear, bound = kernels.advance(x, u, truth, mask, np.int32(k))
                clear = float(clear)
                bound = float(bound)
                minimum = min(minimum, clear)
                count += 1
                cursor = np.float32(proposed)
                previous = u
                if leg == total - 1 and bool(kernels.arrived(x, goal)):
                    status = 1
                if bound > c.qp_tolerance:
                    status = 8
                if clear <= 0:
                    status = 2
                reason = qod.NAMES[status]
        mission = (
            dict(
                waypoint_index=leg,
                waypoint_handoff=handoff,
                mission_goal=goal,
                mission_previous_control=before_control,
                mission_route_cursor_before=before_cursor,
                waypoints_visited=leg + int(status == 1),
            )
            if ordered
            else {}
        )
        records.append(
            dict(
                state=np.asarray(x),
                control=u,
                active=accepted,
                status=status,
                observed_state=sensed,
                observed_obstacles=seen,
                clearance=clear,
                state_bound_violation=bound,
                route_progress=cursor,
                route_target=target,
                route_remaining=remaining,
                solver_attempted=attempted,
                **{"odqp_" + k: v for k, v in result.items()},
                **mission,
            )
        )
        if status:
            break
    if not status:
        status = 4
        reason = qod.NAMES[status]
    data = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    data.update(
        true_initial_state=np.asarray(initial),
        true_obstacles=np.asarray(truth),
        initial_observation=observed,
        observed_obstacles_initial=obs,
        obstacle_mask=mask,
        noise=noise,
        goal=final_goal,
        key=key,
    )
    if not ordered:
        data.update(points=routes[0], route_mask=route_masks[0])
    row = dict(
        group_id=parent["group_id"],
        family=parent["family"],
        obstacles=int(mask.sum()),
        noise_scale=float(parent.get("noise_scale", round(float(noise[0] / 0.015), 6))),
        status=qod.NAMES[status],
        status_code=status,
        steps=count,
        min_clearance=minimum,
        final_state=np.asarray(x).tolist(),
        route_progress=float(cursor),
        termination_reason=reason,
        execution_seconds=time.perf_counter() - start,
        solver_attempts=int(data["solver_attempted"].sum()),
        solver_seconds=float(data["odqp_solve_seconds"].sum()),
        solver_setup_seconds=solver.setup_seconds,
        waypoint_index=leg,
        waypoints_visited=leg + int(status == 1),
        required_waypoints=total,
        waypoint_handoffs=sum((int(r.get("waypoint_handoff", False)) for r in records)),
    )
    if ordered:
        row.update(original_kind=parent["original_kind"], variant_index=parent["variant_index"])
    return (qod.sanitize(row), data)


def _quad2d_bn_rows(state, obstacles, parameters, radius=0.3):
    import jax
    import jax.numpy as jnp
    from oa_cbf_jax.quad2d_barriernet import constraints as static, require_x64

    require_x64()

    def moving(_):
        d = state[:2] - obstacles[:, :2]
        v = state[3:5] - obstacles[:, 3:5]
        h = jnp.sum(d * d, axis=-1) - 1.01 * (obstacles[:, 2] + radius) ** 2
        hd = 2 * jnp.sum(d * v, axis=-1)
        drift = 2 * jnp.sum(v * v, axis=-1) - 2 * 9.81 * d[:, 1]
        coefficient = 2 * (-d[:, 0] * jnp.sin(state[2]) + d[:, 1] * jnp.cos(state[2]))
        return (
            -jnp.repeat(coefficient[:, None], 2, axis=1),
            drift + jnp.sum(parameters, axis=-1) * hd + jnp.prod(parameters, axis=-1) * h,
        )

    return jax.lax.cond(
        jnp.any(obstacles[:, 3:5] != 0),
        moving,
        lambda _: static(state, obstacles, parameters, radius),
        None,
    )


def _quad2d_bn_numpy(ctx, p, radius=0.3, lower=1.0, upper=10.0):
    """Independent CV BN rows; keep original feature/network/input semantics."""
    import numpy as np

    static = qbn.numpy_constraints
    x = ctx[:6]
    obs = ctx[8:].reshape(5, 7)
    if not np.any(obs[:, 3:5]):
        return static(ctx, p, radius, lower, upper)
    delta = x[:2] - obs[:, :2]
    relative = x[3:5] - obs[:, 3:5]
    h = np.sum(delta * delta, axis=1) - 1.01 * (obs[:, 2] + radius) ** 2
    hd = 2 * np.sum(delta * relative, axis=1)
    drift = 2 * np.sum(relative * relative, axis=1) - 2 * 9.81 * delta[:, 1]
    coefficient = 2 * (-delta[:, 0] * np.sin(x[2]) + delta[:, 1] * np.cos(x[2]))
    A = -np.repeat(coefficient[:, None], 2, axis=1)
    b = drift + np.sum(p, axis=1) * hd + np.prod(p, axis=1) * h
    return (np.vstack((A, np.eye(2), -np.eye(2))), np.r_[b, [upper] * 2, [-lower] * 2])


def _quad2d_bn_policy(model, params, mean, std, config=qbn.FlightConfig()):

    def policy(state, goal, obstacles, mask):
        z, ctx = qbn.features(state, goal, obstacles, mask, config.robot.radius)
        reference = qbn.nominal(state, goal, config)
        u_nom, p = model.apply({"params": params}, (z - mean) / std, state, goal, reference)
        G, h = _quad2d_bn_rows(state, ctx[8:].reshape(5, 7), p, config.robot.radius)
        candidate, valid, violation, raw, solver_valid, raw_violation = qbn.deployment_qp(
            u_nom, G, h, config
        )
        return dict(
            candidate=candidate,
            valid=valid,
            violation=violation,
            raw_control=raw,
            solver_valid=solver_valid,
            raw_violation=raw_violation,
            gains=p,
            u_nom=u_nom,
            reference=reference,
        )

    return jax.jit(policy)


def _make_quad2d_bn(bundle, config):
    manifest, model, params, mean, std = qbn.load_bundle(bundle)
    return _quad2d_bn_policy(model, params, mean, std, config), manifest


class Quad2DBarrierNet(qbn.Controller):
    def __init__(self, bundle, capacity=64, config=qbn.FlightConfig()):
        self.config = config
        fn, self.manifest = _make_quad2d_bn(bundle, config)
        start = time.perf_counter()
        self.execute = fn.lower(
            jnp.zeros(6, jnp.float64),
            jnp.zeros(2, jnp.float64),
            jnp.zeros((capacity, 5), jnp.float64),
            jnp.zeros(capacity, bool),
        ).compile()
        self.compile_seconds = time.perf_counter() - start

    def solve(self, state, target, obstacles, mask):
        start = time.perf_counter()
        result = jax.device_get(
            self.execute(
                jnp.asarray(state, jnp.float64),
                jnp.asarray(target, jnp.float64),
                jnp.asarray(obstacles, jnp.float64),
                jnp.asarray(mask),
            )
        )
        c = self.config.robot
        u = result["candidate"].astype(np.float32)
        _, ctx = qbn.numpy_features(
            np.asarray(state, float),
            np.asarray(target, float),
            np.asarray(obstacles, float),
            mask,
            c.radius,
        )
        A, b = _quad2d_bn_numpy(ctx, result["gains"], c.radius, c.force_min, c.force_max)
        stored = float(np.max(A @ u - b)) if np.isfinite(u).all() else np.inf
        result.update(
            feasible=bool(result["valid"] and stored <= c.qp_tolerance),
            stored_violation=stored,
            solve_seconds=time.perf_counter() - start,
        )
        return result


def _quad3d_barrier(
    states, controls, obs, alpha, omegas, config, shared_contract=True, coupled_decay=True
):
    import numpy as np

    if not np.any(np.asarray(obs)[:, 3:5]):
        return tmpc.numpy_barrier(
            states, controls, obs, alpha, omegas, config, shared_contract, coupled_decay
        )
    x, u, o = map(lambda v: np.asarray(v, float), (states, controls, obs))
    a, b = tmpc.matrices(config.robot)
    dt = config.robot.dt
    k1 = x @ a.T + u @ b.T
    k2 = (x + dt / 2 * k1) @ a.T + u @ b.T
    k3 = (x + dt / 2 * k2) @ a.T + u @ b.T
    k4 = (x + dt * k3) @ a.T + u @ b.T
    y = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    stage = np.arange(len(x), dtype=float)[:, None, None]
    current = o[None, :, :2] + stage * dt * o[None, :, 3:5]
    following = o[None, :, :2] + (stage + 1) * dt * o[None, :, 3:5]
    radius = config.robot.radius + (config.clearance_buffer if shared_contract else 0.0)
    r2 = 1.01 * (radius + o[:, 2]) ** 2
    h0 = np.sum((x[:, None, :2] - current) ** 2, axis=-1) - r2
    h1 = np.sum((y[:, None, :2] - following) ** 2, axis=-1) - r2
    gain = alpha * (np.asarray(omegas)[..., 0] if coupled_decay else np.ones(x.shape[:-1]))
    return h1 - h0 + gain[..., None] * h0


def _quad3d_prediction_residual(
    states, controls, omegas, x, obs, mask, alpha, c, shared_contract=True, coupled_decay=True
):
    if not all((np.isfinite(v).all() for v in (states, controls, omegas, x, obs))):
        return (np.inf, np.inf)
    a, b = tmpc.matrices(c.robot)
    expected = states[:-1] + c.robot.dt * (states[:-1] @ a.T + controls @ b.T)
    eq = max(float(np.max(abs(states[0] - x))), float(np.max(abs(states[1:] - expected))))
    barrier = _quad3d_barrier(
        states[:-1], controls, obs, alpha, omegas, c, shared_contract, coupled_decay
    )
    violation = max(
        float(np.max(-barrier[:, mask], initial=-np.inf)),
        float(np.max(c.robot.input_min - controls)),
        float(np.max(controls - c.robot.input_max)),
    )
    if shared_contract:
        limits = np.array(
            [c.tilt_limit, c.tilt_limit, c.yaw_limit] + [c.velocity_limit] * 3 + [c.rate_limit] * 3
        )
        violation = max(
            violation,
            float(np.max(abs(states[1:, 3:]) - limits)),
            float(np.max(c.altitude_min - states[1:, 2])),
            float(np.max(states[1:, 2] - c.altitude_max)),
        )
    return (eq, violation)


class Quad3DMPC(tmpc.Quad3DMPC):
    def __init__(
        self,
        capacity=64,
        method="fixed_low",
        config=tmpc.Quad3DControlConfig(hold_guard="bernstein_v87"),
        *,
        shared_contract=True,
        coupled_decay=True,
    ):
        import casadi as ca

        if method not in tmpc.DEFAULTS or type(capacity) is not int or capacity < 1:
            raise ValueError("Invalid Quad3D MPC method/capacity")
        if config.nominal_bias_observer != "none" or config.qp_refinement != "none":
            raise ValueError("Native baselines do not inherit OA improvements")
        self.capacity = capacity
        self.method = method
        self.config = config
        self.horizon = 10
        self.alpha = tmpc.DEFAULTS[method]
        self.decay = method == "optimal_decay"
        self.shared_contract = shared_contract
        self.coupled_decay = coupled_decay
        self.last_omega = np.zeros(2)
        H = self.horizon
        C = capacity
        c = config
        X = ca.SX.sym("x", 12, H + 1)
        U = ca.SX.sym("u", 4, H)
        W = ca.SX.sym("omega", 2, H) if self.decay else ca.DM.ones(2, H)
        initial = ca.SX.sym("initial", 12)
        goal = ca.SX.sym("goal", 3)
        obs = ca.SX.sym("obs", C, 5)
        active = ca.SX.sym("active", C)
        previous = ca.SX.sym("previous", 4)
        parameters = ca.vertcat(initial, goal, ca.vec(obs), active, previous)
        variables = (
            ca.vertcat(ca.vec(X), ca.vec(U), ca.vec(W))
            if self.decay
            else ca.vertcat(ca.vec(X), ca.vec(U))
        )
        a, b, f, g = map(ca.DM, tmpc.flow(c.robot))
        target = ca.vertcat(goal, ca.DM.zeros(9))
        weights = ca.DM(tmpc.Q)

        def barrier(x, u, w, stage=0):
            radius = c.robot.radius + (c.clearance_buffer if shared_contract else 0.0)

            def h(z):
                return (
                    (obs[:, 0] - z[0]) ** 2
                    + (obs[:, 1] - z[1]) ** 2
                    - 1.01 * (obs[:, 2] + radius) ** 2
                )

            gain = self.alpha * (w[0] if coupled_decay else 1.0)
            static_value = h(f @ x + g @ u) - h(x) + gain * h(x)

            def moving_h(z, k):
                return (
                    (obs[:, 0] + k * c.robot.dt * obs[:, 3] - z[0]) ** 2
                    + (obs[:, 1] + k * c.robot.dt * obs[:, 4] - z[1]) ** 2
                    - 1.01 * (obs[:, 2] + radius) ** 2
                )

            moving_value = (
                moving_h(f @ x + g @ u, stage + 1) - moving_h(x, stage) + gain * moving_h(x, stage)
            )
            return ca.if_else(ca.mmax(ca.fabs(obs[:, 3:5])) > 0, moving_value, static_value)

        equalities = [X[:, 0] - initial]
        inequalities = []
        cost = 0
        limits = ca.DM(
            [c.tilt_limit, c.tilt_limit, c.yaw_limit] + [c.velocity_limit] * 3 + [c.rate_limit] * 3
        )
        for k in range(H):
            cost += ca.dot(weights, (X[:, k] - target) ** 2)
            cost += (
                10 * ca.sumsqr(W[:, k] - 1)
                if self.decay
                else ca.sumsqr(U[:, k] - (previous if k == 0 else U[:, k - 1]))
            )
            equalities.append(X[:, k + 1] - X[:, k] - c.robot.dt * (a @ X[:, k] + b @ U[:, k]))
            inequalities.append(active * barrier(X[:, k], U[:, k], W[:, k], k) + (1 - active))
            if shared_contract:
                inequalities.append(
                    ca.vertcat(
                        limits - X[3:, k + 1],
                        limits + X[3:, k + 1],
                        X[2, k + 1] - c.altitude_min,
                        c.altitude_max - X[2, k + 1],
                    )
                )
        cost += ca.dot(weights, (X[:, H] - target) ** 2)
        constraints = ca.vertcat(*equalities, *inequalities)
        self.equalities = 12 * (H + 1)
        self.lbg = np.zeros(int(constraints.numel()))
        self.ubg = np.r_[
            np.zeros(self.equalities), np.full(len(self.lbg) - self.equalities, np.inf)
        ]
        self.lbx = np.full(int(variables.numel()), -np.inf)
        self.ubx = -self.lbx.copy()
        self.lbx[12 * (H + 1) : 12 * (H + 1) + 4 * H] = c.robot.input_min
        self.ubx[12 * (H + 1) : 12 * (H + 1) + 4 * H] = c.robot.input_max
        self.solver_options = {"ipopt.print_level": 0, "print_time": False}
        start = time.perf_counter()
        self.solver = ca.nlpsol(
            "quad3d_native_mpc",
            "ipopt",
            dict(x=variables, p=parameters, f=cost, g=constraints),
            self.solver_options,
        )
        self.setup_seconds = time.perf_counter() - start
        sx = ca.SX.sym("state", 12)
        su = ca.SX.sym("control", 4)
        sw = ca.SX.sym("omega", 2)
        self.barrier_expression = ca.Function(
            "quad3d_native_barrier", [sx, su, obs, sw], [barrier(sx, su, sw)]
        )

    def solve(self, x, goal, obs, mask, previous):
        x, goal, obs, previous = map(lambda v: np.asarray(v, float), (x, goal, obs, previous))
        mask = np.asarray(mask, bool)
        H = self.horizon
        if (
            x.shape != (12,)
            or goal.shape != (3,)
            or obs.shape != (self.capacity, 5)
            or (mask.shape != (self.capacity,))
            or (previous.shape != (4,))
        ):
            raise ValueError("Quad3D MPC input shape mismatch")
        parameters = np.r_[x, goal, obs.ravel(order="F"), mask.astype(float), previous]
        if not np.isfinite(parameters).all():
            raise ValueError("Nonfinite Quad3D MPC input")
        guess = np.r_[np.tile(x, H + 1), np.tile(previous, H)]
        if self.decay:
            guess = np.r_[guess, np.tile(self.last_omega, H)]
        start = time.perf_counter()
        answer = self.solver(
            x0=guess, p=parameters, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg
        )
        elapsed = time.perf_counter() - start
        stats = self.solver.stats()
        v = np.asarray(answer["x"]).ravel()
        X = v[: 12 * (H + 1)].reshape(H + 1, 12)
        U = v[12 * (H + 1) : 12 * (H + 1) + 4 * H].reshape(H, 4)
        W = v[-2 * H :].reshape(H, 2) if self.decay else np.ones((H, 2))
        eq, violation = _quad3d_prediction_residual(
            X, U, W, x, obs, mask, self.alpha, self.config, self.shared_contract, self.coupled_decay
        )
        feasible = bool(
            stats.get("success", False)
            and eq <= self.config.qp_tolerance
            and (violation <= self.config.qp_tolerance)
        )
        return dict(
            control=U[0],
            omega=W[0],
            states=X,
            controls=U,
            omegas=W,
            feasible=feasible,
            solver_success=bool(stats.get("success", False)),
            solver_status=str(stats.get("return_status")),
            iterations=int(stats.get("iter_count", -1)),
            solve_seconds=elapsed,
            max_equality_error=eq,
            max_constraint_violation=violation,
            objective=float(answer["f"]),
        )


def _quad3d_moving_bn_rows(state, obstacles, parameters, radius=0.25):
    """Exact original reduced-state proxy: fourth control and p2 are unused."""
    delta = jnp.column_stack((state[:2] - obstacles[:, :2], jnp.full(len(obstacles), state[2])))
    barrier = jnp.sum(delta**2, axis=-1) - 1.01 * (obstacles[:, 2] + radius) ** 2
    drift = 2 * jnp.sum(
        delta * (state[3:6] - jnp.column_stack((obstacles[:, 3:5], jnp.zeros(len(obstacles))))),
        axis=-1,
    )
    return (
        -jnp.column_stack((2 * delta, jnp.zeros(len(obstacles)))),
        drift + parameters[:, 0] * barrier,
    )


def _quad3d_bn_rows(state, obstacles, parameters, radius=0.25):
    return jax.lax.cond(
        jnp.any(obstacles[:, 3:5] != 0),
        lambda _: _quad3d_moving_bn_rows(state, obstacles, parameters, radius),
        lambda _: variants.quad3d_constraints(state, obstacles, parameters, radius),
        None,
    )


def _quad3d_bn_problem(model, params, mean, std, robot_model, config):
    """Return one fixed-shape NN/nominal/constraint function; caller AOT compiles."""
    vbn.require_x64()
    if robot_model == "Quad3D":
        if not isinstance(config, vbn.Quad3DConfig):
            raise ValueError("Expected Quad3D physics")
        feature, nominal, rows = (vbn.quad3d_features, vbn.quad3d_nominal, _quad3d_bn_rows)
        state_dim, goal_dim = (6, 3)
    elif robot_model == "KinematicBicycle2D_DPCBF":
        if not isinstance(config, vbn.BicycleConfig):
            raise ValueError("Expected bicycle physics")
        feature, nominal, rows = (
            vbn.bicycle_features,
            vbn.bicycle_nominal,
            vbn.bicycle_constraints,
        )
        state_dim, goal_dim = (4, 2)
    else:
        raise ValueError("Unsupported native BarrierNet variant")
    mean = jnp.asarray(mean, jnp.float64)
    std = jnp.asarray(std, jnp.float64)

    def problem(state, goal, obstacles, mask):
        z, ctx = feature(state, goal, obstacles, mask, config.radius)
        ref = nominal(state, goal, config)
        u_nom, p = model.apply({"params": params}, (z - mean) / std, ctx[:state_dim], goal, ref)
        G, h = rows(ctx[:state_dim], ctx[state_dim + goal_dim :].reshape(5, 7), p, config.radius)
        return (u_nom, G, h, p, ref, z, ctx)

    return jax.jit(problem)


class Quad3DBarrierNet(dbn.Quad3DController):
    def __init__(self, bundle, config=dbn.Quad3DConfig(), capacity=64):
        self.config = config
        self.manifest, model, self.params, self.mean, self.std = dbn.load_bundle(
            bundle, "Quad3D", config
        )
        start = time.perf_counter()
        self.fn = _quad3d_bn_problem(model, self.params, self.mean, self.std, "Quad3D", config)
        args = (
            jnp.zeros(12, jnp.float64),
            jnp.zeros(3, jnp.float64),
            jnp.zeros((capacity, 5), jnp.float64),
            jnp.zeros(capacity, bool),
        )
        self.execute = self.fn.lower(*args).compile()
        self.compile_seconds = time.perf_counter() - start
        self.reset()


def _quad3d_mpc_episode(p, solver, kernels, steps=1600, ordered=False, *, tape):
    c = solver.config
    mask = np.asarray(p["mask"], bool)
    original = np.asarray(p["obstacles"], float)
    x = np.asarray(p["x"], float)
    goal = np.asarray(p["goal"], float)
    noise = np.asarray(p["noise"], float)
    points = np.asarray(p["route"]["points"], float)
    rm = np.asarray(p["route"]["mask"], bool)
    leg = 0
    if ordered:
        goals = np.asarray(p["waypoint_goals"], float)
        total = p["waypoint_count"]
        goal = goals[0]
        routes = np.asarray(p["waypoint_routes"]["points"], float)
        route_masks = np.asarray(p["waypoint_routes"]["mask"], bool)
        points = routes[0]
        rm = route_masks[0]
    bx, bo, ix, io = tuple((tape[name] for name in ("bx", "bo", "ix", "io")))
    cursor = 0.0
    previous = np.zeros(4)
    solver.last_omega = np.zeros(2)
    clear, bound = tmpc.initial_physical(x, original, mask, c)
    status = 4 if clear <= 0 else 5 if bound > c.qp_tolerance else 0
    count = 0
    records = []
    start = time.perf_counter()
    for k in range(steps):
        truth = original.copy()
        truth[:, :2] += k * c.robot.dt * original[:, 3:5]
        seen, so = map(
            np.asarray, tmpc.execute64(kernels.sense, x, truth, mask, bx, bo, noise, ix[k], io[k])
        )
        mission_info = {}
        if ordered:
            handoff = (
                status == 0
                and leg < total - 1
                and bool(tmpc.execute64(kernels.arrived, seen, goals[leg], noise))
            )
            if handoff:
                leg += 1
                cursor = 0.0
            goal = goals[leg]
            points = routes[leg]
            rm = route_masks[leg]
            mission_info = dict(waypoint_index=leg, waypoint_handoff=handoff, mission_goal=goal)
        target, progress, remaining, visible, controlled = map(
            np.asarray,
            tmpc.execute64(
                kernels.route,
                seen,
                goal,
                so,
                mask,
                points,
                rm,
                np.asarray(cursor, np.float64),
                noise,
            ),
        )
        if (
            status == 0
            and (not ordered or leg == total - 1)
            and bool(tmpc.execute64(kernels.arrived, seen, goal, noise))
        ):
            status = 1
        result = tmpc.empty_result()
        attempted = status == 0
        active = False
        before = x.copy()
        cursor_before = cursor
        previous_before = previous.copy()
        omega_before = solver.last_omega.copy()
        if attempted:
            try:
                result = solver.solve(seen, target, controlled, mask, previous)
            except RuntimeError as error:
                result["solver_status"] = "exception:" + str(error)
            if result["feasible"]:
                active = True
                previous = result["control"]
                solver.last_omega = result["omega"]
                cursor = float(progress)
                x, clear, bound = map(
                    np.asarray, tmpc.execute64(kernels.advance, x, previous, truth, mask)
                )
                count += 1
                if clear <= 0:
                    status = 4
                elif bound > c.qp_tolerance:
                    status = 5
            else:
                status = 3
        records.append(
            dict(
                state=before,
                next_state=x.copy(),
                observed=seen,
                control=previous.copy() if active else np.zeros(4),
                active=active,
                status=status,
                attempted=attempted,
                clearance=clear,
                envelope=bound,
                route_target=target,
                route_cursor_before=cursor_before,
                route_progress=cursor,
                route_remaining=remaining,
                route_visible=visible,
                previous_control=previous_before,
                previous_omega=omega_before,
                **mission_info,
                **{"mpc_" + key: value for key, value in result.items()},
            )
        )
        if status:
            break
    if status == 0:
        truth = original.copy()
        truth[:, :2] += steps * c.robot.dt * original[:, 3:5]
        seen, _ = tmpc.execute64(kernels.sense, x, truth, mask, bx, bo, noise, ix[steps], io[steps])
        status = (
            1
            if (not ordered or leg == total - 1)
            and bool(tmpc.execute64(kernels.arrived, seen, goal, noise))
            else 6
        )
    assert all((v == 0 for v in kernels.cache_sizes().values()))
    trace = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    row = dict(
        id=p["id"],
        family=p["family"],
        noise_level=p["noise_level"],
        method=solver.method,
        status=status,
        steps=count,
        final_state=x.tolist(),
        elapsed_seconds=time.perf_counter() - start,
        solver_attempts=int(trace["attempted"].sum()),
        solver_seconds=float(trace["mpc_solve_seconds"].sum()),
        solver_iterations=int(trace["mpc_iterations"].sum()),
        termination_solver_status=str(trace["mpc_solver_status"][-1]),
        implicit_jit_cache_entries=kernels.cache_sizes(),
    )
    if ordered:
        row.update(waypoint_index=leg, waypoints_visited=leg + int(status == 1))
    return (row, trace)


def _quad3d_bn_episode(parent, kernels, solver, steps, config, *, tape):
    c = config
    mask = np.asarray(parent["mask"], bool)
    original = np.asarray(parent["obstacles"], float)
    x = np.asarray(parent["x"], float)
    noise = np.asarray(parent["noise"], float)
    ordered = "waypoint_count" in parent
    leg = 0
    goals = np.asarray(parent["waypoint_goals"] if ordered else [parent["goal"]], float)
    total = parent["waypoint_count"] if ordered else 1
    routes = np.asarray(
        parent["waypoint_routes"]["points"] if ordered else [parent["route"]["points"]], float
    )
    route_masks = np.asarray(
        parent["waypoint_routes"]["mask"] if ordered else [parent["route"]["mask"]], bool
    )
    bx, bo, ix, io = tuple((tape[name] for name in ("bx", "bo", "ix", "io")))
    cursor = 0.0
    solver.reset()
    clear, bound = dbn.initial_physical(x, original, mask, c)
    status = 4 if clear <= 0 else 5 if bound > c.qp_tolerance else 0
    count = 0
    records = []
    start = time.perf_counter()
    for tick in range(steps):
        truth = original.copy()
        truth[:, :2] += tick * c.robot.dt * original[:, 3:5]
        seen, obs = map(
            np.asarray,
            dbn.execute64(kernels.sense, x, truth, mask, bx, bo, noise, ix[tick], io[tick]),
        )
        handoff = (
            status == 0
            and leg < total - 1
            and bool(dbn.execute64(kernels.arrived, seen, goals[leg], noise))
        )
        if handoff:
            leg += 1
            cursor = 0.0
        goal = goals[leg]
        target, proposed, remaining, visible, controlled = map(
            np.asarray,
            dbn.execute64(
                kernels.route,
                seen,
                goal,
                obs,
                mask,
                routes[leg],
                route_masks[leg],
                np.asarray(cursor, np.float64),
                noise,
            ),
        )
        if (
            status == 0
            and leg == total - 1
            and bool(dbn.execute64(kernels.arrived, seen, goal, noise))
        ):
            status = 1
        attempted = status == 0
        active = False
        result = dbn.quad3d_empty_result()
        before = x.copy()
        before_cursor = cursor
        u = np.zeros(4)
        if attempted:
            result = solver.solve(seen, target, controlled, mask)
            if result["accepted"]:
                active = True
                u = result["control"]
                cursor = float(proposed)
                x, clear, bound = map(np.asarray, dbn.execute64(kernels.advance, x, u, truth, mask))
                count += 1
                if clear <= 0:
                    status = 4
                elif bound > c.qp_tolerance:
                    status = 5
            else:
                status = 3
        records.append(
            dict(
                state=before,
                next_state=x.copy(),
                observed=seen,
                control=u,
                active=active,
                attempted=attempted,
                status=status,
                clearance=clear,
                envelope=bound,
                route_target=target,
                route_cursor_before=before_cursor,
                route_progress=cursor,
                route_remaining=remaining,
                route_visible=visible,
                waypoint_index=leg,
                waypoint_handoff=handoff,
                mission_goal=goal,
                **{"native_" + k: v for k, v in result.items()},
                observed_obstacles=obs.copy(),
                world_obstacles=truth.copy(),
                world_time=tick * c.robot.dt,
            )
        )
        if status:
            break
    if status == 0:
        final_truth = original.copy()
        final_truth[:, :2] += steps * c.robot.dt * original[:, 3:5]
        seen, _ = dbn.execute64(
            kernels.sense, x, final_truth, mask, bx, bo, noise, ix[steps], io[steps]
        )
        status = (
            1
            if leg == total - 1 and bool(dbn.execute64(kernels.arrived, seen, goals[leg], noise))
            else 6
        )
    if any(kernels.cache_sizes().values()) or solver.fn._cache_size() != 0:
        raise ValueError("Runtime tracing in native episode")
    trace = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    row = dict(
        id=parent["id"],
        family=parent["family"],
        noise_level=parent["noise_level"],
        method="barriernet",
        status=status,
        steps=count,
        final_state=x.tolist(),
        waypoint_index=leg,
        waypoints_visited=leg + int(status == 1),
        solver_attempts=int(trace["attempted"].sum()),
        solver_seconds=float(trace["native_seconds"].sum()),
        execution_seconds=time.perf_counter() - start,
        implicit_jit_cache_entries=0,
    )
    return (row, trace)


def _conditioned_physics(dynamics, config):
    """Use the same pre-drawn sensor innovations as the learned controllers."""
    from functools import partial

    if dynamics == "unicycle":
        kernels = umpc.PhysicalKernels(64, 64, STEPS, config)
        model = partial(unicycle_sensor_model, config=config, steps=STEPS)
    else:
        kernels = qmpc.FlightPhysicalKernels(64, 64, STEPS, config)
        model = partial(quad2d_sensor_model, steps=STEPS, stationary_obstacles=False)
    condition = jax.jit(model)

    def bind(case):
        inputs = case["prepared_arguments"]
        key = np.asarray(inputs["key"], dtype=np.uint32)
        units = {n: inputs[n] for n in ("unit_x", "unit_o", "unit_i")}

        def prepare(x, obstacles, mask, noise, actual_key):
            if not np.array_equal(key, actual_key):
                raise ValueError("Sensor key differs from the generated case")
            with jax.enable_x64(False):
                return condition(x, obstacles, mask, noise, key, **units)

        kernels.prepare = prepare

    return kernels, bind


def _unicycle(models, spec, method):
    from oa_cbf_jax.config import UnicycleConfig

    config = UnicycleConfig(**dict(spec["robot"], stationary_obstacles=False))
    if method == "optimal_decay_qp":
        return UnicycleDecayQP(config, spec).run
    kernels, bind = _conditioned_physics("unicycle", config)
    if method == "barriernet":
        controller = UnicycleBarrierNet(models / spec["bundle"], 64, config)
        episode = ubn.episode
    else:
        controller = UnicycleMPC(64, method, config)
        episode = _unicycle_mpc_episode

    def run(case):
        bind(case)
        raw, trace, truth = episode(case["parent"], controller, kernels, STEPS, case["noise_scale"])
        trace.update(truth)
        return raw, trace

    return run


def _quad2d(models, spec, method):
    from oa_cbf_jax.quad2d_control import flight_config_from_contract

    config = replace(flight_config_from_contract(spec["config"]), stationary_obstacles=False)
    kernels, bind = _conditioned_physics("quad2d", config)
    if method == "barriernet":
        controller = Quad2DBarrierNet(models / spec["bundle"], config=config)

        def episode(parent):
            return qbn.episode(parent, kernels, controller, STEPS, config)
    elif method == "optimal_decay_qp":

        def episode(parent):
            return _quad2d_od_episode(parent, kernels, STEPS, config)
    else:
        controller = Quad2DMPC(64, method, config)

        def episode(parent):
            return _quad2d_mpc_episode(parent, controller, kernels, STEPS)

    def run(case):
        bind(case)
        return episode(case["parent"])

    return run


def _quad3d(models, spec, method):
    from oa_cbf_jax.quad3d_control import control_config

    config = control_config(spec["config"])
    kernels = tmpc.PhysicalKernels(config)
    if method == "barriernet":
        controller = Quad3DBarrierNet(models / spec["bundle"], config.robot)

        def run(case):
            return _quad3d_bn_episode(
                case["parent"], kernels, controller, STEPS, config, tape=case["prepared_arguments"]
            )
    else:
        controller = Quad3DMPC(64, method, config)

        def run(case):
            return _quad3d_mpc_episode(
                case["parent"], controller, kernels, STEPS, tape=case["prepared_arguments"]
            )

    return run


def _bicycle(models, spec, method):
    from oa_cbf_jax.bicycle_control import control_config
    from oa_cbf_jax import bicycle_native_qp

    config = control_config(spec["config"])
    kernels = bicycle_native_qp.PhysicalKernels(config)
    if method == "barriernet":
        controller = dbn.BicycleController(models / spec["bundle"], config.robot)

        def run(case):
            return dbn.bicycle_episode(case["parent"], controller, kernels, STEPS)
    else:

        def run(case):
            return bicycle_native_qp.episode(
                case["parent"], method, kernels, STEPS, acceptance=spec["acceptance"]
            )

    return run


def build(models, dynamics, method):
    """Create a reusable native controller, with a fresh solver state per case."""
    models = Path(models)
    spec = verify(models)["baselines"][dynamics][method]
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_enable_x64", True)
    return {"unicycle": _unicycle, "quad2d": _quad2d, "quad3d": _quad3d, "bicycle": _bicycle}[
        dynamics
    ](models, spec, method)
