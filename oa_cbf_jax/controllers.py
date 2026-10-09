"""Controllers functions and shared contracts."""

from functools import partial

from typing import NamedTuple

import jax

import jax.numpy as jnp

import numpy as np

from .config import UnicycleConfig

jax.config.update('jax_explicit_x64_dtypes','allow')

class QPResult(NamedTuple):
    control: jax.Array
    feasible: jax.Array
    max_violation: jax.Array
    objective: jax.Array

def nominal_unicycle(x, goal, config: UnicycleConfig):
    delta = goal - x[:2]
    desired_heading = jnp.arctan2(delta[1], delta[0])
    error = jnp.arctan2(jnp.sin(desired_heading - x[2]), jnp.cos(desired_heading - x[2]))
    # Shared nominal controller slows before reaching the goal. Its stopping
    # target and speed feedback avoid overshooting into a low-speed orbit.
    remaining = jnp.maximum(jnp.linalg.norm(delta)-.1,0.)
    desired_speed = jnp.minimum(config.v_max,jnp.minimum(1.2*remaining,jnp.sqrt(2*config.a_max*remaining))) * jnp.maximum(jnp.cos(error), 0.)
    return jnp.stack((2.0*(desired_speed - x[3]), 2.0 * error))

def unicycle_cbf_qp(x, goal, obstacles, mask, alpha, config: UnicycleConfig, clearance_uncertainty=0.):
    """Assemble A u <= b for moving disks and continuous second-order CBF.

    h=||p-o||²-R², hddot+(alpha1+alpha2)hdot+alpha1*alpha2*h>=margin.
    Both gains are constant during the solve. Joint feasibility does not imply
    higher-order initial admissibility: h and psi1 are returned separately.
    """
    # Near a barrier, subtracting squared distances in FP32 loses significant
    # digits; gain products then amplify the error before the QP can check it.
    # Assemble these small geometric rows in FP64, then solve the large candidate
    # enumeration in the original dtype. Explicit FP64 permission does not
    # change model/rollout storage or globally enable 64-bit JAX defaults.
    dtype=x.dtype
    state=x.astype(jnp.float64);seen=obstacles.astype(jnp.float64);gains=alpha.astype(jnp.float64)
    direction = jnp.stack((jnp.cos(state[2]), jnp.sin(state[2])))
    normal = jnp.stack((-direction[1], direction[0]))
    delta = state[:2] - seen[:, :2]
    velocity = state[3] * direction - seen[:, 3:5]
    radii = config.radius + config.clearance_buffer + seen[:, 2] + jnp.asarray(clearance_uncertainty,jnp.float64)
    h64 = jnp.sum(delta**2, axis=-1) - radii**2
    hdot = 2 * jnp.sum(delta * velocity, axis=-1)
    authority64 = 2 * jnp.stack((delta @ direction, state[3] * (delta @ normal)), axis=-1)
    rhs64 = 2 * jnp.sum(velocity**2, axis=-1) + jnp.sum(gains) * hdot + jnp.prod(gains) * h64 - config.cbf_margin
    h=h64.astype(dtype);psi=(hdot+gains[0]*h64).astype(dtype)
    authority=authority64.astype(dtype);rhs=rhs64.astype(dtype)
    # Inactive rows are strictly interior 0*u<=1, safe for an IPM too.
    a_obs = jnp.where(mask[:, None], -authority, 0.)
    b_obs = jnp.where(mask, rhs, 1.)
    lower_a = jnp.maximum(-config.a_max, -x[3] / config.dt)
    upper_a = jnp.minimum(config.a_max, (config.v_max - x[3]) / config.dt)
    a_bounds = jnp.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=x.dtype)
    b_bounds = jnp.stack((upper_a, -lower_a, jnp.asarray(config.w_max, x.dtype), jnp.asarray(config.w_max, x.dtype)))
    return (nominal_unicycle(x, goal, config), jnp.concatenate((a_obs, a_bounds)),
            jnp.concatenate((b_obs, b_bounds)), h, psi)

@jax.jit
def solve_qp2(reference, a, b, weights, tolerance=1e-5):
    """Exact active-set enumeration for strictly convex diagonal two-input QP.

    Enumerate interior, all face projections and all pairwise intersections.
    Reject infeasible/nonfinite candidates. Return NaN control on infeasibility;
    callers must explicitly choose/log any unverified emergency input.
    """
    invw = 1 / weights
    denominator = jnp.sum(a * a * invw, axis=-1)
    face = reference - ((jnp.matmul(a, reference, precision="highest") - b) / jnp.maximum(denominator, 1e-20))[:, None] * (a * invw)
    i, j = jnp.triu_indices(a.shape[0], k=1)
    det = a[i, 0] * a[j, 1] - a[i, 1] * a[j, 0]
    denom = jnp.where(jnp.abs(det) > 1e-10, det, 1.)
    cross = jnp.stack(((b[i] * a[j, 1] - a[i, 1] * b[j]) / denom,
                       (a[i, 0] * b[j] - b[i] * a[j, 0]) / denom), axis=-1)
    candidates = jnp.concatenate((reference[None, :], face, cross))
    valid_geometry = jnp.concatenate((jnp.ones(1, dtype=bool), denominator > 1e-20, jnp.abs(det) > 1e-10))
    # Default GPU dot precision may use TF32. Its rounding can reject the true
    # optimum at a 1e-5 safety tolerance and make actions depend on padding.
    violation = jnp.max(jnp.matmul(candidates, a.T, precision="highest") - b, axis=-1)
    valid = valid_geometry & (violation <= tolerance) & jnp.all(jnp.isfinite(candidates), axis=-1)
    costs = .5 * jnp.sum(weights * (candidates - reference)**2, axis=-1)
    valid_costs=jnp.where(valid,costs,jnp.inf)
    # JAX0.8.2 argmin constructs an FP32 identity for explicit FP64 arrays when
    # global x64 is disabled. Min plus first-true argmax preserves exact tie
    # semantics and works in nested FP32 physical / FP64 small-QP transforms.
    idx=(jnp.argmax(valid_costs==jnp.min(valid_costs)) if costs.dtype==jnp.float64 else jnp.argmin(valid_costs))
    feasible = jnp.any(valid)
    return QPResult(jnp.where(feasible, candidates[idx], jnp.full(2, jnp.nan)), feasible,
                    jnp.where(feasible, violation[idx], jnp.inf), jnp.where(feasible, costs[idx], jnp.inf))

@partial(jax.jit, static_argnames=("config",))
def control_unicycle(x, goal, obstacles, mask, alpha, config=UnicycleConfig()):
    ref, a, b, h, psi1 = unicycle_cbf_qp(x, goal, obstacles, mask, alpha, config)
    result = solve_qp2(ref, a, b, jnp.ones(2, dtype=x.dtype), config.qp_tolerance)
    return result, jnp.min(jnp.where(mask, h, jnp.inf)), jnp.min(jnp.where(mask, psi1, jnp.inf))

class NativeOSQP:
    """Persistent dense-pattern native QP reference, with explicit diagnostics."""
    def __init__(self, n_rows, weights=(1., 1.), tolerance=1e-8, solver_defaults=False):
        import osqp
        from scipy import sparse
        self.solver = osqp.OSQP()
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.ndim != 1 or not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0):
            raise ValueError('QP weights must be a finite positive diagonal')
        self.n_variables = len(self.weights)
        self.n_rows = n_rows
        settings=dict(verbose=False) if solver_defaults else dict(eps_abs=tolerance,eps_rel=tolerance,max_iter=10000,
                                                                  verbose=False,polishing=True,warm_starting=True)
        self.solver.setup(P=sparse.diags(self.weights, format="csc"), q=np.zeros(self.n_variables),
                          A=sparse.csc_matrix(np.ones((n_rows, self.n_variables))), l=np.full(n_rows, -np.inf), u=np.ones(n_rows),
                          **settings)

    def solve(self, reference, a, b):
        a, b, ref = np.asarray(a, float), np.asarray(b, float), np.asarray(reference, float)
        if a.shape != (self.n_rows,self.n_variables) or b.shape != (self.n_rows,) or ref.shape != (self.n_variables,):
            raise ValueError('QP shape changed after setup')
        if not all(np.all(np.isfinite(value)) for value in (a,b,ref)):
            return dict(control=None,feasible=False,violation=float('inf'),status='nonfinite_input',iterations=0)
        reference_violation=float(np.max(a@ref-b))
        if reference_violation<=0:
            # The unconstrained minimizer is already feasible. This exact
            # solution also avoids OSQP's unconditional empty-polish message.
            return dict(control=ref.copy(),feasible=True,violation=reference_violation,status='solved_reference',iterations=0)
        self.solver.update(q=-self.weights * ref, Ax=a.ravel(order="F"), u=b)
        out = self.solver.solve(raise_error=False)
        feasible = out.info.status_val in (1, 2) and out.x is not None and np.isfinite(out.x).all()
        violation = float(np.max(a @ out.x - b)) if feasible else float("inf")
        return dict(control=out.x if feasible else None, feasible=feasible and violation <= 1e-5,
                    violation=violation, status=out.info.status, iterations=out.info.iter)


class NativeClarabel:
    def __init__(self,n_rows,weights=(1.,1.),tolerance=1e-9):
        import clarabel
        from scipy import sparse
        self.api=clarabel;self.sparse=sparse;self.weights=np.asarray(weights,float)
        if self.weights.ndim!=1 or not np.isfinite(self.weights).all() or np.any(self.weights<=0):
            raise ValueError('QP weights must be finite and positive')
        self.scale=np.sqrt(self.weights);self.n_variables=len(self.weights);self.n_rows=n_rows
        self.hessian=sparse.eye(self.n_variables,format='csc');self.linear=np.zeros(self.n_variables)
        self.cones=[clarabel.NonnegativeConeT(n_rows)]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_gap_abs=tolerance;settings.tol_gap_rel=tolerance;settings.tol_feas=tolerance
        self.settings=settings

    def solve(self,reference,a,b):
        reference,a,b=map(lambda value:np.asarray(value,float),(reference,a,b))
        if reference.shape!=(self.n_variables,) or a.shape!=(self.n_rows,self.n_variables) or b.shape!=(self.n_rows,):
            raise ValueError('QP shape changed after setup')
        if not all(np.isfinite(value).all() for value in (reference,a,b)):
            return dict(control=None,feasible=False,violation=float('inf'),status='nonfinite_input',iterations=0)
        violation=float(np.max(a@reference-b))
        if violation<=0:
            return dict(control=reference.copy(),feasible=True,violation=violation,status='solved_reference',iterations=0)
        solver=self.api.DefaultSolver(self.hessian,self.linear,self.sparse.csc_matrix(a/self.scale),b-a@reference,
                                      self.cones,self.settings)
        result=solver.solve();x=reference+np.asarray(result.x)/self.scale
        finite=np.isfinite(x).all();violation=float(np.max(a@x-b)) if finite else float('inf')
        solved=result.status==self.api.SolverStatus.Solved
        return dict(control=x if finite else None,feasible=bool(solved and finite and violation<=1e-5),
                    violation=violation,status=str(result.status),iterations=int(result.iterations))


class NativeProxQP:
    def __init__(self,n_rows,weights=(1.,1.),tolerance=1e-9):
        import proxsuite
        self.api=proxsuite.proxqp;self.weights=np.asarray(weights,float)
        if self.weights.ndim!=1 or not np.isfinite(self.weights).all() or np.any(self.weights<=0):
            raise ValueError('QP weights must be finite and positive')
        self.scale=np.sqrt(self.weights);self.n_variables=len(self.weights);self.n_rows=n_rows
        self.solver=self.api.dense.QP(self.n_variables,0,n_rows)
        self.solver.settings.eps_abs=tolerance;self.solver.settings.eps_rel=0.
        self.solver.settings.eps_primal_inf=1e-12;self.solver.settings.eps_dual_inf=1e-12
        self.solver.settings.max_iter=10000
        self.solver.settings.verbose=False
        self.solver.init(np.eye(self.n_variables),np.zeros(self.n_variables),None,None,
                         np.zeros((n_rows,self.n_variables)),np.full(n_rows,-np.inf),np.ones(n_rows))
        self.previous_solved=False

    def solve(self,reference,a,b):
        reference,a,b=map(lambda v:np.asarray(v,float),(reference,a,b))
        if reference.shape!=(self.n_variables,) or a.shape!=(self.n_rows,self.n_variables) or b.shape!=(self.n_rows,):
            raise ValueError('QP shape changed after setup')
        if not all(np.isfinite(v).all() for v in (reference,a,b)):
            return dict(control=None,feasible=False,violation=float('inf'),status='nonfinite_input',iterations=0)
        violation=float(np.max(a@reference-b))
        if violation<=0:
            return dict(control=reference.copy(),feasible=True,violation=violation,status='solved_reference',iterations=0)
        # ProxQP 0.7.3 can retain an infeasibility status/internal iterate across
        # updates. A clean initialization is cheap for these 3/4-variable QPs
        # and is necessary even when a previously rejected scene becomes empty.
        self.solver.cleanup()
        self.solver.settings.initial_guess=self.api.InitialGuess.NO_INITIAL_GUESS
        self.solver.init(np.eye(self.n_variables),-self.scale*reference,None,None,
                         a/self.scale,np.full(self.n_rows,-np.inf),b)
        self.solver.solve();result=self.solver.results
        x=np.asarray(result.x).copy()/self.scale
        solved=result.info.status==self.api.QPSolverOutput.PROXQP_SOLVED
        finite=np.isfinite(x).all();violation=float(np.max(a@x-b)) if finite else float('inf')
        self.previous_solved=bool(solved and finite and violation<=1e-5)
        return dict(control=x if finite else None,feasible=self.previous_solved,violation=violation,
                    status=str(result.info.status),iterations=int(result.info.iter))
