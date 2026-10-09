"""Initial full-state OA quadrotor QP. No trained policy or baseline substitute.

Four-stage obstacle class-K gains are explicit inputs. Fixed envelope barriers
protect the declared linearized-model domain. QP residual acceptance, cascade
admissibility and true physical safety are distinct quantities.
"""

from dataclasses import dataclass, field, asdict

import math

import numpy as np

import jax

import jax.numpy as jnp

from .quad3d import Quad3DConfig, matrices, cylinder_hocbf

@dataclass(frozen=True)
class Quad3DControlConfig:
    robot: Quad3DConfig = field(default_factory=Quad3DConfig)
    tilt_limit: float = .45
    yaw_limit: float = 1.5
    rate_limit: float = 2.
    velocity_limit: float = 2.5  # componentwise, not a Euclidean speed bound
    altitude_min: float = -2.
    altitude_max: float = 5.
    cruise_speed: float = 1.
    acceleration_limit: float = 1.8
    attitude_kp: float = 16.
    attitude_kd: float = 8.
    envelope_gain: float = 3.
    clearance_buffer: float = .05
    qp_tolerance: float = 1e-5
    goal_tolerance: float = .25
    terminal_speed: float = .2
    terminal_attitude: float = .1
    terminal_rate: float = .2
    nominal_design: str = 'four_pole_v86'
    horizontal_frequency: float = 1.6
    hold_guard: str = 'none'
    nominal_bias_observer: str = 'none'
    observer_time_constant: float = 2.
    qp_refinement: str = 'none'

    def __post_init__(self):
        values = {k:v for k,v in asdict(self).items() if k not in ('robot', 'nominal_design', 'hold_guard', 'nominal_bias_observer', 'qp_refinement')}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Finite control configuration required')
        if any(v <= 0 for k,v in values.items() if k not in ('altitude_min', 'altitude_max', 'clearance_buffer')):
            raise ValueError('Positive controller constants required')
        if not self.altitude_min < self.altitude_max or self.clearance_buffer < 0:
            raise ValueError('Invalid altitude/clearance bounds')
        if self.tilt_limit >= math.pi/2 or self.cruise_speed > self.velocity_limit:
            raise ValueError('Invalid linearized-model envelope')
        if self.nominal_design not in ('cascade_v85','four_pole_v86'):
            raise ValueError('Unknown nominal feedback contract')
        if self.hold_guard not in ('none','bernstein_v87'):
            raise ValueError('Unknown held-input guard contract')
        if self.nominal_bias_observer not in ('none','innovation_ema_v97'):
            raise ValueError('Unknown nominal observation-bias contract')
        if self.qp_refinement not in ('none','active_faces_v99'):
            raise ValueError('Unknown QP refinement contract')

def control_config(value):
    value=dict(value);value['robot']=Quad3DConfig(**value['robot'])
    return Quad3DControlConfig(**value)

def envelope_rows(x, config=Quad3DControlConfig()):
    c = config
    _, bb = matrices(c.robot)
    b = jnp.asarray(bb, x.dtype)
    signs = jnp.asarray(np.array([1., -1.]), x.dtype)
    limits = jnp.asarray(np.array([c.tilt_limit, c.tilt_limit, c.yaw_limit]), x.dtype)
    k = c.envelope_gain
    angle_h = limits[:, None] - x[3:6, None]*signs
    angle_hd = -x[9:12, None]*signs
    angle_a = (b[9:12, None, :]*signs[None, :, None]).reshape(-1, 4)
    angle_rhs = (k*k*angle_h+2*k*angle_hd).reshape(-1)
    rate_h = c.rate_limit - x[9:12, None]*signs
    rate_rhs = (k*rate_h).reshape(-1)
    acc = c.robot.gravity*jnp.stack((x[3], -x[4]))
    jerk = c.robot.gravity*jnp.stack((x[9], -x[10]))
    snap = c.robot.gravity*jnp.stack((b[9], -b[10]))
    velocity_h = c.velocity_limit - x[6:9, None]*signs
    xy_hd = -acc[:, None]*signs
    xy_hdd = -jerk[:, None]*signs
    velocity_a = jnp.concatenate(((snap[:, None, :]*signs[None, :, None]).reshape(-1, 4), signs[:, None]*b[8]))
    velocity_rhs = jnp.concatenate(((k**3*velocity_h[:2]+3*k*k*xy_hd+3*k*xy_hdd).reshape(-1), k*velocity_h[2]))
    z_h = jnp.stack((c.altitude_max-x[2], x[2]-c.altitude_min))
    z_hd = -x[8]*signs
    z_a = signs[:, None]*b[8]
    domain = jnp.concatenate((angle_h.ravel(), (angle_hd+k*angle_h).ravel(), rate_h.ravel(),
                              velocity_h.ravel(), (xy_hd+k*velocity_h[:2]).ravel(),
                              (xy_hdd+2*k*xy_hd+k*k*velocity_h[:2]).ravel(), z_h, z_hd+k*z_h))
    return jnp.concatenate((angle_a, angle_a, velocity_a, z_a)), jnp.concatenate((angle_rhs, rate_rhs, velocity_rhs, k*k*z_h+2*k*z_hd)), domain

def nominal_quad3d(x, goal, config=Quad3DControlConfig()):
    """Three-dimensional goal feedback; clipping desired acceleration is nominal.

    Actual motor commands are determined by the QP; no physical state clipping.
    The four wrench channels maintain altitude and all attitudes continuously.
    """
    c = config; r = c.robot
    delta = goal-x[:3]
    desired_v = delta/jnp.maximum(jnp.linalg.norm(delta), 1e-12)*jnp.minimum(c.cruise_speed, 1.5*jnp.linalg.norm(delta))
    acceleration = jnp.clip(2*(desired_v-x[6:9]), -c.acceleration_limit, c.acceleration_limit)
    desired_angle = jnp.stack((acceleration[0]/r.gravity, -acceleration[1]/r.gravity, jnp.zeros((), x.dtype)))
    angular_acc = c.attitude_kp*(desired_angle-x[3:6])-c.attitude_kd*x[9:12]
    if c.nominal_design == 'four_pole_v86':
        # Exact horizontal relative-degree-four feedback. Near the destination
        # each scalar closed-loop characteristic is (s+w)^4. Far away cap only
        # the nominal position error: w*lookahead/4 = declared cruise speed.
        # This changes neither the physical state nor the hard CBF/QP rows.
        w=jnp.asarray(np.asarray(c.horizontal_frequency,np.float64),x.dtype)
        gravity=jnp.asarray(np.asarray(r.gravity,np.float64),x.dtype)
        error=delta[:2]*jnp.minimum(1.,(4*c.cruise_speed/w)/jnp.maximum(jnp.linalg.norm(delta[:2]),1e-12))
        acc=gravity*jnp.stack((x[3],-x[4]));jerk=gravity*jnp.stack((x[9],-x[10]))
        snap=w**4*error-4*w**3*x[6:8]-6*w*w*acc-4*w*jerk
        angular_acc=angular_acc.at[:2].set(jnp.stack((snap[0]/gravity,-snap[1]/gravity)))
    wrench = jnp.concatenate((jnp.reshape(r.mass*acceleration[2], (1,)),
                              jnp.asarray(np.array([r.inertia_y, r.inertia_x, r.inertia_z]), x.dtype)*angular_acc))
    allocation = np.array([[1, 1, 1, 1], [0, r.arm, 0, -r.arm], [r.arm, 0, -r.arm, 0],
                           [r.yaw_coefficient, -r.yaw_coefficient, r.yaw_coefficient, -r.yaw_coefficient]])
    return jnp.asarray(np.linalg.inv(allocation), x.dtype) @ wrench

def quad3d_rows(x, obstacles, mask, gains, config=Quad3DControlConfig()):
    a, b, psi = cylinder_hocbf(x, obstacles, mask, gains, config.robot, config.clearance_buffer)
    ea, eb, domain = envelope_rows(x, config)
    ia = jnp.asarray(np.concatenate((np.eye(4), -np.eye(4))), x.dtype)
    ib = jnp.asarray(np.array([config.robot.input_max]*4+[-config.robot.input_min]*4), x.dtype)
    return jnp.concatenate((a, ea, ia)), jnp.concatenate((b, eb, ib)), jnp.min(jnp.where(mask[:, None], psi, jnp.inf)), jnp.min(domain)

def solve_qp4(reference, a, b, tolerance=1e-5, box_bounds=None, refinement='none'):
    """Hard JAX QP with original-unit residual checks.

    If supplied, box_bounds must also be present among a,b. The Euclidean box
    projection is the exact optimum whenever it satisfies every additional row.
    This is a control-space QP solution, never clipping a propagated state.
    """
    from .quad3d_qp import solve_actual_iterate
    scale = jnp.maximum(jnp.maximum(jnp.linalg.norm(a, axis=-1), jnp.abs(b)), 1.)
    aa, bb = a/scale[:, None], b/scale
    u, _, dual, converged, iterations = solve_actual_iterate(reference,aa,bb,
        tolerance=2e-7,max_iterations=60)
    # A feasible unconstrained minimizer is the exact constrained minimizer.
    # Avoid accepting a less accurate interior-point approximation in that case.
    direct=reference if box_bounds is None else jnp.clip(reference,*box_bounds)
    exact=jnp.all(jnp.isfinite(direct)) & (jnp.max(a @ direct-b)<=0.)
    if refinement=='active_faces_v99':
        from .quad3d_qp import polish
        polished,certificate=jax.lax.cond(exact|~converged,
            lambda _: (u,jnp.bool_(False)),lambda _:polish(reference,aa,bb,dual),operand=None)
        # The independent full original-unit residual test below still applies.
        u=jnp.where(certificate,polished,u)
    elif refinement!='none':raise ValueError('Unknown QP refinement')
    u=jnp.where(exact,direct,u)
    violation = jnp.max(a @ u-b)
    feasible = (exact | (converged == 1)) & jnp.all(jnp.isfinite(u)) & (violation <= tolerance)
    iterations=jnp.where(exact,jnp.zeros((),iterations.dtype),iterations)
    # Preserve candidate on rejection for diagnosis; callers must gate application.
    return u, feasible, violation, iterations

def quad3d_control(x, goal, obstacles, mask, gains, config=Quad3DControlConfig()):
    reference,a,b,psi,domain=quad3d_problem(x,goal,obstacles,mask,gains,config)
    u, feasible, violation, iterations = solve_qp4(reference, a, b, config.qp_tolerance,
        (config.robot.input_min,config.robot.input_max),refinement=config.qp_refinement)
    return u, feasible, psi, domain, violation, iterations

def quad3d_problem(x, goal, obstacles, mask, gains, config=Quad3DControlConfig(), nominal_state=None):
    reference = nominal_quad3d(x if nominal_state is None else nominal_state, goal, config)
    a, b, psi, domain = quad3d_rows(x, obstacles, mask, gains, config)
    if config.hold_guard=='bernstein_v87':
        from .quad3d_held import hold_rows
        ha,hb=hold_rows(x,reference,obstacles,mask,gains,config)
        a=jnp.concatenate((a,ha));b=jnp.concatenate((b,hb))
    return reference,a,b,psi,domain

def arrived(x, goal, config=Quad3DControlConfig()):
    return ((jnp.linalg.norm(x[:3]-goal) <= config.goal_tolerance)
            & (jnp.linalg.norm(x[6:9]) <= config.terminal_speed)
            & (jnp.max(jnp.abs(x[3:6])) <= config.terminal_attitude)
            & (jnp.max(jnp.abs(x[9:12])) <= config.terminal_rate))
