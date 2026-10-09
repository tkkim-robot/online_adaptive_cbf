"""Shared quad3d implementation."""

from dataclasses import dataclass, asdict

import math

import numpy as np

import jax

import jax.numpy as jnp

@dataclass(frozen=True)
class Quad3DConfig:
    dt: float = .05
    mass: float = 3.
    inertia_x: float = .5
    inertia_y: float = .5
    inertia_z: float = .5
    arm: float = .3
    yaw_coefficient: float = .1
    gravity: float = 9.8
    radius: float = .3
    input_min: float = -2.
    input_max: float = 5.
    integration_substeps: int = 8

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Finite quadrotor constants required')
        positive = ('dt', 'mass', 'inertia_x', 'inertia_y', 'inertia_z', 'arm',
                    'yaw_coefficient', 'gravity', 'radius')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError('Positive physical constants required')
        if not self.input_min < 0 < self.input_max or self.radius < self.arm:
            raise ValueError('Hover-capable signed inputs and enclosing radius required')
        if type(self.integration_substeps) is not int or self.integration_substeps < 1:
            raise ValueError('Positive integer substep count required')

def matrices(config=Quad3DConfig()):
    """Host constants; converting NumPy constants avoids implicit JAX FP32 casts."""
    c = config
    a = np.zeros((12, 12), np.float64)
    a[np.arange(3), np.arange(6, 9)] = 1
    a[np.arange(3, 6), np.arange(9, 12)] = 1
    a[6, 3] = c.gravity
    a[7, 4] = -c.gravity
    allocation = np.array([[1, 1, 1, 1], [0, c.arm, 0, -c.arm],
                           [c.arm, 0, -c.arm, 0],
                           [c.yaw_coefficient, -c.yaw_coefficient,
                            c.yaw_coefficient, -c.yaw_coefficient]], np.float64)
    b = np.zeros((12, 4), np.float64)
    b[8:12] = np.diag([1/c.mass, 1/c.inertia_y, 1/c.inertia_x,
                       1/c.inertia_z]) @ allocation
    return a, b

def held_quad3d_state(x, u, duration, config=Quad3DConfig()):
    """Exact quartic trajectory: A**4=0, but A**3 B is nonzero.

    Duration may be a traced scalar. Keeping the local polynomial explicitly
    also exposes the continuous path to an independent collision auditor.
    """
    a, b = (jnp.asarray(v, x.dtype) for v in matrices(config))
    d1 = a @ x + b @ u
    d2 = a @ d1
    d3 = a @ d2
    d4 = a @ d3
    return x + duration*(d1 + duration*(d2/2 + duration*(d3/6 + duration*d4/24)))

def integrate_quad3d(x, u, config=Quad3DConfig()):
    times = jnp.asarray(np.arange(1, config.integration_substeps+1, dtype=np.float64)
                        * config.dt/config.integration_substeps, x.dtype)
    states = jax.vmap(lambda t: held_quad3d_state(x, u, t, config))(times)
    return states[-1], states

def cylinder_hocbf(x, obstacles, mask, gains, config=Quad3DConfig(), clearance=0.):
    """Four-stage continuous HOCBF for constant-velocity vertical cylinders.

    Return A u <= b and ALL initial cascade values psi0..psi3. Gains are
    positive [k1,k2,k3,k4]; admissibility is distinct from QP feasibility.
    No first-order relative-degree assumption, hazard truncation or slack.
    A sampled command still needs a held-input physical/constraint audit.
    """
    p = x[:2] - obstacles[:, :2]
    v = x[6:8] - obstacles[:, 3:5]
    gravity = jnp.asarray(np.asarray(config.gravity, np.float64), x.dtype)
    a = gravity * jnp.stack((x[3], -x[4]))
    j = gravity * jnp.stack((x[9], -x[10]))
    radius = jnp.asarray(np.asarray(config.radius + clearance, np.float64), x.dtype) + obstacles[:, 2]
    h = jnp.sum(p*p, axis=-1) - radius*radius
    d1 = 2*jnp.sum(p*v, axis=-1)
    d2 = 2*jnp.sum(v*v, axis=-1) + 2*jnp.sum(p*a, axis=-1)
    d3 = 6*jnp.sum(v*a, axis=-1) + 2*jnp.sum(p*j, axis=-1)
    d4_drift = 6*jnp.sum(a*a) + 8*jnp.sum(v*j, axis=-1)
    _, b = matrices(config)
    snap = jnp.asarray(config.gravity*np.stack((b[9], -b[10])), x.dtype)
    authority = 2*p @ snap
    k1, k2, k3, k4 = gains
    s1 = k1+k2+k3+k4
    s2 = k1*k2+k1*k3+k1*k4+k2*k3+k2*k4+k3*k4
    s3 = k1*k2*k3+k1*k2*k4+k1*k3*k4+k2*k3*k4
    s4 = k1*k2*k3*k4
    rhs = d4_drift+s1*d3+s2*d2+s3*d1+s4*h
    psi = jnp.stack((h, d1+k1*h, d2+(k1+k2)*d1+k1*k2*h,
                     d3+(k1+k2+k3)*d2+(k1*k2+k1*k3+k2*k3)*d1+k1*k2*k3*h), -1)
    return jnp.where(mask[:, None], -authority, 0.), jnp.where(mask, rhs, 1.), psi
