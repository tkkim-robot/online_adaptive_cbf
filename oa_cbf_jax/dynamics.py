"""Pure dynamics. Unicycle state [x,y,heading,speed], input [accel,yaw_rate]."""

from functools import partial
import jax
import jax.numpy as jnp


def unicycle_flow(x, u):
    return jnp.stack((x[3] * jnp.cos(x[2]), x[3] * jnp.sin(x[2]), u[1], u[0]))


def rk4_step(x, u, dt):
    k1 = unicycle_flow(x, u)
    k2 = unicycle_flow(x + dt * k1 / 2, u)
    k3 = unicycle_flow(x + dt * k2 / 2, u)
    k4 = unicycle_flow(x + dt * k3, u)
    y = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # Heading and speed have constant derivatives for the held input. Advancing
    # them analytically avoids accumulating round-off through redundant stages.
    y=y.at[2].set(x[2]+dt*u[1]).at[3].set(x[3]+dt*u[0])
    heading=jnp.where(jnp.abs(y[2])<=jnp.pi,y[2],jnp.arctan2(jnp.sin(y[2]),jnp.cos(y[2])))
    return y.at[2].set(heading)


@partial(jax.jit, static_argnames=("substeps",))
def integrate_unicycle(x, u, dt, substeps=4):
    """RK4 positions, exact linear heading/speed flow; no physical projection."""
    def step(carry, index):
        y = rk4_step(carry, u, dt / substeps)
        time=(index+1)*(dt/substeps)
        y=y.at[2].set(x[2]+time*u[1]).at[3].set(x[3]+time*u[0])
        return y, y
    _,states=jax.lax.scan(step,x,jnp.arange(substeps))
    heading=states[:,2]
    heading=jnp.where(jnp.abs(heading)<=jnp.pi,heading,jnp.arctan2(jnp.sin(heading),jnp.cos(heading)))
    states=states.at[:,2].set(heading)
    return states[-1],states


def obstacle_positions(obstacles, time):
    """Obstacles have columns [x,y,radius,vx,vy], with constant-velocity motion."""
    return obstacles[:, :2] + time * obstacles[:, 3:5]


def signed_clearance(position, obstacles, mask, radius, time=0.0):
    distance = jnp.linalg.norm(position[None, :] - obstacle_positions(obstacles, time), axis=-1)
    return jnp.where(mask, distance - radius - obstacles[:, 2], jnp.inf)


def swept_disk_clearance(start, end, obstacles, mask, radius, t0, t1):
    """Exact relative-segment distance for disk translation over a substep.

    RK4 curves are piecewise-linearly audited; finer-integration audits check
    the approximation separately. Includes moving obstacles and endpoints.
    """
    a = start[:2] - obstacle_positions(obstacles, t0)
    b = end[:2] - obstacle_positions(obstacles, t1)
    delta = b - a
    tau = jnp.clip(-jnp.sum(a * delta, axis=-1) / jnp.maximum(jnp.sum(delta**2, axis=-1), 1e-20), 0., 1.)
    distance = jnp.linalg.norm(a + tau[:, None] * delta, axis=-1)
    return jnp.min(jnp.where(mask, distance - radius - obstacles[:, 2], jnp.inf))
