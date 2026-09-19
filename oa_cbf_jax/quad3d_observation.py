"""Causal full-state sensor contract; latent quantities belong to the simulator.

Position/velocity errors are isotropic balls/disks, attitude/rate errors have
componentwise bounds. Each episode has one persistent bias plus independent
15% innovations. Public bounds, not realized errors, can enter the controller.
"""
import numpy as np
import jax.numpy as jnp

SCHEMA = 'quad3d_bounded_bias_innovation_v92'
FIELDS = ('position_ball3', 'attitude_component', 'velocity_ball3',
          'rate_component', 'cylinder_position_disk', 'cylinder_velocity_disk',
          'cylinder_radius')
BASE_NOISE = np.array([.015, .004, .015, .008, .02, .02, .008], np.float64)
INNOVATION_FRACTION = .15


def _unit_draws(rng, count, capacity):
    # A fixed number of draws per tick makes prefixes independent of horizon.
    raw = rng.random((count, 12+5*capacity))
    x = 2*raw[:, :12]-1
    for start in (0, 6):
        u = raw[:, start:start+3]
        z = 2*u[:, 0]-1
        angle = 2*np.pi*u[:, 1]
        radius = np.cbrt(u[:, 2])
        xy = np.sqrt(np.maximum(1-z*z, 0))
        x[:, start:start+3] = radius[:, None]*np.stack((xy*np.cos(angle), xy*np.sin(angle), z), -1)
    raw_o = raw[:, 12:].reshape(count, capacity, 5)
    o = 2*raw_o-1
    for start in (0, 3):
        angle = 2*np.pi*raw_o[..., start+1]
        radius = np.sqrt(raw_o[..., start])
        o[..., start:start+2] = radius[..., None]*np.stack((np.cos(angle), np.sin(angle)), -1)
    return x, o


def unit_tape(seed, steps, capacity=64):
    """Host-owned reproducible tape, never exposed to control/model functions."""
    if steps < 0 or capacity < 1:
        raise ValueError('Invalid sensor tape dimensions')
    bias = np.random.default_rng(np.random.SeedSequence([int(seed), 9201]))
    innovation = np.random.default_rng(np.random.SeedSequence([int(seed), 9202]))
    bx, bo = _unit_draws(bias, 1, capacity)
    ix, io = _unit_draws(innovation, steps+1, capacity)
    return bx[0], bo[0], ix, io


def scales(noise):
    return noise[jnp.asarray([0]*3+[1]*3+[2]*3+[3]*3)], noise[jnp.asarray([4, 4, 6, 5, 5])]


def observe(physical, obstacles, mask, bias_x, bias_o, noise, innovation_x, innovation_o):
    """Only current observations leave this simulator boundary; explicit FP64."""
    xs, os = scales(noise)
    fraction = jnp.asarray(np.asarray(INNOVATION_FRACTION, np.float64), physical.dtype)
    x = physical + xs*(bias_x+fraction*innovation_x)
    o = obstacles + os*(bias_o+fraction*innovation_o)
    return x, jnp.where(mask[:, None], o, jnp.zeros_like(o))


def numpy_observe(physical, obstacles, mask, bias_x, bias_o, noise, innovation_x, innovation_o):
    xs = np.repeat(np.asarray(noise)[:4], 3)
    os = np.asarray(noise)[[4, 4, 6, 5, 5]]
    return (np.asarray(physical)+xs*(bias_x+.15*innovation_x),
            np.where(np.asarray(mask)[:, None], obstacles+os*(bias_o+.15*innovation_o), 0.))


def geometry_inflation(noise):
    # Euclidean errors: triangle inequality, no extra sqrt(d) for ball radii.
    return jnp.asarray(np.asarray(1.15, np.float64), noise.dtype)*(noise[0]+noise[4]+noise[6])


def controller_obstacles(obstacles, mask, noise):
    inflated = obstacles.at[:, 2].add(geometry_inflation(noise))
    return jnp.where(mask[:, None], inflated, jnp.zeros_like(inflated))


def guidance_obstacles(obstacles, mask, noise):
    """Static-compatible means zero velocity lies in measured error support.

    This is a guidance hypothesis only. Original measured velocities remain in
    every QP row; all cylinders remain present. No true static/moving flag.
    """
    o = controller_obstacles(obstacles, mask, noise)
    bound = jnp.asarray(np.asarray(1.15, np.float64), noise.dtype)*noise[5]+1e-10
    compatible = jnp.linalg.norm(o[:, 3:5], axis=-1) <= bound
    return o.at[:, 3:5].set(jnp.where(compatible[:, None], 0., o[:, 3:5]))


def numpy_obstacles(obstacles, mask, noise, guidance=False):
    o = np.array(obstacles, np.float64, copy=True)
    o[:, 2] += 1.15*sum(np.asarray(noise)[[0, 4, 6]])
    if guidance:
        o[np.linalg.norm(o[:, 3:5], axis=-1) <= 1.15*noise[5]+1e-10, 3:5] = 0.
    return np.where(np.asarray(mask)[:, None], o, 0.)


def observed_arrived(x, goal, noise, config):
    """Sufficient physical arrival condition under the declared sensor bounds."""
    bound = jnp.asarray(np.asarray(1.15, np.float64), noise.dtype)*noise
    return ((jnp.linalg.norm(x[:3]-goal)+bound[0] <= config.goal_tolerance)
            & (jnp.linalg.norm(x[6:9])+bound[2] <= config.terminal_speed)
            & (jnp.max(jnp.abs(x[3:6]))+bound[1] <= config.terminal_attitude)
            & (jnp.max(jnp.abs(x[9:12]))+bound[3] <= config.terminal_rate))


def numpy_arrived(x, goal, noise, config):
    b = 1.15*np.asarray(noise)
    return bool(np.linalg.norm(x[:3]-goal)+b[0] <= config.goal_tolerance
                and np.linalg.norm(x[6:9])+b[2] <= config.terminal_speed
                and np.max(abs(x[3:6]))+b[1] <= config.terminal_attitude
                and np.max(abs(x[9:12]))+b[3] <= config.terminal_rate)
