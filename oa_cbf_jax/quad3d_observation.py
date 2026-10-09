"""Quad3d observation functions and shared contracts."""

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


from functools import lru_cache

from .quad3d import matrices

IDENTIFIABLE=np.array([3,4,6,7,8,9,10,11])

def coefficients(robot):
    a,b=matrices(robot);dt=robot.dt;identity=np.eye(12)
    aa=a@a;aaa=aa@a
    f=identity+dt*a+dt**2/2*aa+dt**3/6*aaa
    g=(dt*identity+dt**2/2*a+dt**3/6*aa+dt**4/24*aaa)@b
    design=(identity-f)[:,IDENTIFIABLE]
    assert np.linalg.matrix_rank(design)==len(IDENTIFIABLE)
    return f,g,np.linalg.pinv(design)

def update_bias(estimate,previous,current,applied,noise,config):
    f,g,inverse=(jnp.asarray(v,current.dtype) for v in coefficients(config.robot))
    residual=current-f@previous-g@applied
    sample=jnp.zeros(12,current.dtype).at[IDENTIFIABLE].set(inverse@residual)
    alpha=jnp.asarray(np.asarray(-np.expm1(-config.robot.dt/config.observer_time_constant),np.float64),current.dtype)
    updated=estimate+alpha*(sample-estimate)
    # Project the estimated persistent bias onto its public sensor support.
    # This is estimator regularization, never physical-state clipping.
    updated=updated.at[3:5].set(jnp.clip(updated[3:5],-noise[1],noise[1]))
    updated=updated.at[6:9].set(updated[6:9]*jnp.minimum(1.,noise[2]/jnp.maximum(jnp.linalg.norm(updated[6:9]),1e-24)))
    updated=updated.at[9:12].set(jnp.clip(updated[9:12],-noise[3],noise[3]))
    return updated.at[jnp.asarray([0,1,2,5])].set(0.)

@lru_cache(maxsize=8)
def numpy_coefficients(robot):
    from scipy.linalg import expm
    a,b=matrices(robot);aug=np.zeros((16,16));aug[:12,:12]=a;aug[:12,12:]=b
    fg=expm(aug*robot.dt)
    return fg[:12,:12],fg[:12,12:]

def numpy_update_bias(estimate,previous,current,applied,noise,config):
    """Independent reconstruction using scipy matrix exponential and least squares."""
    f,g=numpy_coefficients(config.robot)
    residual=np.asarray(current)-f@previous-g@applied
    value=np.linalg.lstsq((np.eye(12)-f)[:,IDENTIFIABLE],residual,rcond=None)[0]
    sample=np.zeros(12);sample[IDENTIFIABLE]=value
    updated=np.asarray(estimate)+(-np.expm1(-config.robot.dt/config.observer_time_constant))*(sample-estimate)
    updated[3:5]=np.clip(updated[3:5],-noise[1],noise[1])
    updated[6:9]*=min(1.,noise[2]/max(np.linalg.norm(updated[6:9]),1e-24))
    updated[9:12]=np.clip(updated[9:12],-noise[3],noise[3]);updated[[0,1,2,5]]=0.
    return updated
