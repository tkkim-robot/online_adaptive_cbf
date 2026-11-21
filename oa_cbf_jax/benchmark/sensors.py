"""Create the common host-side inputs consumed by every method on a scene."""

import numpy as np


def planar_units(sensor_seed, state_size):
    """The same initial offsets and 1,600 innovations for every controller."""
    import jax
    import jax.numpy as jnp

    with jax.default_device(jax.devices("cpu")[0]):
        keys = jax.random.split(jax.random.PRNGKey(sensor_seed), 3)
        shapes = ((state_size,), (64, 5), (1600, state_size + 64 * 5))
        x, o, innovations = (
            jax.random.uniform(key, shape, dtype=jnp.float32, minval=-1.0, maxval=1.0)
            for key, shape in zip(keys, shapes)
        )
        innovations = innovations.at[0].set(0.0)
        return dict(
            unit_x=np.asarray(x),
            unit_o=np.asarray(o),
            unit_i=np.asarray(innovations),
            seed=sensor_seed,
            capacity=64,
            steps=1600,
        )


def arguments(case):
    """Return one shared typed packet, independent of method and backend."""
    if "prepared_arguments" in case:
        return case["prepared_arguments"]
    dynamics = case["dynamics"]
    parent = case["parent"]
    route = parent["route"]
    dtype = np.float64 if dynamics == "quad3d" else np.float32
    scene = parent["scene"] if dynamics == "unicycle" else parent
    args = dict(
        points=np.asarray(route["points"], dtype),
        route_mask=np.asarray(route["mask"], bool),
        ready=np.asarray(route["status"] == "ready"),
        goal=np.asarray(scene["goal"], dtype),
    )
    if dynamics in ("unicycle", "quad2d"):
        args.update(
            initial=np.asarray(scene["initial_state"], dtype),
            obstacles=np.asarray(scene["obstacles"], dtype),
            mask=np.asarray(scene["obstacle_mask"], bool),
            noise=np.asarray(case["noise"], dtype),
            key=np.asarray([0, case["sensor_seed"]], np.uint32),
        )
        args.update({name: case["sensor_units"][name] for name in ("unit_x", "unit_o", "unit_i")})
    elif dynamics == "quad3d":
        args.update(
            initial=np.asarray(parent["x"], dtype),
            obstacles=np.asarray(parent["obstacles"], dtype),
            mask=np.asarray(parent["mask"], bool),
            noise=np.asarray(parent["noise"], dtype),
            **case["sensor_units"],
        )
    elif dynamics == "bicycle":
        args.update(
            initial=np.asarray(parent["initial"], np.float64),
            obstacles=np.asarray(parent["obstacles"], np.float64),
            mask=np.asarray(parent["mask"], bool),
            noise=np.asarray(parent["noise"], dtype),
            bx=np.asarray(parent["bias_x"], dtype),
            bo=np.asarray(parent["bias_o"], dtype),
            first_x=np.asarray(parent["first_x"], dtype),
            first_o=np.asarray(parent["first_o"], dtype),
            key=np.asarray([0, parent["seed"] + 4], np.uint32),
        )
    else:
        raise ValueError(f"Unknown dynamics {dynamics}")
    return args


def unicycle_sensor_model(
    observed_x, observed_obstacles, mask, noise, key, config, steps, *, unit_x, unit_o, unit_i
):
    """Apply the public unicycle observation model to the shared sensor draws."""
    import jax.numpy as jnp

    x_scale = jnp.array([noise[0], noise[0], noise[1], noise[2]])
    obs_scale = jnp.array([noise[3], noise[3], noise[5], noise[4], noise[4]])
    physical_x = observed_x + unit_x * x_scale
    physical_x = physical_x.at[3].set(jnp.clip(physical_x[3], 0.0, config.v_max))
    x_bias = physical_x - observed_x
    obs_bias = jnp.where(mask[:, None], unit_o * obs_scale, 0.0)
    physical_obs = observed_obstacles + obs_bias
    physical_obs = physical_obs.at[:, 2].set(jnp.maximum(physical_obs[:, 2], 1e-4))
    if config.stationary_obstacles:
        physical_obs = physical_obs.at[:, 3:5].set(0.0)
    obs_bias = physical_obs - observed_obstacles
    innovations = jnp.asarray(unit_i).at[0].set(0.0)
    return physical_x, physical_obs, x_bias, obs_bias, x_scale, obs_scale, innovations


def quad2d_sensor_model(
    observed, obstacles, mask, noise, key, steps, stationary_obstacles=False, *, unit_x, unit_o, unit_i
):
    """Apply the public flight observation model to the shared sensor draws."""
    import jax.numpy as jnp

    xs = jnp.stack((noise[0], noise[0], noise[1], noise[2], noise[2], noise[3]))
    os = jnp.stack((noise[4], noise[4], noise[6], noise[5], noise[5]))
    xb = unit_x * xs
    ob = jnp.where(mask[:, None], unit_o * os, 0.0)
    truth = obstacles + ob
    truth = truth.at[:, 2].set(jnp.maximum(truth[:, 2], 1e-4))
    if stationary_obstacles:
        truth = truth.at[:, 3:5].set(0.0)
    ob = truth - obstacles
    innovations = jnp.asarray(unit_i).at[0].set(0.0)
    return observed + xb, truth, xb, ob, xs, os, innovations
