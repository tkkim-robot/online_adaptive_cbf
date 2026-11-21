"""Generate the four fixed obstacle populations, nominal routes and sensor inputs."""

from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .sensors import arguments, planar_units


@lru_cache(maxsize=4)
def _spec(models, dynamics):
    manifest = json.loads((models / "release.json").read_text())
    directory = manifest["models"][dynamics]["gat"]
    return json.loads((models / directory / "policy.json").read_text())


def _disk_field(q):
    """The original density-conditioned sampler, including draw order and FP32 storage."""
    families = ("random_field", "paired_clusters", "staggered_layers", "random_pockets")
    family = families.index(q["family"])
    count = (24, 36, 48, 60)[q["density"]]
    rng = np.random.default_rng(q["geometry_seed"])
    disks = []
    centers = np.array([(x, z) for x in (4.0, 8.0, 12.0, 16.0) for z in (-2.0, 2.0)])
    centers += rng.uniform(-0.5, 0.5, centers.shape)
    cell_order = rng.permutation(60)
    attempts = 0
    while len(disks) < count:
        attempts += 1
        if attempts > 20000:
            raise ValueError("Disk placement exhausted for the reserved seed")
        radius = rng.uniform(0.19, 0.36)
        if family == 1 and len(disks) % 2:
            anchor = np.asarray(disks[-1])
            theta = rng.uniform(-np.pi, np.pi)
            distance = anchor[2] + radius + rng.uniform(0.10, 0.38)
            position = anchor[:2] + distance * np.array([np.cos(theta), np.sin(theta)])
        elif family == 2:
            column, row = divmod(int(cell_order[len(disks)]), 6)
            position = np.array([2.8 + 1.4 * column, -3.5 + 1.35 * row])
            position += np.array([rng.uniform(-0.18, 0.18), 0.22 * (-1) ** column + rng.uniform(-0.18, 0.18)])
        elif family == 3:
            position = centers[rng.integers(len(centers))] + rng.normal(0.0, 1.0, 2)
        else:
            position = rng.uniform([2.5, -4.3], [16.5, 4.3])
        if not (2.3 <= position[0] <= 17.0 and -4.5 <= position[1] <= 4.5):
            continue
        if disks:
            previous = np.asarray(disks)
            if np.min(np.linalg.norm(previous[:, :2] - position, axis=1) - previous[:, 2] - radius) < 0.08:
                continue
        disks.append([*position, radius, 0.0, 0.0])
    obstacles = np.zeros((64, 5), np.float32)
    obstacles[:count] = np.asarray(disks, np.float32)[rng.permutation(count)]
    angle = rng.uniform(-np.pi / 12, np.pi / 12)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = rng.uniform(-3.0, 3.0, 2)
    obstacles[:count, :2] = obstacles[:count, :2] @ rotation.T + shift
    goal = np.array([19.5, rng.uniform(-0.5, 0.5)]) @ rotation.T + shift
    state = np.r_[shift, rng.uniform(-0.08, 0.08), rng.uniform(-0.15, 0.15, 2), rng.uniform(-0.1, 0.1)]
    return state.astype(np.float32), goal.astype(np.float32), obstacles, np.arange(64) < count


def _route_dict(route):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(route).items()}


def _order(initial_xy, obstacles, mask):
    return sorted(
        np.flatnonzero(mask),
        key=lambda i: (
            float(np.sum((obstacles[i, :2] - initial_xy) ** 2)),
            *map(float, obstacles[i, :3]),
            int(i),
        ),
    )


def _initial_far_motion(state, goal, obstacles, mask):
    order = _order(state[:2], obstacles.astype(float), mask)
    obstacles[:, 3:5] = 0.0
    if order:
        distance = np.linalg.norm(obstacles[:, :2].astype(float) - state[:2], axis=1)
        slots = [i for i in order[1:] if distance[i] >= distance[order[0]] + 0.25][:2]
        direction = np.asarray(goal - state[:2], np.float32)
        length = np.linalg.norm(direction)
        direction = direction / max(float(length), 1e-20) if length else np.array([1.0, 0.0], np.float32)
        for slot, velocity in zip(slots, 0.2 * np.stack((direction, -direction)).astype("float32")):
            obstacles[slot, 3:5] = velocity
    return order


def _planar_case(case, q, config):
    from oa_cbf_jax.routing import plan_route

    is_unicycle = case["dynamics"] == "unicycle"
    if is_unicycle:
        from oa_cbf_jax.config import UnicycleConfig
        from oa_cbf_jax.scenes import multiscale_scenes_scene, stationary_scene

        scene = stationary_scene(multiscale_scenes_scene(q["geometry_seed"], q["family"], 64)).json_record()
        start, goal = np.asarray(scene["initial_state"]), np.asarray(scene["goal"])
        geometry, mask = np.asarray(scene["obstacles"]), np.asarray(scene["obstacle_mask"], bool)
        robot = UnicycleConfig(**dict(config, stationary_obstacles=False))
        noise = [0.02, 0.03, 0.02, 0.03, 0.025, 0.01]
    else:
        from oa_cbf_jax.quad2d_control import flight_config_from_contract

        start, goal, geometry, mask = _disk_field(q)
        robot = flight_config_from_contract(dict(config, stationary_obstacles=False)).robot
        noise = [0.015, 0.01, 0.015, 0.015, 0.02, 0.02, 0.008]
    state, target, obstacles = (np.asarray(v, np.float32).copy() for v in (start, goal, geometry))
    order = _initial_far_motion(state, target, obstacles, mask)
    # The planner consumes the original double-precision world before overriding the two nearest velocities.
    planning = geometry.astype(float).copy()
    planning[:, 3:5] = obstacles[:, 3:5]
    route = plan_route(
        start.astype(float)[:2],
        goal.astype(float),
        planning,
        mask,
        robot,
        capacity=64,
        visibility_batch_nodes=32,
    )
    route = dict(
        points=np.asarray(route.points, np.float32).tolist(), mask=route.mask.tolist(), status=route.status
    )
    if is_unicycle and order:
        first = order[0]
        radial = obstacles[first, :2].astype(float) - state[:2].astype(float)
        norm = np.linalg.norm(radial)
        if norm:
            obstacles[first, 3:5] = 0.7 * (radial / norm)
            # The braking-window eligibility and its separate ordering are explicitly FP32.
            narrow_order = _order(state[:2], obstacles, mask)
            threshold = np.float32(1) + np.float32(0.35) / np.float32(0.5) + np.float32(0.05)
            if narrow_order[0] == first:
                for i in narrow_order[1:]:
                    delta = state[:2] - obstacles[i, :2]
                    distance = np.sqrt(np.sum(delta * delta))
                    if np.float32(distance - obstacles[i, 2] - np.float32(0.3)) > threshold:
                        velocity = np.float32(0.2) * delta / distance
                        obstacles[i, 3:5] = np.where(velocity == 0.0, np.float32(0), velocity)
                        break
    elif not is_unicycle and len(order) >= 2:
        radial = [obstacles[i, :2].astype(float) - state[:2].astype(float) for i in order[:2]]
        if all(np.linalg.norm(delta) for delta in radial):
            for slot, delta, sign in zip(order[:2], radial, (1.0, -1.0)):
                velocity = np.asarray(sign * 0.7 * delta / np.linalg.norm(delta), np.float32)
                obstacles[slot, 3:5] = np.where(velocity == 0.0, np.float32(0), velocity)
    physical = dict(
        initial_state=state.tolist(),
        goal=target.tolist(),
        obstacles=obstacles.tolist(),
        obstacle_mask=mask.tolist(),
        family=q["family"],
        seed=q["geometry_seed"],
    )
    case["noise"] = (np.float32(q["noise_scale"]) * np.asarray(noise, np.float32)).tolist()
    if is_unicycle:
        physical["scene_id"] = case["trial_id"]
        case["parent"] = dict(scene=physical, route=route)
    else:
        case["parent"] = dict(physical, group_id=case["trial_id"], noise=case["noise"], route=route)
    case["sensor_key"] = [0, case["sensor_seed"]]
    case["sensor_units"] = planar_units(case["sensor_seed"], len(state))


def _point_at_distance(points, mask, arc):
    distance = 0.0
    for j in range(len(points) - 1):
        delta = points[j + 1] - points[j]
        length = float(np.linalg.norm(delta)) if mask[j] and mask[j + 1] else 0.0
        if length > 0 and distance <= arc < distance + length:
            tangent = delta / length
            return points[j] + (arc - distance) * tangent, tangent
        distance += length
    return None


def _closing_gate(a, seen, observed):
    """Give an outward-moving decoy and two crossing cylinders their specified observed velocities."""
    order = _order(seen[:2], observed, a["mask"])
    if len(order) < 3 or not a["ready"]:
        return
    nearest, others = order[0], order[1:]
    radial = observed[nearest, :2] - seen[:2]
    radius = float(np.linalg.norm(radial))
    if radius == 0:
        return
    for arc in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5):
        spot = _point_at_distance(a["points"], a["route_mask"], arc)
        if spot is None:
            continue
        target, tangent = spot
        normal = np.array([-tangent[1], tangent[0]], np.float64)
        seconds = arc + 15 / 16
        noise = a["noise"]
        center_error = 1.15 * (float(noise[4]) + seconds * float(noise[5]))
        total_error = center_error + 1.15 * float(noise[6]) + 1.15 * float(noise[0])
        halfwidth = {i: float(observed[i, 2]) + 0.3 + 0.05 + total_error + 0.05 for i in others}
        left = {i: (target - halfwidth[i] * normal - observed[i, :2]) / seconds for i in others}
        right = {i: (target + halfwidth[i] * normal - observed[i, :2]) / seconds for i in others}
        left_speed = {i: float(np.linalg.norm(left[i])) for i in others}
        right_speed = {i: float(np.linalg.norm(right[i])) for i in others}
        left_slots = [i for i in others if 0 < left_speed[i] <= 0.7 and left[i] @ normal >= 0.2]
        right_slots = [i for i in others if 0 < right_speed[i] <= 0.7 and right[i] @ normal <= -0.2]
        pairs = []
        for i in left_slots:
            for j in right_slots:
                if i == j:
                    continue
                u, v = observed[i, :2] - target, observed[j, :2] - target
                denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
                if denominator and np.arccos(np.clip(u @ v / denominator, -1.0, 1.0)) >= np.pi / 3:
                    pairs.append((i, j))
        if pairs:

            def rank(pair):
                i, j = pair
                return (
                    max(left_speed[i], right_speed[j]),
                    left_speed[i] ** 2 + right_speed[j] ** 2,
                    *map(float, observed[i, :3]),
                    i,
                    *map(float, observed[j, :3]),
                    j,
                )

            i, j = min(pairs, key=rank)
            slots = [nearest, i, j]
            desired = np.array([0.4 * radial / radius, left[i], right[j]])
            desired = np.where(desired == 0.0, 0.0, desired)
            errors = noise[5] * (a["bo"][slots, 3:5] + 0.15 * a["io"][0, slots, 3:5])
            a["obstacles"][slots, 3:5] = desired - errors
            return


def _quad3d_case(case, q, spec):
    from oa_cbf_jax.quad3d_control import control_config
    from oa_cbf_jax.quad3d_observation import BASE_NOISE, numpy_obstacles, numpy_observe, unit_tape
    from oa_cbf_jax.quad3d_routing import observed_route

    start, target, obstacles, mask = _disk_field(q)
    seed = q["geometry_seed"]
    rng = np.random.default_rng(seed + 100000)
    state = np.zeros(12)
    state[:2] = start[:2]
    state[2] = rng.uniform(0.4, 2.4)
    broad = bool(q["ordinal"] % 2)
    tilt, velocity, vertical, rate = (0.1, 0.4, 0.2, 0.15) if broad else (0.025, 0.08, 0.05, 0.025)
    state[3:5] = rng.uniform(-tilt, tilt, 2)
    state[5] = rng.uniform(-0.2, 0.2)
    state[6:8] = rng.uniform(-velocity, velocity, 2)
    state[8] = rng.uniform(-vertical, vertical)
    state[9:12] = rng.uniform(-rate, rate, 3)
    goal = np.array([*target, rng.uniform(0.2, 2.8)], np.float64)
    noise = q["noise_scale"] * BASE_NOISE
    obstacles = obstacles.astype(np.float64)
    units = dict(zip(("bx", "bo", "ix", "io"), unit_tape(q["sensor_seed"], 1600, 64)))
    seen, observed = numpy_observe(
        state, obstacles, mask, units["bx"], units["bo"], noise, units["ix"][0], units["io"][0]
    )
    guide = numpy_obstacles(observed, mask, noise, True)
    route = observed_route(seen, goal, guide, mask, control_config(spec["config"]))
    parent = dict(
        x=state.tolist(),
        goal=goal.tolist(),
        obstacles=obstacles.tolist(),
        mask=mask.tolist(),
        sensor_seed=q["sensor_seed"],
        noise=noise.tolist(),
        route=_route_dict(route),
        seed=seed,
        family=q["family"],
        gains=[spec["policy"]["initial_gain"]] * 4,
        id=q["case_id"],
        group_id=q["case_id"],
        noise_level=q["noise_scale"],
        index=q["ordinal"],
        density=q["density"],
        obstacle_count=int(mask.sum()),
        replica=q["replica"],
        motion_stratum="broad" if broad else "small",
    )
    case.update(parent=parent, sensor_units=units)
    a = arguments(case)
    _closing_gate(a, seen, observed)
    parent["obstacles"] = a["obstacles"].tolist()
    case["prepared_arguments"] = a


def _bicycle_case(case, q, config):
    import jax
    import jax.numpy as jnp
    from oa_cbf_jax.bicycle_control import control_config
    from oa_cbf_jax.bicycle_observation import BASE_NOISE, sample_bias, observe
    from oa_cbf_jax.routing import plan_route

    start, goal, obstacles, mask = _disk_field(q)
    seed = q["geometry_seed"]
    rng = np.random.default_rng(seed + 100000)
    count = int(mask.sum())
    moving = rng.permutation(count)[: count // 4]
    angle = rng.uniform(-np.pi, np.pi, len(moving))
    speed = rng.uniform(0.05, 0.18, len(moving))
    obstacles[moving, 3:5] = speed[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    position = start[:2]
    direction = goal - position
    heading = np.arctan2(direction[1], direction[0]) + rng.uniform(-0.08, 0.08)
    initial = np.r_[position, heading, rng.uniform(0.8, 1.2)].astype(np.float32)
    noise = BASE_NOISE * np.float32(q["noise_scale"])
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        bx, bo = sample_bias(jax.random.PRNGKey(seed + 1), jnp.asarray(noise), jnp.asarray(mask))
        sx, so = observe(
            jnp.asarray(initial, dtype=jnp.float64),
            jnp.asarray(obstacles, dtype=jnp.float64),
            jnp.asarray(mask),
            bx,
            bo,
            jnp.asarray(noise),
            jnp.zeros(4, jnp.float32),
            jnp.zeros((64, 5), jnp.float32),
        )
        bx, bo, sx, so = map(np.asarray, (bx, bo, sx, so))

    planning = so.astype(float)
    bound = 1.15 * noise[4] + (1e-7 if noise[4] > 0 else 0.0)
    speeds = np.linalg.norm(planning[:, 3:5], axis=1)
    planning[mask & ((speeds <= bound) | (speeds < 1e-10)), 3:5] = 0.0
    route = plan_route(
        sx[:2], goal, planning, mask, control_config(config), capacity=64, visibility_batch_nodes=32
    )
    observed = so.astype(float)
    measured_xy = sx[:2].astype(float)
    nearest = min(
        np.flatnonzero(mask),
        key=lambda i: (
            (float(observed[i, 0] - measured_xy[0]) ** 2 + float(observed[i, 1] - measured_xy[1]) ** 2),
            *map(float, observed[i, :3]),
            int(i),
        ),
    )
    radial = observed[nearest, :2] - measured_xy
    length = float(np.hypot(*radial))
    obstacles = obstacles.astype(np.float64)
    so = so.copy()
    if length:
        velocity = np.asarray(0.15 * (radial / length), np.float32)
        so[nearest, 3:5] = np.where(velocity == 0.0, np.float32(0), velocity)
        obstacles[nearest, 3:5] = so[nearest, 3:5].astype(float) - bo[nearest, 3:5].astype(float)
    case["parent"] = dict(
        initial=initial.tolist(),
        goal=goal.tolist(),
        mask=mask.tolist(),
        obstacles=obstacles.tolist(),
        route=_route_dict(route),
        noise=noise.tolist(),
        first_x=sx.tolist(),
        first_o=so.tolist(),
        bias_x=bx.tolist(),
        bias_o=bo.tolist(),
        seed=seed,
        family=q["family"],
        group_id=q["case_id"],
        density=q["density"],
        obstacle_count=count,
        moving_obstacles=len(moving),
        noise_level=q["noise_scale"],
        replica=q["replica"],
        layout_seed=seed,
    )


def make_case(request, models):
    """Generate one independent parent and its shared numerical inputs from its reserved seeds."""
    q = request
    dynamics = q["dynamics"]
    spec = _spec(Path(models), dynamics)
    case = dict(
        dynamics=dynamics,
        trial_id=q["case_id"],
        parent_id=q["case_id"],
        seed=q["geometry_seed"],
        sensor_seed=q["geometry_seed"] + 1 if dynamics == "bicycle" else q["sensor_seed"],
        noise_scale=q["noise_scale"],
        ordinal=q["ordinal"],
        steps=1600,
        strata={k: q[k] for k in ("family", "density", "noise_index", "noise_scale", "replica")},
    )
    if dynamics in ("unicycle", "quad2d"):
        _planar_case(case, q, spec["config"])
    elif dynamics == "quad3d":
        _quad3d_case(case, q, spec)
    elif dynamics == "bicycle":
        _bicycle_case(case, q, spec["config"])
    else:
        raise ValueError(f"Unknown dynamics {dynamics}")
    case["metadata"] = dict(
        geometry_seed=q["geometry_seed"],
        sensor_seed=case["sensor_seed"],
        route_status=case["parent"]["route"]["status"],
    )
    case["prepared_arguments"] = arguments(case)
    return case
