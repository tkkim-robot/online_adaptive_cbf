"""Reserved random fields for the unchanged moving-obstacle bicycle policy.

Geometry is generated without observing any model output. A quarter of the
obstacles move at constant velocities; the remaining disks are static. Every
parent, failed route and failed mission stays in the evaluation denominator.
"""
import numpy as np

from .quad2d_random_density import fields, FAMILIES, COUNTS, NOISE

SCHEMA = 'bicycle_random_density_development'
SEED = 2610011000
REPLICAS = 12
GROUPS = len(FAMILIES) * len(COUNTS) * len(NOISE) * REPLICAS


def geometry(index, *, seed=SEED, schema=SCHEMA):
    if type(index) is not int or not 0 <= index < GROUPS:
        raise ValueError('Index outside the complete 768-parent reservation')
    cell, replica = divmod(index, REPLICAS)
    # Reuse the outcome-independent field generator with a fresh namespace and
    # seed. Select 12 independent replicas in EVERY cell, not just early cells.
    layout_index = cell * 32 + replica
    base = fields(layout_index, seed=seed, schema=schema)
    rng = np.random.default_rng(seed + 100000 + index)
    obstacles = np.asarray(base['obstacles'], np.float32)
    mask = np.asarray(base['obstacle_mask'], bool)
    count = int(mask.sum())
    moving = rng.permutation(count)[:count // 4]
    angle = rng.uniform(-np.pi, np.pi, len(moving))
    speed = rng.uniform(.05, .18, len(moving))
    obstacles[moving, 3:5] = speed[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    position = np.asarray(base['initial_state'][:2], np.float32)
    goal = np.asarray(base['goal'], np.float32)
    direction = goal - position
    heading = np.arctan2(direction[1], direction[0]) + rng.uniform(-.08, .08)
    initial = np.r_[position, heading, rng.uniform(.8, 1.2)].astype(np.float32)
    return dict(initial=initial, goal=goal, obstacles=obstacles, mask=mask,
                family=base['family'], density=base['density'],
                obstacle_count=count, moving_obstacles=len(moving),
                noise_level=base['declared_noise_scale'], replica=replica,
                geometry_proposals=base['geometry_proposals'],
                layout_seed=seed + layout_index, seed=seed + 200000 + index,
                group_id=f'{schema}:{seed + index}')


def geometry_metrics(parent, config):
    obstacles = np.asarray(parent['obstacles'], float)[np.asarray(parent['mask'], bool)]
    distances = np.linalg.norm(obstacles[:, None, :2] - obstacles[None, :, :2], axis=-1)
    separation = distances - obstacles[:, None, 2] - obstacles[None, :, 2]
    np.fill_diagonal(separation, np.inf)
    inflated = (obstacles[:, 2] + config['robot']['radius'] + config['clearance_buffer']) * config['barrier_inflation']
    clearance = [float(np.min(np.linalg.norm(obstacles[:, :2] - np.asarray(p)[:2], axis=1) - inflated))
                 for p in (parent['initial'], parent['goal'])]
    speeds = np.linalg.norm(obstacles[:, 3:5], axis=1)
    if (len(obstacles) != parent['obstacle_count'] or separation.min() < .07999
            or min(clearance) <= .2 or np.count_nonzero(speeds) != len(obstacles) // 4
            or speeds.max() > .180001 or not np.isfinite(obstacles).all()):
        raise ValueError('Invalid random moving-obstacle geometry')
    return dict(obstacles=len(obstacles), moving_obstacles=int(np.count_nonzero(speeds)),
                endpoint_inflated_clearance=clearance,
                minimum_surface_spacing=float(separation.min()),
                mean_other_obstacles_within_3m=float(((distances < 3.) & (distances > 0.)).sum(1).mean()),
                geometry_only=True, dynamic_solvability='unknown')
