"""Single-nearest-obstacle inputs for the paper's FC/PENN comparator.

This is an information-limited baseline, distinct from the full-scene
``matched_fc`` encoder ablation. All obstacles remain in the controller QP.
No route, goal, history, obstacle count, or other-obstacle statistic enters
this network. Inputs follow online_adaptive_cbf.get_rel_state_wt_obs.
"""
import math
import jax
import jax.numpy as jnp

# Keep neural arithmetic FP32 while allowing explicit FP64 geometry keys.
jax.config.update('jax_explicit_x64_dtypes', 'allow')

SCHEMA = 'nearest_obstacle_fc_inputs_v2'
SHAPES = {'unicycle': (18, 31), 'bicycle': (35,), 'quad2d': (40,), 'quad3d': (50, 58)}


def from_manifest(manifest):
    columns = manifest['graph_features']
    matches = [name for name, widths in SHAPES.items() if columns in widths]
    if len(matches) != 1:
        raise ValueError('Unregistered nearest-FC observation schema')
    dynamics = matches[0]
    gain_dimension = manifest.get('gain_dimension', 2)
    if gain_dimension != {'unicycle': 2, 'bicycle': 1, 'quad2d': 2, 'quad3d': 4}[dynamics]:
        raise ValueError('Nearest-FC graph/gain dynamics mismatch')
    yaw_scale = manifest['config']['yaw_limit'] if dynamics == 'quad3d' else 1.
    return dynamics, float(yaw_scale)


def contract(dynamics, yaw_scale=1., constraint_features=False):
    if dynamics not in SHAPES or not math.isfinite(yaw_scale) or yaw_scale <= 0:
        raise ValueError('Invalid nearest-FC feature contract')
    velocities = {'unicycle': ['speed'], 'bicycle': ['speed'],
                  'quad2d': ['velocity_x', 'velocity_z'],
                  'quad3d': ['velocity_x', 'velocity_y', 'velocity_z']}[dynamics]
    result=dict(schema=SCHEMA, dynamics=dynamics, neural_obstacles=1,
        nearest='Minimum observed center distance computed in FP64 from observed positions; geometric tie breakers; masked obstacles excluded.',
        context=['signed_surface_clearance', *velocities, 'sin_relative_bearing', 'cos_relative_bearing'],
        candidate='Log of each queried positive class-K gain; same candidate domain as OA.',
        scaling='Dataset fixed physical units: clearance/3 and velocity/declared speed limit.',
        yaw_scale=yaw_scale,
        no_obstacle='Clearance 100/3, bearing sin=0 cos=1; robot velocity retained.',
        hidden_layers='ReLU W, 2W, 3W, W, as original FC/PENN.',
        controller_obstacles='All observed obstacles; restriction applies only to neural prediction.',
        excluded=['goal', 'route', 'history', 'noise_level', 'other_obstacles', 'obstacle_count'],
        quad3d_adaptation='Linearized 3D plant uses three translational velocities, actual yaw, and four queried gains; never altitude as heading.',
        comparison='Paper nearest-obstacle FC baseline; not a full-scene encoder-only ablation.')
    if constraint_features:
        if dynamics!='unicycle':raise ValueError('Coefficient variant requires unicycle')
        from .unicycle_constraint_features import contract as coefficient_contract
        result.update(schema='nearest_obstacle_fc_unicycle_coefficients_v1',
            observed_coefficient_features=coefficient_contract(),
            comparison='Augmented nearest-obstacle FC comparator; original FC results remain separate.',
            candidate='Log gains plus observed-coefficient gain basis, identical to the GAT treatment.')
    return result


def validate_metadata(metadata):
    arch = metadata['architecture']
    dynamics = arch.get('nearest_dynamics')
    expected = contract(dynamics, arch.get('nearest_yaw_scale', 1.),arch.get('unicycle_constraint_features',False))
    if (arch['encoder'] != 'nearest_fc' or metadata.get('nearest_fc_contract') != expected
            or metadata.get('graph_features') not in SHAPES[dynamics]
            or metadata.get('gain_dimension', 2) != {'unicycle': 2, 'bicycle': 1, 'quad2d': 2, 'quad3d': 4}[dynamics]):
        raise ValueError('Changed nearest-obstacle FC input contract')
    return expected


def encode(features, mask, dynamics, yaw_scale=1.):
    """Fixed-shape, JIT-compatible nearest selection; no pooled scene context."""
    if dynamics not in SHAPES or features.shape[-1] not in SHAPES[dynamics]:
        raise ValueError('Wrong dynamics-specific nearest-FC graph')
    clean = jnp.where(mask[..., None], features, 0.)
    ego, obstacles, valid = clean[:, 0], clean[:, 2:], mask[:, 2:]
    # A zero-capacity graph also has a well-defined finite empty observation.
    obstacles = jnp.pad(obstacles, ((0, 0), (0, 1), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, 1)))
    # A rounded FP32 sum can turn unequal distances into a tie on CPU, while
    # GPU multiply-add fusion preserves their order. That changes the selected
    # obstacle discontinuously. FP64 products of the FP32 observations retain
    # their exact mantissas; this key does not change the neural compute dtype.
    position = obstacles[..., 3:5].astype(jnp.float64)
    distance2 = jnp.sum(position ** 2, axis=-1)
    order = jnp.lexsort((obstacles[..., 7], obstacles[..., 4], obstacles[..., 3],
                        jnp.where(valid, distance2, jnp.inf)), axis=1)
    nearest = jnp.take_along_axis(obstacles, order[:, :1, None], axis=1)[:, 0]
    present = valid.any(axis=1)
    bearing = jnp.arctan2(nearest[:, 4], nearest[:, 3])
    if dynamics in ('unicycle', 'quad2d'):
        bearing -= jnp.arctan2(ego[:, 9], ego[:, 10])
        velocity = ego[:, 11:12] if dynamics == 'unicycle' else ego[:, 11:13]
    elif dynamics == 'bicycle':
        # Bicycle graph positions are already in the robot-heading frame.
        velocity = ego[:, 9:10]
    else:
        bearing -= ego[:, 20] * yaw_scale
        velocity = ego[:, 21:24]
    clearance = jnp.where(present, nearest[:, 8], 100. / 3.)[:, None]
    angles = jnp.stack((jnp.where(present, jnp.sin(bearing), 0.),
                        jnp.where(present, jnp.cos(bearing), 1.)), axis=-1)
    return jnp.concatenate((clearance, velocity, angles), axis=-1)
