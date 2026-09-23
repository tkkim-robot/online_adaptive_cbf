"""Causal route geometry for an offline, matched bicycle learning experiment.

The controller already owns this route. No candidate rollout, realized future
obstacle motion, physical state, or label enters the feature calculation.
"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from .routing import route_geometry, route_target_from_position

SCHEMA = 'bicycle_observed_route_samples59'
DISTANCES = (0., .5, 1., 2., 3., 4., 5.5, 7.)
FEATURES = 35 + 3 * len(DISTANCES)


def contract():
    return dict(schema=SCHEMA, graph_features=FEATURES,
                arclength_offsets_metres=list(DISTANCES), coordinate_scale_metres=5.,
                frame='current observed ego heading',
                fields_per_sample=['relative_x', 'relative_y', 'route_reaches_offset'],
                route='already committed observed route; local monotone cursor update',
                endpoint='clamp sample to endpoint, retain explicit reaches-offset bit',
                privilege='No physical state, future observation, obstacle forecast or target label',
                use='Offline learning pilot; runtime policy qualification remains required')


def route_context(x, points, route_mask, cursor):
    """Eight static-shape samples of the committed route, in the observed frame."""
    # Padded route coordinates must not affect any arithmetic, even if a caller
    # uses arbitrary padding. The original route itself is never modified.
    points = jnp.where(route_mask[:, None], points, 0.)
    _, updated, _ = route_target_from_position(x[:2], x[3], points, route_mask, cursor)
    vectors, valid, lengths, cumulative = route_geometry(points, route_mask)
    valid = valid & (lengths > 1e-8)
    requested = updated + jnp.asarray(DISTANCES, x.dtype)
    desired = jnp.minimum(requested, cumulative[-1])
    eligible = valid[None, :] & (cumulative[None, 1:] >= desired[:, None] - 1e-6)
    index = jnp.argmax(eligible, axis=1)
    fraction = jnp.clip((desired - cumulative[index]) / jnp.maximum(lengths[index], 1e-12), 0., 1.)
    samples = points[index] + fraction[:, None] * vectors[index]
    rotation = jnp.stack((jnp.stack((jnp.cos(x[2]), -jnp.sin(x[2]))),
                          jnp.stack((jnp.sin(x[2]), jnp.cos(x[2])))))
    relative = jnp.matmul(samples - x[:2], rotation, precision='highest') / 5.
    reaches = (requested <= cumulative[-1] + 1e-6) & jnp.any(valid)
    return jnp.concatenate((relative, reaches[:, None].astype(x.dtype)), axis=1).reshape(-1)


def append_context(features, node_mask, x, points, route_mask, cursor):
    if features.shape[-1] != 35:
        raise ValueError('Route augmentation requires the original observed graph35')
    context = route_context(x, points, route_mask, cursor)
    extended = jnp.concatenate((features, jnp.broadcast_to(context, (len(features), len(context)))), axis=-1)
    return jnp.where(node_mask[:, None], extended, 0.)


def numpy_context(x, points, mask, cursor):
    """Independent scalar reference: walk the active route segment by segment."""
    x = np.asarray(x, float); points = np.asarray(points, float)[np.asarray(mask, bool)]
    segments = np.diff(points, axis=0); lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0., lengths.cumsum()]
    best_distance, updated = float('inf'), float(cursor)
    for i, (segment, length) in enumerate(zip(segments, lengths)):
        if length <= 1e-8:
            continue
        f = np.clip(np.dot(x[:2] - points[i], segment) / length**2, 0., 1.)
        s = cumulative[i] + f * length
        distance = np.sum((points[i] + f * segment - x[:2])**2)
        if cursor - .05 <= s <= cursor + 1. and distance < best_distance:
            best_distance, updated = distance, max(float(cursor), s)
    result = []
    for offset in DISTANCES:
        requested = updated + offset; desired = min(requested, cumulative[-1])
        for i, length in enumerate(lengths):
            if length > 1e-8 and cumulative[i + 1] >= desired - 1e-6:
                fraction = np.clip((desired - cumulative[i]) / length, 0., 1.)
                delta = points[i] + fraction * segments[i] - x[:2]
                c, s = np.cos(x[2]), np.sin(x[2])
                result.extend(((c * delta[0] + s * delta[1]) / 5.,
                               (-s * delta[0] + c * delta[1]) / 5.,
                               float(requested <= cumulative[-1] + 1e-6)))
                break
        else:
            raise ValueError('Route has no nonzero active segment')
    return np.asarray(result, np.float32)


def augment_development(data, role, batch=256):
    """Transform every audited development query, with an independent check.

    Calibration and benchmark records are deliberately refused by this pilot.
    Both encoders call the exact same transform; original arrays stay intact.
    """
    if role not in ('train', 'validation') or not np.all(data['partition'] == role):
        raise ValueError('Only original TRAIN/validation queries may enter this pilot')
    if data['features'].shape[-1] != 35 or not len(data['features']):
        raise ValueError('Nonempty original graph35 data required')
    rm = np.asarray(data['route_mask'], bool)
    if np.any(rm[:, 1:] & ~rm[:, :-1]) or not (rm.sum(1) >= 2).all():
        raise ValueError('A contiguous committed route with two active points is required')
    inputs = [data[k] for k in ('features', 'node_mask', 'observed_state', 'points', 'route_mask', 'cursor')]
    # CPU is sufficient for this small preprocessing step; keep accelerator
    # memory available for actual optimizer and inference workloads.
    with jax.default_device(jax.devices('cpu')[0]):
        compute = jax.jit(jax.vmap(append_context))
        parts = []
        for start in range(0, len(rm), batch):
            count = min(batch, len(rm) - start)
            block = [np.pad(a[start:start + count], [(0, batch-count)] + [(0, 0)]*(a.ndim-1), mode='edge') for a in inputs]
            parts.append(np.asarray(compute(*block))[:count])
        result = np.concatenate(parts)
        if compute._cache_size() != 1:
            raise ValueError('Unexpected route feature recompilation')
    references = np.stack([numpy_context(x, p, m, c) for x, p, m, c in zip(
        data['observed_state'], data['points'], rm, data['cursor'])])
    np.testing.assert_allclose(result[:, 0, 35:], references, atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(result[..., :35], data['features'])
    if not np.isfinite(result).all() or np.any(result[~data['node_mask']] != 0.):
        raise ValueError('Invalid route features or padded nodes')
    proof = dict(contract=contract(), role=role, parents=len(np.unique(data['group_id'])), queries=len(rm),
                 source_features_sha256=hashlib.sha256(data['features'].tobytes()).hexdigest(),
                 augmented_features_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                 independent_numpy_max_error=float(np.max(np.abs(result[:, 0, 35:] - references))),
                 original_features_unchanged=True, all_queries_checked=True, feature_signature_count=1)
    return dict(data, features=result), proof
