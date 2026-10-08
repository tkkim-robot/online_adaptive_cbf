"""Robot-heading coordinates for the observed 18-column unicycle graph.

This deterministic encoder transform preserves every observed obstacle, goal,
relative velocity, clearance and physical limit. It removes only the arbitrary
world orientation, which the static circular-obstacle unicycle task does not
depend on. Saved physical graph data and the CBF-QP are unchanged.
"""


import jax.numpy as jnp


def ego_frame(features, mask):
    if features.shape[-1] != 18 or features.shape[:-1] != mask.shape:
        raise ValueError('Robot-heading transform requires the masked unicycle graph18')
    clean = jnp.where(mask[..., None], features, 0.)
    sine = clean[..., :1, 9]; cosine = clean[..., :1, 10]
    result = clean
    for first in (3, 5, 16):
        x, y = clean[..., first], clean[..., first+1]
        result = result.at[..., first].set(cosine*x+sine*y)
        result = result.at[..., first+1].set(-sine*x+cosine*y)
    result = result.at[..., 9].set(0.).at[..., 10].set(1.)
    return jnp.where(mask[..., None], result, 0.)
