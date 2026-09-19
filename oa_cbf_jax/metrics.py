"""Statistics giving each physical scene equal weight across its samples."""

import numpy as np


def parent_mean(values, ids, mask=None):
    values = np.asarray(values)
    ids = np.asarray(ids)
    mask = np.ones_like(values, bool) if mask is None else np.asarray(mask, bool)
    results = []
    for group in np.unique(ids):
        keep = ids == group
        valid = mask[keep]
        if valid.any():
            results.append(float(values[keep][valid].mean()))
    return float(np.mean(results)) if results else None
