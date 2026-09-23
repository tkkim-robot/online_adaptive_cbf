"""Statistics giving each physical scene equal weight across its samples."""

import numpy as np


def binary_parent_rate(values, ids, mask=None, *, legacy_float32=False):
    """Equal-parent binary rate from integer counts, with explicit report rounding.

    Historical ``parent_mean`` reports on FP32 event arrays rounded each
    parent's rate to FP32 before the final FP64 average. The optional mode
    independently reproduces that convention without changing old artifacts;
    the default keeps both averaging stages in FP64.
    """
    values=np.asarray(values);ids=np.asarray(ids)
    mask=np.ones_like(values,bool) if mask is None else np.asarray(mask,bool)
    if values.ndim<1 or ids.shape!=(len(values),) or mask.shape!=values.shape:
        raise ValueError('Aligned parent IDs and binary observations required')
    if not np.isin(values[mask],[0,1]).all():
        raise ValueError('Observed events must be binary')
    if legacy_float32 and values.dtype!=np.float32:
        raise ValueError('Legacy event rounding requires an FP32 source array')
    rates=[]
    for group in np.unique(ids):
        keep=ids==group;selected=values[keep][mask[keep]]
        if selected.size:
            rate=np.count_nonzero(selected)/selected.size
            rates.append(float(np.float32(rate)) if legacy_float32 else rate)
    return float(np.mean(rates,dtype=np.float64)) if rates else None


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
