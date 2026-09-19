"""Explicit gain-domain expansion at unchanged physical parent snapshots."""

import hashlib
import numpy as np
from scipy.stats import qmc


def wide_pairs(upper):
    return np.asarray([(upper,upper),(upper/2,upper/2),(.75*upper,.75*upper),
        (upper/2,upper/4),(upper,upper/2),(upper,upper/4),(upper/4,upper/2),(upper/2,upper)],np.float32)


def requery(original,group_id,queries,replicas,upper):
    """Keep twelve source queries and replica identity; sample the wider domain.

    Query design is fixed independently of labels, episode outcomes and splits.
    Stable parent-specific Sobol seeds make this independent of worker ordering.
    A source prior must not be regenerated when query gains change.
    """
    if queries<32 or queries&(queries-1) or replicas<2 or not np.isfinite(upper) or upper<16:
        raise ValueError('Wide requery needs >=32 power-of-two queries, >=2 replicas and finite upper>=16')
    original=np.asarray(original,np.float32).reshape(queries,replicas,2)
    if not np.all(original==original[:,:1]):raise ValueError('Gain replicas disagree')
    if np.any(original<.3) or np.any(original>upper):raise ValueError('New domain must include source queries')
    seed=int.from_bytes(hashlib.sha256(('wide_gain_requery_v1:'+str(group_id)).encode()).digest()[:4],'little')
    values=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.log2(queries)))
    query=np.exp(np.log(.3)+values*np.log(upper/.3)).astype(np.float32)
    query[:12]=original[:12,0];query[12:20]=wide_pairs(upper)
    return np.repeat(query,replicas,axis=0)
