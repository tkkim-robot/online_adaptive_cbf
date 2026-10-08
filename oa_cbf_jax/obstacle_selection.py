"""Deterministic observed-neighborhood selection shared by graph and QP.

Selection is a controller observation boundary, not a collision-check boundary.
Callers must retain the full physical world for outcome measurements.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


def contract(count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('A positive static neighborhood capacity is required')
    return dict(schema='observed_nearest_center_neighborhood_v1', capacity=count,
        ordering='Observed center distance in FP64, then x/y/radius/vx/vy geometric ties.',
        field_of_view='360 degrees; explicit difference from historical unicycle forward sector',
        refresh='Every actual controller tick; no future truth or outcome-dependent choice.',
        graph_obstacles=count, qp_obstacles=count, fc_neural_obstacles=1,
        collision_check='Every physical obstacle, including obstacles outside the selected neighborhood.',
        padding='Masked zero rows with index -1; never silently drop a real obstacle within capacity.')


@partial(jax.jit, static_argnames=('count',))
def nearest_obstacles(position, obstacles, mask, count):
    """Return fixed [count,5] rows, their mask and original scene identities."""
    contract(count)
    if obstacles.ndim != 2 or obstacles.shape[1] != 5 or mask.shape != obstacles.shape[:1]:
        raise ValueError('Expected observed five-column obstacles and a matching mask')
    n = len(obstacles)
    padding = max(0, count-n)
    rows = jnp.pad(obstacles, ((0,padding),(0,0)))
    valid = jnp.pad(mask, (0,padding))
    delta = rows[:, :2].astype(jnp.float64)-position[:2].astype(jnp.float64)
    distance2 = jnp.sum(delta**2, axis=-1)
    order = jnp.lexsort((rows[:,4],rows[:,3],rows[:,2],rows[:,1],rows[:,0],
                        jnp.where(valid,distance2,jnp.inf)))[:count]
    selected = valid[order]
    return jnp.where(selected[:,None],rows[order],0.), selected, jnp.where(selected,order,-1)


def nearest_numpy(position, obstacles, mask, count):
    contract(count)
    rows=np.asarray(obstacles);valid=np.asarray(mask,bool)
    delta=rows[:, :2].astype(float)-np.asarray(position[:2],float)
    keys=(rows[:,4],rows[:,3],rows[:,2],rows[:,1],rows[:,0],np.where(valid,np.sum(delta**2,axis=-1),np.inf))
    indices=np.lexsort(keys)[:count]
    indices=indices[valid[indices]]
    out=np.zeros((count,5),rows.dtype);out[:len(indices)]=rows[indices]
    present=np.arange(count)<len(indices);ids=np.full(count,-1,int);ids[:len(indices)]=indices
    return out,present,ids
