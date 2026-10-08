"""Diagnose saturated bicycle labels using a physical route-to-go potential.

This module reads recorded, independently audited development futures. It does
not select an online gain, change a controller, or authorize model training.
"""


import numpy as np


def contract():
    return dict(schema='bicycle_physical_route_to_go_progress',
        potential='Distance to the nearest local route projection plus remaining route arclength.',
        projection='FP64 physical position; valid positive-length segments clipped to cursor+-1 metre; first nearest-distance tie.',
        target='Initial potential minus final potential, divided by horizon*dt*cruise_speed.',
        task_stop='Final potential is zero ONLY on a recorded physically verified GOAL event.',
        censoring='Actual retained prefix on every branch, including adverse termination; never invent later progress.',
        new_features=False, gain_search=False, controller_changed=False,
        limitation='Geometric task progress, not a reachability, safety or navigation-success certificate.')


def route_potential(position, points, mask, cursor):
    """Vectorized distance via the same local route branch used for progress.

    The cursor is recorded controller memory. It cannot itself credit arrival:
    lateral distance remains even when the projected coordinate reaches the end.
    Route suffix length preserves the reward for following a necessary detour.
    """
    x=np.asarray(position, np.float64); p=np.asarray(points, np.float64)
    m=np.asarray(mask, bool); hint=np.asarray(cursor, np.float64)
    if (x.shape[-1:]!=(2,) or p.ndim!=2 or p.shape[-1]!=2
            or m.shape!=p.shape[:1] or not np.isfinite(x).all()
            or not np.isfinite(p).all() or not np.isfinite(hint).all()):
        raise ValueError('Invalid physical route projection input')
    hint=np.broadcast_to(hint, x.shape[:-1])
    vectors=np.diff(p, axis=0); valid=m[:-1]&m[1:]
    lengths=np.where(valid, np.linalg.norm(vectors, axis=-1), 0.)
    cumulative=np.r_[0., np.cumsum(lengths)]
    if not np.any(lengths>0):
        raise ValueError('A positive-length recorded route is required')
    low=np.maximum(cumulative[:-1], hint[...,None]-1.)
    high=np.minimum(cumulative[1:], hint[...,None]+1.)
    eligible=valid&(lengths>0)&(low<=high)
    if not np.all(np.any(eligible, axis=-1)):
        raise ValueError('Recorded cursor has no local route segment')
    along=np.sum((x[...,None,:]-p[:-1])*vectors, axis=-1)/np.maximum(lengths, 1e-300)
    coordinate=np.maximum(low, np.minimum(high, cumulative[:-1]+along))
    projection=p[:-1]+((coordinate-cumulative[:-1])/np.maximum(lengths, 1e-300))[...,None]*vectors
    distance=np.linalg.norm(x[...,None,:]-projection, axis=-1)
    index=np.argmin(np.where(eligible, distance, np.inf), axis=-1)
    chosen=np.take_along_axis(coordinate, index[...,None], axis=-1)[...,0]
    cross=np.take_along_axis(distance, index[...,None], axis=-1)[...,0]
    return cumulative[-1]-chosen+cross


def reference_potential(position, points, mask, cursor):
    """Independent scalar construction using clipped Cartesian segment ends."""
    import math
    x=tuple(map(float, position)); hint=float(cursor); candidates=[]; total=0.
    for i in range(len(points)-1):
        if not (mask[i] and mask[i+1]):
            continue
        a=tuple(map(float, points[i])); b=tuple(map(float, points[i+1]))
        length=math.dist(a,b)
        start=total; total+=length
        if length==0:
            continue
        lo=max(start,hint-1.); hi=min(total,hint+1.)
        if lo>hi:
            continue
        direction=((b[0]-a[0])/length,(b[1]-a[1])/length)
        left=(a[0]+(lo-start)*direction[0],a[1]+(lo-start)*direction[1])
        right=(a[0]+(hi-start)*direction[0],a[1]+(hi-start)*direction[1])
        v=(right[0]-left[0],right[1]-left[1]); square=v[0]**2+v[1]**2
        t=min(1.,max(0.,((x[0]-left[0])*v[0]+(x[1]-left[1])*v[1])/square)) if square else 0.
        projected=(left[0]+t*v[0],left[1]+t*v[1])
        candidates.append((math.dist(x,projected),lo+t*(hi-lo)))
    if not candidates:
        raise ValueError('No local route segment in independent construction')
    distance,coordinate=min(candidates,key=lambda value:value[0])
    return total-coordinate+distance


def physical_progress(initial, final, points, mask, cursor, final_cursor, status, goal, config):
    """Validate actual task arrival before giving an absorbing goal its value."""
    initial=np.asarray(initial,float); final=np.asarray(final,float); status=np.asarray(status)
    at_goal=(np.linalg.norm(final[...,:2]-goal,axis=-1)<=config['goal_tolerance'])&(final[...,3]<=config['terminal_speed'])
    if np.any((status==1)&~at_goal):
        raise ValueError('Recorded GOAL does not satisfy unchanged physical task')
    if not np.allclose(np.asarray(points)[np.asarray(mask,bool)][-1],goal,atol=1e-6,rtol=0):
        raise ValueError('Route endpoint differs from physical task goal')
    start=route_potential(initial[:2],points,mask,cursor)
    end=route_potential(final[...,:2],points,mask,final_cursor)
    end=np.where(status==1,0.,end)
    return start-end
