"""Shared detour implementation."""

import jax

import jax.numpy as jnp

from .routing import route_geometry

def segment_gap(starts,ends,obstacles,mask,radii):
    delta=ends-starts;relative=obstacles[:,:2]-starts[...,None,:]
    fraction=jnp.clip(jnp.sum(relative*delta[...,None,:],axis=-1)/jnp.maximum(jnp.sum(delta**2,axis=-1)[...,None],1e-12),0.,1.)
    closest=starts[...,None,:]+fraction[...,None]*delta[...,None,:]
    gap=jnp.linalg.norm(closest-obstacles[:,:2],axis=-1)-radii
    return jnp.min(jnp.where(mask,gap,jnp.inf),axis=-1)

def local_detour(x,target,points,route_mask,cursor,obstacles,mask,config,clearance_credit=0.):
    """Return a current-center two-segment nominal target and diagnostics.

    No physical or route-progress state is committed by this function. A failed
    geometric proposal retains the original reference. The caller still checks
    observed moving obstacles in its bounded motion preview and actual CBF-QP.
    """
    radii=obstacles[:,2]+config.radius+config.clearance_buffer
    slack=jnp.min(jnp.where(mask,jnp.linalg.norm(x[:2]-obstacles[:,:2],axis=-1)-radii,jnp.inf))
    padding=jnp.minimum(jnp.maximum(.2-clearance_credit,0.),jnp.maximum(slack-.02,0.))
    padded=radii+padding
    original_gap=segment_gap(x[:2],target,obstacles,mask,padded)
    _,valid,lengths,arc=route_geometry(points,route_mask)
    coordinates=jnp.minimum(cursor+jnp.asarray([.75,1.5,3.,5.],x.dtype),arc[-1])
    def interpolate(s):
        index=jnp.argmax(valid&(arc[1:]>=s-1e-6))
        fraction=jnp.clip((s-arc[index])/jnp.maximum(lengths[index],1e-12),0.,1.)
        return points[index]+fraction*(points[index+1]-points[index])
    anchors=jax.vmap(interpolate)(coordinates)
    # Choose candidate disks by closeness to the current-to-farthest-anchor
    # segment. This limits graph nodes only, never collision/CBF obstacle rows.
    delta=anchors[-1]-x[:2]
    fraction=jnp.clip(jnp.sum((obstacles[:,:2]-x[:2])*delta,axis=-1)/jnp.maximum(jnp.sum(delta**2),1e-12),0.,1.)
    distance=jnp.linalg.norm(obstacles[:,:2]-(x[:2]+fraction[:,None]*delta),axis=-1)-padded
    # Geometric tie-breaks keep node selection independent of obstacle order.
    # Frame coordinates rotate with the route; identical geometry yields
    # identical nodes even when velocity/radius records are permuted.
    relative=obstacles[:,:2]-x[:2]
    along=jnp.sum(relative*delta,axis=-1)
    lateral=relative[:,0]*delta[1]-relative[:,1]*delta[0]
    indices=jnp.lexsort((lateral,along,jnp.sum(relative**2,axis=-1),jnp.where(mask,distance,jnp.inf)))[:min(8,len(mask))]
    bearing=jnp.arctan2(delta[1],delta[0])
    angles=bearing+jnp.arange(8,dtype=x.dtype)*(2*jnp.pi/8)
    directions=jnp.stack((jnp.cos(angles),jnp.sin(angles)),axis=-1)
    # An outer ring permits a two-segment bend around a convex disk; one
    # circumscribed ring alone can require two intermediate polygon vertices.
    rings=jnp.asarray([1.,1.8],x.dtype)
    nodes=(obstacles[indices,None,None,:2]+(padded[indices,None,None,None]/jnp.cos(jnp.pi/8)*rings[None,:,None,None]+.02)*directions[None,None]).reshape(-1,2)
    node_mask=jnp.repeat(mask[indices],16)
    first_gap=segment_gap(x[:2],nodes,obstacles,mask,padded)
    second_gap=segment_gap(nodes[:,None,:],anchors[None,:,:],obstacles,mask,padded)
    usable=node_mask[:,None]&(first_gap[:,None]>=0)&(second_gap>=0)&(coordinates[None,:]>cursor+.05)
    first_length=jnp.linalg.norm(nodes-x[:2],axis=-1)
    length=first_length[:,None]+jnp.linalg.norm(nodes[:,None,:]-anchors[None,:,:],axis=-1)
    # Length-to-reconnect plus remaining route, with a small turn preference.
    # No positive reward for assigning an unvisited future route coordinate.
    angles_to_nodes=jnp.arctan2(nodes[:,1]-x[1],nodes[:,0]-x[0])-x[2]
    turn=jnp.arctan2(jnp.sin(angles_to_nodes),jnp.cos(angles_to_nodes))
    cost=length+arc[-1]-coordinates[None,:]+.1*jnp.abs(turn[:,None])
    index=jnp.argmin(jnp.where(usable,cost,jnp.inf));node_index=index//len(anchors);anchor_index=index%len(anchors)
    # Direct reconnection is also allowed if every disk clears the segment.
    direct_gap=segment_gap(x[:2],anchors,obstacles,mask,padded)
    direct_cost=jnp.linalg.norm(anchors-x[:2],axis=-1)+arc[-1]-coordinates
    direct_valid=(direct_gap>=0)&(coordinates>cursor+.05)
    direct_index=jnp.argmin(jnp.where(direct_valid,direct_cost,jnp.inf))
    direct=jnp.any(direct_valid)&(direct_cost[direct_index]<=jnp.where(jnp.any(usable),cost[node_index,anchor_index],jnp.inf))
    proposed=jnp.where(direct,anchors[direct_index],nodes[node_index])
    available=jnp.any(usable)|jnp.any(direct_valid)
    changed=(original_gap<0)&available&(slack>0)&(jnp.linalg.norm(proposed-x[:2])>.05)
    return jnp.where(changed,proposed,target),dict(changed=changed,available=available,original_gap=original_gap,
        selected_gap=jnp.where(direct,direct_gap[direct_index],jnp.minimum(first_gap[node_index],second_gap[node_index,anchor_index])),
        reconnect=jnp.where(direct,anchors[direct_index],anchors[anchor_index]),padding=padding)

def detour_reference(x,goal,reference,target,points,route_mask,cursor,obstacles,mask,config,clearance_credit=0.):
    chosen,info=local_detour(x,target,points,route_mask,cursor,obstacles,mask,config,clearance_credit)
    delta=chosen-x[:2];bearing=jnp.arctan2(delta[1],delta[0])-x[2]
    error=jnp.arctan2(jnp.sin(bearing),jnp.cos(bearing))
    curvature=2*jnp.sin(error)/jnp.maximum(jnp.linalg.norm(delta),.15)
    corner=.8*config.w_max/jnp.maximum(jnp.abs(curvature),.01)
    distance=jnp.maximum(jnp.linalg.norm(goal-x[:2])-.1,0.)
    speed=jnp.minimum(config.v_max,jnp.minimum(corner,jnp.minimum(1.2*distance,jnp.sqrt(2*config.a_max*distance))))*jnp.maximum(jnp.cos(error),0.)
    command=jnp.stack((2*(speed-x[3]),2*error))
    return jnp.where(info['changed'],command,reference),chosen
