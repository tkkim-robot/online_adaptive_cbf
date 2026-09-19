"""Bounded observed-map pose search for a forward bicycle escape turn.

This geometric guide accounts for initial heading and bounded slip. It does
not certify CBF feasibility or unknown true obstacle motion. A failed bounded
search is unknown, never proof of physical infeasibility.
"""
from dataclasses import dataclass
import heapq
import math
import time
import numpy as np

from .bicycle_control import BicycleControlConfig
from .bicycle_observed_routing import stationary_compatible
from .routing import Route, segment_clearances


@dataclass(frozen=True)
class PoseRouteConfig:
    step: float = .2
    xy_resolution: float = .08
    heading_bins: int = 48
    samples: int = 8
    max_expansions: int = 20000
    target_tolerance: float = .18
    heading_tolerance: float = math.pi/8

    def __post_init__(self):
        if any(not np.isfinite(v) or v<=0 for v in (self.step,self.xy_resolution,self.target_tolerance,self.heading_tolerance)):
            raise ValueError('Pose search distances and tolerances must be positive and finite')
        if any(type(v) is not int or v<1 for v in (self.heading_bins,self.samples,self.max_expansions)):
            raise ValueError('Pose search counts must be positive integers')


def spatial_primitive(pose,slips,distances,rear_axle):
    """Exact constant-slip motion, parameterized by integral of forward speed."""
    pose=np.asarray(pose,float);slips=np.asarray(slips,float)
    distance=np.asarray(distances,float)
    phase=slips[:,None]*distance[None,:]/rear_axle
    middle=pose[2]+phase/2
    scale=distance[None,:]*np.sinc(phase/(2*np.pi))
    x=pose[0]+scale*(np.cos(middle)-slips[:,None]*np.sin(middle))
    y=pose[1]+scale*(np.sin(middle)+slips[:,None]*np.cos(middle))
    return np.stack((x,y,pose[2]+phase),axis=-1)


def primitive_clearances(pose,paths,slips,centers,radii,config,search):
    if not len(radii):return np.full(len(paths),np.inf)
    starts=np.concatenate((np.broadcast_to(pose[:2],(len(paths),1,2)),paths[:,:-1,:2]),axis=1)
    clear=segment_clearances(starts,paths[:,:,:2],centers,radii).min(axis=(1,2))
    # A curved primitive can deviate from a subsegment chord by this exact
    # circular sagitta. Subtracting it makes disk clearance conservative.
    beta=np.abs(slips);turn_radius=config.robot.rear_axle_distance*np.sqrt(1+beta**2)/np.maximum(beta,1e-12)
    half_angle=beta*search.step/(2*config.robot.rear_axle_distance*search.samples)
    sagitta=np.where(beta>0,2*turn_radius*np.sin(half_angle/2)**2,0.)
    return clear-sagitta


def search_pose_path(state,target,target_heading,obstacles,mask,noise,
                     config=BicycleControlConfig(),search=PoseRouteConfig()):
    start_time=time.monotonic()
    state=np.asarray(state,float);target=np.asarray(target,float);obstacles=np.asarray(obstacles,float)
    included=stationary_compatible(mask,obstacles,noise)
    # Work in the initial body frame so the discretization itself does not
    # encode a preferred global orientation or origin.
    theta=state[2];rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    goal=(target-state[:2])@rotation
    centers=(obstacles[included,:2]-state[:2])@rotation
    radii=(obstacles[included,2]+config.radius+config.clearance_buffer)*config.barrier_inflation
    yaw=target_heading-theta
    slips=np.array([-1.,-.5,0.,.5,1.])*config.robot.slip_max
    distances=np.linspace(search.step/search.samples,search.step,search.samples)
    def finish(status,poses,controls,expanded):
        poses=np.asarray(poses,float).reshape(-1,3)
        if len(poses):
            poses[:,:2]=poses[:,:2]@rotation.T+state[:2];poses[:,2]+=theta
        return dict(status=status,poses=poses,slips=np.asarray(controls,float),step=search.step,
                    expanded=expanded,seconds=time.monotonic()-start_time,included_mask=included,
                    certified_physical_feasible=False)
    if len(radii) and np.min(np.linalg.norm(centers,axis=1)-radii)<=0:
        return finish('initial_outside_geometric_domain',[],[],0)
    if len(radii) and np.min(np.linalg.norm(centers-goal,axis=1)-radii)<=0:
        return finish('target_outside_geometric_domain',[],[],0)
    def key(p):
        return (int(np.rint(p[0]/search.xy_resolution)),int(np.rint(p[1]/search.xy_resolution)),
                int(np.rint(p[2]*search.heading_bins/(2*np.pi)))%search.heading_bins)
    # Every node has its own immutable parent; improving a cell must not alter
    # a descendant's recorded control sequence or reconstructed geometry.
    nodes=[(np.zeros(3),-1,0.,0.)];best={key(nodes[0][0]):0.}
    queue=[(float(np.linalg.norm(goal)),0.,0)];expanded=0
    while queue and expanded<search.max_expansions:
        _,cost,index=heapq.heappop(queue);pose,_,_,_=nodes[index]
        if cost>best.get(key(pose),np.inf)+1e-12:continue
        distance=np.linalg.norm(pose[:2]-goal)
        angle=abs(math.atan2(math.sin(pose[2]-yaw),math.cos(pose[2]-yaw)))
        if distance<=search.target_tolerance and angle<=search.heading_tolerance:
            chain=[];controls=[]
            while index>=0:
                p,parent,beta,_=nodes[index];chain.append(p)
                if parent>=0:controls.append(beta)
                index=parent
            return finish('ready',chain[::-1],controls[::-1],expanded)
        expanded+=1
        paths=spatial_primitive(pose,slips,distances,config.robot.rear_axle_distance)
        clearance=primitive_clearances(pose,paths,slips,centers,radii,config,search)
        for i in np.flatnonzero(clearance>1e-8):
            end=paths[i,-1];new_cost=cost+search.step*np.sqrt(1+slips[i]**2);cell=key(end)
            if new_cost+1e-12>=best.get(cell,np.inf):continue
            best[cell]=new_cost;nodes.append((end,index,slips[i],new_cost))
            heapq.heappush(queue,(new_cost+float(np.linalg.norm(end[:2]-goal)),new_cost,len(nodes)-1))
    return finish('search_budget_exhausted' if queue else 'discretized_frontier_exhausted',[],[],expanded)


def escape_route(state,obstacles,mask,noise,route,config=BicycleControlConfig(),search=PoseRouteConfig()):
    """Try a pose-compatible prefix only when the first route edge is behind.

    A failure leaves the old guide intact and is recorded. No parent is dropped.
    The local controller still processes every original measured obstacle.
    """
    points=route.points[route.mask]
    delta=points[1]-state[:2]
    heading=np.array([np.cos(state[2]),np.sin(state[2])])
    if route.status!='ready' or np.dot(delta,heading)>=0:
        return route,dict(status='unchanged_forward_or_unready',expanded=0,seconds=0.,changed=False)
    target_heading=np.arctan2(delta[1],delta[0])
    result=search_pose_path(state,points[1],target_heading,obstacles,mask,noise,config,search)
    result['changed']=False
    if result['status']!='ready':return route,result
    prefix=result['poses'][:,:2]
    proposed=np.concatenate((prefix,points[1:]))
    # Do not silently truncate or shortcut curved paths to fit graph storage.
    if len(proposed)>len(route.points):
        result['status']='route_capacity_exceeded';return route,result
    included=result['included_mask'];obs=np.asarray(obstacles)[included]
    radii=(obs[:,2]+config.radius+config.clearance_buffer)*config.barrier_inflation
    clear=float(segment_clearances(proposed[:-1],proposed[1:],obs[:,:2],radii).min()) if len(obs) else None
    result['minimum_segment_domain_clearance']=clear
    if clear is not None and clear<=0:
        result['status']='connector_or_chord_not_clear';return route,result
    padded=np.broadcast_to(points[-1],route.points.shape).copy();padded[:len(proposed)]=proposed
    value=Route(padded,np.arange(len(padded))<len(proposed),float(np.linalg.norm(np.diff(proposed,axis=0),axis=1).sum()),0.,'ready')
    result['changed']=True
    return value,result
