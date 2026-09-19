"""OA map guidance with explicit bounded-velocity observation uncertainty.

An obstacle whose measured velocity disk contains zero can be stationary. Its
current observed disk participates in geometric routing. This is guidance, not
a motion classification certificate; original measurements stay in every QP.
"""
import numpy as np
from .bicycle_control import BicycleControlConfig
from .routing import plan_route

SCHEMA='bicycle_stationary_compatible_observed_map_v76'


def stationary_compatible(mask,obstacles,noise):
    obs=np.asarray(obstacles,float);mask=np.asarray(mask,bool);noise=np.asarray(noise,float)
    if (obs.ndim!=2 or obs.shape[1]!=5 or mask.shape!=obs.shape[:1] or noise.shape!=(6,)
            or not np.isfinite(obs).all() or not np.isfinite(noise).all() or np.any(noise<0)):
        raise ValueError('Finite observed obstacles and six nonnegative noise ranges required')
    # Initial acquisitions have no innovation; retain the full current support
    # so the same function also accepts a real later observed map. No true
    # velocity, family or identity is supplied to this decision.
    bound=1.15*noise[4]+(1e-7 if noise[4]>0 else 0.)
    return mask&((np.linalg.norm(obs[:,3:5],axis=1)<=bound)|(np.linalg.norm(obs[:,3:5],axis=1)<1e-10))


def plan_observed_bicycle_route(state,goal,obstacles,mask,noise,config=BicycleControlConfig(),**options):
    original=np.asarray(obstacles);planning=np.array(original,dtype=float,copy=True)
    included=stationary_compatible(mask,planning,noise)
    # Only the private planning copy changes; sensed velocities remain intact
    # for graph features, physical prediction and every moving-obstacle CBF row.
    planning[included,3:5]=0.
    route=plan_route(np.asarray(state)[:2],goal,planning,mask,config,**options)
    return route,included
