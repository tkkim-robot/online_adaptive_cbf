"""Observed six-state world-vertical planar flight graph; schema40, not unicycle."""
import jax.numpy as jnp
from .quad2d_control import FlightConfig
from .routing import route_target_from_position

SCHEMA='quad2d_route_observed_40_v1'


def flight_graph(x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise,config=FlightConfig()):
    c=config.robot;n=len(obstacles)+2
    positions=jnp.concatenate((x[None,:2],goal[None],obstacles[:,:2]))-x[:2]
    velocity=x[3:5]
    velocities=jnp.concatenate((velocity[None],jnp.zeros((1,2),x.dtype),obstacles[:,3:5]))-velocity
    radii=jnp.concatenate((jnp.array([c.radius,0.],x.dtype),obstacles[:,2]))
    clearance=jnp.sqrt(jnp.sum(positions**2,axis=-1)+1e-12)-c.radius-radii
    types=jnp.concatenate((jnp.eye(3,dtype=x.dtype)[jnp.array([0,2])],jnp.tile(jnp.array([0.,1.,0.],x.dtype),(n-2,1))))
    ego=jnp.array([jnp.sin(x[2]),jnp.cos(x[2]),x[3]/config.velocity_limit,x[4]/config.velocity_limit,x[5]/config.pitch_rate_limit,
        c.mass,c.inertia/.05,c.arm,c.force_min/10,c.force_max/10,c.gravity/9.81,c.dt,config.pitch_limit,config.pitch_rate_limit,config.velocity_limit],x.dtype)
    target,_,remaining=route_target_from_position(x[:2],jnp.linalg.norm(velocity),points,route_mask,cursor)
    context=jnp.concatenate(((goal-x[:2])/5,(target-x[:2])/5,jnp.array([remaining/10],x.dtype),
        (2*previous_control-c.force_max-c.force_min)/(c.force_max-c.force_min),jnp.log(previous_gain),noise))
    features=jnp.concatenate((types,positions/5,velocities/config.velocity_limit,radii[:,None],clearance[:,None]/3,
        jnp.broadcast_to(ego,(n,len(ego))),jnp.broadcast_to(context,(n,len(context)))),axis=-1)
    node_mask=jnp.concatenate((jnp.ones(2,bool),mask))
    return jnp.where(node_mask[:,None],features,0.),node_mask
