"""35-feature ego-heading-frame graph from observable bicycle context only."""
import jax.numpy as jnp
from .bicycle_control import BicycleControlConfig
from .routing import route_target_from_position

SCHEMA='bicycle_ego_route_observed35_scalar_gain_v67'


def bicycle_inference_graph(*args,config=BicycleControlConfig(),compute_dtype='float32'):
    """Same observed graph, with explicit, bundle-bound numerical precision."""
    if compute_dtype not in ('float32','float64'):raise ValueError('Unknown graph precision')
    if compute_dtype=='float64':
        args=tuple(jnp.asarray(v,dtype=bool if i in (3,5) else jnp.float64) for i,v in enumerate(args))
    return bicycle_graph(*args,config=config)


def bicycle_graph(x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise,config=BicycleControlConfig()):
    c=config.robot;n=len(obstacles)+2
    rot=jnp.stack((jnp.stack((jnp.cos(x[2]),-jnp.sin(x[2]))),jnp.stack((jnp.sin(x[2]),jnp.cos(x[2])))))
    positions=jnp.matmul(jnp.concatenate((x[None,:2],goal[None],obstacles[:,:2]))-x[:2],rot,precision='highest')
    velocity=x[3]*jnp.stack((jnp.cos(x[2]),jnp.sin(x[2])))
    velocities=jnp.matmul(jnp.concatenate((velocity[None],jnp.zeros((1,2),x.dtype),obstacles[:,3:5]))-velocity,rot,precision='highest')
    radii=jnp.concatenate((jnp.array([c.radius,0.],x.dtype),obstacles[:,2]))
    clearance=jnp.sqrt(jnp.sum(positions**2,axis=1)+1e-12)-c.radius-radii
    types=jnp.concatenate((jnp.eye(3,dtype=x.dtype)[jnp.array([0,2])],jnp.tile(jnp.array([0.,1.,0.],x.dtype),(n-2,1))))
    ego=jnp.array([x[3]/c.speed_max,c.radius,c.wheel_base,c.rear_axle_distance,c.acceleration_max,c.slip_max,c.speed_min,c.speed_max,c.dt,
        config.clearance_buffer,config.barrier_inflation,config.relative_speed_epsilon],x.dtype)
    target,_,remaining=route_target_from_position(x[:2],x[3],points,route_mask,cursor)
    context=jnp.concatenate((jnp.matmul(goal-x[:2],rot,precision='highest')/5,jnp.matmul(target-x[:2],rot,precision='highest')/5,jnp.array([remaining/10],x.dtype),
        previous_control/jnp.array([c.acceleration_max,c.slip_max],x.dtype),jnp.reshape(jnp.log(previous_gain),(1,)),noise))
    features=jnp.concatenate((types,positions/5,velocities/c.speed_max,radii[:,None],clearance[:,None]/3,jnp.broadcast_to(ego,(n,12)),jnp.broadcast_to(context,(n,14))),axis=1)
    node_mask=jnp.concatenate((jnp.ones(2,bool),mask));return jnp.where(node_mask[:,None],features,0.),node_mask
