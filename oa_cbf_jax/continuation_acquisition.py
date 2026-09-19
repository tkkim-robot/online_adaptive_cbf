"""Extract actual simulator state and causal observer memory at one visited tick."""
import jax
import jax.numpy as jnp
from .motion_observer import MotionState,initialize,replay,effective_noise
from .continuation import Snapshot
from .position_observer import PositionState,initialize as position_initialize,replay as position_replay


def make_extractor(robot,window,filter_obstacle_position=False):
    if filter_obstacle_position and not window:raise ValueError('Position filtering requires velocity intervals')
    def one(trace,truth,snapshot,mask,noise):
        raw=trace['raw_observed_obstacles']
        if window:
            _,observed=replay(raw,mask,noise,robot.dt,window)
            estimated=observed['obstacles'][snapshot]
            feature_noise=effective_noise(noise,observed['bound'][snapshot],mask)
            index=jnp.maximum(snapshot-1,0)
            previous=MotionState(observed['current_anchor'][index],observed['previous_anchor'][index],observed['ticks'][index])
            observer=jax.tree.map(lambda a,b:jnp.where(snapshot>0,a,b),previous,initialize(raw[0]))
        else:
            estimated=raw[snapshot];feature_noise=noise;observer=initialize(raw[snapshot])
        position=None
        if filter_obstacle_position:
            history=observed['obstacles']
            velocity_error=1.15*(jnp.max(observed['bound'],axis=(1,2))/1.15)
            _,(filtered,width,_)=position_replay(history,history[:,:,3:5],velocity_error[:,None,None],mask,noise,robot.dt)
            previous_position=PositionState(filtered[index,:,:2],width[index],mask)
            position=jax.tree.map(lambda a,b:jnp.where(snapshot>0,a,b),previous_position,position_initialize(raw[0]))
            estimated=filtered[snapshot]
        physical_x=jnp.where(snapshot>0,trace['state'][jnp.maximum(snapshot-1,0)],truth['initial_state'])
        physical_obs=truth['obstacles'].at[:,:2].add(snapshot*robot.dt*truth['obstacles'][:,3:5])
        saved=Snapshot(physical_x,physical_obs,truth['x_bias'],truth['obs_bias'],trace['observed_state'][snapshot],raw[snapshot],observer,position)
        return saved,estimated,feature_noise
    return jax.jit(jax.vmap(one))
