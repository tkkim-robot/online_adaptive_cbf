"""Observed CBF coefficients for an optional local-unicycle learning study.

These are deterministic current-state features, not feasibility decisions or
future trajectories. The actual controller continues to assemble its own rows.
The FC variant exposes coefficients of only its single nearest obstacle.
"""
import jax.numpy as jnp


def contract():
    return dict(schema='local_unicycle_observed_coefficients_v1', graph_columns=18,
        clearance_buffer=.05, node_features=['h/25', 'h_dot/10', 'Lf2h/8',
            'acceleration_authority/10', 'turn_authority/20'],
        gain_features=['log_alpha1', 'log_alpha2', 'alpha1/4',
            '(alpha1+alpha2)/8', 'alpha1*alpha2/16'],
        physical_rows='Static circular obstacles, quadratic h; derivatives use current observed positions and velocity.',
        observations='GAT: every masked local obstacle. FC: its original single nearest obstacle, including that obstacle radius.',
        extra_fc_information='Radius of the same observed nearest obstacle; no second obstacle, goal, history or pooled statistic.',
        safety='Neural features only; no online solve, guarantee, rollout or changed physical controller.')


def coefficients(features, mask):
    """[B,N,18] -> five coefficients per obstacle; ego/goal/padding are zero."""
    if features.shape[-1] != 18 or features.shape[:-1] != mask.shape:
        raise ValueError('Observed coefficients require masked local graph18')
    clean=jnp.where(mask[...,None],features,0.)
    ego=clean[:,0]
    position=5.*clean[...,3:5]  # obstacle minus robot
    relative_velocity=clean[...,5:7]*ego[:,None,14:15]
    direction=jnp.stack((ego[:,10],ego[:,9]),-1)
    normal=jnp.stack((-ego[:,9],ego[:,10]),-1)
    speed=ego[:,11]*ego[:,14]
    radius=clean[...,7]+ego[:,None,7]+.05
    h=jnp.sum(position**2,-1)-radius**2
    hdot=2.*jnp.sum(position*relative_velocity,-1)
    drift=2.*jnp.sum(relative_velocity**2,-1)
    acceleration=-2.*jnp.sum(position*direction[:,None],-1)
    turning=-2.*speed[:,None]*jnp.sum(position*normal[:,None],-1)
    value=jnp.stack((h/25.,hdot/10.,drift/8.,acceleration/10.,turning/20.),-1)
    valid=mask&(jnp.arange(mask.shape[-1])[None,:]>=2)
    return jnp.where(valid[...,None],value,0.)


def nearest_coefficients(features, mask):
    """Identical geometric nearest/tie rule to the paper FC feature encoder."""
    values=coefficients(features,mask)[:,2:]
    clean=jnp.where(mask[...,None],features,0.)[:,2:]
    valid=mask[:,2:]
    clean=jnp.pad(clean,((0,0),(0,1),(0,0)))
    values=jnp.pad(values,((0,0),(0,1),(0,0)))
    valid=jnp.pad(valid,((0,0),(0,1)))
    position=clean[...,3:5].astype(jnp.float64)
    distance2=jnp.sum(position**2,-1)
    order=jnp.lexsort((clean[...,7],clean[...,4],clean[...,3],
        jnp.where(valid,distance2,jnp.inf)),axis=1)
    result=jnp.take_along_axis(values,order[:,:1,None],axis=1)[:,0]
    return jnp.where(valid.any(-1,keepdims=True),result,0.)


def gain_coordinates(gains):
    if gains.shape[-1]!=2:
        raise ValueError('Unicycle coefficient head requires two ordered gains')
    return jnp.concatenate((jnp.log(jnp.maximum(gains,1e-6)),gains[...,:1]/4.,
        gains.sum(-1,keepdims=True)/8.,jnp.prod(gains,axis=-1,keepdims=True)/16.),-1)
