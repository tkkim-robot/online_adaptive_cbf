"""Residual graph attention with scene encoding reused over queried CBF gains."""

from dataclasses import dataclass
from functools import partial
import math
import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import UnicycleConfig

# Fixed FP32 reference; lower-precision neural variants require their own audit.
Dense = partial(nn.Dense, precision="highest")


@dataclass(frozen=True)
class GATConfig:
    width: int = 64
    layers: int = 2
    heads: int = 4
    continuous_outputs: int = 2  # Named target schema is stored with the bundle.
    event_outputs: int = 2
    encoder: str = 'gat'
    flight_history_invariant: bool = False
    risk_log_variance_min: float = -10.
    scalar_gain_quadratic: bool = False
    compute_dtype: str = 'float32'
    quad3d_history_invariant: bool = False
    paired_gain_quadratic: bool = False
    continuous_log_variance_min: float = -10.
    quad3d_obstacle_pooling: bool = False

    def __post_init__(self):
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads or self.layers < 1:
            raise ValueError("Invalid attention configuration")
        if self.encoder not in ('gat','legacy_fc','full_fc'):
            raise ValueError('Unknown model encoder')
        if self.encoder!='gat' and self.layers!=4:
            raise ValueError('FC baselines use the repository four-hidden-layer architecture')
        if not isinstance(self.flight_history_invariant,bool) or (self.flight_history_invariant and self.encoder!='gat'):
            raise ValueError('Flight history invariance is an explicit OA GAT variant')
        if (not math.isfinite(self.risk_log_variance_min) or not -10.<=self.risk_log_variance_min<=0.
                or (self.encoder!='gat' and self.risk_log_variance_min!=-10.)):
            raise ValueError('Risk variance floor is an explicit bounded OA GAT setting')
        if not isinstance(self.scalar_gain_quadratic,bool) or (self.scalar_gain_quadratic and self.encoder!='gat'):
            raise ValueError('Scalar gain basis is an explicit OA GAT variant')
        if self.compute_dtype not in ('float32','float64') or (self.compute_dtype!='float32' and self.encoder!='gat'):
            raise ValueError('Higher precision is an explicit OA GAT variant')
        if any(not isinstance(v,bool) for v in (self.quad3d_history_invariant,self.paired_gain_quadratic,self.quad3d_obstacle_pooling)):
            raise ValueError('Quad3D feature variants must be explicit booleans')
        if (self.quad3d_history_invariant or self.paired_gain_quadratic or self.quad3d_obstacle_pooling) and self.encoder!='gat':
            raise ValueError('Quad3D feature variants are restricted to OA GAT')
        if self.quad3d_history_invariant and self.flight_history_invariant or self.paired_gain_quadratic and self.scalar_gain_quadratic:
            raise ValueError('Incompatible dynamics feature transforms')
        if not math.isfinite(self.continuous_log_variance_min) or not -10.<=self.continuous_log_variance_min<=0. or self.continuous_log_variance_min!=-10. and self.encoder!='gat':
            raise ValueError('Continuous variance floor is an explicit OA GAT variant')


def unicycle_graph(x,goal,obstacles,obstacle_mask,config=UnicycleConfig()):
    """One observed scene -> nodes [M+2,18], mask. No gain-dependent encoding.

    World axes retained here with explicit orientation; translation invariant.
    Caller supplies observed obstacle velocities, never privileged future noise.
    """
    n=obstacles.shape[0]+2
    positions=jnp.concatenate((x[None,:2],goal[None,:],obstacles[:,:2]))-x[:2]
    velocity=x[3]*jnp.stack((jnp.cos(x[2]),jnp.sin(x[2])))
    velocities=jnp.concatenate((velocity[None,:],jnp.zeros((1,2),x.dtype),obstacles[:,3:5]))-velocity
    radii=jnp.concatenate((jnp.asarray([config.radius,0.],x.dtype),obstacles[:,2]))
    distance=jnp.sqrt(jnp.sum(positions**2,axis=-1)+1e-12)
    clearance=distance-config.radius-radii
    types=jnp.concatenate((jnp.eye(3,dtype=x.dtype)[jnp.array([0,2])],jnp.broadcast_to(jnp.array([0.,1.,0.],x.dtype),(n-2,3))))
    ego=jnp.asarray([jnp.sin(x[2]),jnp.cos(x[2]),x[3]/config.v_max,config.a_max,config.w_max,config.v_max,config.dt],x.dtype)
    features=jnp.concatenate((types,positions/5,velocities/config.v_max,radii[:,None],clearance[:,None]/3,
                              jnp.broadcast_to(ego,(n,7)),jnp.broadcast_to((goal-x[:2])/5,(n,2))),axis=-1)
    mask=jnp.concatenate((jnp.ones(2,bool),obstacle_mask))
    return jnp.where(mask[:,None],features,0.),mask


def route_graph(x,goal,obstacles,obstacle_mask,points,route_mask,progress,previous_control,
                previous_gain,noise,config=UnicycleConfig()):
    """Schema v2: 31 features, adding observable route/controller/sensor context.

    Noise ranges describe the known sensor model; realized latent errors and
    future innovations never enter the network. Old 18-feature bundles cannot
    be used with this graph schema.
    """
    from .routing import route_target
    features,mask=unicycle_graph(x,goal,obstacles,obstacle_mask,config)
    target,_,remaining=route_target(x,points,route_mask,progress,config)
    context=jnp.concatenate(((target-x[:2])/5,jnp.asarray([remaining/10],x.dtype),
                             previous_control/jnp.array([config.a_max,config.w_max]),jnp.log(previous_gain),noise))
    expanded=jnp.concatenate((features,jnp.broadcast_to(context,(features.shape[0],context.shape[0]))),axis=-1)
    return jnp.where(mask[:,None],expanded,0.),mask


class AttentionBlock(nn.Module):
    width:int
    heads:int

    @nn.compact
    def __call__(self,nodes,mask,relative):
        z=nn.LayerNorm()(nodes)
        qkv=Dense(3*self.width)(z)
        q,k,v=jnp.split(qkv,3,axis=-1)
        shape=(*nodes.shape[:-1],self.heads,self.width//self.heads)
        q,k,v=(a.reshape(shape) for a in (q,k,v))
        logits=jnp.einsum('bihd,bjhd->bhij',q,k,precision='highest')/jnp.sqrt(self.width//self.heads)
        edge_bias=Dense(self.heads)(relative).transpose(0,3,1,2)
        logits=logits+edge_bias
        valid=mask[:,None,None,:]
        masked=jnp.where(valid,logits,-1e30)
        exp=jnp.where(valid,jnp.exp(masked-jnp.max(masked,axis=-1,keepdims=True)),0.)
        weights=exp/jnp.maximum(jnp.sum(exp,axis=-1,keepdims=True),1e-12)
        messages=jnp.einsum('bhij,bjhd->bihd',weights,v,precision='highest').reshape(nodes.shape)
        nodes=nodes+Dense(self.width)(messages)
        residual=Dense(2*self.width)(nn.LayerNorm()(nodes))
        residual=Dense(self.width)(nn.gelu(residual))
        return jnp.where(mask[:,:,None],nodes+residual,0.)


class CandidateGAT(nn.Module):
    config:GATConfig=GATConfig()

    def setup(self):
        self.project=Dense(self.config.width)
        self.blocks=[AttentionBlock(self.config.width,self.config.heads) for _ in range(self.config.layers)]
        self.norm=nn.LayerNorm()
        self.head1=Dense(self.config.width)
        self.head2=Dense(self.config.width)
        self.output=Dense(2*self.config.continuous_outputs+self.config.event_outputs)

    def encode(self,features,mask):
        if self.config.compute_dtype=='float64':
            features=features.astype(jnp.float64)
        if self.config.quad3d_history_invariant:
            if features.shape[-1]!=58:raise ValueError('Quad3D history invariance requires the58-column observer graph')
            # The fixed-candidate branch skips the observer update at tick zero,
            # inherits its already updated memory, and subsequently uses only
            # its own applied controls. Old control/gain have no label influence.
            # Keep all observer memory, attitude, velocity and route context.
            features=features.at[...,27:35].set(0.)
        if self.config.flight_history_invariant:
            if features.shape[-1] not in (40,50):raise ValueError('Flight command-history invariance requires an observed flight graph')
            # This label plant has no actuator lag, command-change cost or
            # previous-gain dependence: its future fixed-gain branches are
            # conditioned on the observed state/route/noise and (for graph50)
            # causal obstacle history. That observer history remains intact.
            # Keep true controller history in the graph/artifact for auditing,
            # but enforce the exact target invariance before neural encoding.
            features=features.at[...,29:33].set(0.)
        clean=jnp.where(mask[:,:,None],features,0.)
        nodes=self.project(clean)
        positions=clean[:,:,3:5]
        delta=positions[:,:,None,:]-positions[:,None,:,:]
        relative=jnp.concatenate((delta,jnp.sqrt(jnp.sum(delta**2,axis=-1,keepdims=True)+1e-12)),axis=-1)
        for block in self.blocks:
            nodes=block(nodes,mask,relative)
        if not self.config.quad3d_obstacle_pooling:
            return self.norm(nodes[:,0,:])
        if features.shape[-1]!=58:raise ValueError('Quad3D obstacle pooling requires the58-column observer graph')
        # Retain ego attention and expose obstacle extrema/means directly to
        # the candidate head. Raw observed context bypasses the attention
        # bottleneck; no physical future, target or gain search enters here.
        z=self.norm(nodes);om=mask[:,2:];ob=z[:,2:]
        count=jnp.maximum(om.sum(-1,keepdims=True),1)
        average=jnp.where(om[...,None],ob,0.).sum(1)/count
        maximum=jnp.max(jnp.where(om[...,None],ob,-jnp.inf),axis=1)
        maximum=jnp.where(om.any(-1,keepdims=True),maximum,0.)
        clearance=clean[:,2:,8]
        nearest=jnp.min(jnp.where(om,clearance,jnp.inf),axis=1,keepdims=True)
        nearest=jnp.where(om.any(-1,keepdims=True),nearest,0.)
        # Smooth proximity pooling avoids order-dependent ties. Column8 is
        # clearance/3m, so this has a fixed3m length scale, not a fitted label.
        weights=jnp.where(om,jnp.exp(-jnp.maximum(clearance-nearest,0.)),0.)
        weights=weights/jnp.maximum(weights.sum(-1,keepdims=True),1e-12)
        geometry=jnp.sum(weights[...,None]*clean[:,2:,3:9],axis=1)
        return jnp.concatenate((z[:,0,:],average,maximum,clean[:,0,:],geometry,nearest),axis=-1)

    def score(self,context,gains):
        """context [B,W], gains [B,K,D]; one scalar or a declared gain pair."""
        broadcast=jnp.broadcast_to(context[:,None,:],(*gains.shape[:2],context.shape[-1]))
        log_gain=jnp.log(jnp.maximum(gains,1e-6))
        if self.config.paired_gain_quadratic:
            if gains.shape[-1]!=4:raise ValueError('Paired gain basis requires four Quad3D gains')
            # Keep all four coordinates so unequal pairs cannot silently alias.
            # Center/scale the registered2..8 bank at4 in log space.
            coordinate=(log_gain-math.log(4.))/math.log(2.)
            log_gain=jnp.concatenate((coordinate,coordinate**2,coordinate[...,:2]*coordinate[...,2:]),axis=-1)
        if self.config.scalar_gain_quadratic:
            if gains.shape[-1]!=1:raise ValueError('Quadratic gain basis requires one scalar gain')
            # Fixed log-domain coordinates for the audited bicycle .5..8 bank.
            # Exposes curvature to the learned head without fitting a label
            # polynomial or introducing any physical rollout in inference.
            coordinate=(log_gain-math.log(2.))/math.log(4.)
            log_gain=jnp.concatenate((coordinate,coordinate**2),axis=-1)
        z=jnp.concatenate((broadcast,log_gain),axis=-1)
        z=nn.gelu(self.head1(z));z=z+nn.gelu(self.head2(z))
        out=self.output(z);d=self.config.continuous_outputs
        variance=jnp.clip(out[...,d:2*d],self.config.continuous_log_variance_min,3.)
        if self.config.risk_log_variance_min!=-10.:
            # Normalized risk head only. The default path and all FC outputs
            # remain exact; a changed floor requires a new trained/calibrated
            # bundle, but does not change the physical target or control law.
            variance=variance.at[...,0].set(jnp.clip(out[...,d],self.config.risk_log_variance_min,3.))
        return dict(mean=out[...,:d],log_variance=variance,event_logits=out[...,2*d:])

    def __call__(self,features,mask,gains):
        return self.score(self.encode(features,mask),gains)


class CandidateFC(nn.Module):
    """Repository FC/PENN family: ReLU widths W,2W,3W,W, Gaussian outputs.

legacy_fc observes nearest signed clearance, speed and relative bearing as in
the original unicycle FC. full_fc is a stronger expanded-input comparator with
all current graph/route/noise information, sorted and flattened without GAT.
Both predict the current branch targets and use the same ensemble/calibration/
selection framework. Neither is BarrierNet or a direct-gain imitation policy.
"""
    config:GATConfig

    def setup(self):
        self.hidden=[Dense(self.config.width*factor) for factor in (1,2,3,1)]
        self.output=Dense(2*self.config.continuous_outputs+self.config.event_outputs)

    def encode(self,features,mask):
        clean=jnp.where(mask[...,None],features,0.)
        obstacles=clean[:,2:];obstacle_mask=mask[:,2:]
        if self.config.encoder=='legacy_fc':
            index=jnp.argmin(jnp.where(obstacle_mask,obstacles[:,:,8],jnp.inf),axis=1)
            nearest=obstacles[jnp.arange(len(obstacles)),index]
            bearing=jnp.arctan2(nearest[:,4],nearest[:,3])-jnp.arctan2(clean[:,0,9],clean[:,0,10])
            valid=jnp.any(obstacle_mask,axis=1)
            # Fixed physical scaling replaces the old fitted StandardScaler;
            # angular sin/cos and the nearest-obstacle information are retained.
            return jnp.stack((jnp.where(valid,nearest[:,8],100./3),clean[:,0,11],
                              jnp.where(valid,jnp.sin(bearing),0.),jnp.where(valid,jnp.cos(bearing),1.)),axis=-1)
        # Deterministic obstacle ordering, including geometric tie breakers.
        # All active obstacles are retained. This FC explicitly has a fixed
        # input capacity; an unwarmed/different capacity must not be truncated.
        keys=(obstacles[:,:,6],obstacles[:,:,5],obstacles[:,:,7],obstacles[:,:,4],obstacles[:,:,3],
              jnp.where(obstacle_mask,obstacles[:,:,8],jnp.inf))
        order=jnp.lexsort(keys,axis=1)
        ordered=jnp.take_along_axis(obstacles,order[...,None],axis=1)
        return jnp.concatenate((clean[:,:2],ordered),axis=1).reshape(len(clean),-1)

    def score(self,context,gains):
        context=jnp.broadcast_to(context[:,None,:],(*gains.shape[:2],context.shape[-1]))
        values=jnp.concatenate((context,jnp.log(jnp.maximum(gains,1e-6))),axis=-1)
        for layer in self.hidden:values=nn.relu(layer(values))
        output=self.output(values);d=self.config.continuous_outputs
        return dict(mean=output[...,:d],log_variance=jnp.clip(output[...,d:2*d],-10.,3.),event_logits=output[...,2*d:])

    def __call__(self,features,mask,gains):
        return self.score(self.encode(features,mask),gains)


def make_model(config):
    return CandidateGAT(config) if config.encoder=='gat' else CandidateFC(config)


def initialize_ensemble(model,key,features,mask,gains,members=4):
    """Independent entire encoders and heads, not just independent last layers."""
    keys=jax.random.split(key,members)
    return jax.vmap(lambda k:model.init(k,features,mask,gains)['params'])(keys)


def predict_ensemble(model,params,features,mask,gains):
    if model.config.encoder=='full_fc':
        # The first learned matrix fixes the flattened observation width. Pad
        # neural inputs only: resizing physical obstacles would change the
        # simulator's PRNG draws and invalidate paired comparisons. This is
        # static shape handling at compilation, never truncation or selection.
        width=params['hidden_0']['kernel'].shape[-2]-gains.shape[-1]
        nodes,remainder=divmod(width,features.shape[-1])
        if remainder or nodes<2:raise ValueError('Invalid full-FC learned input width')
        if features.shape[1]>nodes:
            raise ValueError('Full-FC observation exceeds trained capacity; truncation is forbidden')
        extra=nodes-features.shape[1]
        if extra:
            features=jnp.pad(features,((0,0),(0,extra),(0,0)))
            mask=jnp.pad(mask,((0,0),(0,extra)),constant_values=False)
    out=jax.vmap(lambda p:model.apply({'params':p},features,mask,gains))(params)
    # The versioned inference variant promotes the authentic parameters once
    # at load. Return raw FP32 heads before the unchanged statistical selector;
    # gain-bank coordinates retain their original FP32 representation.
    if model.config.compute_dtype=='float64':
        out=jax.tree.map(lambda v:v.astype(jnp.float32),out)
    return out
