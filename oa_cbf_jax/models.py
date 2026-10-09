"""Residual graph attention with scene encoding reused over queried CBF gains."""

from dataclasses import dataclass

from functools import partial

import math

import flax.linen as nn

import jax

import jax.numpy as jnp

from .config import UnicycleConfig

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
    flight_obstacle_pooling: bool = False
    flight_gain_basis: bool = False
    bicycle_history_invariant: bool = False
    bicycle_obstacle_pooling: bool = False
    bicycle_route_context: bool = False
    bicycle_reserve_auxiliary: bool = False
    bicycle_constraint_features: bool = False
    bicycle_affine_gain: bool = False
    bicycle_motion_history: bool = False
    bicycle_candidate_encoding: bool = False
    nearest_dynamics: str = ''
    nearest_yaw_scale: float = 1.
    flight_local_residual: bool = False
    unicycle_ego_frame: bool = False
    unicycle_constraint_features: bool = False
    flight_constraint_features: bool = False
    flight_gain_attention: bool = False

    def __post_init__(self):
        if type(self.flight_gain_attention) is not bool or (self.flight_gain_attention and
                (self.encoder!='gat' or not self.flight_history_invariant or not self.flight_obstacle_pooling
                 or self.flight_constraint_features or self.flight_local_residual or self.bicycle_history_invariant
                 or self.quad3d_history_invariant or self.unicycle_ego_frame or self.flight_gain_basis)):
            raise ValueError('Gain attention requires the original pooled history-invariant flight GAT')
        if type(self.flight_constraint_features) is not bool or (self.flight_constraint_features and
                (self.encoder != 'gat' or not self.flight_history_invariant
                 or not self.flight_obstacle_pooling
                 or self.flight_local_residual or self.bicycle_history_invariant
                 or self.quad3d_history_invariant or self.unicycle_ego_frame)):
            raise ValueError('Observed flight coefficients require the pooled history-invariant flight GAT')
        if type(self.unicycle_constraint_features) is not bool or (self.unicycle_constraint_features and
                (self.encoder not in ('gat','nearest_fc') or self.compute_dtype!='float32'
                 or (self.encoder=='gat' and not self.unicycle_ego_frame)
                 or (self.encoder=='nearest_fc' and self.nearest_dynamics!='unicycle'))):
            raise ValueError('Observed unicycle coefficients require heading-frame GAT or nearest-unicycle FC')
        if type(self.unicycle_ego_frame) is not bool or (self.unicycle_ego_frame and
                (self.encoder!='gat' or self.compute_dtype!='float32')):
            raise ValueError('Robot-heading unicycle graph is an explicit FP32 GAT variant')
        if type(self.flight_local_residual) is not bool or (self.flight_local_residual and
                (self.encoder!='gat' or not self.flight_history_invariant or
                 self.compute_dtype!='float32' or self.bicycle_candidate_encoding or
                 self.bicycle_history_invariant or self.quad3d_history_invariant)):
            raise ValueError('Local residual flight model requires FP32 graph40 GAT and flight history invariance')
        if self.encoder == 'nearest_fc':
            from .nearest_fc import contract
            contract(self.nearest_dynamics, self.nearest_yaw_scale)
        elif self.nearest_dynamics or self.nearest_yaw_scale != 1.:
            raise ValueError('Nearest-FC feature settings require its explicit encoder')
        if (not isinstance(self.bicycle_candidate_encoding,bool) or self.bicycle_candidate_encoding and
                (not self.bicycle_constraint_features or self.bicycle_affine_gain or self.bicycle_motion_history)):
            raise ValueError('Candidate encoding requires the matched bicycle current-constraint graph35 treatment')
        if (not isinstance(self.bicycle_motion_history,bool) or self.bicycle_motion_history and
                (not self.bicycle_constraint_features or self.bicycle_affine_gain)):
            raise ValueError("Motion history requires the original matched bicycle constraint-feature model")
        if (not isinstance(self.bicycle_affine_gain,bool) or self.bicycle_affine_gain and
                (self.encoder not in ('gat','matched_fc') or not self.bicycle_constraint_features)):
            raise ValueError('Affine gain coordinates require matched bicycle constraint-feature encoders')
        if (not isinstance(self.bicycle_constraint_features,bool) or self.bicycle_constraint_features and
                (self.encoder not in ('gat','matched_fc') or not self.bicycle_history_invariant
                 or not self.bicycle_obstacle_pooling or not self.scalar_gain_quadratic
                 or self.bicycle_route_context or self.bicycle_reserve_auxiliary)):
            raise ValueError('Current constraint features require matched bicycle graph35 without other feature/auxiliary treatments')
        if (not isinstance(self.bicycle_reserve_auxiliary, bool) or self.bicycle_reserve_auxiliary and
                (self.encoder not in ('gat', 'matched_fc') or not self.bicycle_history_invariant
                 or not self.bicycle_obstacle_pooling or not self.scalar_gain_quadratic or self.bicycle_route_context)):
            raise ValueError('Reserve auxiliary requires matched bicycle graph35 learning')
        if (not isinstance(self.bicycle_route_context, bool) or self.bicycle_route_context and
                (self.encoder not in ('gat', 'matched_fc') or not self.bicycle_history_invariant
                 or not self.bicycle_obstacle_pooling or not self.scalar_gain_quadratic)):
            raise ValueError('Bicycle route context requires its matched history-invariant pooled encoder')
        if (not isinstance(self.bicycle_obstacle_pooling,bool) or self.bicycle_obstacle_pooling and
                (self.encoder not in ('gat','matched_fc') or self.flight_obstacle_pooling
                 or self.quad3d_obstacle_pooling or self.flight_history_invariant
                 or self.quad3d_history_invariant or self.flight_gain_basis or self.paired_gain_quadratic)):
            raise ValueError('Bicycle pooling requires a matched encoder and its declared observed graph')
        if (not isinstance(self.bicycle_history_invariant,bool) or self.bicycle_history_invariant and
                (self.encoder not in ('gat','matched_fc') or self.flight_history_invariant or self.quad3d_history_invariant)):
            raise ValueError('Bicycle history invariance requires its matched observed-state encoder')
        if (not isinstance(self.flight_gain_basis,bool) or self.flight_gain_basis and
                (self.encoder not in ('gat','matched_fc') or self.scalar_gain_quadratic or self.paired_gain_quadratic)):
            raise ValueError('Planar gain basis requires its matched two-gain head')
        if (not isinstance(self.flight_obstacle_pooling,bool)
                or self.flight_obstacle_pooling and (self.encoder not in ('gat','matched_fc') or self.quad3d_obstacle_pooling)):
            raise ValueError('Planar flight pooling requires a matched encoder and its own feature contract')
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads or self.layers < 1:
            raise ValueError("Invalid attention configuration")
        if self.encoder not in ('gat','legacy_fc','full_fc','matched_fc','nearest_fc'):
            raise ValueError('Unknown model encoder')
        if self.encoder in ('legacy_fc','full_fc','nearest_fc') and self.layers!=4:
            raise ValueError('FC baselines use the repository four-hidden-layer architecture')
        if not isinstance(self.flight_history_invariant,bool) or (self.flight_history_invariant and self.encoder not in ('gat','matched_fc')):
            raise ValueError('Flight history invariance is an explicit OA GAT variant')
        if (not math.isfinite(self.risk_log_variance_min) or not -10.<=self.risk_log_variance_min<=0.
                or (self.encoder not in ('gat','matched_fc') and self.risk_log_variance_min!=-10.)):
            raise ValueError('Risk variance floor is an explicit bounded OA GAT setting')
        if not isinstance(self.scalar_gain_quadratic,bool) or (self.scalar_gain_quadratic and self.encoder not in ('gat','matched_fc')):
            raise ValueError('Scalar gain basis is an explicit OA GAT variant')
        if self.compute_dtype not in ('float32','float64') or (self.compute_dtype!='float32' and self.encoder not in ('gat','matched_fc','nearest_fc')):
            raise ValueError('Higher precision is an explicit OA GAT variant')
        if any(not isinstance(v,bool) for v in (self.quad3d_history_invariant,self.paired_gain_quadratic,self.quad3d_obstacle_pooling)):
            raise ValueError('Quad3D feature variants must be explicit booleans')
        if (self.quad3d_history_invariant or self.paired_gain_quadratic or self.quad3d_obstacle_pooling) and self.encoder not in ('gat','matched_fc'):
            raise ValueError('Quad3D feature variants are restricted to OA GAT')
        if self.quad3d_history_invariant and self.flight_history_invariant or self.paired_gain_quadratic and self.scalar_gain_quadratic:
            raise ValueError('Incompatible dynamics feature transforms')
        if not math.isfinite(self.continuous_log_variance_min) or not -10.<=self.continuous_log_variance_min<=0. or self.continuous_log_variance_min!=-10. and self.encoder not in ('gat','matched_fc'):
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

def flight_gain_coordinates(gains):
    """Two ordered gains, log curvature, and the actual HOCBF coefficients.

    Center at the declared gain4 reference. These are deterministic candidate
    features, with no physical rollout, selected gain, fitted target or search.
    Ordered coordinates retain the distinct first-stage domain dependence.
    """
    if gains.shape[-1]!=2:raise ValueError('Planar gain basis requires two ordered gains')
    coordinate=(jnp.log(jnp.maximum(gains,1e-6))-math.log(4.))/math.log(2.)
    return jnp.concatenate((coordinate,coordinate**2,
        (coordinate[...,0]*coordinate[...,1])[...,None],
        (gains.sum(-1)/8.-1.)[...,None],(gains.prod(-1)/16.-1.)[...,None]),axis=-1)

def clipped_mixture_heads(output):
    """Two latent risk components; all ten head outputs have a learned role."""
    means=jnp.stack((output[...,0],output[...,6]),axis=-1)
    log_variances=jnp.clip(jnp.stack((output[...,2],output[...,7]),axis=-1),-10.,3.)
    logits=output[...,8:10]
    weights=jax.nn.softmax(logits,axis=-1)
    mean=jnp.sum(weights*means,axis=-1)
    variance=jnp.sum(weights*(jnp.exp(log_variances)+(means-mean[...,None])**2),axis=-1)
    return dict(mean=jnp.stack((mean,output[...,1]),axis=-1),
        log_variance=jnp.stack((jnp.log(variance),jnp.clip(output[...,3],-10.,3.)),axis=-1),
        event_logits=output[...,4:6],risk_component_mean=means,
        risk_component_log_variance=log_variances,risk_component_logits=logits)

class CandidateGAT(nn.Module):
    config:GATConfig=GATConfig()
    risk_components:int=1

    def setup(self):
        self.project=Dense(self.config.width)
        self.blocks=[AttentionBlock(self.config.width,self.config.heads) for _ in range(self.config.layers)]
        self.norm=nn.LayerNorm()
        self.head1=Dense(self.config.width)
        self.head2=Dense(self.config.width)
        output_options=dict(kernel_init=nn.initializers.zeros_init()) if self.config.flight_local_residual else {}
        self.output=Dense(2*self.config.continuous_outputs+self.config.event_outputs+(4 if self.risk_components==2 else 0),**output_options)
        if self.config.flight_gain_attention:
            self.gain_query=Dense(self.config.width)
            self.obstacle_key=Dense(self.config.width)
            self.gain_context_residual=Dense(self.config.width,kernel_init=nn.initializers.zeros_init())
        if self.config.bicycle_reserve_auxiliary:
            self.prefix_reserve=Dense(1)

    def prepare_features(self,features,mask):
        if self.config.unicycle_ego_frame:
            from .unicycle_features import ego_frame
            features=ego_frame(features,mask)
        if self.config.flight_gain_basis and features.shape[-1]!=40:
            raise ValueError('Planar gain basis requires the declared flight graph40')
        if self.config.compute_dtype=='float64':
            features=features.astype(jnp.float64)
        if self.config.bicycle_history_invariant:
            expected = 59 if self.config.bicycle_route_context else 39 if self.config.bicycle_motion_history else 35
            if features.shape[-1]!=expected:raise ValueError(f'Bicycle history invariance requires its declared observed graph ({expected}-column)')
            # The fixed-candidate branch takes its new gain separately and has
            # no actuator lag, past-input cost, or previous-gain argument. Keep
            # current observations, committed route memory and noise features;
            # raw acquisition history remains in the independently audited data.
            features=features.at[...,26:29].set(0.)
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
        if self.config.flight_constraint_features:
            from .quad2d_features import coefficients
            clean=jnp.concatenate((clean,coefficients(clean,mask)),axis=-1)
        if self.config.unicycle_constraint_features:
            from .unicycle_features import coefficients
            clean=jnp.concatenate((clean,coefficients(clean,mask)),axis=-1)
        if self.config.bicycle_constraint_features:
            from .bicycle_features import append_constraints
            derived=append_constraints(clean[...,:35],mask)[...,35:]
            clean=jnp.concatenate((clean,derived),axis=-1)
        return clean

    def encode(self,features,mask,candidate=None):
        clean=self.prepare_features(features,mask)
        if self.config.bicycle_candidate_encoding:
            from .bicycle_features import condition_nodes
            clean=condition_nodes(clean,mask,candidate)
        nodes=self.project(clean)
        positions=clean[:,:,3:5]
        delta=positions[:,:,None,:]-positions[:,None,:,:]
        relative=jnp.concatenate((delta,jnp.sqrt(jnp.sum(delta**2,axis=-1,keepdims=True)+1e-12)),axis=-1)
        for block in self.blocks:
            nodes=block(nodes,mask,relative)
        if not (self.config.quad3d_obstacle_pooling or self.config.flight_obstacle_pooling or self.config.bicycle_obstacle_pooling):
            return self.norm(nodes[:,0,:])
        expected=(59 if self.config.bicycle_route_context else 39 if self.config.bicycle_motion_history else 35) if self.config.bicycle_obstacle_pooling else 40 if self.config.flight_obstacle_pooling else 58
        if features.shape[-1]!=expected:raise ValueError(f'Obstacle pooling requires the declared {"bicycle" if self.config.bicycle_obstacle_pooling else "flight"} graph ({expected}-column)')
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
        context=jnp.concatenate((z[:,0,:],average,maximum,clean[:,0,:features.shape[-1]],geometry,nearest),axis=-1)
        if self.config.flight_gain_attention:
            return dict(scene=context,obstacles=ob,mask=om)
        return context

    def score(self,context,gains):
        """context [B,W], gains [B,K,D]; one scalar or a declared gain pair."""
        encoded=context['scene'] if self.config.flight_gain_attention else context
        broadcast=jnp.broadcast_to(encoded[:,None,:],(*gains.shape[:2],encoded.shape[-1]))
        if self.config.flight_gain_attention:
            width,heads=self.config.width,self.config.heads
            query=self.gain_query(flight_gain_coordinates(gains)).reshape(*gains.shape[:2],heads,width//heads)
            nodes=context['obstacles'];mask=context['mask']
            keys=self.obstacle_key(nodes).reshape(*nodes.shape[:2],heads,width//heads)
            logits=jnp.einsum('bkhd,bnhd->bkhn',query,keys)/math.sqrt(width//heads)
            weights=jax.nn.softmax(jnp.where(mask[:,None,None,:],logits,-1e30),axis=-1)
            weights=jnp.where(mask[:,None,None,:],weights,0.)
            values=nodes.reshape(*nodes.shape[:2],heads,width//heads)
            attended=jnp.einsum('bkhn,bnhd->bkhd',weights,values).reshape(*gains.shape[:2],width)
            delta=self.gain_context_residual(attended)
            delta=jnp.where(mask.any(-1)[:,None,None],delta,0.)
            broadcast=broadcast.at[...,:width].add(delta)
        log_gain=jnp.log(jnp.maximum(gains,1e-6))
        if self.config.unicycle_constraint_features:
            from .unicycle_features import gain_coordinates
            log_gain=gain_coordinates(gains)
        if self.config.flight_gain_basis:
            log_gain=flight_gain_coordinates(gains)
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
        if self.config.bicycle_affine_gain:
            from .bicycle_control import coordinates
            log_gain=coordinates(gains)
        z=jnp.concatenate((broadcast,log_gain),axis=-1)
        z=nn.gelu(self.head1(z));z=z+nn.gelu(self.head2(z))
        out=self.output(z);d=self.config.continuous_outputs
        if self.risk_components==2:return clipped_mixture_heads(out)
        variance=jnp.clip(out[...,d:2*d],self.config.continuous_log_variance_min,3.)
        if self.config.risk_log_variance_min!=-10.:
            # Normalized risk head only. The default path and all FC outputs
            # remain exact; a changed floor requires a new trained/calibrated
            # bundle, but does not change the physical target or control law.
            variance=variance.at[...,0].set(jnp.clip(out[...,d],self.config.risk_log_variance_min,3.))
        result=dict(mean=out[...,:d],log_variance=variance,event_logits=out[...,2*d:])
        if self.config.bicycle_reserve_auxiliary:
            result['prefix_reserve']=self.prefix_reserve(z)
        return result

    def __call__(self,features,mask,gains):
        if self.config.bicycle_candidate_encoding:
            if gains.shape[-1]!=1:raise ValueError('Bicycle candidate encoding requires one scalar gain')
            batch,count=gains.shape[:2]
            expanded=jnp.broadcast_to(features[:,None],(batch,count,*features.shape[1:])).reshape(batch*count,*features.shape[1:])
            expanded_mask=jnp.broadcast_to(mask[:,None],(batch,count,mask.shape[1])).reshape(batch*count,mask.shape[1])
            candidates=gains.reshape(batch*count,1)
            encoded=self.encode(expanded,expanded_mask,candidates)
            prediction=self.score(encoded,candidates[:,None])
            return jax.tree.map(lambda a:a.reshape(batch,count,a.shape[-1]),prediction)
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
    risk_components:int=1

    def setup(self):
        self.hidden=[Dense(self.config.width*factor) for factor in (1,2,3,1)]
        self.output=Dense(2*self.config.continuous_outputs+self.config.event_outputs+(4 if self.risk_components==2 else 0))

    def encode(self,features,mask):
        if self.config.encoder == 'nearest_fc':
            from .nearest_fc import encode
            if self.config.compute_dtype == 'float64':
                features = features.astype(jnp.float64)
            context=encode(features, mask, self.config.nearest_dynamics, self.config.nearest_yaw_scale)
            if self.config.unicycle_constraint_features:
                from .unicycle_features import nearest_coefficients
                context=jnp.concatenate((context,nearest_coefficients(features,mask)),axis=-1)
            return context
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
        gain_features=jnp.log(jnp.maximum(gains,1e-6))
        if self.config.unicycle_constraint_features:
            from .unicycle_features import gain_coordinates
            gain_features=gain_coordinates(gains)
        values=jnp.concatenate((context,gain_features),axis=-1)
        for layer in self.hidden:values=nn.relu(layer(values))
        output=self.output(values);d=self.config.continuous_outputs
        if self.risk_components==2:return clipped_mixture_heads(output)
        if self.config.encoder == 'nearest_fc' and self.config.compute_dtype == 'float64':
            output = output.astype(jnp.float32)
        return dict(mean=output[...,:d],log_variance=jnp.clip(output[...,d:2*d],-10.,3.),event_logits=output[...,2*d:])

    def __call__(self,features,mask,gains):
        return self.score(self.encode(features,mask),gains)

class CandidateMatchedFC(CandidateGAT):
    """Encoder-only ablation: all observed nodes -> dense scene embedding.

    Uses the same preprocessing, candidate basis, head, precision and variance
    floors as CandidateGAT. Original FC/PENN checkpoints remain separate.
    """

    def setup(self):
        self.project=Dense(self.config.width)
        self.blocks=[Dense(self.config.width) for _ in range(self.config.layers)]
        self.context_projection=Dense(self.config.width*(3 if self.config.quad3d_obstacle_pooling or self.config.flight_obstacle_pooling or self.config.bicycle_obstacle_pooling else 1))
        self.norm=nn.LayerNorm()
        self.head1=Dense(self.config.width)
        self.head2=Dense(self.config.width)
        self.output=Dense(2*self.config.continuous_outputs+self.config.event_outputs)
        if self.config.bicycle_reserve_auxiliary:
            self.prefix_reserve=Dense(1)

    def encode(self,features,mask,candidate=None):
        clean=self.prepare_features(features,mask)
        if self.config.bicycle_candidate_encoding:
            from .bicycle_features import condition_nodes
            clean=condition_nodes(clean,mask,candidate)
        obstacles=clean[:,2:];om=mask[:,2:]
        keys=(obstacles[:,:,6],obstacles[:,:,5],obstacles[:,:,7],obstacles[:,:,4],obstacles[:,:,3],
              jnp.where(om,obstacles[:,:,8],jnp.inf))
        order=jnp.lexsort(keys,axis=1)
        ordered=jnp.take_along_axis(obstacles,order[...,None],axis=1)
        values=jnp.concatenate((clean[:,:2],ordered),axis=1).reshape(len(clean),-1)
        values=nn.gelu(self.project(values))
        for layer in self.blocks:values=values+nn.gelu(layer(values))
        context=self.norm(self.context_projection(values))
        if not (self.config.quad3d_obstacle_pooling or self.config.flight_obstacle_pooling or self.config.bicycle_obstacle_pooling):return context
        expected=(59 if self.config.bicycle_route_context else 39 if self.config.bicycle_motion_history else 35) if self.config.bicycle_obstacle_pooling else 40 if self.config.flight_obstacle_pooling else 58
        if features.shape[-1]!=expected:raise ValueError(f'Obstacle pooling requires the declared {"bicycle" if self.config.bicycle_obstacle_pooling else "flight"} graph ({expected}-column)')
        clearance=clean[:,2:,8]
        nearest=jnp.min(jnp.where(om,clearance,jnp.inf),axis=1,keepdims=True)
        nearest=jnp.where(om.any(-1,keepdims=True),nearest,0.)
        weights=jnp.where(om,jnp.exp(-jnp.maximum(clearance-nearest,0.)),0.)
        weights=weights/jnp.maximum(weights.sum(-1,keepdims=True),1e-12)
        geometry=jnp.sum(weights[...,None]*clean[:,2:,3:9],axis=1)
        return jnp.concatenate((context,clean[:,0,:features.shape[-1]],geometry,nearest),axis=-1)

class CandidateFlightLocalResidualGAT(CandidateGAT):
    """Frozen nearest-obstacle predictor plus a trainable scene GAT correction.

    The correction starts at zero. Both branches are neural predictors of the
    existing targets; neither executes a controller or searches gain rollouts.
    """
    def setup(self):
        super().setup()
        self.local_model=CandidateFC(GATConfig(width=40,layers=4,encoder='nearest_fc',nearest_dynamics='quad2d'))

    def encode(self,features,mask,candidate=None):
        if features.shape[-1]!=40:raise ValueError('Local residual flight model requires graph40')
        context=super().encode(features,mask,candidate)
        local=self.local_model.encode(features,mask)
        return jnp.concatenate((context,local),axis=-1)

    def score(self,context,gains):
        # Quad2D local contract: clearance, vx, vz, bearing sine and cosine.
        base=self.local_model.score(context[...,-5:],gains)
        correction=super().score(context[...,:-5],gains)
        return dict(mean=base['mean']+correction['mean'],
            log_variance=jnp.clip(base['log_variance']+correction['log_variance'],-10.,3.),
            event_logits=base['event_logits']+correction['event_logits'])

def make_model(config,risk_components=1):
    if risk_components != 1:
        if (risk_components!=2 or config.encoder not in ('gat','nearest_fc')
                or config.compute_dtype!='float32' or config.continuous_outputs!=2 or config.event_outputs!=2
                or config.flight_local_residual or config.bicycle_reserve_auxiliary or config.flight_gain_attention):
            raise ValueError('Clipped mixture requires the two-head FP32 GAT or nearest FC')
        return CandidateGAT(config,risk_components) if config.encoder=='gat' else CandidateFC(config,risk_components)
    if config.flight_local_residual:return CandidateFlightLocalResidualGAT(config)
    if config.encoder=='matched_fc':return CandidateMatchedFC(config)
    return CandidateGAT(config) if config.encoder=='gat' else CandidateFC(config)

def initialize_ensemble(model,key,features,mask,gains,members=4):
    """Independent entire encoders and heads, not just independent last layers."""
    keys=jax.random.split(key,members)
    return jax.vmap(lambda k:model.init(k,features,mask,gains)['params'])(keys)

def predict_ensemble(model,params,features,mask,gains):
    if model.config.encoder=='matched_fc':
        width=params['project']['kernel'].shape[-2]
        encoded_width=features.shape[-1]+(6 if model.config.bicycle_constraint_features else 0)+(2 if model.config.bicycle_candidate_encoding else 0)
        nodes,remainder=divmod(width,encoded_width)
        if remainder or nodes<2 or features.shape[1]>nodes:
            raise ValueError('Matched-FC input capacity mismatch; truncation is forbidden')
        extra=nodes-features.shape[1]
        if extra:
            features=jnp.pad(features,((0,0),(0,extra),(0,0)))
            mask=jnp.pad(mask,((0,0),(0,extra)),constant_values=False)
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
