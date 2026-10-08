"""Observed planar-flight barrier coefficients, without a solver or lookahead.

The fixed graph40 schema supplies relative position/velocity, attitude, mass,
gravity and radii. These features expose the existing disk HOCBF polynomial to
the graph encoder. They do not change its input information or the QP itself.
"""
from dataclasses import asdict
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from .dataset import sha256
from .quad2d_static_inputs import read


def contract():
    return dict(schema='quad2d_observed_barrier_coefficients_v1',
        input='Observed graph40 only; no latent physical state, future, labels or candidate search.',
        columns=['h/25', 'hdot/20', 'free_hddot/100', 'rotor_sum_authority/10'],
        equation='free_hddot + authority*(right_thrust+left_thrust) + (k1+k2)*hdot + k1*k2*h >= 0',
        geometry='Disk radii plus existing .05m clearance buffer; no hard uncertainty inflation.',
        masked='Ego, goal and padded obstacle coefficients are zero.',
        scope='Static observed Quad2D graph40 and original guided Gaussian GAT only.',
        controller_changed=False, gain_domain_changed=False, labels_changed=False)


def coefficients(features, mask):
    if features.ndim != 3 or features.shape[-1] != 40 or features.shape[1] < 2:
        raise ValueError('Expected observed graph40 with ego and goal nodes')
    clean=jnp.where(mask[...,None],features,0.)
    ego=clean[:,0]
    delta=-5.*clean[:,2:,3:5]
    velocity=-clean[:,2:,5:7]*ego[:,None,23:24]
    radius=ego[:,None,7]+clean[:,2:,7]+.05
    h=jnp.sum(delta**2,axis=-1)-radius**2
    hd=2.*jnp.sum(delta*velocity,axis=-1)
    gravity=ego[:,19]*9.81
    free=2.*jnp.sum(velocity**2,axis=-1)-2.*gravity[:,None]*delta[...,1]
    axis=jnp.stack((-ego[:,9],ego[:,10]),axis=-1)/jnp.maximum(ego[:,14:15],1e-6)
    authority=2.*jnp.sum(delta*axis[:,None],axis=-1)
    values=jnp.stack((h/25.,hd/20.,free/100.,authority/10.),axis=-1)
    values=jnp.where(mask[:,2:,None],values,0.)
    return jnp.pad(values,((0,0),(2,0),(0,0)))


def validate_source(meta, architecture, normalization, dataset_sha256):
    from .models import GATConfig
    expected=asdict(GATConfig(**meta['architecture']))
    expected['flight_constraint_features']=architecture.flight_constraint_features
    expected['flight_gain_attention']=architecture.flight_gain_attention
    if (asdict(architecture)!=expected or meta['architecture'].get('flight_constraint_features',False)
            or meta['architecture'].get('flight_gain_attention',False)
            or meta['architecture']['encoder']!='gat' or meta['graph_features']!=40
            or meta['normalization']!=normalization or meta['dataset_manifest_sha256']!=dataset_sha256
            or meta.get('risk_distribution_contract') or meta.get('flight_local_residual_contract')):
        raise ValueError('Warm start requires the unchanged original Gaussian flight GAT')
    config=meta['controller']['config']
    if (config['robot']['clearance_buffer']!=.05 or config['robot']['cbf_margin']!=0
            or not config['stationary_obstacles']):
        raise ValueError('Unregistered coefficient geometry')
    guidance=meta['controller']['predictive_guidance']
    if guidance.get('inflate_geometry',False) or guidance.get('hard_prediction_clearance',False):
        raise ValueError('Hard inflated geometry needs its own coefficient contract')


def initialize(model, data, seed, normalization, bundle, dataset_sha256, freeze_original=False):
    from .models import GATConfig, make_model
    from .bicycle_motion_adapter import frozen_digest
    root=Path(bundle).resolve();meta=read(root/'manifest.json')
    if freeze_original and not model.config.flight_gain_attention:
        raise ValueError('Frozen flight predictor requires the added gain attention')
    validate_source(meta,model.config,normalization,dataset_sha256)
    weights=root/'weights.msgpack'
    if sha256(weights)!=meta['weights_sha256']:
        raise ValueError('Changed warm-start source weights')
    seeds=[row['seed'] for row in meta['members']]
    if seeds.count(seed)!=1:raise ValueError('Corresponding independent seed required')
    slot=seeds.index(seed)
    original=serialization.msgpack_restore(weights.read_bytes())
    base=jax.tree.map(lambda v:np.array(v[slot],copy=True),original)
    params=jax.tree.map(np.copy,base)
    if model.config.flight_constraint_features:
        if params['project']['kernel'].shape[0]!=40:raise ValueError('Expected original graph40 projection')
        params['project']['kernel']=np.pad(params['project']['kernel'],((0,4),(0,0)))
    args=tuple(jnp.asarray(data[k][:8]) for k in ('features','node_mask','gains'))
    if model.config.flight_gain_attention:
        initialized=model.init(jax.random.key(seed+87291),*args)['params']
        expected_new={'gain_query','obstacle_key','gain_context_residual'}
        if set(initialized)-set(base)!=expected_new or set(base)-set(initialized):
            raise ValueError('Unexpected gain-attention warm-start parameter changes')
        params.update({k:np.array(v) if not isinstance(v,dict) else jax.tree.map(np.array,v)
                       for k,v in initialized.items() if k in expected_new})
    reference=make_model(GATConfig(**meta['architecture']))
    actual=model.apply({'params':jax.tree.map(jnp.asarray,params)},*args)
    expected=reference.apply({'params':jax.tree.map(jnp.asarray,base)},*args)
    differences={}
    for key in expected:
        differences[key]=float(np.max(np.abs(np.asarray(actual[key])-np.asarray(expected[key]))))
        np.testing.assert_allclose(actual[key],expected[key],atol=3e-5,rtol=3e-5,
            err_msg='Zero added coefficient weights must preserve initial predictions')
    allowed=jax.tree.map(lambda v:np.ones(v.shape,bool),params)
    if freeze_original:
        from .quad2d_gain_attention import trainable_attention_mask
        allowed=trainable_attention_mask(params)
    proof=dict(source_bundle=str(root),source_manifest_sha256=sha256(root/'manifest.json'),
        source_weights_sha256=meta['weights_sha256'],source_seed=seed,source_member=slot,
        coefficient_features=model.config.flight_constraint_features,
        gain_attention=model.config.flight_gain_attention,
        frozen_original=freeze_original,
        initialization_max_errors=differences,initial_checkpoint_eligible=True,
        frozen_parameters_sha256=frozen_digest(params,allowed),
        trainable_parameters=int(sum(np.count_nonzero(v) for v in jax.tree.leaves(allowed))),
        frozen_parameters=int(sum(np.count_nonzero(~v) for v in jax.tree.leaves(allowed))))
    return params,allowed,proof
