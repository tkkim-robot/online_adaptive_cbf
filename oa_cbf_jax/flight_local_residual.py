"""Bind and freeze the local predictor for a scene-residual GAT treatment."""
from pathlib import Path
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from .dataset import sha256
from .nearest_fc_qualification import read
from .bicycle_motion_adapter import frozen_digest
from .models import GATConfig,make_model


def contract():
    return dict(schema='flight_local_scene_residual_v1',
        local_input='Robot plus the one nearest observed obstacle; same explicit paper FC contract.',
        scene_input='Full observed graph40 with existing history invariance and obstacle pooling.',
        predictions='Local Gaussian means/log variances and event logits plus zero-initialized GAT corrections.',
        frozen='All original local predictor parameters, including optimizer weight decay.',
        trained='Scene attention and residual prediction head, on existing training parents only.',
        checkpoint='Epoch zero is eligible; checkpoint selection uses the existing validation objective.',
        comparison='A new hybrid OA variant; graph contribution requires comparison to its frozen local-only predictor.',
        controller_changed=False,labels_changed=False,physical_truth_used=False)


def initialize(model,data,seed,normalization,bundle,dataset_sha256):
    root=Path(bundle).resolve();meta=read(root/'manifest.json')
    from .nearest_fc import validate_metadata
    validate_metadata(meta)
    if (not model.config.flight_local_residual or meta['architecture'].get('nearest_dynamics')!='quad2d'
            or meta['normalization']!=normalization or meta['dataset_manifest_sha256']!=dataset_sha256):
        raise ValueError('Mismatched frozen local model, training data or normalization')
    weights=root/'weights.msgpack'
    if sha256(weights)!=meta['weights_sha256']:raise ValueError('Changed frozen local weights')
    seeds=[r['seed'] for r in meta['members']]
    if seeds.count(seed)!=1:raise ValueError('Use the corresponding independent local member seed')
    member=seeds.index(seed);original=serialization.msgpack_restore(weights.read_bytes())
    local=jax.tree.map(lambda v:np.array(v[member],copy=True),original)
    params=model.init(jax.random.key(seed),data['features'][:1],data['node_mask'][:1],data['gains'][:1])['params']
    params=jax.tree.map(lambda v:np.array(v,copy=True),params)
    if jax.tree.structure(local)!=jax.tree.structure(params['local_model']) or any(
            a.shape!=b.shape for a,b in zip(jax.tree.leaves(local),jax.tree.leaves(params['local_model']))):
        raise ValueError('Frozen local parameter shape mismatch')
    params['local_model']=local
    mask=jax.tree.map(lambda v:np.ones(v.shape,bool),params)
    mask['local_model']=jax.tree.map(lambda v:np.zeros(v.shape,bool),local)
    local_model=make_model(GATConfig(**meta['architecture']))
    args=(data['features'][:2],data['node_mask'][:2],data['gains'][:2])
    a=model.apply({'params':jax.tree.map(jnp.asarray,params)},*args)
    b=local_model.apply({'params':jax.tree.map(jnp.asarray,local)},*args)
    for k in b:np.testing.assert_array_equal(a[k],b[k],err_msg='Epoch-zero local identity')
    proof=dict(contract=contract(),source_bundle=str(root),source_manifest_sha256=sha256(root/'manifest.json'),
        source_weights_sha256=sha256(weights),source_member=member,source_seed=seed,
        frozen_parameters_sha256=frozen_digest(params,mask),epoch_zero_exact_identity=True,
        trainable_parameters=int(sum(np.count_nonzero(x) for x in jax.tree.leaves(mask))))
    return params,mask,proof
