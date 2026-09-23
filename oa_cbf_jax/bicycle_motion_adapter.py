"""Learn observed-history input connections while freezing the original predictor."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .bicycle_experiment import read
from .dataset import sha256
from .models import GATConfig


def contract():
    return dict(schema='bicycle_frozen_predictor_motion_inputs',
        initialization='Copy corresponding reviewed original member; insert zero input weights for four causal-history columns.',
        trainable='Only project/kernel rows for the four added columns, per node for matched FC.',
        frozen='Every original weight and bias, all attention/dense blocks and output heads; new ego-context head rows also remain zero.',
        optimizer='Original warmup/cosine AdamW; mask gradients before clipping and mask updates after AdamW including weight decay.',
        epoch_zero='Eligible validation checkpoint, preserving the original predictor as the no-learning choice.',
        labels_changed=False,controller_changed=False,physical_truth_used=False)


def expand_parameters(original,encoder,width):
    """Exact algebraic embedding from encoded41 to45 and raw context35 to39."""
    if encoder not in ('gat','matched_fc'):raise ValueError('Expected matched bicycle encoder')
    params=jax.tree.map(lambda a:np.array(a,copy=True),original)
    kernel=params['project']['kernel']
    if kernel.shape[1]!=width:raise ValueError('Projection width mismatch')
    nodes=1 if encoder=='gat' else kernel.shape[0]//41
    if kernel.shape[0]!=nodes*41 or nodes<1:raise ValueError('Original encoded41 projection required')
    expanded=np.zeros((nodes,45,width),kernel.dtype)
    old=kernel.reshape(nodes,41,width)
    expanded[:,:35]=old[:,:35];expanded[:,39:]=old[:,35:]
    params['project']['kernel']=expanded.reshape(nodes*45,width)
    cut=3*width+35
    head=params['head1']['kernel']
    if head.shape!=(cut+7+2,width):raise ValueError('Original pooled scalar quadratic head required')
    enlarged=np.zeros((len(head)+4,width),head.dtype)
    enlarged[:cut]=head[:cut];enlarged[cut+4:]=head[cut:]
    params['head1']['kernel']=enlarged
    mask=jax.tree.map(lambda a:np.zeros(a.shape,bool),params)
    mask['project']['kernel'].reshape(nodes,45,width)[:,35:39]=True
    return params,mask


def frozen_digest(params,mask):
    h=hashlib.sha256()
    for (path,array),(_,allowed) in zip(jax.tree_util.tree_flatten_with_path(params)[0],jax.tree_util.tree_flatten_with_path(mask)[0]):
        value=np.asarray(array);allowed=np.asarray(allowed,bool)
        h.update(str(path).encode());h.update(str((value.shape,str(value.dtype))).encode());h.update(value[~allowed].tobytes())
    return h.hexdigest()


def mask_transform(mask):
    """Element masks also exclude AdamW decay on the frozen parameter subset."""
    mask=jax.tree.map(jnp.asarray,mask)
    def update(updates,state,params=None):
        return jax.tree.map(lambda u,m:jnp.where(m,u,jnp.zeros_like(u)),updates,mask),state
    return optax.GradientTransformation(lambda _:optax.EmptyState(),update)


def initialize(model,seed,normalization,descriptor):
    source=read(descriptor)
    for path,digest in source['bound_files'].items():
        if sha256(path)!=digest:raise ValueError('Changed frozen predictor source: '+path)
    review=read(source['review'])
    if review['status']!='passed' or not review['matched_encoder_only_training_verified']:
        raise ValueError('Reviewed matched original source required')
    if not model.config.bicycle_motion_history:raise ValueError('Adapter requires observed history schema')
    if seed not in (411,412,413,414):raise ValueError('Only corresponding original member seeds')
    member=Path(source['training'])/model.config.encoder/f'member_{seed-411}'
    settings=read(member/'settings.json');best=read(member/'best.json')
    if GATConfig(**settings['architecture'])!=replace(model.config,bicycle_motion_history=False):
        raise ValueError('Adapter changed the original model beyond history columns')
    if settings['seed']!=seed or settings['normalization']!=normalization:raise ValueError('Original seed/normalization mismatch')
    path=member/'checkpoints'/best['state_file']
    if sha256(path)!=best['state_sha256'] or sha256(member/'settings.json')!=best['settings_sha256']:
        raise ValueError('Original checkpoint binding mismatch')
    original=serialization.msgpack_restore(path.read_bytes())['params']
    params,mask=expand_parameters(original,model.config.encoder,model.config.width)
    proof=dict(contract=contract(),descriptor=str(Path(descriptor).resolve()),descriptor_sha256=sha256(descriptor),
        source_settings=str(member/'settings.json'),source_settings_sha256=sha256(member/'settings.json'),
        source_checkpoint=str(path),source_checkpoint_sha256=sha256(path),source_epoch=best['epoch'],
        trainable_parameters=int(sum(np.count_nonzero(v) for v in jax.tree.leaves(mask))),
        frozen_parameters=int(sum(np.count_nonzero(~v) for v in jax.tree.leaves(mask))),
        frozen_parameters_sha256=frozen_digest(params,mask),zero_new_input_weights=True)
    return params,mask,proof


def verify_settings(before,after):
    from .bicycle_learning_contracts import verify_motion_only_settings
    after=copy.deepcopy(after)
    if after.pop('bicycle_motion_adapter_contract')!=contract():raise ValueError('Changed adapter treatment')
    descriptor=after.pop('bicycle_motion_adapter_source')
    if after.pop('bicycle_motion_adapter_source_sha256')!=sha256(descriptor):raise ValueError('Changed source descriptor')
    if after.pop('initial_checkpoint_eligible') is not True:raise ValueError('Missing unchanged initial predictor')
    return verify_motion_only_settings(before,after)
