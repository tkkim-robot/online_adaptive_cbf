"""Bicycle training functions and shared contracts."""

from pathlib import Path

import numpy as np

from .bicycle_control import read

from .io import sha256

def contract():
    return dict(schema='bicycle_observed_prefix_reserve_auxiliary',
        target='minimum recorded observed h through the actual stop, including the first rejected decision',
        mask='finite recorded h/domain and strictly positive barrier domain at every recorded observation',
        semantics='observed prefix only; no post-stop states or uncensored future safety claim',
        primary_targets_changed=False, primary_loss_weights_changed=False,
        graph_features=35, head='one separate linear scalar from the shared candidate hidden representation',
        objective='parent-weighted normalized mean squared error', weight=.25,
        normalization='original TRAIN physical parents equal total weight over valid branches',
        checkpoint_selection='unchanged primary validation NLL+BCE+progress contrast; auxiliary loss excluded',
        deployment_use=False)

def reserve_normalization(data):
    if not np.all(data['partition']=='train'):
        raise ValueError('Auxiliary normalization uses original TRAIN parents only')
    valid=np.asarray(data['reserve_mask'],bool)[...,0];values=np.asarray(data['reserve_target'])[...,0]
    if not valid.any() or not np.isfinite(values[valid]).all():raise ValueError('No finite auxiliary observations')
    _,inverse=np.unique(data['group_id'],return_inverse=True)
    counts=np.bincount(inverse,weights=valid.sum(1))
    weights=np.broadcast_to(1/np.maximum(counts[inverse,None],1),valid.shape)[valid]
    mean=float(np.average(values[valid],weights=weights))
    scale=max(.05,float(np.sqrt(np.average((values[valid]-mean)**2,weights=weights))))
    return dict(reserve_mean=mean,reserve_scale=scale)

def identities(data):
    keys=list(zip(data['group_id'].tolist(),data['query_origin'].tolist(),data['query_tick'].tolist()))
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate parent/history/query identity')
    return keys

def attach_labels(raw, directory, dataset, role):
    if role not in ('train','validation') or not np.all(raw['partition']==role):
        raise ValueError('Only complete original development partitions may use auxiliary labels')
    root=Path(directory);report=read(root/'report.json');source=Path(dataset)
    if (report.get('status')!='passed' or report['contract']!=contract()
            or not report['every_prefix_independently_checked']
            or report['dataset_manifest_sha256']!=sha256(source/'manifest.json')
            or report['dataset_index_sha256']!=sha256(source/'index.json')
            or report['labels_sha256']!=sha256(root/'labels.npz')):
        raise ValueError('Verified full development sidecar required')
    for path,digest in report['bound_files'].items():
        if sha256(path)!=digest:raise ValueError('Changed auxiliary trace/source binding')
    with np.load(root/'labels.npz') as z:all_data={k:z[k] for k in z.files}
    keep=all_data['partition']==role;data={k:v[keep] for k,v in all_data.items()}
    lookup={key:i for i,key in enumerate(identities(data))};keys=identities(raw)
    if set(keys)!=set(lookup):raise ValueError('Missing or additional original development queries')
    order=np.array([lookup[k] for k in keys])
    for k in ('group_id','partition','query_tick','query_origin','gains','status','steps','target','target_mask','events','event_mask'):
        np.testing.assert_array_equal(raw[k],data[k][order])
    valid=data['observed_reserve_valid'][order] & (data['minimum_observed_domain'][order]>0)
    values=data['minimum_observed_h'][order]
    if not np.isfinite(values[valid]).all():raise ValueError('Nonfinite unmasked auxiliary observation')
    result=dict(raw,reserve_target=np.where(valid,values,0.)[...,None].astype(np.float32),reserve_mask=valid[...,None])
    proof=dict(directory=str(root.resolve()),report_sha256=sha256(root/'report.json'),labels_sha256=report['labels_sha256'])
    return result,proof


from dataclasses import replace

import hashlib


from flax import serialization

import jax

import jax.numpy as jnp


import optax


from .models import GATConfig

def motion_adapter_contract():
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
    proof=dict(contract=motion_adapter_contract(),descriptor=str(Path(descriptor).resolve()),descriptor_sha256=sha256(descriptor),
        source_settings=str(member/'settings.json'),source_settings_sha256=sha256(member/'settings.json'),
        source_checkpoint=str(path),source_checkpoint_sha256=sha256(path),source_epoch=best['epoch'],
        trainable_parameters=int(sum(np.count_nonzero(v) for v in jax.tree.leaves(mask))),
        frozen_parameters=int(sum(np.count_nonzero(~v) for v in jax.tree.leaves(mask))),
        frozen_parameters_sha256=frozen_digest(params,mask),zero_new_input_weights=True)
    return params,mask,proof


def safety_opportunity_weights(data, replicas):
    """Equal expected mass for mixed-outcome and other physical parents.

    A mixed query has at least one candidate whose recorded replicas all reach
    goal/horizon, and at least one candidate with an adverse recorded replica.
    Any such query puts its whole parent in that stratum. All of a parent's
    visits receive the same multiplier, applied after the existing parent
    bootstrap and inverse visit count. No query/branch is filtered. This is an
    empirical training label, never an inference feature or safety certificate.
    """
    ids=np.asarray(data['group_id']);gains=np.asarray(data['gains'])
    n=len(ids)
    if (n==0 or ids.ndim!=1 or type(replicas) is not int or replicas<1
            or gains.ndim!=3 or gains.shape[0]!=n or gains.shape[-1]!=1
            or gains.shape[1]%replicas or gains.shape[1]//replicas<2):
        raise ValueError('Complete scalar-gain parent queries and replicas required')
    partition=np.asarray(data['partition'])
    if partition.shape!=(n,) or not np.all(partition=='train'):
        raise ValueError('Safety opportunity weights use training parents only')
    bank=gains.reshape(n,-1,replicas)
    if (not np.isfinite(bank).all() or np.any(bank<=0)
            or not np.array_equal(bank,np.broadcast_to(bank[:,:,:1],bank.shape))
            or np.any(np.diff(np.sort(bank[:,:,0],axis=1),axis=1)<=0)):
        raise ValueError('Distinct candidate gains with identical paired replicas required')
    status=np.asarray(data['status'])
    if status.shape!=gains.shape[:2] or not np.isin(status,np.arange(1,8)).all():
        raise ValueError('Complete observed terminal outcomes required')
    nonadverse=np.isin(status,[1,4]).reshape(bank.shape).all(-1)
    mixed=nonadverse.any(-1)&~nonadverse.all(-1)
    parents,inverse=np.unique(ids,return_inverse=True)
    opportunity=np.zeros(len(parents),bool)
    np.logical_or.at(opportunity,inverse,mixed)
    positive=int(opportunity.sum());negative=len(parents)-positive
    if not positive or not negative:raise ValueError('Both parent strata required')
    multipliers=np.where(opportunity,.5*len(parents)/positive,.5*len(parents)/negative).astype(np.float32)
    return multipliers[inverse],dict(schema='bicycle_observed_safety_opportunity_weight_v1',
        parents=len(parents),queries=n,opportunity_parents=positive,other_parents=negative,
        mixed_outcome_queries=int(mixed.sum()),replicas=replicas,
        opportunity_weight=float(multipliers[opportunity][0]),other_weight=float(multipliers[~opportunity][0]),
        opportunity_group_ids=parents[opportunity].tolist(),
        sampling='Equal expected stratum mass before the unchanged parent bootstrap; identical multiplier for every visit and branch of a parent. No query selection.',
        validation_and_calibration='Original parent-weighted held-out partitions; no safety-opportunity reweighting or filtering.',
        runtime_use=False)
