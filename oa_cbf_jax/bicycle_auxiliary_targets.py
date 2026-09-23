"""Validated observed-prefix auxiliary targets for bicycle training."""
from pathlib import Path
import numpy as np
from .bicycle_experiment import read
from .dataset import sha256


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
