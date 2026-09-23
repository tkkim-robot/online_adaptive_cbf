"""Training-only weighting of rare observed opportunities for gain adaptation."""
import numpy as np


def gain_opportunity_weights(data, replicas):
    """Equal expected mass for opportunity/other parents before bootstrapping.

    A candidate is empirically nonadverse only if all saved replicas reach the
    goal or the horizon. An opportunity improves observed progress by >0.01
    over gain[4,4], or avoids that reference's adverse outcome. This is an
    offline training stratum, not a safety certificate or runtime oracle.
    Every parent and branch remains present. Validation/calibration stay
    unweighted; their outcomes never enter these weights.
    """
    n=len(data['group_id']);gains=np.asarray(data['gains'])
    if (n==0 or len(set(data['group_id'].tolist()))!=n or replicas<1
            or gains.ndim!=3 or gains.shape[0]!=n or gains.shape[-1]!=2
            or gains.shape[1]%replicas):
        raise ValueError('One independent two-gain training query per parent required')
    q=gains.shape[1]//replicas;bank=gains.reshape(n,q,replicas,2)
    if not np.array_equal(bank,np.broadcast_to(bank[:,:,:1],bank.shape)):
        raise ValueError('Paired replicas must use the same gain')
    reference=np.all(bank[:,:,0]==4.,axis=-1)
    if not np.all(reference.sum(-1)==1):raise ValueError('Exactly one gain[4,4] reference required')
    progress=np.asarray(data['target'])[...,1]
    if (progress.shape!=(n,q*replicas) or not np.isfinite(progress).all()
            or not np.asarray(data['target_mask'])[...,1].all()):
        raise ValueError('Complete observed-prefix progress labels required')
    status=np.asarray(data['status'])
    if status.shape!=progress.shape or not np.isin(status,np.arange(1,9)).all():
        raise ValueError('Complete terminal branch outcomes required')
    safe=np.isin(status,[1,4]).reshape(n,q,replicas).all(-1)
    actual=progress.reshape(n,q,replicas).mean(-1);ref=reference.argmax(-1);rows=np.arange(n)
    useful=(safe&((actual-actual[rows,ref,None]>.01)|~safe[rows,ref,None])).any(-1)
    positive=int(useful.sum());negative=n-positive
    if not positive or not negative:raise ValueError('Both training opportunity strata required')
    weights=np.where(useful,.5*n/positive,.5*n/negative).astype(np.float32)
    return weights,dict(schema='quad2d_observed_gain_opportunity_weight_v1',parents=n,
        opportunity_parents=positive,other_parents=negative,reference_gain=[4.,4.],
        physical_progress_difference=.01,replicas=replicas,
        opportunity_weight=float(weights[useful][0]),other_weight=float(weights[~useful][0]),
        opportunity_group_ids=data['group_id'][useful].tolist(),
        sampling='Equal expected stratum mass before the original parent bootstrap; all branches share their parent weight.',
        validation_and_calibration='Original unweighted held-out partitions; no opportunity-based filtering.',
        runtime_use=False)
