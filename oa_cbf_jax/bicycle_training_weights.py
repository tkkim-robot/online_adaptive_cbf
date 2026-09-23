"""Training-only parent weighting for observed gain-dependent adverse outcomes."""
import numpy as np


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
