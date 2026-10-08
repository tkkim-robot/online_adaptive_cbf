"""Offline progress contrasts among observed nonadverse candidate replicas.

This auxiliary never admits a gain at runtime. All original continuous and
event losses, including failed trials, remain intact. Checkpoint selection
continues to use the original primary validation loss.
"""
import jax.numpy as jnp


def contract():
    return dict(schema='observed_nonadverse_progress_contrast_v1',weight=1.0,
        eligibility='All paired replicas have observed progress and both observed event flags false; at least two eligible gains per parent.',
        objective='Parent-weighted mean squared centered progress error over eligible gains, after averaging common-noise replicas. Normalized original progress units; no spread division.',
        effect='Add only to the optimization objective, alongside every original continuous/event/all-gain-contrast loss. Never remove failed trials from original losses.',
        selection='Original primary validation loss; auxiliary is logged but not a checkpoint selection criterion.',
        inference='No label-conditioned mask, new feature, safety threshold, gain eligibility or architecture change at inference.',
        limitations='Empirical nonadversity of four offline replicas is not a safety certificate.')


def loss(prediction,target,valid,events,event_mask,group_weight,replicas):
    if (replicas<1 or prediction.ndim!=2 or prediction.shape!=target.shape
            or valid.shape!=target.shape or prediction.shape[1]%replicas
            or events.shape!=(*target.shape,2) or event_mask.shape!=events.shape
            or group_weight.shape!=(target.shape[0],)):
        raise ValueError('Complete paired progress/event layout required')
    n,k=prediction.shape; q=k//replicas
    eligible=(valid.reshape(n,q,replicas).all(-1)
        & event_mask.reshape(n,q,replicas,2).all((-1,-2))
        & ~events.astype(bool).reshape(n,q,replicas,2).any((-1,-2)))
    counts=eligible.sum(-1)
    predicted=prediction.reshape(n,q,replicas).mean(-1)
    actual=target.reshape(n,q,replicas).mean(-1)
    error=predicted-actual
    center=jnp.where(eligible,error,0.).sum(-1)/jnp.maximum(counts,1)
    per_parent=jnp.where(eligible,(error-center[:,None])**2,0.).sum(-1)/jnp.maximum(counts,1)
    weight=group_weight*(counts>=2)
    return jnp.sum(per_parent*weight)/jnp.where(weight.sum()>0,weight.sum(),1.)
