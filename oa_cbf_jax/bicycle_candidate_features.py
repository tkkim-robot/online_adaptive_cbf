"""Candidate-conditioned observed constraint inputs, with no controller solve."""
import math
import jax.numpy as jnp
import numpy as np


def contract():
    return dict(schema='bicycle_candidate_conditioned_scene_encoding',raw_graph_features=35,
        prepared_graph_features=41,encoded_graph_features=43,
        added_fields=['log_gain_center2_scale_log4','asinh_candidate_constraint_rhs_over_speed_max'],
        candidate_rhs='drift + candidate_gain * h; recover h and drift from original observed constraint coordinates.',
        input='Current observed graph35, mask, original6 deterministic constraint features, queried scalar gain.',
        treatment='Encode each scene-candidate pair before the unchanged Gaussian/event head; identical transform for GAT and matched FC.',
        padding='Both added fields zero on padding; rhs zero on ego/goal or invalid geometric domain.',
        learned_parameters='Original encoder and head; only input projection gains two columns per node.',
        gain_domain=[.5,8.],controller_solve=False,physical_truth_used=False,training_labels_used=False,
        limitation='Observed instantaneous inequality coefficients, not a forecast or safety certificate.')


def validate_metadata(metadata):
    from .bicycle_constraint_features import contract as constraint_contract
    from .bicycle_features import SCHEMA as graph_schema
    architecture=metadata.get('architecture',{})
    if (metadata.get('graph_features')!=35 or metadata.get('gain_dimension')!=1
            or architecture.get('bicycle_candidate_encoding') is not True
            or architecture.get('bicycle_constraint_features') is not True
            or architecture.get('bicycle_motion_history',False)
            or architecture.get('bicycle_affine_gain',False)
            or architecture.get('encoder') not in ('gat','matched_fc')
            or metadata.get('bicycle_candidate_encoding_contract')!=contract()
            or metadata.get('bicycle_constraint_features_contract')!=constraint_contract()
            or metadata.get('bicycle_contract',{}).get('graph_schema')!=graph_schema):
        raise ValueError('Changed candidate-conditioned observed model contract')


def validate_numerical_mode(fit,reference_recording,numerical_test):
    # Development success alone does not authorize a calibrated/live policy.
    if not (reference_recording is True and numerical_test is True
            and fit.get('diagnostic_identity_only') is True
            and fit.get('calibration_fitted') is False):
        raise ValueError('Candidate-encoding pilot requires runtime qualification and calibration')


def condition_nodes(prepared,mask,candidate):
    if candidate is None or candidate.shape!=(len(prepared),1) or prepared.shape[-1]!=41:
        raise ValueError('Explicit scalar candidate and prepared observed graph41 required')
    gain=candidate.astype(prepared.dtype)
    coordinate=(jnp.log(jnp.maximum(gain,1e-6))-math.log(2.))/math.log(4.)
    coordinate=jnp.broadcast_to(coordinate,mask.shape)
    valid=mask&(prepared[...,36]>.5)
    h=jnp.sinh(prepared[...,37]);drift=jnp.sinh(prepared[...,38])
    rhs=jnp.where(valid,jnp.arcsinh(drift+gain*h),0.)
    added=jnp.stack((coordinate,rhs),axis=-1)
    return jnp.concatenate((prepared,jnp.where(mask[...,None],added,0.)),axis=-1)


def reference_features(prepared,mask,candidate):
    """Independent host construction from the exact rounded source features."""
    result=np.zeros((*prepared.shape[:2],2),float)
    for batch,gain in enumerate(np.asarray(candidate).reshape(-1)):
        for node in np.flatnonzero(mask[batch]):
            result[batch,node,0]=math.log(float(gain)/2.)/math.log(4.)
            if prepared[batch,node,36]>.5:
                h=math.sinh(float(prepared[batch,node,37]));drift=math.sinh(float(prepared[batch,node,38]))
                result[batch,node,1]=math.asinh(drift+float(gain)*h)
    return result


def verify_conditioning(features,mask,gains,batch=128):
    """Check every original development node/candidate against host arithmetic."""
    import jax
    from .bicycle_constraint_features import append_constraints
    if features.shape[-1]!=35 or gains.shape[-1]!=1:
        raise ValueError('Original observed bicycle inputs required')
    # Replica gains are intentionally kept, including duplicates.
    def transform(f,m,g):
        f=f.at[...,26:29].set(0.)
        prepared=append_constraints(jnp.where(m[...,None],f,0.),m)
        b,k=g.shape[:2]
        ff=jnp.broadcast_to(prepared[:,None],(b,k,*prepared.shape[1:])).reshape(b*k,*prepared.shape[1:])
        mm=jnp.broadcast_to(m[:,None],(b,k,m.shape[1])).reshape(b*k,m.shape[1])
        extra=condition_nodes(ff,mm,g.reshape(b*k,1))[...,-2:].reshape(b,k,m.shape[1],2)
        return prepared,extra
    fn=jax.jit(transform);exe=None;maximum=0.;valid_count=0
    for start in range(0,len(features),batch):
        stop=min(start+batch,len(features));n=stop-start
        arrays=[np.pad(v[start:stop],[(0,batch-n)]+[(0,0)]*(v.ndim-1),mode='edge') for v in (features,mask,gains)]
        args=(jnp.asarray(arrays[0],dtype=jnp.float64 if features.dtype==np.float64 else jnp.float32),
            jnp.asarray(arrays[1],dtype=bool),jnp.asarray(arrays[2],dtype=jnp.float32))
        if exe is None:exe=fn.lower(*args).compile()
        prepared,actual=jax.device_get(exe(*args));prepared=prepared[:n].astype(float);actual=actual[:n]
        mm=mask[start:stop];gg=gains[start:stop,:,0].astype(float)
        expected=np.zeros(actual.shape,float)
        expected[...,0]=np.log(gg[:,:,None]/2.)/math.log(4.)
        h=np.sinh(prepared[...,37]);drift=np.sinh(prepared[...,38]);valid=mm&(prepared[...,36]>.5)
        expected[...,1]=np.where(valid[:,None,:],np.arcsinh(drift[:,None,:]+gg[:,:,None]*h[:,None,:]),0.)
        expected=np.where(mm[:,None,:,None],expected,0.)
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
        maximum=max(maximum,float(np.max(np.abs(actual-expected))));valid_count+=int(mm.sum())*gains.shape[1]
    if fn._cache_size():raise ValueError('Unexpected candidate-feature audit recompilation')
    return dict(every_candidate_feature_independently_verified=True,queries=len(features),
        candidate_values=int(np.prod(gains.shape[:2])),valid_node_candidates=valid_count,
        maximum_absolute_error=maximum,implicit_jit_cache_entries=0,
        feature_input_dtype=str(args[0].dtype),
        no_physics_solver_labels_or_future_inputs=True)
