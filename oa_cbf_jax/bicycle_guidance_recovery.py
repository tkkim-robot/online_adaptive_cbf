"""Explicit frozen-policy guidance ablation and independent action-bound audit."""
from dataclasses import asdict, replace
import numpy as np


def runtime_guidance(source, original, phase):
    requested=source.get('runtime_guidance')
    if requested is None:return original
    preserve=requested.get('recovery_preserve_preview',False)
    if type(preserve) is not bool:raise ValueError('Explicit preview preservation flag required')
    deterministic=requested.get('deterministic_witness',False)
    if type(deterministic) is not bool:raise ValueError('Explicit deterministic witness flag required')
    expected=replace(original,recover_margin=True,recovery_preserve_preview=preserve,deterministic_witness=deterministic)
    # Historical sources predate the added disabled flag; missing means False.
    normalized=dict(requested,recovery_preserve_preview=preserve,deterministic_witness=deterministic)
    ordinary=(phase=='policy_audit' and source.get('data_role')=='frozen_policy_margin_recovery_development'
              and source.get('weight_fit_authorized') is False)
    acquisition=(phase=='dense_prediction_acquisition'
                 and source.get('data_role')=='reserved_dense_bicycle_prediction_acquisition'
                 and source.get('weight_fit_authorized') is True
                 and source.get('training_use') is True and preserve and not deterministic)
    reference=False
    if phase=='gate_calibration' and source.get('data_role')=='reserved_dense_readout_trajectory_calibration':
        from .bicycle_dense_gate import validate_declaration
        validate_declaration(source)
        reference=preserve and not deterministic
    if (not (ordinary or acquisition or reference) or source.get('final_test') is not False
            or normalized!=asdict(expected)):
        raise ValueError('Unreserved or changed runtime guidance ablation')
    return expected


def audit_recovery(data, config, guidance=None):
    """Recompute every candidate bound and changed first input independently.

    Controls remain subject to the original CBF rows and sensor-speed bounds.
    This validates the experiment, not recursive feasibility or calibration.
    """
    from .bicycle_margin_audit import reference_margin
    from .bicycle_audit import reference_rows
    from .bicycle_observed_audit import recorded_gain
    controls=np.asarray(data['guidance_recovery_first_controls'],float)
    saved=np.asarray(data['guidance_recovery_profile_lower'],float)
    n,p,_=controls.shape
    if 'controller_gain' in data:
        gains=np.asarray(data['controller_gain'],float)
    else:
        # Held-gain label branches record a scalar alpha, whereas online
        # trajectories record the gain chosen at every query. Audit both
        # against their actual applied gain without changing saved traces.
        gain=recorded_gain(data,data['mask'])
        if np.ndim(gain):raise ValueError('Recovery audit requires a scalar bicycle gain')
        gains=np.full(n,gain,dtype=float)
    if gains.shape!=(n,) or not np.isfinite(gains).all() or np.any(gains<=0):
        raise ValueError('Invalid recovery gain history')
    if saved.shape!=(n,p) or np.isnan(saved).any():raise ValueError('Missing recovery profile bounds')
    original=np.asarray(data['guidance_recovery_original_index'],int)
    selected=np.asarray(data['guidance_selected'],int)
    if np.any((original<0)|(original>=p)):raise ValueError('Invalid original profile')
    noise=np.asarray(data['noise'],float);c=config.robot;alternate_count=0
    for start in range(0,n,8):
        end=min(start+8,n);u=controls[start:end].copy();valid=np.isfinite(u).all(-1)
        admissible=(data['h'][start:end]>=-config.qp_tolerance)&(data['domain'][start:end]>0)
        valid &= admissible[:,None];u[~valid]=0
        x=np.repeat(data['observed_state'][start:end],p,axis=0)
        o=np.repeat(data['observed_obstacles'][start:end],p,axis=0)
        predicted=reference_margin(x,u.reshape(-1,2),o,data['mask'],noise,config)['lower'].reshape(end-start,p)
        predicted=np.where(valid,predicted,-np.inf)
        different=~np.isclose(predicted,saved[start:end],atol=2e-10,rtol=2e-10)
        if different.any():
            alternate=reference_margin(x,u.reshape(-1,2),o,data['mask'],noise,config,linear_fused=False)['lower'].reshape(end-start,p)
            alternate=np.where(valid,alternate,-np.inf)
            np.testing.assert_allclose(alternate[different],saved[start:end][different],atol=2e-10,rtol=2e-10)
            predicted[different]=alternate[different];alternate_count+=int(different.sum())
        np.testing.assert_allclose(predicted,saved[start:end],atol=2e-10,rtol=2e-10)
        for k in range(start,end):
            good=valid[k-start]
            if not good.any():continue
            a,b,_,_=reference_rows(data['observed_state'][k],data['observed_obstacles'][k],data['mask'],float(gains[k]),config)
            error=1.15*noise[2]+(5e-7 if noise[2]>0 else 0.)
            speed=float(data['observed_state'][k,3])
            b[-4]=min(c.acceleration_max,(c.speed_max-speed-error)/c.dt)
            b[-3]=-max(-c.acceleration_max,(c.speed_min-speed+error)/c.dt)
            if np.max(controls[k,good]@a.T-b)>config.qp_tolerance+1e-10:
                raise ValueError('Recovery profile violates original QP constraints')
    rows=np.arange(n);supported=saved
    if guidance is not None and guidance.recovery_preserve_preview:
        steps=np.asarray(data['guidance_recovery_profile_steps'])
        complete=np.asarray(data['guidance_recovery_profile_complete'])
        if (steps.shape!=saved.shape or complete.shape!=saved.shape or complete.dtype!=bool
                or not np.issubdtype(steps.dtype,np.integer) or np.any((steps<0)|(steps>guidance.horizon))
                or np.any(complete&(steps!=guidance.horizon))):
            raise ValueError('Invalid preview survival accounting')
        eligible=(steps>=steps[rows,original,None]) & (~complete[rows,original,None]|complete)
        supported=np.where(eligible,saved,-np.inf)
        np.testing.assert_array_equal(data['guidance_steps'],steps[rows,selected])
        np.testing.assert_array_equal(data['guidance_complete'],complete[rows,selected])
        np.testing.assert_array_equal(data['guidance_complete_profiles'],complete.sum(1))
    elif 'guidance_recovery_profile_steps' in data or 'guidance_recovery_profile_complete' in data:
        raise ValueError('Unreported preview-preservation variant')
    best=np.argmax(supported,axis=1)
    recover=(np.any(noise>0)&(saved[rows,original]<0)&~np.any(saved>=0,axis=1)
             &np.isfinite(supported[rows,best])&(saved[rows,best]>saved[rows,original]+1e-10))
    if guidance is not None and guidance.deterministic_witness:
        witnesses=np.where(np.isfinite(supported)&(supported>=0),supported,-np.inf)
        witness_best=np.argmax(witnesses,axis=1)
        deterministic=(not np.any(noise>0)) & (saved[rows,original]<0) & np.isfinite(witnesses[rows,witness_best])
        best=np.where(deterministic,witness_best,best)
        recover=recover|deterministic
    np.testing.assert_array_equal(data['guidance_recovery_applied'],recover)
    np.testing.assert_array_equal(selected,np.where(recover,best,original))
    np.testing.assert_array_equal(data['guidance_next_observation_lower'],saved[rows,selected])
    active=np.asarray(data['active'],bool)
    np.testing.assert_array_equal(data['control'][active],controls[rows[active],selected[active]])
    return dict(audit_passed=True,queries=n,profiles=p,recovery_decisions=int(recover.sum()),
        applied_recoveries=int((recover&active).sum()),alternate_rounding_checks=alternate_count)
