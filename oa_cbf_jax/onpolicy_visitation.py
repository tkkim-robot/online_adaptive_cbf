"""Frozen OA-CBF acquisition policy and causal extraction of visited contexts."""

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .adaptive import PolicyConfig
from .dataset import sha256


def acquisition_contract(bundle, calibration, *, experiment=None, visitation_steps=400,
                         fixed_gain=None, robot=None, gain_upper=4., controller=None):
    """Resolve learned acquisition or an explicitly declared fixed behavior.

    A fixed controller provides causal noisy histories for a fresh dataset without
    pretending old moving-obstacle weights were calibrated for the static plant.
    It is collection behavior only, never a learned policy or gain-search oracle.
    """
    if fixed_gain is None:
        return visitation_contract(bundle, calibration, experiment=experiment,
                                   visitation_steps=visitation_steps)
    if bundle is not None or calibration is not None or experiment is not None:
        raise ValueError('Fixed acquisition cannot also use learned acquisition assets')
    if robot is None or not isinstance(visitation_steps, int) or isinstance(visitation_steps, bool) or visitation_steps < 2:
        raise ValueError('Fixed acquisition requires robot and at least two observations')
    gain = np.asarray(fixed_gain, dtype=float)
    if gain.shape != (2,) or not np.isfinite(gain).all() or np.any(gain < .3) or np.any(gain > gain_upper):
        raise ValueError('Fixed acquisition gain must lie inside the queried gain domain')
    config = PolicyConfig(mode='fixed', fixed_gain=tuple(gain), backup_gains=(),
                          **(controller or {}))
    return dict(mode='fixed_behavior', policy=json.loads(json.dumps(asdict(config))),
                robot=asdict(robot), pool=[gain.tolist()], queries=1,
                gain_domain=dict(lower=.3, upper=float(gain_upper)),
                behavior_horizon=visitation_steps, noise_seed_offset=12819,
                visitation_seed_offset=25903,
                acquisition='50% initial; 50% uniformly selected available pre-action observations under a declared fixed gain; one observation per independent parent; no outcome rejection sampling',
                noise='Actual latent acquisition state, bounded raw readings and causal observer memory are copied; replicas vary future sensor innovations only.',
                interpretation='Fixed collection behavior, no neural weights or calibration, no online gain search; all candidate labels use actual simulator continuations.')


def visitation_contract(bundle,calibration,queries=16,experiment=None,visitation_steps=400):
    if isinstance(visitation_steps,bool) or not isinstance(visitation_steps,int) or visitation_steps<1:
        raise ValueError('Visitation steps must be a positive integer')
    if not bundle and not calibration and experiment is None:
        if visitation_steps!=400:raise ValueError('Custom visitation steps require a frozen acquisition policy')
        return None
    if not bundle or not calibration:raise ValueError('Both visitation bundle and calibration are required')
    bundle=Path(bundle).resolve();calibration=Path(calibration).resolve()
    model=json.loads((bundle/'manifest.json').read_text());cal=json.loads(calibration.read_text())
    if model['weights_sha256']!=cal['weights_sha256']:raise ValueError('Visitation calibration does not match the model')
    domain=model.get('gain_domain',dict(lower=.3,upper=4.))
    gains=tuple((g,g) for g in (1.,2.,3.,4.,8.,12.,16.) if domain['lower']<=g<=domain['upper'])
    config=PolicyConfig(mode='learned',fixed_gain=(4.,4.),backup_gains=gains,reactive_reselection=True,
        sensor_margin_scale=model.get('controller',{}).get('sensor_margin_scale',0.),
        margin_guidance=model.get('controller',{}).get('margin_guidance',False),
        shared_clearance_budget=model.get('controller',{}).get('shared_clearance_budget',False),
        motion_observer_window=model.get('controller',{}).get('motion_observer_window',0),
        filter_obstacle_position=model.get('controller',{}).get('filter_obstacle_position',False))
    result=dict(mode='frozen_oa_cbf',bundle=str(bundle),calibration=str(calibration),queries=queries,
        weights_sha256=model['weights_sha256'],calibration_sha256=sha256(calibration),gain_domain=domain,
        policy=json.loads(json.dumps(asdict(config))),behavior_horizon=visitation_steps,
        acquisition='50% initial; 50% uniformly selected available visited observations; one observation per independent parent; no outcome rejection sampling',
        noise='Known group noise bounds also used during visitation; independent key from the branch replicas. Branches restart the documented conditional physical prior at the sensed observation.',
        noise_seed_offset=12819,visitation_seed_offset=25903)
    if experiment is not None:
        path=Path(experiment).resolve()/'manifest.json'
        saved=json.loads(path.read_text())
        if saved.get('stage')!='development_adaptive_closed_loop' or saved['policy']['mode']!='learned':
            raise ValueError('Visitation requires a recorded learned closed-loop experiment')
        if saved['weights_sha256']!=model['weights_sha256'] or saved['calibration_sha256']!=sha256(calibration):
            raise ValueError('Visitation experiment/model/calibration mismatch')
        from .config import UnicycleConfig
        recorded_robot=UnicycleConfig(**saved['robot'])
        if recorded_robot!=UnicycleConfig(**cal['robot']):
            raise ValueError('Visitation experiment/calibration robot mismatch')
        config=PolicyConfig(**saved['policy'])
        from .sensor_margin import require_matching_controller
        require_matching_controller(model,cal,config.sensor_margin_scale,config.margin_guidance,
                                    config.shared_clearance_budget,config.motion_observer_window,config.filter_obstacle_position)
        if cal.get('gain_domain',dict(lower=.3,upper=4.))!=domain:
            raise ValueError('Visitation model/calibration gain-domain mismatch')
        pool=np.asarray(saved['pool'],np.float32)
        if pool.shape!=(saved['queries'],2) or saved['queries']+1<config.shortlist:
            raise ValueError('Invalid recorded visitation candidate shape or shortlist')
        for values in (pool,np.asarray([config.fixed_gain,*config.backup_gains])):
            if not np.isfinite(values).all() or np.any(values<domain['lower']) or np.any(values>domain['upper']):
                raise ValueError('Recorded visitation gains exceed the calibrated domain')
        result.update(policy=json.loads(json.dumps(asdict(config))),queries=saved['queries'],
            pool=pool.tolist(),robot=asdict(recorded_robot),candidate_design=saved.get('candidate_design','legacy'),
            source_experiment=str(path.parent),source_experiment_manifest_sha256=sha256(path),
            source_experiment_fingerprint=saved.get('source_fingerprint'),
            noise='Known group noise bounds used during visitation; independent acquisition key from branch replicas. Dataset schema records whether branches preserve acquired physical states or restart the conditional prior.',
            policy_capture=f'Exact saved decision settings and candidate values; acquisition still uses its declared{visitation_steps}-step budget and fresh scenes/noise. Package source fingerprint records the executing implementation.')
    return result


def select_visited_contexts(trace,counts,rngs,initial_gain):
    """Use pre-action observation at t and ONLY committed context from t-1.

    The final available observation may be at a terminal decision. It is kept
    without pretending its censored physical future is known. Branch targets
    are subsequently obtained from fresh real simulations of every query.
    """
    limit=trace['observed_state'].shape[1]-1
    snapshot=np.array([rng.integers(1,min(int(n),limit)+1) if min(int(n),limit)>0 and rng.random()<.5 else 0
                       for rng,n in zip(rngs,counts)],np.int32)
    indices=np.arange(len(snapshot))
    observed_x=trace['observed_state'][indices,snapshot].copy()
    observed_obs=trace['observed_obstacles'][indices,snapshot].copy()
    gain=np.broadcast_to(np.asarray(initial_gain,np.float32),(len(snapshot),2)).copy()
    control=np.zeros_like(gain);cursor=np.zeros(len(snapshot),np.float32)
    for i,t in enumerate(snapshot):
        if t:
            if not trace['active'][i,t-1]:raise ValueError('Selected context has no applied predecessor')
            gain[i]=trace['gains'][i,t-1];control[i]=trace['control'][i,t-1];cursor[i]=trace['route_progress'][i,t-1]
    return observed_x,observed_obs,gain,control,cursor,snapshot
