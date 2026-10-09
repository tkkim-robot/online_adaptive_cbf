"""Bicycle features functions and shared contracts."""

import jax.numpy as jnp

from .bicycle_control import BicycleControlConfig

from .routing import route_target_from_position

SCHEMA='bicycle_ego_route_observed35_scalar_gain_v67'

def bicycle_inference_graph(*args,config=BicycleControlConfig(),compute_dtype='float32'):
    """Same observed graph, with explicit, bundle-bound numerical precision."""
    if compute_dtype not in ('float32','float64'):raise ValueError('Unknown graph precision')
    if compute_dtype=='float64':
        args=tuple(jnp.asarray(v,dtype=bool if i in (3,5) else jnp.float64) for i,v in enumerate(args))
    return bicycle_graph(*args,config=config)

def bicycle_graph(x,goal,obstacles,mask,points,route_mask,cursor,previous_control,previous_gain,noise,config=BicycleControlConfig()):
    c=config.robot;n=len(obstacles)+2
    rot=jnp.stack((jnp.stack((jnp.cos(x[2]),-jnp.sin(x[2]))),jnp.stack((jnp.sin(x[2]),jnp.cos(x[2])))))
    positions=jnp.matmul(jnp.concatenate((x[None,:2],goal[None],obstacles[:,:2]))-x[:2],rot,precision='highest')
    velocity=x[3]*jnp.stack((jnp.cos(x[2]),jnp.sin(x[2])))
    velocities=jnp.matmul(jnp.concatenate((velocity[None],jnp.zeros((1,2),x.dtype),obstacles[:,3:5]))-velocity,rot,precision='highest')
    radii=jnp.concatenate((jnp.array([c.radius,0.],x.dtype),obstacles[:,2]))
    clearance=jnp.sqrt(jnp.sum(positions**2,axis=1)+1e-12)-c.radius-radii
    types=jnp.concatenate((jnp.eye(3,dtype=x.dtype)[jnp.array([0,2])],jnp.tile(jnp.array([0.,1.,0.],x.dtype),(n-2,1))))
    ego=jnp.array([x[3]/c.speed_max,c.radius,c.wheel_base,c.rear_axle_distance,c.acceleration_max,c.slip_max,c.speed_min,c.speed_max,c.dt,
        config.clearance_buffer,config.barrier_inflation,config.relative_speed_epsilon],x.dtype)
    target,_,remaining=route_target_from_position(x[:2],x[3],points,route_mask,cursor)
    context=jnp.concatenate((jnp.matmul(goal-x[:2],rot,precision='highest')/5,jnp.matmul(target-x[:2],rot,precision='highest')/5,jnp.array([remaining/10],x.dtype),
        previous_control/jnp.array([c.acceleration_max,c.slip_max],x.dtype),jnp.reshape(jnp.log(previous_gain),(1,)),noise))
    features=jnp.concatenate((types,positions/5,velocities/c.speed_max,radii[:,None],clearance[:,None]/3,jnp.broadcast_to(ego,(n,12)),jnp.broadcast_to(context,(n,14))),axis=1)
    node_mask=jnp.concatenate((jnp.ones(2,bool),mask));return jnp.where(node_mask[:,None],features,0.),node_mask


import math


def contract(gain_domain=None):
    domain=dict(lower=.5,upper=8.) if gain_domain is None else gain_domain
    if domain not in (dict(lower=.5,upper=8.),dict(lower=.0625,upper=64.)):
        raise ValueError('Unqualified candidate encoding gain domain')
    return dict(schema='bicycle_candidate_conditioned_scene_encoding',raw_graph_features=35,
        prepared_graph_features=41,encoded_graph_features=43,
        added_fields=['log_gain_center2_scale_log4','asinh_candidate_constraint_rhs_over_speed_max'],
        candidate_rhs='drift + candidate_gain * h; recover h and drift from original observed constraint coordinates.',
        input='Current observed graph35, mask, original6 deterministic constraint features, queried scalar gain.',
        treatment='Encode each scene-candidate pair before the unchanged Gaussian/event head; identical transform for GAT and matched FC.',
        padding='Both added fields zero on padding; rhs zero on ego/goal or invalid geometric domain.',
        learned_parameters='Original encoder and head; only input projection gains two columns per node.',
        gain_domain=[domain['lower'],domain['upper']],controller_solve=False,physical_truth_used=False,training_labels_used=False,
        limitation='Observed instantaneous inequality coefficients, not a forecast or safety certificate.')

def validate_metadata(metadata):
    from .bicycle_gain_contract import validate_target_metadata
    validate_target_metadata(metadata)
    from .bicycle_features import constraint_features_contract as constraint_contract
    from .bicycle_features import SCHEMA as graph_schema
    architecture=metadata.get('architecture',{})
    domain=metadata.get('gain_domain',dict(lower=.5,upper=8.))
    if domain==dict(lower=.0625,upper=64.):
        from .bicycle_gain_contract import contract as gain_contract, TRAIN_SCHEMA
        if (metadata.get('bicycle_gain_contract')!=gain_contract()
                or metadata.get('dataset_schema')!=TRAIN_SCHEMA
                or metadata.get('offline_wide_gain_pilot') is not True):
            raise ValueError('Wide candidate encoding requires explicit new training contract')
    if (metadata.get('graph_features')!=35 or metadata.get('gain_dimension')!=1
            or architecture.get('bicycle_candidate_encoding') is not True
            or architecture.get('bicycle_constraint_features') is not True
            or architecture.get('bicycle_motion_history',False)
            or architecture.get('bicycle_affine_gain',False)
            or architecture.get('encoder') not in ('gat','matched_fc')
            or metadata.get('bicycle_candidate_encoding_contract')!=contract(domain)
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


def constraint_features_contract():
    return dict(schema='bicycle_graph35_current_constraint_transform',input_features=35,encoded_features=41,
        added_fields=['asinh_domain_over_inflated_radius_squared','positive_domain',
            'asinh_h_over_speed_max','asinh_drift_over_speed_max',
            'asinh_acceleration_authority_times_limit_over_speed_max',
            'asinh_slip_authority_times_limit_over_speed_max'],
        source='Existing observed ego-frame graph35 only; reconstruct its rounded geometry and declared dynamics.',
        precision='FP64 feature arithmetic, cast to original neural dtype; same transform for both encoders.',
        invalid_domain='Keep signed domain and explicit validity; zero undefined barrier derivatives.',
        invalid_or_padded_nodes='Padded nodes and ego/goal added fields are zero.',
        gain_dependent=False,controller_solve=False,training_labels_used=False,
        limitation='Describes instantaneous constraints at graph-rounded observations; not a future feasibility certificate.')

def append_constraints(features,mask):
    """B,N,35 -> B,N,41. Stateless and permutation equivariant."""
    if features.ndim!=3 or features.shape[-1]!=35 or mask.shape!=features.shape[:2]:
        raise ValueError('Current bicycle constraints require batched graph35 and its mask')
    f=jnp.where(mask[...,None],features,0.).astype(jnp.float64);ego=f[:,0,:];ob=f[:,2:,:];valid=mask[:,2:]
    vmax=ego[:,16,None];speed=ego[:,9,None]*vmax;rear=ego[:,12,None]
    radius=(ego[:,10,None]+ego[:,18,None]+ob[...,7])*ego[:,19,None]
    p=ob[...,3:5]*5.;relative=ob[...,5:7]*vmax[...,None]
    distance2=jnp.sum(p*p,-1);distance=jnp.sqrt(jnp.maximum(distance2,1e-24));unit=p/distance[...,None]
    normal=jnp.stack((-unit[...,1],unit[...,0]),axis=-1)
    radial=jnp.sum(unit*relative,-1);lateral=jnp.sum(normal*relative,-1)
    domain=distance2-radius**2;admissible=valid&(domain>0)
    # Benign guards are only arithmetic guards; validity is an explicit input.
    root=jnp.sqrt(jnp.maximum(domain,1e-12));r=jnp.maximum(radius,1e-12)
    speed_relative=jnp.sqrt(jnp.sum(relative*relative,-1)+ego[:,20,None]**2)
    speed_relative=jnp.maximum(speed_relative,1e-12)
    factor=jnp.sqrt(jnp.maximum(ego[:,19,None]**2-1,0))/r;lam=.5*factor
    h=radial+lam*root*lateral**2/speed_relative+factor*root
    dz=p/root[...,None];drad=lateral[...,None]*normal/distance[...,None];dlat=-radial[...,None]*normal/distance[...,None]
    dp=drad+(lam*lateral**2/speed_relative)[...,None]*dz+(2*lam*root*lateral/speed_relative)[...,None]*dlat+factor[...,None]*dz
    dv=unit+(lam*root)[...,None]*(2*lateral[...,None]*normal/speed_relative[...,None]-lateral[...,None]**2*relative/speed_relative[...,None]**3)
    drift=jnp.sum(dp*relative,-1);aa=-dv[...,0];ab=-speed*dp[...,1]-speed**2/jnp.maximum(rear,1e-12)*dv[...,1]
    derived=jnp.stack((h,drift,aa*ego[:,13,None],ab*ego[:,14,None]),axis=-1)/jnp.maximum(vmax[...,None],1e-12)
    extra=jnp.concatenate((jnp.arcsinh(domain/jnp.maximum(radius**2,1e-12))[...,None],admissible[...,None].astype(f.dtype),
        jnp.where(admissible[...,None],jnp.arcsinh(derived),0.)),axis=-1)
    extra=jnp.where(valid[...,None],extra,0.)
    extra=jnp.concatenate((jnp.zeros((len(f),2,6),f.dtype),extra),axis=1).astype(features.dtype)
    return jnp.concatenate((jnp.where(mask[...,None],features,0.),extra),axis=-1)


from pathlib import Path

import hashlib

import numpy as np

import jax


from .bicycle_observation import velocity, numpy_velocity, contract as observer_contract

from .bicycle_control import read

from .io import sha256

MOTION_FEATURES_SCHEMA='bicycle_ego_route_observed39_motion_history'

def motion_features_contract():
    return dict(schema=MOTION_FEATURES_SCHEMA,input_features=35,graph_features=39,encoded_constraint_features=45,
        fields=['estimated_minus_raw_obstacle_velocity_ego_x_over_vmax','estimated_minus_raw_obstacle_velocity_ego_y_over_vmax',
            'secant_fusion_weight','history_available'],observer=observer_contract(),
        ego_goal_padding='All four added fields zero.',raw_graph_unchanged=True,
        controller_sensing_unchanged=True,physical_truth_used=False,future_observations_used=False)

def append_history(features,node_mask,x,current,past,noise,elapsed):
    if features.shape[-1]!=35:raise ValueError('Motion history requires original graph35')
    mask=node_mask[2:];estimate=velocity(current,past,mask,noise,elapsed)
    angle=x[2].astype(jnp.float64);c,s=jnp.cos(angle),jnp.sin(angle)
    rotation=jnp.array([[c,-s],[s,c]],dtype=jnp.float64)
    difference=estimate['velocity']-current[:,3:5].astype(jnp.float64)
    local=jnp.matmul(difference,rotation,precision='highest')/features[0,16].astype(jnp.float64)
    extras=jnp.concatenate((local,jnp.full((len(mask),1),estimate['weight']),
        jnp.full((len(mask),1),estimate['history_available'],dtype=jnp.float64)),axis=1)
    extras=jnp.concatenate((jnp.zeros((2,4),jnp.float64),extras),axis=0).astype(features.dtype)
    return jnp.where(node_mask[:,None],jnp.concatenate((features,extras),axis=1),0.)

def numpy_features(features,node_mask,x,current,past,noise,elapsed):
    ref=numpy_velocity(current,past,node_mask[2:],noise,elapsed);extra=np.zeros((len(node_mask),4),float)
    delta=ref['velocity']-current[:,3:5].astype(float);c,s=np.cos(float(x[2])),np.sin(float(x[2]))
    extra[2:,0]=(c*delta[:,0]+s*delta[:,1])/float(features[0,16])
    extra[2:,1]=(-s*delta[:,0]+c*delta[:,1])/float(features[0,16])
    extra[2:,2]=ref['weight'];extra[2:,3]=ref['history_available']
    return np.where(node_mask[:,None],np.concatenate((features,extra.astype(features.dtype)),axis=-1),0.)

def augment_development(data,role,descriptor,batch=256):
    if role not in ('train','validation') or not (data['partition']==role).all():raise ValueError('Only original development roles')
    source=read(descriptor);proof=read(source['review']);report=read(source['report'])
    if (sha256(source['review'])!=source['review_sha256'] or sha256(source['report'])!=source['report_sha256']
            or proof['status']!='passed' or not proof['warrants_observed_history_learning']
            or not proof['every_query_source_and_causal_history_verified'] or proof['report_sha256']!=source['report_sha256']):
        raise ValueError('Independently reviewed causal history required')
    entry=report['inputs'][role]
    if sha256(entry['path'])!=entry['sha256']:raise ValueError('Changed observed history')
    # Deliberate whitelist: no physical velocity, state, bias, event or future is loaded.
    with np.load(entry['path']) as z:
        history={k:z[k] for k in ('group_id','query_tick','query_origin','observed_obstacles','obstacle_mask','noise','past_positions','elapsed')}
    for k in ('group_id','query_tick','query_origin','observed_obstacles','obstacle_mask','noise'):
        np.testing.assert_array_equal(data[k],history[k],err_msg='History identity: '+k)
    fn=jax.jit(jax.vmap(append_history));exe=None;parts=[];maximum=0.
    columns=[data['features'],data['node_mask'],data['observed_state'],data['observed_obstacles'],history['past_positions'],data['noise'],history['elapsed']]
    with jax.default_device(jax.devices('cpu')[0]):
        for offset in range(0,len(data['group_id']),batch):
            count=min(batch,len(data['group_id'])-offset);ix=np.minimum(np.arange(offset,offset+batch),len(data['group_id'])-1)
            args=tuple(jnp.asarray(a[ix],dtype=bool if i==1 else jnp.float64 if i==6 else jnp.float32) for i,a in enumerate(columns))
            if exe is None:exe=fn.lower(*args).compile()
            actual=np.asarray(exe(*args))[:count]
            refs=np.stack([numpy_features(*(a[i] for a in columns)) for i in range(offset,offset+count)])
            np.testing.assert_allclose(actual,refs,atol=3e-6,rtol=2e-6)
            maximum=max(maximum,float(np.max(abs(actual-refs))));parts.append(actual)
    features=np.concatenate(parts);np.testing.assert_array_equal(features[...,:35],data['features'])
    if fn._cache_size() or not np.isfinite(features).all() or np.any(features[~data['node_mask']]!=0):raise ValueError('Invalid feature or implicit JIT')
    evidence=dict(role=role,queries=len(features),parents=len(set(data['group_id'])),descriptor=str(Path(descriptor).resolve()),
        descriptor_sha256=sha256(descriptor),history_sha256=entry['sha256'],contract=motion_features_contract(),source_features_sha256=hashlib.sha256(data['features'].tobytes()).hexdigest(),
        augmented_features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),every_feature_independently_verified=True,
        original_features_unchanged=True,independent_max_error=maximum,signatures=1,implicit_jit_cache_entries=0,
        loaded_history_fields=list(history),physical_truth_loaded=False)
    return dict(data,features=features),evidence


from .routing import route_geometry

ROUTE_CONTEXT_SCHEMA = 'bicycle_observed_route_samples59'

DISTANCES = (0., .5, 1., 2., 3., 4., 5.5, 7.)

FEATURES = 35 + 3 * len(DISTANCES)

def route_context_contract():
    return dict(schema=ROUTE_CONTEXT_SCHEMA, graph_features=FEATURES,
                arclength_offsets_metres=list(DISTANCES), coordinate_scale_metres=5.,
                frame='current observed ego heading',
                fields_per_sample=['relative_x', 'relative_y', 'route_reaches_offset'],
                route='already committed observed route; local monotone cursor update',
                endpoint='clamp sample to endpoint, retain explicit reaches-offset bit',
                privilege='No physical state, future observation, obstacle forecast or target label',
                use='Offline learning pilot; runtime policy qualification remains required')

def route_context(x, points, route_mask, cursor):
    """Eight static-shape samples of the committed route, in the observed frame."""
    # Padded route coordinates must not affect any arithmetic, even if a caller
    # uses arbitrary padding. The original route itself is never modified.
    points = jnp.where(route_mask[:, None], points, 0.)
    _, updated, _ = route_target_from_position(x[:2], x[3], points, route_mask, cursor)
    vectors, valid, lengths, cumulative = route_geometry(points, route_mask)
    valid = valid & (lengths > 1e-8)
    requested = updated + jnp.asarray(DISTANCES, x.dtype)
    desired = jnp.minimum(requested, cumulative[-1])
    eligible = valid[None, :] & (cumulative[None, 1:] >= desired[:, None] - 1e-6)
    index = jnp.argmax(eligible, axis=1)
    fraction = jnp.clip((desired - cumulative[index]) / jnp.maximum(lengths[index], 1e-12), 0., 1.)
    samples = points[index] + fraction[:, None] * vectors[index]
    rotation = jnp.stack((jnp.stack((jnp.cos(x[2]), -jnp.sin(x[2]))),
                          jnp.stack((jnp.sin(x[2]), jnp.cos(x[2])))))
    relative = jnp.matmul(samples - x[:2], rotation, precision='highest') / 5.
    reaches = (requested <= cumulative[-1] + 1e-6) & jnp.any(valid)
    return jnp.concatenate((relative, reaches[:, None].astype(x.dtype)), axis=1).reshape(-1)

def append_context(features, node_mask, x, points, route_mask, cursor):
    if features.shape[-1] != 35:
        raise ValueError('Route augmentation requires the original observed graph35')
    context = route_context(x, points, route_mask, cursor)
    extended = jnp.concatenate((features, jnp.broadcast_to(context, (len(features), len(context)))), axis=-1)
    return jnp.where(node_mask[:, None], extended, 0.)

def numpy_context(x, points, mask, cursor):
    """Independent scalar reference: walk the active route segment by segment."""
    x = np.asarray(x, float); points = np.asarray(points, float)[np.asarray(mask, bool)]
    segments = np.diff(points, axis=0); lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0., lengths.cumsum()]
    best_distance, updated = float('inf'), float(cursor)
    for i, (segment, length) in enumerate(zip(segments, lengths)):
        if length <= 1e-8:
            continue
        f = np.clip(np.dot(x[:2] - points[i], segment) / length**2, 0., 1.)
        s = cumulative[i] + f * length
        distance = np.sum((points[i] + f * segment - x[:2])**2)
        if cursor - .05 <= s <= cursor + 1. and distance < best_distance:
            best_distance, updated = distance, max(float(cursor), s)
    result = []
    for offset in DISTANCES:
        requested = updated + offset; desired = min(requested, cumulative[-1])
        for i, length in enumerate(lengths):
            if length > 1e-8 and cumulative[i + 1] >= desired - 1e-6:
                fraction = np.clip((desired - cumulative[i]) / length, 0., 1.)
                delta = points[i] + fraction * segments[i] - x[:2]
                c, s = np.cos(x[2]), np.sin(x[2])
                result.extend(((c * delta[0] + s * delta[1]) / 5.,
                               (-s * delta[0] + c * delta[1]) / 5.,
                               float(requested <= cumulative[-1] + 1e-6)))
                break
        else:
            raise ValueError('Route has no nonzero active segment')
    return np.asarray(result, np.float32)

def route_context_augment_development(data, role, batch=256):
    """Transform every audited development query, with an independent check.

    Calibration and benchmark records are deliberately refused by this pilot.
    Both encoders call the exact same transform; original arrays stay intact.
    """
    if role not in ('train', 'validation') or not np.all(data['partition'] == role):
        raise ValueError('Only original TRAIN/validation queries may enter this pilot')
    if data['features'].shape[-1] != 35 or not len(data['features']):
        raise ValueError('Nonempty original graph35 data required')
    rm = np.asarray(data['route_mask'], bool)
    if np.any(rm[:, 1:] & ~rm[:, :-1]) or not (rm.sum(1) >= 2).all():
        raise ValueError('A contiguous committed route with two active points is required')
    inputs = [data[k] for k in ('features', 'node_mask', 'observed_state', 'points', 'route_mask', 'cursor')]
    # CPU is sufficient for this small preprocessing step; keep accelerator
    # memory available for actual optimizer and inference workloads.
    with jax.default_device(jax.devices('cpu')[0]):
        compute = jax.jit(jax.vmap(append_context))
        parts = []
        for start in range(0, len(rm), batch):
            count = min(batch, len(rm) - start)
            block = [np.pad(a[start:start + count], [(0, batch-count)] + [(0, 0)]*(a.ndim-1), mode='edge') for a in inputs]
            parts.append(np.asarray(compute(*block))[:count])
        result = np.concatenate(parts)
        if compute._cache_size() != 1:
            raise ValueError('Unexpected route feature recompilation')
    references = np.stack([numpy_context(x, p, m, c) for x, p, m, c in zip(
        data['observed_state'], data['points'], rm, data['cursor'])])
    np.testing.assert_allclose(result[:, 0, 35:], references, atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(result[..., :35], data['features'])
    if not np.isfinite(result).all() or np.any(result[~data['node_mask']] != 0.):
        raise ValueError('Invalid route features or padded nodes')
    proof = dict(contract=route_context_contract(), role=role, parents=len(np.unique(data['group_id'])), queries=len(rm),
                 source_features_sha256=hashlib.sha256(data['features'].tobytes()).hexdigest(),
                 augmented_features_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                 independent_numpy_max_error=float(np.max(np.abs(result[:, 0, 35:] - references))),
                 original_features_unchanged=True, all_queries_checked=True, feature_signature_count=1)
    return dict(data, features=result), proof
