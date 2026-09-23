"""Append four observed-history features without replacing controller sensing."""
from pathlib import Path
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
from .bicycle_motion_observer import velocity,numpy_velocity,contract as observer_contract
from .bicycle_experiment import read
from .dataset import sha256

SCHEMA='bicycle_ego_route_observed39_motion_history'


def contract():
    return dict(schema=SCHEMA,input_features=35,graph_features=39,encoded_constraint_features=45,
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
        descriptor_sha256=sha256(descriptor),history_sha256=entry['sha256'],contract=contract(),source_features_sha256=hashlib.sha256(data['features'].tobytes()).hexdigest(),
        augmented_features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),every_feature_independently_verified=True,
        original_features_unchanged=True,independent_max_error=maximum,signatures=1,implicit_jit_cache_entries=0,
        loaded_history_fields=list(history),physical_truth_loaded=False)
    return dict(data,features=features),evidence
