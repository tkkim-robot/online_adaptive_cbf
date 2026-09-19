"""Grouped route/visited-state data with observation-conditioned noise replicas.

One observation per independently split scene group keeps whole-group bootstrap
exact. Gain queries and noise replicas are descendants of that observation, not
independent rows. This development version still fixes robot dynamics/limits and
does not supply the fresh final calibration or a deployment certificate.
"""

import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from .config import UnicycleConfig, config_hash
from .dataset import sha256, source_fingerprint, load_dataset
from .io import write_json
from .models import route_graph
from .route_control import rollout_route, INADMISSIBLE
from .routing import plan_routes
from .scenes import DIVERSE_FAMILIES, diverse_scene
from .simulation import STATUS_NAMES, COLLISION, INFEASIBLE, GOAL, TIMEOUT
from .stochastic import stochastic_branch, STATE_BOUND_VIOLATION
from .sensor_margin import controller_contract

SCHEMA='oa_cbf_route_sensor_v4'
TARGETS=['negative_min_clearance_div_0.3_capped_below_minus_2','physical_route_progress_div_horizon_distance']
EVENTS=['collision_first','controller_failure_first']
PLANNER_FAILURE=7
NAMES={**STATUS_NAMES,INADMISSIBLE:'hocbf_inadmissible',PLANNER_FAILURE:'planner_failure',STATE_BOUND_VIOLATION:'state_bound_violation'}


def manifest_for(groups,capacity,queries,replicas,horizon,seed,shard_groups,robot=UnicycleConfig(),gain_upper=4.,visitation=None,sensor_margin_scale=0.,margin_guidance=False,shared_clearance_budget=False,physical_continuation=False,motion_observer_window=0,filter_obstacle_position=False,scene_profile='legacy',visibility_batch_nodes=None,route_capacity=32,reuse_dataset=None):
    if scene_profile not in ('legacy','multiscale_v1') or (scene_profile=='multiscale_v1' and capacity!=64):
        raise ValueError('Unknown scene profile or incompatible obstacle capacity')
    if isinstance(route_capacity,bool) or not isinstance(route_capacity,int) or route_capacity<2:raise ValueError('Invalid route capacity')
    if visibility_batch_nodes is not None and visibility_batch_nodes<1:raise ValueError('Invalid visibility batch')
    if groups<80 or groups%8 or groups%shard_groups or queries<8 or queries&(queries-1) or replicas<2 or capacity<16 or horizon<1:
        raise ValueError('Require >=80 groups divisible by 8 and shard size, >=8 power-of-two queries, >=2 replicas, capacity>=16')
    if not np.isfinite(gain_upper) or gain_upper<4. or (gain_upper>4. and queries<16):
        raise ValueError('Expanded gain domain requires finite upper>=4 and at least16 queries')
    records=[]
    for family_index,family in enumerate(DIVERSE_FAMILIES):
        count=groups//8
        for i in range(count):
            scene_seed=seed*100000+family_index*count+i
            group_id=(f'{scene_profile}:' if scene_profile!='legacy' else '')+f'{family}:{scene_seed}'
            # Stable across dataset sizes/revisions: a parent's descendants may
            # never change split merely because more scenes are collected.
            fraction=int.from_bytes(hashlib.sha256(group_id.encode()).digest()[:8],'big')/2**64
            partition='train' if fraction<.7 else 'validation' if fraction<.85 else 'development_calibration'
            records.append(dict(group_id=group_id,family=family,seed=scene_seed,partition=partition))
    result=dict(schema=SCHEMA,stage='route_sensor_development',production_eligible=False,groups=records,
                capacity=capacity,queries=queries,replicas=replicas,gains=queries*replicas,horizon_steps=horizon,
                seed=seed,shard_groups=shard_groups,config=asdict(robot),config_hash=config_hash(robot),
                targets=TARGETS,events=EVENTS,source_fingerprint=source_fingerprint(),graph_features=31,
                gain_domain=dict(lower=.3,upper=float(gain_upper)),
                controller=controller_contract(sensor_margin_scale,margin_guidance,shared_clearance_budget),
                query_design='fixed gains .5/1/2/3, previous and nearby gains, upper-end diagonal, expanded diagonals/asymmetric endpoints when upper>4, remaining log-Sobol queries',
                behavior_gains=behavior_gains(gain_upper).tolist(),
                recorded_audit='one uniformly indexed parent and gain/replica branch in every shard; actual collection controls and observations serialized',
                assumptions=dict(observation='shared branch observation; latent bounded sensor biases plus 15% innovations thereafter',
                   noise='documented uniform conditional synthetic sensor prior in stochastic.py; not an empirically fitted real sensor posterior',
                   speed_bounds='QP bounds tightened for the known sensor speed-error interval; true speed checked after every physical step',
                   acquisition='50% initial observations; 50% uniform available visited steps under random fixed gains; no rejection sampling on outcomes',
                   lineage='one observation per scene; all query/replica descendants remain in the same group and partition',
                   route='shared visibility graph from initial static observations, no future motion oracle',
                   gain_hold='entire branch horizon',obstacle_prediction='observed constant velocity',physical_motion='latent constant velocity',
                   risk='observed minimum until collision or goal/horizon; censored by earlier controller/planner failure',
                   progress='physical route arclength gain near committed route branch, can decrease; masked on earlier collision/controller/planner failure',
                   event_semantics='mutually exclusive first terminal events; no claim about events after termination',
                   limitations='fixed unicycle physics; no online-adaptive visited states yet; development calibration only'))
    if scene_profile!='legacy':
        from .multiscale_scenes import contract
        result['scene_distribution']=contract()
    if visibility_batch_nodes is not None:result['visibility_batch_nodes']=visibility_batch_nodes
    if visitation is not None:
        if 'robot' in visitation and visitation['robot']!=asdict(robot):
            raise ValueError('Recorded visitation robot differs from collection physics/nominal')
        if 'pool' in visitation:
            used=np.asarray([*visitation['pool'],visitation['policy']['fixed_gain'],*visitation['policy']['backup_gains']])
            if np.any(used<.3) or np.any(used>gain_upper):
                raise ValueError('Collection gain domain excludes recorded acquisition gains')
        result['visitation']=visitation
        result.pop('behavior_gains')
        result['assumptions']['acquisition']=visitation['acquisition']
        result['assumptions']['limitations']='fixed unicycle physics; frozen acquisition policy; synthetic conditional sensor prior; development calibration only'
    if physical_continuation:
        if visitation is None:raise ValueError('Physical continuation requires a recorded acquisition policy')
        result['schema']='oa_cbf_route_continuation_v6' if filter_obstacle_position else 'oa_cbf_route_continuation_v5'
        result['controller']=controller_contract(sensor_margin_scale,margin_guidance,shared_clearance_budget,motion_observer_window,filter_obstacle_position)
        result['assumptions'].update(observation='Raw sensor state and causal tracked velocity; feature noise[4] is the conservative observer velocity bound divided by1.15',
            noise='One actual acquired latent physical state per parent; replicas vary future sensor innovations only. No restarted physical prior or claimed exact posterior.',
            limitations='fixed unicycle physics and stable obstacle identities with constant physical velocity; actual acquisition snapshots; development calibration only')
        result['branch_snapshot']='Physical state/obstacles, original constant sensor biases, current raw reading, and observer memory BEFORE processing that reading. Every gain branch receives the same snapshot.'
        if filter_obstacle_position:
            result['assumptions']['position_filter']='Causal bounded innovation intersection, tracking position plus unknown constant bias; no bias removal and no reduction in original geometric uncertainty allowance. Actual position interval memory is copied before the branch reading.'
    elif motion_observer_window or filter_obstacle_position:
        raise ValueError('Observer targets require physical continuation, not a restarted prior')
    if route_capacity!=32:result['route_capacity']=route_capacity
    if reuse_dataset is not None:
        from .route_recovery import origin_for
        result['recovery_origin']=origin_for(reuse_dataset,result)
    return result


def behavior_gains(upper):
    base=[.5,1.,1.5,2.,3.]
    return np.unique(np.r_[base,[4.,upper/2,.75*upper,upper]]).astype(np.float32) if upper>4. else np.array(base,np.float32)


def query_gains(seed,rng,previous,queries,upper):
    sobol=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.log2(queries)))
    query=np.exp(np.log(.3)+sobol*np.log(upper/.3)).astype(np.float32)
    query[:4]=[[.5,.5],[1.,1.],[2.,2.],[3.,3.]];query[4]=previous
    query[5]=np.clip(previous*np.exp(rng.uniform(-.2,.2,2)),.3,upper)
    query[6]=[upper,upper]
    if upper>4.:
        query[7:12]=[[4.,4.],[upper/2,upper/2],[.75*upper,.75*upper],[4.,upper],[upper,4.]]
    return query


def audit(directory):
    root=Path(directory);manifest=json.loads((root/'manifest.json').read_text())
    expected={r['group_id']:r['partition'] for r in manifest['groups']};seen=set();parts={};checks=[]
    for partition in ('train','validation','development_calibration'):
        data=load_dataset(root,partition);ids=data['group_id'].tolist();seen.update(ids)
        valid=data['target_mask'];status=data['status'];observed=(status==GOAL)|(status==TIMEOUT)
        checks.extend([len(ids)==len(set(ids)),all(expected.get(i)==partition for i in ids),
                       bool(np.isfinite(data['features']).all()),bool(np.isfinite(data['target']).all()),
                       bool(np.array_equal(valid[...,1],observed)),bool(np.sum(data['steps'])>0)])
        domain=manifest.get('gain_domain',dict(lower=.3,upper=4.))
        checks.extend(bool(np.isfinite(data[key]).all() and np.all(data[key]>=np.float32(domain['lower'])) and
                           np.all(data[key]<=np.float32(domain['upper']))) for key in ('gains','previous_gain'))
        q,r=manifest['queries'],manifest['replicas']
        truth=data['true_initial_state'].reshape(len(ids),q,r,4)
        checks.append(bool(np.all(truth==truth[:,:1])))
        std=[float(np.std(data['target'][...,i][valid[...,i]])) for i in range(2)]
        checks.append(min(std)>1e-6)
        parts[partition]=dict(groups=len(ids),branches=int(status.size),observed_steps=int(data['steps'].sum()),
             outcomes={NAMES[int(k)]:int(v) for k,v in zip(*np.unique(status,return_counts=True))},
             valid_target_std=std,visited_observations=int((data['snapshot_step']>0).sum()),
             unique_query_sets=len({a.tobytes() for a in data['gains']}),
             replica_risk_variation=float(np.nanmean(np.nanstd(np.where(valid[...,0],data['target'][...,0],np.nan).reshape(len(ids),q,r),axis=-1))))
    checks.append(seen==set(expected))
    return dict(schema=manifest['schema'],contract_valid=all(checks),partitions=parts,unique_groups=len(seen),
                production_eligible=False,group_bootstrap_contract='one observation per independent parent group')


def collect(output,groups=1024,capacity=16,queries=16,replicas=4,horizon_steps=80,seed=4139,shard_groups=32,
            guidance_horizon=0.,guidance_min_speed=0.,worker_id=0,workers=1,
            guidance_kernel='scan',guidance_goal_braking=False,gain_upper=4.,visitation_bundle=None,visitation_calibration=None,sensor_margin_scale=0.,margin_guidance=False,shared_clearance_budget=False,physical_continuation=False,motion_observer_window=0,visitation_experiment=None,filter_obstacle_position=False,scene_profile='legacy',visibility_batch_nodes=None,route_capacity=32,reuse_dataset=None,guidance_wide_turns=False,visitation_steps=400,route_workers=1,guidance_detour=False,guidance_turn_return=False):
    if not 0<=worker_id<workers:raise ValueError('Invalid collection worker assignment')
    if isinstance(route_workers,bool) or not isinstance(route_workers,int) or route_workers<1:raise ValueError('Invalid route worker count')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    robot=UnicycleConfig(guidance_horizon=guidance_horizon,guidance_min_speed=guidance_min_speed,
                         guidance_kernel=guidance_kernel,guidance_goal_braking=guidance_goal_braking,
                         guidance_wide_turns=guidance_wide_turns,guidance_detour=guidance_detour,guidance_turn_return=guidance_turn_return)
    from .onpolicy_visitation import visitation_contract,select_visited_contexts
    visitation=visitation_contract(visitation_bundle,visitation_calibration,experiment=visitation_experiment,visitation_steps=visitation_steps)
    manifest=manifest_for(groups,capacity,queries,replicas,horizon_steps,seed,shard_groups,robot,gain_upper,visitation,sensor_margin_scale,margin_guidance,shared_clearance_budget,physical_continuation,motion_observer_window,filter_obstacle_position,scene_profile,visibility_batch_nodes,route_capacity,reuse_dataset)
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Dataset contract changed; use a fresh output directory')
    if not path.exists():write_json(path,manifest)
    behavior_horizon=visitation['behavior_horizon'] if visitation is not None else 400
    visit=jax.jit(jax.vmap(lambda x,g,o,m,a,p,r:rollout_route(x,g,o,m,a,p,r,config=robot,steps=behavior_horizon)))
    if visitation is not None:
        from .adaptive import DevelopmentPolicy,PolicyConfig
        from .adaptive_experiment import candidate_pool
        from .closed_loop import make_closed_loop
        teacher=DevelopmentPolicy(visitation_bundle,visitation_calibration,PolicyConfig(**visitation['policy']),robot=robot,allow_development=True)
        pool=jnp.asarray(visitation['pool'] if 'pool' in visitation else candidate_pool(visitation['queries'],visitation['gain_domain']['upper']),jnp.float32)
        visit=jax.jit(jax.vmap(make_closed_loop(teacher,behavior_horizon,batch_axis='scenes'),
                    in_axes=(None,None,0,0,0,0,None,0,0,0,0,0),axis_name='scenes'))
    encode=jax.jit(jax.vmap(lambda *args:route_graph(*args,config=robot)))
    def branches(x,g,o,m,alpha,p,r,c,noise,keys):
        return jax.vmap(lambda a,key:stochastic_branch(x,g,o,m,a,p,r,c,noise,key,config=robot,steps=horizon_steps,
            sensor_margin_scale=sensor_margin_scale,margin_guidance=margin_guidance,shared_clearance_budget=shared_clearance_budget))(alpha,keys)
    if physical_continuation:
        from .continuation import rollout as continuation_rollout,snapshot_payload
        from .continuation_acquisition import make_extractor
        extract=make_extractor(robot,motion_observer_window,filter_obstacle_position)
        def branches(snapshot,g,m,alpha,p,r,c,noise,keys):
            return jax.vmap(lambda a,key:continuation_rollout(snapshot,g,m,a,p,r,c,noise,key,config=robot,steps=horizon_steps,
                sensor_margin_scale=sensor_margin_scale,margin_guidance=margin_guidance,shared_clearance_budget=shared_clearance_budget,
                motion_observer_window=motion_observer_window,filter_obstacle_position=filter_obstacle_position))(alpha,keys)
    @jax.jit
    def simulate(audit_parent,audit_branch,*args):
        result,trace,truth=jax.vmap(branches)(*args)
        return result,truth,jax.tree.map(lambda a:a[audit_parent,audit_branch],trace)
    index=[];start=time.perf_counter()
    for number,offset in enumerate(range(0,groups,shard_groups)):
        if number%workers!=worker_id:continue
        shard=root/f'shard_{number:05d}.npz';meta=shard.with_suffix('.json')
        if shard.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if sha256(shard)!=record['sha256']:raise ValueError('Corrupt dataset shard')
            if 'visitation_file' in record and sha256(root/record['visitation_file'])!=record['visitation_sha256']:
                raise ValueError('Corrupt acquisition trace')
            index.append(record);continue
        tick=time.perf_counter();entries=manifest['groups'][offset:offset+shard_groups]
        audit_rng=np.random.default_rng(entries[0]['seed']+51941)
        audit_parent=int(audit_rng.integers(shard_groups));audit_branch=int(audit_rng.integers(queries*replicas))
        scene_function=diverse_scene
        if scene_profile=='multiscale_v1':
            from .multiscale_scenes import scene as scene_function
        scenes=[scene_function(e['seed'],e['family'],capacity) for e in entries]
        planning_start=time.perf_counter()
        routes,route_times=plan_routes(scenes,robot,workers=route_workers,capacity=route_capacity,visibility_batch_nodes=visibility_batch_nodes)
        planning_wall_seconds=time.perf_counter()-planning_start
        arrays=[np.stack([getattr(s,k) for s in scenes]) for k in ('initial_state','goal','obstacles','obstacle_mask')]
        x,g,o,m=arrays;points=np.stack([r.points for r in routes]);route_mask=np.stack([r.mask for r in routes])
        rngs=[np.random.default_rng(e['seed']+6163) for e in entries]
        behavior=np.array([[rng.choice(behavior_gains(gain_upper))]*2 for rng in rngs],np.float32)
        as_jax=lambda a:jnp.asarray(a,dtype=bool if a.dtype==bool else jnp.float32)
        if visitation is None:
            visited,trace=visit(*(as_jax(a) for a in (x,g,o,m,behavior,points,route_mask)))
            visited,trace=jax.device_get((visited,trace))
            snapshot=np.array([rng.integers(1,int(n)+1) if n>0 and rng.random()<.5 else 0 for rng,n in zip(rngs,visited.steps)],np.int32)
            cursor=np.zeros(shard_groups,np.float32);previous_u=np.zeros((shard_groups,2),np.float32)
            observed_x=x.copy();observed_obs=o.copy()
            for i,t in enumerate(snapshot):
                if t:
                    observed_x[i]=trace['state'][i,t-1];cursor[i]=trace['route_progress'][i,t-1];previous_u[i]=trace['control'][i,t-1]
                    observed_obs[i,:,:2]+=t*robot.dt*o[i,:,3:5]
        else:
            noise_values=np.array([(np.random.default_rng(e['seed']+visitation['noise_seed_offset']).uniform(.25,2.) if e['seed']%8 else 0.)
                *np.array([.02,.03,.02,.03,.025,.01]) for e in entries],np.float32)
            visit_keys=np.stack([np.asarray(jax.random.PRNGKey(e['seed']+visitation['visitation_seed_offset'])) for e in entries])
            ready=np.array([r.status=='ready' for r in routes])
            visited,trace,visit_truth=visit(teacher.params,teacher.calibration,*(as_jax(a) for a in (x,g,o,m)),pool,
                                          as_jax(points),as_jax(route_mask),as_jax(noise_values),jnp.asarray(visit_keys),jnp.asarray(ready))
            visited,trace,visit_truth=jax.device_get((visited,trace,visit_truth))
            observed_x,observed_obs,behavior,previous_u,cursor,snapshot=select_visited_contexts(trace,visited.steps,rngs,teacher.config.fixed_gain)
            # Random parent avoids the zero-noise seed at every shard boundary.
            ap=audit_parent
            np.savez_compressed(root/f'visitation_trace_{number:05d}.npz',**{k:v[ap] for k,v in trace.items()},
                true_initial_state=visit_truth['initial_state'][ap],true_obstacles=visit_truth['obstacles'][ap],
                obstacle_mask=m[ap],noise=noise_values[ap],key=visit_keys[ap],group_id=entries[ap]['group_id'],parent=np.int32(ap),
                selected_snapshot=snapshot[ap],expected_steps=visited.steps[ap],expected_status=visited.status[ap])
        gains=[];keys=[];noise=[]
        for group_index,(rng,e,a) in enumerate(zip(rngs,entries,behavior)):
            query=query_gains(e['seed'],rng,a,queries,gain_upper)
            gains.append(np.repeat(query,replicas,axis=0))
            replica_keys=np.asarray(jax.random.split(jax.random.PRNGKey(e['seed']+761),replicas))
            keys.append(np.tile(replica_keys,(queries,1)))
            if visitation is None:
                scale=rng.uniform(.25,2.) if e['seed']%8 else 0.
                noise.append(scale*np.array([.02,.03,.02,.03,.025,.01]))
            else:noise.append(noise_values[group_index])
        gains=np.asarray(gains,np.float32);noise=np.asarray(noise,np.float32);keys=np.asarray(keys,np.uint32)
        extra={};estimated_obs=observed_obs;feature_noise=noise
        if physical_continuation:
            copied,estimated_obs,feature_noise=extract(jax.tree.map(jnp.asarray,trace),jax.tree.map(jnp.asarray,visit_truth),
                jnp.asarray(snapshot),as_jax(m),as_jax(noise))
            extra=jax.tree.map(np.asarray,snapshot_payload(copied))
            estimated_obs,feature_noise=map(np.asarray,(estimated_obs,feature_noise))
            result,truth,audit_trace=simulate(jnp.int32(audit_parent),jnp.int32(audit_branch),copied,
                *(as_jax(a) for a in (g,m,gains,points,route_mask,cursor,noise)),jnp.asarray(keys))
        else:
            result,truth,audit_trace=simulate(jnp.int32(audit_parent),jnp.int32(audit_branch),*(as_jax(a) for a in (observed_x,g,observed_obs,m,gains,points,route_mask,cursor,noise)),jnp.asarray(keys))
        result,truth,audit_trace=jax.device_get((result,truth,audit_trace))
        status=np.asarray(result.status).copy();status[[r.status!='ready' for r in routes]]=PLANNER_FAILURE
        risk=-np.minimum(result.min_clearance,.6)/.3
        progress=truth['route_progress_delta']/(horizon_steps*robot.dt*robot.v_max)
        complete=(status==GOAL)|(status==TIMEOUT);risk_valid=complete|(status==COLLISION)
        valid=np.stack((risk_valid,complete),axis=-1)
        target=np.where(valid,np.stack((risk,progress),axis=-1),0.).astype(np.float32)
        failure=(status==INFEASIBLE)|(status==INADMISSIBLE)|(status==STATE_BOUND_VIOLATION)
        events=np.stack((status==COLLISION,failure),axis=-1).astype(np.float32)
        event_mask=np.broadcast_to((status!=PLANNER_FAILURE)[...,None],events.shape)
        features,node_mask=encode(*(as_jax(a) for a in (observed_x,g,estimated_obs,m,points,route_mask,cursor,previous_u,behavior,feature_noise)))
        payload=dict(features=np.asarray(features),node_mask=np.asarray(node_mask),gains=gains,target=target,target_mask=valid,
           events=events,event_mask=event_mask,status=status,steps=np.where(status==PLANNER_FAILURE,0,result.steps),
           min_clearance=result.min_clearance,min_psi1=result.min_psi1,worst_qp_violation=result.worst_qp_violation,
           final_state=result.final_state,true_initial_state=truth['initial_state'],true_obstacles=truth['obstacles'],
           raw_goal_progress=result.progress,raw_route_progress=truth['route_progress_delta'],
           group_id=np.asarray([e['group_id'] for e in entries]),partition=np.asarray([e['partition'] for e in entries]),
           initial_state=observed_x,goal=g,obstacles=observed_obs,obstacle_mask=m,route_points=points,route_mask=route_mask,
           route_progress=cursor,previous_control=previous_u,previous_gain=behavior,noise=noise,snapshot_step=snapshot,
           parent_initial_state=x,parent_obstacles=o,behavior_status=visited.status,
           route_status=np.asarray([r.status for r in routes]),planning_seconds=np.asarray(route_times),
           replica_key=keys,replica_id=np.broadcast_to(np.tile(np.arange(replicas),queries),(shard_groups,queries*replicas)))
        if physical_continuation:
            payload.update(extra,estimated_obstacles=estimated_obs,feature_noise=feature_noise)
            if not np.all(np.asarray(truth['initial_state'])==extra['snapshot_physical_state'][:,None]):
                raise ValueError('Physical continuation initial states differ from acquisition')
            if not np.all(np.asarray(truth['obstacles'])==extra['snapshot_physical_obstacles'][:,None]):
                raise ValueError('Physical continuation obstacles differ from acquisition')
        if not np.isfinite(target).all() or not np.isfinite(payload['features']).all():raise ValueError('Nonfinite unmasked training values')
        temp=shard.with_suffix('.tmp')
        with temp.open('wb') as file:np.savez_compressed(file,**payload)
        temp.replace(shard)
        np.savez_compressed(root/f'audit_trace_{number:05d}.npz',**audit_trace,
            true_initial_state=truth['initial_state'][audit_parent,audit_branch],true_obstacles=truth['obstacles'][audit_parent,audit_branch],
            obstacle_mask=m[audit_parent],alpha=gains[audit_parent,audit_branch],group_id=np.asarray(entries[audit_parent]['group_id']),
            partition=np.asarray(entries[audit_parent]['partition']),branch=np.int32(audit_branch),parent=np.int32(audit_parent),
            noise=noise[audit_parent],route_ready=np.asarray(routes[audit_parent].status=='ready'),
            expected_final_state=result.final_state[audit_parent,audit_branch],expected_min_clearance=result.min_clearance[audit_parent,audit_branch],
            expected_steps=result.steps[audit_parent,audit_branch],expected_status=result.status[audit_parent,audit_branch])
        record=dict(file=shard.name,sha256=sha256(shard),groups=shard_groups,branches=int(status.size),
            route_workers=route_workers,planning_wall_seconds=planning_wall_seconds,
            collection_source_fingerprint=manifest['source_fingerprint'],collection_route_capacity=route_capacity,stored_route_capacity=route_capacity,
                    audit_file=f'audit_trace_{number:05d}.npz',audit_sha256=sha256(root/f'audit_trace_{number:05d}.npz'),
                    observed_steps=int(payload['steps'].sum()),visited_observations=int((snapshot>0).sum()),
                    outcomes={NAMES[int(k)]:int(v) for k,v in zip(*np.unique(status,return_counts=True))},wall_seconds=time.perf_counter()-tick)
        if visitation is not None:
            record.update(visitation_file=f'visitation_trace_{number:05d}.npz',visitation_sha256=sha256(root/f'visitation_trace_{number:05d}.npz'))
        write_json(meta,record);index.append(record)
        progress_path=root/('progress.json' if workers==1 else f'progress_worker_{worker_id}.json')
        write_json(progress_path,dict(completed_groups=len(index)*shard_groups,total_groups=groups,worker_id=worker_id,workers=workers,elapsed_seconds=time.perf_counter()-start))
        print(json.dumps(record),flush=True)
    signatures=dict(visit=visit._cache_size(),simulate=simulate._cache_size(),encode=encode._cache_size())
    if physical_continuation:signatures["extract"]=extract._cache_size()
    if workers>1:
        write_json(root/f'worker_{worker_id}_complete.json',dict(worker_id=worker_id,shards=len(index),elapsed_seconds=time.perf_counter()-start,jit_signatures=signatures))
        return
    finalize(root,time.perf_counter()-start,signatures)


def finalize(root,elapsed,signatures):
    root=Path(root);path=root/'manifest.json';manifest=json.loads(path.read_text())
    expected=len(manifest['groups'])//manifest['shard_groups']
    index=[]
    for number in range(expected):
        record=json.loads((root/f'shard_{number:05d}.json').read_text())
        if sha256(root/record['file'])!=record['sha256']:raise ValueError('Missing/corrupt collection shard')
        if 'audit_file' in record and sha256(root/record['audit_file'])!=record['audit_sha256']:raise ValueError('Corrupt recorded controls')
        if 'visitation_file' in record and sha256(root/record['visitation_file'])!=record['visitation_sha256']:raise ValueError('Corrupt acquisition trace')
        index.append(record)
    write_json(root/'index.json',index);report=audit(root);write_json(root/'audit.json',report)
    write_json(root/'complete.json',dict(status='completed',elapsed_seconds=elapsed,audit_passed=report['contract_valid'],
               manifest_sha256=sha256(path),jit_signatures=signatures))
    print(json.dumps(report),flush=True)
    if not report['contract_valid']:raise ValueError('Route dataset contract audit failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    for name,default in [('groups',1024),('capacity',16),('queries',16),('replicas',4),('horizon_steps',80),('seed',4139),('shard_groups',32)]:
        parser.add_argument('--'+name.replace('_','-'),type=int,default=default)
    parser.add_argument('--guidance-horizon',type=float,default=0.)
    parser.add_argument('--guidance-min-speed',type=float,default=0.)
    parser.add_argument('--guidance-kernel',choices=['scan','vectorized'],default='scan')
    parser.add_argument('--guidance-goal-braking',action='store_true')
    parser.add_argument('--guidance-wide-turns',action='store_true')
    parser.add_argument('--guidance-detour',action='store_true')
    parser.add_argument('--guidance-turn-return',action='store_true')
    parser.add_argument('--visitation-steps',type=int,default=400)
    parser.add_argument('--route-workers',type=int,default=1)
    parser.add_argument('--gain-upper',type=float,default=4.)
    parser.add_argument('--sensor-margin-scale',type=float,default=0.)
    parser.add_argument('--margin-guidance',action='store_true')
    parser.add_argument('--shared-clearance-budget',action='store_true')
    parser.add_argument('--physical-continuation',action='store_true')
    parser.add_argument('--motion-observer-window',type=int,default=0)
    parser.add_argument('--filter-obstacle-position',action='store_true')
    parser.add_argument('--visitation-bundle');parser.add_argument('--visitation-calibration')
    parser.add_argument('--visitation-experiment',help='Capture exact decision settings and candidate values from a saved learned experiment')
    parser.add_argument('--scene-profile',choices=['legacy','multiscale_v1'],default='legacy')
    parser.add_argument('--visibility-batch-nodes',type=int)
    parser.add_argument('--route-capacity',type=int,default=32);parser.add_argument('--reuse-dataset')
    parser.add_argument('--worker-id',type=int,default=0);parser.add_argument('--workers',type=int,default=1)
    collect(**vars(parser.parse_args()))
