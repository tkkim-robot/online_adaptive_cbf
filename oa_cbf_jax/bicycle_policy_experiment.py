"""Fresh fixed-reference gate trajectories and actual adaptive bicycle rollouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
import json
import shutil
import time
import jax
import jax.numpy as jnp
import numpy as np
from .bicycle_data import prepare as prepare_training_source,args as episode_args,FIELDS
from .bicycle_experiment import read,control_config
from .bicycle_observation import observe,unit_errors
from .bicycle_observed_rollout import make_observed_episode
from .bicycle_control import constant
from .bicycle_policy import BicycleSelector,BicyclePolicyConfig
from .bicycle_rollout import GOAL,COLLISION,TIMEOUT,STATE_BOUND,NAMES
from .dataset import sha256,source_fingerprint
from .io import write_json
from .uncertainty import conformal_threshold


def prepare(output,bundle,prediction_fit,groups=768,seed=6701):
    if groups!=768 or type(seed) is not int or seed<1 or seed*100000+groups+4>=2**32:
        raise ValueError('Reserve768parents and a valid fresh seed before outcomes')
    # Do not regenerate previously inspected policy parents under another name.
    # This checks identities only; no prior outcomes enter source selection.
    proposed=set(range(seed*100000,seed*100000+groups));old=set()
    for path in Path('artifacts/experiments').glob('bicycle_*inputs/scenes.json'):
        if path.parent.resolve()==Path(output).resolve():continue
        old.update(p['seed'] for p in read(path) if 'seed' in p)
    if old&proposed:raise ValueError('Prospective policy seed overlaps prior bicycle parents')
    metadata=read(Path(bundle)/'manifest.json');fit=read(prediction_fit)
    if metadata['controller'].get('routing') is not None:
        raise ValueError('Explicit routing bundle requires its matching source builder')
    from .bicycle_guidance import guidance_from_controller
    guidance_from_controller(metadata['controller'])
    if fit['weights_sha256']!=metadata['weights_sha256'] or fit['controller']!=metadata['controller'] or fit['bundle_manifest_sha256']!=sha256(Path(bundle)/'manifest.json'):
        raise ValueError('Prospective policy requires an exact model/controller fit')
    prepare_training_source(output,groups,seed);root=Path(output);m=read(root/'manifest.json');parents=read(root/'scenes.json')
    write_json(root/'generator_manifest.json',m);rng=np.random.default_rng(seed+11)
    for family in sorted({p['family'] for p in parents}):
        for noise in sorted({p['noise'][0] for p in parents}):
            indices=[i for i,p in enumerate(parents) if p['family']==family and p['noise'][0]==noise]
            if len(indices)!=24:raise ValueError('Wrong family/noise source allocation')
            for k,i in enumerate(rng.permutation(indices)):
                parents[i]['partition']='gate_calibration' if k<8 else 'policy_audit'
                parents[i]['calibration_role']=parents[i]['partition'];parents[i]['acquisition_gain']=BicyclePolicyConfig().initial_gain
    if old&{p['seed'] for p in parents}:raise ValueError('Fresh policy source overlaps training')
    write_json(root/'scenes.json',parents)
    order={phase:rng.permutation([i for i,p in enumerate(parents) if p['partition']==phase]).tolist() for phase in ('gate_calibration','policy_audit')}
    m.update(training_use=False,weight_fit_authorized=False,final_test=False,data_role='fresh_bicycle_trajectory_calibration_and_policy_development_audit',controller=metadata['controller'],
        scenes_sha256=sha256(root/'scenes.json'),generator_manifest_sha256=sha256(root/'generator_manifest.json'),phase_order=order,
        split='Before outcomes: within each family/noise cell8gate-reference and16adaptive-audit parents.256gate/512audit total. No weight training. Seeded within-phase order distributes compute without outcome selection.',
        weights_sha256=read(Path(bundle)/'manifest.json')['weights_sha256'],prediction_fit_sha256=sha256(prediction_fit),policy_config=asdict(BicyclePolicyConfig()),
        limitation='Fresh noisy development trajectories. Static geometric route is not a dynamic feasibility witness. All parents and rejections retained. Fixed-reference calibration and adaptive audit have different trajectory laws; no automatic exchangeable adaptive-coverage claim.')
    write_json(root/'manifest.json',m);print(str(root),flush=True)


def collect(source,bundle,prediction_fit,output,phase,gate=None,shard=0,shards=4,steps=1600,limit_batches=None,storage_floor_gib=150.35,query_every_ticks=1):
    if not np.isfinite(storage_floor_gib) or storage_floor_gib<1:raise ValueError('Positive disk safety buffer required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);source=Path(source);sm=read(source/'manifest.json');parents=read(source/'scenes.json')
    cadence=phase=='cadence_diagnosis'
    from .bicycle_cadence import validate_cadence, held_selection
    validate_cadence(phase,query_every_ticks,sm)
    if sm.get('schema')=='bicycle_selected_package_fresh_confirmation':
        from .bicycle_package_confirmation import validate_source
        validate_source(source)
        if phase!='policy_audit' or gate is None or sha256(gate)!=sm['gate_sha256']:
            raise ValueError('Selected-package confirmation phase/gate required')
    if sm.get('schema')=='bicycle_original_readout_matched_navigation':
        from .bicycle_readout_comparison import validate_source
        validate_source(source)
        if phase!='policy_audit' or gate is None or sha256(gate)!=sm['gate_sha256']:
            raise ValueError('Matched original-policy phase/gate required')
    if sm.get('schema')=='bicycle_dense_readout_fresh_navigation':
        from .bicycle_dense_navigation import validate_source
        validate_source(source)
        if phase!='policy_audit' or gate is None or sha256(gate)!=sm['gate_sha256']:
            raise ValueError('Frozen navigation role/gate required')
    if sm.get('schema')=='bicycle_dense_readout_gate_reference':
        from .bicycle_dense_gate import validate_source
        validate_source(source)
        if phase!='gate_calibration':raise ValueError('Reserved references cannot become adaptive test parents')
    acquisition=phase in ('training_acquisition','cadence_diagnosis','dense_prediction_acquisition')
    if acquisition:
        if phase=='dense_prediction_acquisition':
            from .bicycle_dense_prediction_data import validate_source
        else:
            from .bicycle_acquisition_contracts import validate_source
        validate_source(source)
    if (sm['weight_fit_authorized'] is not acquisition or sha256(source/'scenes.json')!=sm['scenes_sha256'] or sm['prediction_fit_sha256']!=sha256(prediction_fit) or sm['weights_sha256']!=read(Path(bundle)/'manifest.json')['weights_sha256']):
        raise ValueError('Changed fresh policy source or learned lineage')
    if phase not in ('gate_calibration','policy_audit','training_acquisition','cadence_diagnosis','dense_prediction_acquisition') or not 0<=shard<shards:raise ValueError('Invalid policy phase/worker')
    indices=sm['phase_order']['training_acquisition' if cadence else phase][shard::shards]
    if limit_batches is not None:indices=indices[:8*limit_batches]
    if not indices or len(indices)%8 or steps<1:raise ValueError('Fixed full batch8 required')
    reference=phase=='gate_calibration';policy=BicycleSelector(bundle,prediction_fit,gate,reference_recording=reference,config=BicyclePolicyConfig(**sm['policy_config']))
    if sm.get('controller',policy.metadata['controller'])!=policy.metadata['controller'] or (policy.guidance.observation_margin and 'controller' not in sm):
        raise ValueError('Fresh source/learned guidance semantics mismatch')
    validate_routing_contract(sm,policy.metadata)
    from .bicycle_guidance_recovery import runtime_guidance
    policy.guidance=runtime_guidance(sm,policy.guidance,phase)
    c=policy.robot;start=time.monotonic();policy_cold=policy.warm(8,64)
    m=dict(source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),phase=phase,shard=shard,shards=shards,selected_indices=indices,
        config=asdict(c),policy_config=asdict(policy.config),weights_sha256=policy.metadata['weights_sha256'],bundle=str(Path(bundle).resolve()),bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),inference_compute_dtype=policy.predictor.model.config.compute_dtype,prediction_fit=str(Path(prediction_fit).resolve()),prediction_fit_sha256=sha256(prediction_fit),
        gate=None if gate is None else str(Path(gate).resolve()),gate_sha256=None if gate is None else sha256(gate),cs_threshold=None if reference else float(policy.threshold),
        steps=steps,batch=8,query_every_ticks=query_every_ticks,smoke_only=limit_batches is not None,source_fingerprint=source_fingerprint(),controller=policy.metadata['controller'],
        sensor='Same bounded persistent-bias and15percent innovation law; actual current observations. Common parent seed+4 and global-tick fold_in across policies. Initial acquired observation retained.',
        physics='Original audited one-tick physical/GUIDANCE/CBF kernel with the newly selected scalar gain each tick. No physical gain-search rollout. Prior gain fallback is rechecked by actual QP/CBF; rejection censors the task, no fake stationary hold.',
        model_promoted=False,whole_goal_complete=False)
    if cadence:
        m.update(cadence_development=sm['cadence_development'],
            physics='Physical guidance/CBF-QP is re-evaluated every tick. Only neural gain decisions use the declared cadence; intervening ticks hold the last applied gain and still terminate on actual rejection.',
            limitation='Frozen-threshold development component test only. The old trajectory gate is not claimed calibrated for changed cadence; no weight fitting, benchmark result or model promotion.')
    if sm.get('runtime_guidance') is not None:
        m.update(runtime_guidance=asdict(policy.guidance),
            limitation='Explicit frozen-policy guidance ablation. Original learned predictions and thresholds remain fixed; no calibrated-coverage claim for the changed physical trajectory law. Same change for OA and nearest FC, original safety checks retained.')
    if phase=='dense_prediction_acquisition':
        m.update(data_role=sm['data_role'],reservation_sha256=sm['reservation_sha256'],
            weight_fit_rule=sm['weight_fit_rule'],
            limitation='Reserved dense prediction data only. Both frozen policies visit all new parents. Roles were assigned before outcomes; only TRAIN parents may fit later weights. No training or model promotion in this collection.')
    if sm.get('schema')=='bicycle_dense_readout_gate_reference':
        m.update(data_role=sm['data_role'],reservation_sha256=sm['reservation_sha256'],limitation=sm['limitation'])
    if sm.get('schema')=='bicycle_dense_readout_fresh_navigation':
        m.update(data_role=sm['data_role'],reservation_sha256=sm['reservation_sha256'],limitation=sm['limitation'])
    if sm.get('schema')=='bicycle_original_readout_matched_navigation':
        m.update(data_role=sm['data_role'],reservation_sha256=sm['reservation_sha256'],limitation=sm['limitation'])
    if sm.get('schema')=='bicycle_selected_package_fresh_confirmation':
        m.update(data_role=sm['data_role'],reservation_sha256=sm['reservation_sha256'],limitation=sm['limitation'])
    write_json(root/'manifest.json',m)
    def observe_all(x,original,mask,bx,bo,noise,key,t,first_x,first_o):
        current=original.at[:,:,:2].add(t.astype(jnp.float64)*constant(c.robot.dt,jnp.float64)*original[:,:,3:5])
        ix,io=jax.vmap(lambda k:unit_errors(jax.random.fold_in(k,t),64))(key)
        ix=jnp.where(t==0,0.,ix);io=jnp.where(t==0,0.,io)
        seen_x,seen_o=jax.vmap(observe)(x,current,mask,bx,bo,noise,ix,io)
        return current,jnp.where(t==0,first_x,seen_x),jnp.where(t==0,first_o,seen_o),ix,io
    observation_fn=jax.jit(observe_all);physical_fn=jax.jit(jax.vmap(make_observed_episode(c,steps=1,guidance=policy.guidance)));observation_exe=physical_exe=None;index=[];compile_seconds=policy_cold
    for block in range(0,len(indices),8):
        if shutil.disk_usage(root).free/2**30<storage_floor_gib:raise ValueError('Policy experiment disk buffer reached')
        rows=[]
        for i in indices[block:block+8]:
            p=parents[i];rows.append(dict(initial=np.asarray(p['initial'],np.float64),goal=np.asarray(p['goal'],np.float32),obstacles=np.asarray(p['obstacles'],np.float64),mask=np.asarray(p['mask'],bool),
                alpha=np.float32(policy.config.initial_gain),points=np.asarray(p['route']['points'],np.float32),route_mask=np.asarray(p['route']['mask'],bool),ready=p['route']['status']=='ready',cursor=np.float32(0),
                first_x=np.asarray(p['first_x'],np.float32),first_o=np.asarray(p['first_o'],np.float32),bias_x=np.asarray(p['bias_x'],np.float32),bias_o=np.asarray(p['bias_o'],np.float32),noise=np.asarray(p['noise'],np.float32),key=np.array(jax.random.PRNGKey(p['seed']+4))))
        values={key:np.stack([row[key] for row in rows]) for key in FIELDS};state=values['initial'].copy();cursor=values['cursor'].copy();previous_control=np.zeros((8,2),np.float32);previous_gain=values['alpha'].copy()
        alive=np.ones(8,bool);histories=[[] for _ in range(8)];status=np.full(8,TIMEOUT,int)
        motion_history=None
        if policy.motion_history:
            from .bicycle_motion_runtime import ObservationHistory
            motion_history=ObservationHistory(8,64,c.robot.dt)
        def observation_arguments(t):
            raw=(state,values['obstacles'],values['mask'],values['bias_x'],values['bias_o'],values['noise'],values['key'],np.int32(t),values['first_x'],values['first_o'])
            return tuple(jnp.asarray(v,dtype=jnp.float64 if i in (0,1) else None) for i,v in enumerate(raw))
        if observation_exe is None:
            t=time.monotonic();observation_exe=observation_fn.lower(*observation_arguments(0)).compile();compile_seconds+=time.monotonic()-t
        for tick in range(steps):
            current,seen_x,seen_o,ix,io=jax.device_get(observation_exe(*observation_arguments(tick)))
            history_arguments={}
            if motion_history is not None:
                past,elapsed=motion_history.observe(seen_o,tick)
                history_arguments=dict(past_positions=past,history_elapsed=elapsed)
            neural_query=tick%query_every_ticks==0
            if neural_query:
                selection=jax.tree.map(np.asarray,policy.predict(seen_x,values['goal'],seen_o,values['mask'],values['points'],values['route_mask'],cursor,previous_control,previous_gain,values['noise'],**history_arguments))
            else:
                selection=held_selection(selection,previous_gain)
            step_rows=[dict(rows[i],initial=state[i],obstacles=current[i],alpha=selection['controller_gain'][i],cursor=cursor[i],first_x=seen_x[i],first_o=seen_o[i]) for i in range(8)]
            arguments=[episode_args(r) for r in step_rows];arguments=tuple(jnp.stack([a[k] for a in arguments]) for k in range(len(FIELDS)))
            if physical_exe is None:
                t=time.monotonic();physical_exe=physical_fn.lower(*arguments).compile();compile_seconds+=time.monotonic()-t
                print(json.dumps(dict(stage='ready',compiled_core_signatures=3,compile_seconds=compile_seconds,device=str(jax.devices()[0]))),flush=True)
            summaries,traces=jax.device_get(physical_exe(*arguments))
            accepted=traces['active'][:,0]&alive
            for i in np.flatnonzero(alive):
                record={k:v[i,0] for k,v in traces.items()};record.update({k:v[i] for k,v in selection.items()})
                # The one-tick kernel intentionally accepts an acquired first
                # observation; store its real global innovation for full replay.
                record.update(innovation_x=ix[i],innovation_o=io[i],previous_control=previous_control[i].copy(),previous_gain=previous_gain[i],global_tick=np.int32(tick))
                if cadence:record['neural_query']=np.bool_(neural_query)
                histories[i].append(record);status[i]=int(summaries['status'][i])
            state=np.where(accepted[:,None],summaries['final_state'],state);cursor=np.where(accepted,summaries['final_cursor'],cursor)
            previous_control=np.where(accepted[:,None],traces['control'][:,0],previous_control);previous_gain=np.where(accepted,selection['controller_gain'],previous_gain)
            alive&=summaries['status']==TIMEOUT
            if not alive.any():break
        for i,source_index in enumerate(indices[block:block+8]):
            history={k:np.asarray([r[k] for r in histories[i]]) for k in histories[i][0]};count=int(history['active'].sum())
            payload=dict(history,**{k:np.asarray(rows[i][k]) for k in FIELDS},final_status=np.int32(status[i]),expected_steps=np.int32(count),horizon=np.int32(steps))
            # Four collection lanes can write concurrently. Reserve their
            # uncompressed payload bounds before each compressed trace write;
            # preserve every completed trace if unrelated disk use grows.
            bound=sum(np.asarray(v).nbytes for v in payload.values())+4096*len(payload)
            if shutil.disk_usage(root).free < storage_floor_gib*2**30+4*bound+64*2**20:
                raise ValueError('Policy trace write would exhaust the storage reserve')
            filename=f'parent_{source_index:04d}.npz';np.savez_compressed(root/filename,**payload)
            entry=dict(file=filename,sha256=sha256(root/filename),source_index=source_index,group_id=parents[source_index]['group_id'],family=parents[source_index]['family'],status=NAMES[status[i]],status_code=int(status[i]),steps=count,queries=int(history['neural_query'].sum()) if cadence else len(histories[i]),
                maximum_cs=float(np.max(history['cs_score'])),uncertainty_fallback_queries=int(history['uncertainty_fallback'].sum()),learned_queries=int(np.sum(history['selected_index']>=0)),
                applied_gain_changes=int(np.sum(history['active']&(history['controller_gain']!=history['previous_gain']))))
            if cadence:entry['control_ticks']=len(histories[i])
            if 'incumbent_progress_hold' in history:
                entry.update(incumbent_progress_hold_queries=int(history['incumbent_progress_hold'].sum()),
                    incumbent_witness_queries=int(history['incumbent_witness_checked'].sum()),
                    incumbent_recovery_queries=int(np.sum(history['incumbent_witness_checked']&~history['incumbent_witness_feasible'])))
            index.append(entry)
        write_json(root/'index.json',index);print(json.dumps(dict(completed_parents=len(index),total_parents=len(indices),physical_steps=sum(e['steps'] for e in index),elapsed_seconds=time.monotonic()-start)),flush=True)
    caches=dict(observation=observation_fn._cache_size(),physical=physical_fn._cache_size(),policy=policy._function._cache_size())
    if any(caches.values()) or len(policy._compiled)!=1:raise ValueError('Unexpected implicit policy JIT or new runtime signature')
    write_json(root/'summary.json',dict(complete=True,parents=len(index),physical_steps=sum(e['steps'] for e in index),compiled_core_signatures=3,implicit_jit_cache_entries=caches,compile_seconds=compile_seconds,execution_seconds=time.monotonic()-start))


def validate_routing_contract(source_manifest,metadata):
    expected=metadata['controller'].get('routing')
    if source_manifest.get('routing_contract')!=expected:
        raise ValueError('Source routing does not match the trained controller')


def audit(directory,workers=12):
    from .bicycle_policy_audit import audit_one
    root=Path(directory);m=read(root/'manifest.json');source=Path(m['source']);sm=read(source/'manifest.json');parents=read(source/'scenes.json');index=read(root/'index.json')
    from .bicycle_cadence import validate_cadence
    validate_cadence(m['phase'],m.get('query_every_ticks',1),sm)
    if sm.get('schema')=='bicycle_selected_package_fresh_confirmation':
        from .bicycle_package_confirmation import validate_source
        validate_source(source)
        if m['phase']!='policy_audit' or m['gate_sha256']!=sm['gate_sha256']:
            raise ValueError('Changed selected-package confirmation phase/gate')
    if sm.get('schema')=='bicycle_original_readout_matched_navigation':
        from .bicycle_readout_comparison import validate_source
        validate_source(source)
        if m['phase']!='policy_audit' or m['gate_sha256']!=sm['gate_sha256']:
            raise ValueError('Changed matched original-policy phase/gate')
    if sm.get('schema')=='bicycle_dense_readout_fresh_navigation':
        from .bicycle_dense_navigation import validate_source
        validate_source(source)
        if m['phase']!='policy_audit' or m['gate_sha256']!=sm['gate_sha256']:
            raise ValueError('Changed frozen navigation role/gate')
    if sm.get('schema')=='bicycle_dense_readout_gate_reference':
        from .bicycle_dense_gate import validate_source
        validate_source(source)
        if m['phase']!='gate_calibration':raise ValueError('Reserved references used outside calibration')
    if m['phase'] in ('training_acquisition','cadence_diagnosis','dense_prediction_acquisition'):
        if m['phase']=='dense_prediction_acquisition':
            from .bicycle_dense_prediction_data import validate_source
        else:
            from .bicycle_acquisition_contracts import validate_source
        validate_source(source)
    if sha256(source/'manifest.json')!=m['source_manifest_sha256'] or sha256(source/'scenes.json')!=sm['scenes_sha256'] or sha256(m['prediction_fit'])!=m['prediction_fit_sha256']:raise ValueError('Changed policy source/fit')
    if 'bundle_manifest_sha256' in m and (m['bundle_manifest_sha256']!=sha256(Path(m['bundle'])/'manifest.json') or m['bundle_manifest_sha256']!=read(m['prediction_fit'])['bundle_manifest_sha256']):raise ValueError('Changed numerical inference bundle')
    validate_routing_contract(sm,read(Path(m['bundle'])/'manifest.json'))
    if m['gate'] is not None and sha256(m['gate'])!=m['gate_sha256']:raise ValueError('Changed trajectory gate')
    if [e['source_index'] for e in index]!=m['selected_indices']:raise ValueError('Missing or reordered policy parents')
    with ProcessPoolExecutor(workers,mp_context=get_context('spawn')) as pool:
        results=list(pool.map(audit_one,[(str(root),e,m,parents[e['source_index']]) for e in index]))
    from .bicycle_guidance import guidance_from_controller
    margin=guidance_from_controller(read(m['prediction_fit'])['controller']).observation_margin
    if margin and any('margin' not in r for r in results):raise ValueError('Missing applied observation-margin audit')
    if m['policy_config'].get('incumbent_progress',False) and m['phase']!='gate_calibration' and any(not r.get('incumbent',{}).get('incumbent_witness_audit_passed') for r in results):
        raise ValueError('Missing independent incumbent witness audit')
    result=dict(audit_passed=True,manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),all_physical_prefixes_replayed=True,all_current_gain_rows_checked=True,all_policy_graphs_checked=True,all_live_gate_decisions_checked=True,all_recorded_margin_bounds_checked=margin,
        parents=len(index),physical_steps=sum(r['steps'] for r in results),rows=results)
    write_json(root/'independent_replay.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def gate(source,directories,prediction_fit,output):
    source=Path(source);sm=read(source/'manifest.json');parents=read(source/'scenes.json');entries=[];proof=[]
    for directory in map(Path,directories):
        m=read(directory/'manifest.json');a=read(directory/'independent_replay.json')
        if m['phase']!='gate_calibration' or m['smoke_only'] or m['steps']!=1600 or m['source_manifest_sha256']!=sha256(source/'manifest.json') or not a['audit_passed'] or a['manifest_sha256']!=sha256(directory/'manifest.json') or a['index_sha256']!=sha256(directory/'index.json'):raise ValueError('Complete frozen reference trajectories required')
        entries.extend(read(directory/'index.json'));proof.append(dict(directory=str(directory),audit_sha256=sha256(directory/'independent_replay.json'),index_sha256=sha256(directory/'index.json')))
    expected={p['group_id'] for p in parents if p['partition']=='gate_calibration'}
    if len(entries)!=256 or {e['group_id'] for e in entries}!=expected:raise ValueError('Gate reference parent coverage changed')
    by_family={}
    for family in sorted({e['family'] for e in entries}):
        values=[e['maximum_cs'] for e in entries if e['family']==family];threshold=conformal_threshold(values,.95)
        if len(values)!=32 or threshold['status']!='calibrated':raise ValueError('Insufficient family trajectory ranks')
        by_family[family]=threshold
    result=dict(schema='oa_cbf_bicycle_trajectory_gate_v70',weights_sha256=sm['weights_sha256'],prediction_fit_sha256=sha256(prediction_fit),policy_config=sm['policy_config'],source_manifest_sha256=sha256(source/'manifest.json'),
        threshold=max(v['threshold'] for v in by_family.values()),family_thresholds=by_family,reference_parents=256,reference_policy='Frozen scalar gain from policy_config.initial_gain, with the same observed controller and neural query cadence.',
        proof=proof,group_ids=sorted(expected),production_eligible=False,whole_goal_complete=False,
        limitation='Family trajectory-max rank construction on a fixed reference policy; not automatic coverage of the changed adaptive trajectory law, nor universal OOD/physical safety evidence. Fresh512parent adaptive audit follows without retuning.')
    write_json(output,result);print(json.dumps(dict(stage='trajectory_gate_frozen',threshold=result['threshold'],parents=256)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for f in ('output','bundle','prediction_fit'):q.add_argument('--'+f.replace('_','-'),required=True)
    q.add_argument('--seed',type=int,default=6701)
    q=sub.add_parser('collect')
    for f in ('source','bundle','prediction_fit','output','phase'):q.add_argument('--'+f.replace('_','-'),required=True)
    q.add_argument('--gate');q.add_argument('--shard',type=int,default=0);q.add_argument('--shards',type=int,default=4);q.add_argument('--steps',type=int,default=1600);q.add_argument('--limit-batches',type=int)
    q.add_argument('--storage-floor-gib',type=float,default=150.35)
    q.add_argument('--query-every-ticks',type=int,default=1)
    q=sub.add_parser('audit');q.add_argument('--directory',required=True);q.add_argument('--workers',type=int,default=12)
    q=sub.add_parser('gate');q.add_argument('--source',required=True);q.add_argument('--directories',nargs='+',required=True);q.add_argument('--prediction-fit',required=True);q.add_argument('--output',required=True)
    args=vars(p.parse_args());command=args.pop('command');dict(prepare=prepare,collect=collect,audit=audit,gate=gate)[command](**args)
