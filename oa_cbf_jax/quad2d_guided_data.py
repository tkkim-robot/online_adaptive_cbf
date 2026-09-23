"""Observed contexts and actual-workload timing for predictive flight labels.

Exactly one query per parent keeps parent bootstrap/calibration semantics intact.
Sampling rules are assigned before outcomes. Fixed ticks clamp at termination;
episode fractions sample the recorded prefix, using duration only offline.
Requeried labels sample a fresh declared observation-conditioned prior. They are
not a posterior continuation of the acquisition episode's latent physical state.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from .quad2d_control import FlightConfig,flight_config_from_contract
from .quad2d_guidance import GuidanceConfig,NoiseClearanceGuidanceConfig,TerminalGuidanceConfig
from .quad2d_policy import FlightPolicy,FlightPolicyConfig
from .quad2d_rollout import flight_branch,NAMES
from .quad2d_features import flight_graph
from .quad2d_audit import check_trace,check_guidance_trace,check_gain_sources,check_observed_graph
from .quad2d_data import load_gain_bank
from .dataset import sha256,source_fingerprint
from .io import write_json
from .cli import sanitize


def arrays(rows):
    values=[np.asarray([r[k] for r in rows],bool if k=='obstacle_mask' else np.float32) for k in ['initial_state','goal','obstacles','obstacle_mask']]
    values.extend([np.asarray([r['route']['points'] for r in rows],np.float32),np.asarray([r['route']['mask'] for r in rows],bool),
        np.asarray([r['noise'] for r in rows],np.float32),np.asarray([r['seed'] for r in rows],np.uint32),
        np.asarray([r['route']['status']=='ready' for r in rows],bool),np.asarray([r.get('cursor',0.) for r in rows],np.float32)])
    return tuple(values)


def behavior_for_batch(number,mode):
    """75% declared adaptive behavior / 25% fixed over each16batches.

    The staggered schedule gives each of four GPU workers both behaviors. It
    depends only on the original batch number, never the worker or result.
    """
    if mode not in ('fixed','learned','mixed','backup','backup_mixed','matched_learned','matched_held'):raise ValueError('Unknown acquisition behavior')
    if mode=='matched_held':
        return ('learned','matched_fc','backup','backup')[(number+number//4)%4]
    if mode=='matched_learned':
        return 'learned' if (number+number//4)%2==0 else 'matched_fc'
    if mode in ('mixed','backup_mixed'):
        return 'fixed' if (number+number//4)%4==0 else ('learned' if mode=='mixed' else 'backup')
    return mode


def behavior_policy(mode, horizon, fallback_mode):
    if fallback_mode not in ('fixed_set','hold_previous'):raise ValueError('Unknown behavior fallback')
    config = FlightPolicyConfig(mode='learned' if mode=='matched_fc' else mode,
        validation_horizon=horizon) if mode!='fixed' else FlightPolicyConfig(mode='fixed')
    if fallback_mode=='hold_previous':
        if mode=='fixed':raise ValueError('Use held backup for the constant-gain context')
        config=replace(config,backup_gains=())
    return config


def observation_tick(seed,ticks):
    if not ticks or ticks[0]!=0 or list(ticks)!=sorted(set(ticks)) or any(isinstance(t,bool) or not isinstance(t,int) or not 0<=t<1600 for t in ticks):
        raise ValueError('Prespecified unique ascending observation ticks starting at zero required')
    return int(np.random.default_rng(seed+113).choice(ticks))


def acquisition_horizon(ticks, calibration=None):
    """Observation sampling cannot shorten a trajectory-calibrated mission."""
    observation_tick(0,ticks)
    minimum=max(ticks)+1
    if calibration is None or 'trajectory_gate' not in calibration:
        return minimum
    horizon=calibration['trajectory_gate']['steps']
    if isinstance(horizon,bool) or not isinstance(horizon,int) or not minimum<=horizon<=1600:
        raise ValueError('Observation ticks exceed the calibrated mission horizon')
    return horizon


def select_observation(seed,ticks,length,selection='fixed_tick'):
    """One retained parent, with deterministic sampling independent of its score.

    Fractional sampling uses a saved episode's length offline. It never supplies
    future duration to a deployed policy or discards failures/short episodes.
    """
    requested=observation_tick(seed,ticks)
    if isinstance(length,bool) or not isinstance(length,(int,np.integer)) or length<1:
        raise ValueError('A nonempty recorded prefix is required')
    if selection=='fixed_tick':return requested,min(requested,length-1),None
    if selection!='episode_fraction':raise ValueError('Unknown observation selection')
    numerator=int(np.random.default_rng(seed+271).integers(0,9))
    return requested,(numerator*(length-1))//8,numerator/8.


def excluded_parents(directory=Path('artifacts/experiments'),calibration_source=None,pilot_source=None):
    """Keep every explicitly held development/test input out of acquisition."""
    if pilot_source is not None:
        from .quad2d_static_inputs import verify
        if verify(pilot_source)['role']!='pilot':
            raise ValueError('Only independently reserved pilot inputs may use the pilot acquisition exception')
    excluded=set();legacy={'quad2d_v30_policy_pilot_inputs','quad2d_v31_fresh_inputs','quad2d_v35_fresh_inputs','quad2d_v36_new_inputs'}
    for path in sorted(Path(directory).glob('*/manifest.json')):
        scene_file=path.parent/'scenes.json'
        if not scene_file.exists():continue
        m=json.loads(path.read_text())
        own_calibration=(calibration_source is not None and m.get('data_role')=='fresh_predictive_calibration'
            and m.get('weight_fit_authorized') is False and
            (path.parent.resolve()==Path(calibration_source).resolve() or
             (m.get('source') and Path(m['source']).resolve()==Path(calibration_source).resolve())))
        own_pilot=(pilot_source is not None and m.get('data_role')=='pilot'
            and m.get('weight_fit_authorized') is False and m.get('training_use') is True
            and (path.parent.resolve()==Path(pilot_source).resolve() or
                 (m.get('source') and Path(m['source']).resolve()==Path(pilot_source).resolve())))
        if own_calibration or own_pilot:continue
        if path.parent.name in legacy or m.get('training_use') is False or m.get('weight_fit_authorized') is False or m.get('final_test') is True:
            for r in json.loads(scene_file.read_text()):
                parent=r.get('group_id',r.get('scene',{}).get('scene_id'))
                if not isinstance(parent,str) or not parent:raise ValueError('Unknown held evaluation parent schema: '+str(scene_file))
                excluded.add(parent)
    return sorted(excluded)


def prepare(source,output,batch=8,shard_index=0,shards=1,mode='fixed',bundle=None,calibration=None,gain_dataset=None,ticks=(0,40,120,240),noise_clearance_weight=0.,terminal_transition_distance=0.,observation_selection='fixed_tick',fc_bundle=None,fc_calibration=None,fallback_mode='fixed_set'):
    if mode=='matched_held' and fallback_mode!='hold_previous':raise ValueError('Matched held acquisition requires the held-gain contract')
    source=Path(source);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    sm=json.loads((source/'manifest.json').read_text());rows=json.loads((source/'scenes.json').read_text());c=flight_config_from_contract(sm['config'])
    g=NoiseClearanceGuidanceConfig(noise_clearance_weight=noise_clearance_weight) if noise_clearance_weight else GuidanceConfig()
    if not np.isfinite(terminal_transition_distance) or terminal_transition_distance<0:raise ValueError('Invalid terminal transition')
    if terminal_transition_distance:
        if not noise_clearance_weight:raise ValueError('Terminal guidance requires the noise-aware parent contract')
        g=TerminalGuidanceConfig(noise_clearance_weight=noise_clearance_weight,terminal_transition_distance=terminal_transition_distance)
    reserved=sm.get('data_role')=='fresh_predictive_calibration' and sm.get('weight_fit_authorized') is False and sm.get('training_use') is False
    if sha256(source/'scenes.json')!=sm['scenes_sha256'] or len(rows)%batch or not (sm.get('training_use') is True or reserved) or sm.get('final_test'):raise ValueError('Invalid acquisition source')
    if not 0<=shard_index<shards or shards>len(rows)//batch:raise ValueError('Invalid acquisition sharding')
    # Reject all inspected flight evaluation parents, irrespective of generated
    # partition tags: those IDs cannot silently return to training.
    excluded=excluded_parents(calibration_source=source if reserved else None,
        pilot_source=source if sm.get('data_role')=='pilot' else None)
    if {r['group_id'] for r in rows}&set(excluded):raise ValueError('Development evaluation parent in training acquisition')
    select_observation(rows[0]['seed'],ticks,1,observation_selection);horizon=acquisition_horizon(ticks)
    behaviors=sorted({behavior_for_batch(i,mode) for i in range(len(rows)//batch) if i%shards==shard_index})
    candidates=np.array([[1,1],[2,2],[4,4],[8,8]],np.float32);teacher=None;fc_teacher=None;policies={};contracts={}
    if mode in ('learned','mixed','matched_learned','matched_held'):
        if not all((bundle,calibration,gain_dataset)):raise ValueError('Trained behavior needs bundle, calibration and audited candidate data')
        candidates,bank=load_gain_bank(gain_dataset,c)
        dm=json.loads((Path(gain_dataset)/'manifest.json').read_text())
        if {r['group_id'] for r in rows}&{r['group_id'] for r in dm['groups']}:raise ValueError('Acquisition parents overlap behavior training/calibration')
        learned=FlightPolicy(bundle,calibration,c,behavior_policy('learned',g.horizon,fallback_mode),g)
        if learned.metadata['dataset_manifest_sha256']!=bank['manifest_sha256']:raise ValueError('Behavior model candidate source mismatch')
        policies['learned']=learned
        teacher=dict(bundle=str(Path(bundle).resolve()),calibration=str(Path(calibration).resolve()),
            bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),weights_sha256=learned.metadata['weights_sha256'],calibration_sha256=sha256(calibration),gain_bank=bank)
        contracts['learned']=asdict(learned.policy)
        horizon=acquisition_horizon(ticks,json.loads(Path(calibration).read_text()))
    elif any((bundle,calibration,gain_dataset)):raise ValueError('Nonlearned acquisition cannot silently ignore a teacher')
    if mode in ('matched_learned','matched_held'):
        if not all((fc_bundle,fc_calibration)):raise ValueError('Matched behavior requires both frozen encoders')
        fc=FlightPolicy(fc_bundle,fc_calibration,c,behavior_policy('matched_fc',g.horizon,fallback_mode),g)
        if (fc.metadata['dataset_manifest_sha256']!=bank['manifest_sha256']
                or learned.metadata['architecture']['encoder']!='gat'
                or fc.metadata['architecture']['encoder']!='matched_fc'
                or acquisition_horizon(ticks,json.loads(Path(fc_calibration).read_text()))!=horizon):
            raise ValueError('Matched behavior requires shared data, mission and correct encoders')
        policies['matched_fc']=fc;contracts['matched_fc']=asdict(fc.policy)
        fc_teacher=dict(bundle=str(Path(fc_bundle).resolve()),calibration=str(Path(fc_calibration).resolve()),
            bundle_manifest_sha256=sha256(Path(fc_bundle)/'manifest.json'),weights_sha256=fc.metadata['weights_sha256'],
            calibration_sha256=sha256(fc_calibration),gain_bank=bank)
    elif any((fc_bundle,fc_calibration)):raise ValueError('Unexpected FC behavior arguments')
    if mode in ('backup','backup_mixed','matched_held'):
        policies['backup']=FlightPolicy(config=c,policy=behavior_policy('backup',g.horizon,fallback_mode),guidance=g);contracts['backup']=asdict(policies['backup'].policy)
    if mode in ('fixed','mixed','backup_mixed'):
        policies['fixed']=FlightPolicy(config=c,policy=behavior_policy('fixed',g.horizon,fallback_mode),guidance=g);contracts['fixed']=asdict(policies['fixed'].policy)
    compile_seconds=0.
    for behavior in behaviors:
        seconds=policies[behavior].warm(candidates,batch,64,64,horizon);compile_seconds+=seconds
        print(json.dumps(dict(stage='acquisition_compiled',behavior=behavior,seconds=seconds,device=str(jax.devices()[0]))),flush=True)
    output_rows=[];index=[];execution_seconds=0.
    for number,offset in enumerate(range(0,len(rows),batch)):
        if number%shards!=shard_index:continue
        behavior=behavior_for_batch(number,mode);policy=policies[behavior]
        subset=rows[offset:offset+batch];x,goal,obs,mask,points,rm,noise,seeds,ready,_=arrays(subset)
        keys=np.asarray([jax.random.PRNGKey(int(s)+7331) for s in seeds]);start=time.perf_counter()
        summaries,traces,truth=jax.device_get(policy.run(x,goal,obs,mask,points,rm,noise,keys,ready,horizon));execution_seconds+=time.perf_counter()-start
        for i,r in enumerate(subset):
            summary={k:v[i] for k,v in summaries.items()};count=int(summary['steps']);status=int(summary['status'])
            length=max(1,min(horizon,count+int(status not in (1,2,8))))
            trace={k:v[i,:length] for k,v in traces.items()}
            requested,selected,fraction=select_observation(r['seed'],ticks,length,observation_selection)
            previous_control=trace['control'][selected-1] if selected else np.full(2,c.robot.mass*c.robot.gravity/2,np.float32)
            previous_gain=trace['gain'][selected-1] if selected else np.array([4.,4.],np.float32)
            cursor=float(trace['route_progress'][selected-1]) if selected else 0.
            record=dict(r,initial_state=trace['observed_state'][selected].tolist(),obstacles=trace['observed_obstacles'][selected].tolist(),
                cursor=cursor,previous_control=previous_control.tolist(),previous_gain=previous_gain.tolist(),
                observation_origin=dict(requested_tick=requested,selected_tick=selected,behavior_mode=behavior,behavior_initial_gain=[4.,4.],terminal_status=NAMES[status],
                    source_group_id=r['group_id'],source_record=offset+i,source_key=keys[i].tolist(),trace_file=f'behavior_{offset+i:05d}.npz'))
            if fraction is not None:record['observation_origin']['selected_fraction']=fraction
            data=dict(trace,true_initial_state=truth['initial_state'][i],true_obstacles=truth['obstacles'][i],
                initial_observation=x[i],observed_obstacles_initial=obs[i],obstacle_mask=mask[i],noise=noise[i],goal=goal[i],
                group_id=r['group_id'],key=keys[i],expected_steps=count,final_status=status,min_clearance=summary['min_clearance'])
            path=root/record['observation_origin']['trace_file'];np.savez_compressed(path,**data)
            index.append(dict(group_id=r['group_id'],file=path.name,sha256=sha256(path),steps=count,status=status,min_clearance=float(summary['min_clearance'])))
            output_rows.append(record)
        write_json(root/'index.json',index)
        print(json.dumps(dict(stage='acquisition',parents=len(index),behavior=behavior,execution_seconds=execution_seconds)),flush=True)
    write_json(root/'scenes.json',output_rows)
    manifest=dict(sm,stage='quad2d_guided_observation_acquisition',source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),
        schema='oa_cbf_quad2d_guided_observation_source_v2',groups=len(output_rows),
        acquisition_sharding=dict(shard_index=shard_index,shards=shards,batch=batch,total_parents=len(rows)),
        source_fingerprint=source_fingerprint(),scenes_sha256=sha256(root/'scenes.json'),index_sha256=sha256(root/'index.json'),
        visited_observations=True,predictive_guidance=asdict(g),initial_previous_gain=[4.,4.],
        acquisition_mode=mode,acquisition_batch=batch,observation_ticks=list(ticks),acquisition_horizon=horizon,behavior_teacher=teacher,behavior_policies=contracts,candidates=candidates.tolist(),fallback_mode=fallback_mode,
        observation_selection=observation_selection,behavior_fc_teacher=fc_teacher,
        observation_sampling='One query per original parent. Fixed ticks clamp at the last recorded decision; episode_fraction samples uniformly from0,1/8,...,1 of the recorded prefix using a prespecified seed. Episode length is used offline only. No parent/failure is discarded. Original-batch schedule selects behavior before outcomes; matched_learned uses equal frozen GAT/FC behavior, staggered across all workers. Full physical prefixes retained.',
        requery_prior='Fresh synthetic latent prior around saved observed state/obstacles using the original declared ranges, with independent keys. Not the acquisition trajectory posterior or an unobserved physical continuation.',
        excluded_development_evaluation_parents=excluded,compile_seconds=compile_seconds,execution_seconds=execution_seconds,
        limitations='One initial or visited observed context per independent development parent; fresh synthetic requery prior, not posterior continuation or final evaluation.')
    manifest['compiled_signatures']=sum(len(p.compiled) for p in policies.values())
    write_json(root/'manifest.json',manifest)
    print(json.dumps(dict(stage='acquisition_finished',parents=len(output_rows),compile_seconds=compile_seconds,execution_seconds=execution_seconds,physical_audit_pending=True)),flush=True)


def audit(source):
    root=Path(source);m=json.loads((root/'manifest.json').read_text());c=flight_config_from_contract(m['config'])
    if m['scenes_sha256']!=sha256(root/'scenes.json') or m['index_sha256']!=sha256(root/'index.json'):raise ValueError('Acquisition binding changed')
    original=Path(m['source']);sm=json.loads((original/'manifest.json').read_text())
    if flight_config_from_contract(sm['config'])!=c:raise ValueError('Acquisition physical contract changed')
    if sha256(original/'manifest.json')!=m['source_manifest_sha256'] or sha256(original/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Raw acquisition parents changed')
    raw=json.loads((original/'scenes.json').read_text());rows=json.loads((root/'scenes.json').read_text());index=json.loads((root/'index.json').read_text())
    positions={r['group_id']:i for i,r in enumerate(raw)}
    if len(positions)!=len(raw) or set(positions)&set(m['excluded_development_evaluation_parents']):raise ValueError('Duplicate/excluded acquisition parents')
    if m.get('schema')=='oa_cbf_quad2d_guided_observation_source_v2':
        reserved=sm.get('data_role')=='fresh_predictive_calibration' and sm.get('weight_fit_authorized') is False and sm.get('training_use') is False
        if not (sm.get('training_use') is True or reserved) or sm.get('final_test'):raise ValueError('Acquisition of held evaluation inputs')
        ticks=m['observation_ticks'];observation_tick(raw[0]['seed'],ticks)
        teacher=m['behavior_teacher']
        expected_horizon=acquisition_horizon(ticks,json.loads(Path(teacher['calibration']).read_text()) if teacher else None)
        if m['acquisition_horizon']!=expected_horizon:raise ValueError('Changed acquisition horizon')
        fc_teacher=m.get('behavior_fc_teacher')
        for teacher in (t for t in (m['behavior_teacher'],fc_teacher) if t is not None):
            candidates,bank=load_gain_bank(teacher['gain_bank']['dataset'],c)
            if bank!=teacher['gain_bank'] or candidates.tolist()!=m['candidates'] or sha256(teacher['calibration'])!=teacher['calibration_sha256']:raise ValueError('Acquisition teacher binding changed')
            cal=json.loads(Path(teacher['calibration']).read_text())
            metadata=json.loads((Path(teacher['bundle'])/'manifest.json').read_text())
            if sha256(Path(teacher['bundle'])/'manifest.json')!=teacher['bundle_manifest_sha256'] or cal['weights_sha256']!=teacher['weights_sha256'] or metadata['weights_sha256']!=teacher['weights_sha256'] or sha256(Path(teacher['bundle'])/'weights.msgpack')!=teacher['weights_sha256'] or metadata['dataset_manifest_sha256']!=bank['manifest_sha256']:
                raise ValueError('Acquisition behavior weights changed')
            dm=json.loads((Path(bank['dataset'])/'manifest.json').read_text())
            if set(positions)&{r['group_id'] for r in dm['groups']}:raise ValueError('Teacher training/calibration parent reused')
            if acquisition_horizon(ticks,cal)!=expected_horizon:raise ValueError('Behavior mission mismatch')
        teacher=m['behavior_teacher']
        behavior_for_batch(0,m['acquisition_mode'])
        if (teacher is not None)!=(m['acquisition_mode'] in ('learned','mixed','matched_learned','matched_held')):raise ValueError('Missing/unexpected acquisition teacher')
        if (fc_teacher is not None)!=(m['acquisition_mode'] in ('matched_learned','matched_held')):raise ValueError('Missing/unexpected FC behavior teacher')
        if fc_teacher:
            for t,encoder in ((teacher,'gat'),(fc_teacher,'matched_fc')):
                if json.loads((Path(t['bundle'])/'manifest.json').read_text())['architecture']['encoder']!=encoder:
                    raise ValueError('Incorrect behavior encoder')
        expected_modes={'learned','matched_fc'} if m['acquisition_mode']=='matched_learned' else {'fixed','learned'} if m['acquisition_mode']=='mixed' else {'fixed','backup'} if m['acquisition_mode']=='backup_mixed' else {m['acquisition_mode']}
        if m['acquisition_mode']=='matched_held':
            expected_modes={'learned','matched_fc','backup'}
            if m.get('fallback_mode')!='hold_previous':raise ValueError('Changed matched held behavior')
        if set(m['behavior_policies'])!=expected_modes:raise ValueError('Missing/unexpected acquisition policy')
        if teacher is None and m['candidates']!=[[1.,1.],[2.,2.],[4.,4.],[8.,8.]]:raise ValueError('Changed nonlearned acquisition candidates')
        for mode,contract in m['behavior_policies'].items():
            expected=behavior_policy(mode,40,m.get('fallback_mode','fixed_set'))
            if contract!=json.loads(json.dumps(asdict(expected))):raise ValueError('Changed behavior selector contract')
    else:ticks=[0,40,120,240]
    if 'acquisition_sharding' in m:
        s=m['acquisition_sharding'];raw=[r for i,r in enumerate(raw) if (i//s['batch'])%s['shards']==s['shard_index']]
    if not len(raw)==len(rows)==len(index) or [r['group_id'] for r in rows]!=[r['group_id'] for r in raw]:raise ValueError('Lost acquisition parents')
    audits=[];graph=jax.jit(lambda *args:flight_graph(*args,c))
    for r,entry,before in zip(rows,index,raw):
        p=root/entry['file']
        if sha256(p)!=entry['sha256'] or entry['group_id']!=r['group_id']:raise ValueError('Acquisition trace changed')
        with np.load(p) as f:d={k:f[k] for k in f.files}
        if str(d['group_id'])!=r['group_id']:raise ValueError('Trace parent mismatch')
        summary=dict(steps=entry['steps'],status=entry['status'],min_clearance=entry['min_clearance'])
        result=check_trace(d,summary,np.asarray(before['initial_state']),np.asarray(before['obstacles']),np.asarray(before['obstacle_mask']),np.asarray(before['noise']),d['gain'],c)
        check_guidance_trace(d,m['predictive_guidance'],goal=before['goal'])
        requested,k,fraction=select_observation(before['seed'],ticks,len(d['active']),m.get('observation_selection','fixed_tick'))
        origin=r['observation_origin']
        if origin['requested_tick']!=requested or origin['selected_tick']!=k or origin['trace_file']!=entry['file']:raise ValueError('Changed observation selection')
        if origin.get('selected_fraction')!=fraction:raise ValueError('Changed fractional observation selection')
        expected_key=np.asarray(jax.random.PRNGKey(before['seed']+7331))
        if origin['source_record']!=positions[r['group_id']] or origin['source_group_id']!=r['group_id'] or origin['terminal_status']!=NAMES[entry['status']]:raise ValueError('Changed acquisition ancestry')
        np.testing.assert_array_equal(d['key'],expected_key);np.testing.assert_array_equal(origin['source_key'],expected_key)
        if m.get('schema')=='oa_cbf_quad2d_guided_observation_source_v2':
            behavior=behavior_for_batch(positions[r['group_id']]//m['acquisition_batch'],m['acquisition_mode'])
            if origin['behavior_mode']!=behavior or origin['behavior_initial_gain']!=[4.,4.]:raise ValueError('Outcome-dependent behavior reassignment')
            check_gain_sources(d,m['behavior_policies'][behavior],m['candidates'])
        for key in ['group_id','family','seed','partition','goal','noise','obstacle_mask','route']:
            if r[key]!=before[key]:raise ValueError('Changed parent context: '+key)
        for key,expected in [('initial_state',d['observed_state'][k]),('obstacles',d['observed_obstacles'][k]),
                ('cursor',d['route_progress'][k-1] if k else 0.),
                ('previous_control',d['control'][k-1] if k else np.full(2,c.robot.mass*c.robot.gravity/2,np.float32)),
                ('previous_gain',d['gain'][k-1] if k else np.array([4.,4.],np.float32))]:
            np.testing.assert_array_equal(np.asarray(r[key],np.float32),np.asarray(expected,np.float32))
        # Audit the visited graph independently before any label/training stage.
        args=tuple(jnp.asarray(a) for a in (r['initial_state'],r['goal'],r['obstacles'],np.asarray(r['obstacle_mask'],bool),r['route']['points'],np.asarray(r['route']['mask'],bool),np.float32(r['cursor']),r['previous_control'],r['previous_gain'],r['noise']))
        actual=jax.device_get(graph(*args))
        rounded=check_observed_graph(actual[0],actual[1],np.asarray(r['initial_state']),np.asarray(r['goal']),np.asarray(r['obstacles']),np.asarray(r['obstacle_mask']),np.asarray(r['route']['points']),np.asarray(r['route']['mask']),np.asarray(r['noise']),c,r['cursor'],r['previous_control'],r['previous_gain'])
        audits.append(dict(group_id=r['group_id'],selected_tick=k,graph_route_rounding_segment=rounded,**result))
    report=dict(audit_passed=bool(audits) and all(a['audit_passed'] for a in audits),parents=len(rows),steps=sum(a['steps'] for a in audits),
        auditor_source_fingerprint=source_fingerprint(),
        manifest_sha256=sha256(root/'manifest.json'),scenes_sha256=sha256(root/'scenes.json'),index_sha256=sha256(root/'index.json'),
        all_behavior_prefixes_replayed=True,all_observed_contexts_independently_checked=True,all_behavior_gain_sources_checked=m.get('schema')=='oa_cbf_quad2d_guided_observation_source_v2',rows=audits)
    write_json(root/'independent_replay.json',sanitize(report));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if not report['audit_passed']:raise ValueError('Observed acquisition audit failed')


def merge(parts,output):
    """Join independently audited disjoint workers without replaying twice."""
    parts=list(map(Path,parts));root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifests=[json.loads((p/'manifest.json').read_text()) for p in parts];m=manifests[0]
    fields=['source','source_manifest_sha256','source_fingerprint','config','predictive_guidance','seed','excluded_development_evaluation_parents','observation_sampling','requery_prior']
    if m.get('data_role')=='fresh_predictive_calibration':fields+=['data_role','training_use','weight_fit_authorized','calibration_reservation']
    if m.get('schema')=='oa_cbf_quad2d_guided_observation_source_v2':fields+=['schema','acquisition_mode','acquisition_batch','observation_ticks','acquisition_horizon','behavior_teacher','behavior_policies','candidates']
    for field in ('observation_selection','behavior_fc_teacher','fallback_mode'):
        if any(field in v for v in manifests):fields.append(field)
    if any(any(v[k]!=m[k] for k in fields) for v in manifests):raise ValueError('Mismatched acquisition contracts')
    sharding=[v['acquisition_sharding'] for v in manifests]
    if any(s['shards']!=len(parts) or s['batch']!=sharding[0]['batch'] for s in sharding) or sorted(s['shard_index'] for s in sharding)!=list(range(len(parts))):
        raise ValueError('Incomplete/duplicate acquisition worker set')
    original=Path(m['source']);sm=json.loads((original/'manifest.json').read_text())
    if sha256(original/'manifest.json')!=m['source_manifest_sha256'] or sha256(original/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Acquisition source changed')
    raw=json.loads((original/'scenes.json').read_text());records={};entries={};audits={};workers=[]
    for p,v in zip(parts,manifests):
        a=json.loads((p/'independent_replay.json').read_text())
        if not a['audit_passed'] or not a.get('all_behavior_prefixes_replayed') or not a.get('all_observed_contexts_independently_checked') or a['manifest_sha256']!=sha256(p/'manifest.json') or a['scenes_sha256']!=sha256(p/'scenes.json') or a['index_sha256']!=sha256(p/'index.json'):
            raise ValueError('Exactly audited acquisition worker required')
        if m.get('schema')=='oa_cbf_quad2d_guided_observation_source_v2' and not a.get('all_behavior_gain_sources_checked'):raise ValueError('Missing behavior gain-source audit')
        for r in json.loads((p/'scenes.json').read_text()):
            if r['group_id'] in records:raise ValueError('Duplicate observation parent')
            records[r['group_id']]=r
        for e in json.loads((p/'index.json').read_text()):
            if sha256(p/e['file'])!=e['sha256']:raise ValueError('Changed acquisition trace')
            (root/e['file']).hardlink_to(p/e['file']);entries[e['group_id']]=e
        audits.update({r['group_id']:r for r in a['rows']})
        workers.append(dict(source=str(p.resolve()),manifest_sha256=sha256(p/'manifest.json'),audit_sha256=sha256(p/'independent_replay.json'),compile_seconds=v['compile_seconds'],execution_seconds=v['execution_seconds']))
    ids=[r['group_id'] for r in raw]
    if len(set(ids))!=len(ids) or set(ids)!=set(records) or set(ids)!=set(entries) or set(ids)!=set(audits):raise ValueError('Lost/extra acquisition parents')
    write_json(root/'scenes.json',[records[i] for i in ids]);write_json(root/'index.json',[entries[i] for i in ids])
    m=dict(m);m.pop('acquisition_sharding');m.update(groups=len(ids),scenes_sha256=sha256(root/'scenes.json'),index_sha256=sha256(root/'index.json'),
        acquisition_workers=workers,compile_seconds=sum(w['compile_seconds'] for w in workers),execution_seconds=sum(w['execution_seconds'] for w in workers),
        timing_interpretation='Compile/execution sums are worker compute totals; supervisor job wall time measures parallel elapsed time.')
    write_json(root/'manifest.json',m)
    report=dict(audit_passed=True,parents=len(ids),steps=sum(r['steps'] for r in audits.values()),manifest_sha256=sha256(root/'manifest.json'),scenes_sha256=sha256(root/'scenes.json'),index_sha256=sha256(root/'index.json'),
        merge_auditor_source_fingerprint=source_fingerprint(),
        all_behavior_prefixes_replayed=True,all_observed_contexts_independently_checked=True,all_behavior_gain_sources_checked=m.get('schema')=='oa_cbf_quad2d_guided_observation_source_v2',rows=[audits[i] for i in ids],independently_audited_workers=workers)
    write_json(root/'independent_replay.json',sanitize(report));print(json.dumps({k:v for k,v in report.items() if k not in ('rows','independently_audited_workers')}),flush=True)


def benchmark(source,output,parents=4,queries=16,replicas=4,horizon=160,noise_clearance_weight=0.,terminal_transition_distance=0.):
    root=Path(source);sm=json.loads((root/'manifest.json').read_text());rows=json.loads((root/'scenes.json').read_text())[:parents]
    if len(rows)!=parents or sha256(root/'scenes.json')!=sm['scenes_sha256']:raise ValueError('Invalid timing parents')
    c=flight_config_from_contract(sm['config']);g=NoiseClearanceGuidanceConfig(noise_clearance_weight=noise_clearance_weight) if noise_clearance_weight else GuidanceConfig()
    if not np.isfinite(terminal_transition_distance) or terminal_transition_distance<0:raise ValueError('Invalid terminal transition')
    if terminal_transition_distance:
        if not noise_clearance_weight:raise ValueError('Terminal guidance requires the noise-aware parent contract')
        g=TerminalGuidanceConfig(noise_clearance_weight=noise_clearance_weight,terminal_transition_distance=terminal_transition_distance)
    if sm.get('visited_observations') and sm['predictive_guidance']!=json.loads(json.dumps(asdict(g))):raise ValueError('Timing guidance differs from observed-context source')
    canonical=np.exp(np.log(.5)+qmc.Sobol(2,scramble=True,seed=sm['seed']+1).random_base2(int(np.log2(queries)))*np.log(16)).astype(np.float32)
    canonical[:4]=[[.5,.5],[1,1],[2,2],[4,4]];gains=jnp.asarray(np.repeat(canonical,replicas,axis=0))
    def one(x,goal,obs,mask,points,rm,noise,seed,ready,cursor):
        keys=jnp.tile(jax.random.split(jax.random.PRNGKey(seed),replicas),(queries,1))
        return jax.vmap(lambda a,k:flight_branch(x,goal,obs,mask,a,points,rm,cursor,noise,k,ready,c,horizon,g)[0])(gains,keys)
    fn=jax.jit(jax.vmap(one));args=tuple(jnp.asarray(v) for v in arrays(rows));start=time.perf_counter();exe=fn.lower(*args).compile();cold=time.perf_counter()-start
    print(json.dumps(dict(stage='compiled',seconds=cold,device=str(jax.devices()[0]),parents=parents,branches=parents*queries*replicas)),flush=True)
    # Two identical physical workloads distinguish first dispatch from warm time;
    # no hot-loop surrogate and no early-failure denominator substitution.
    times=[];summary=None
    for _ in range(2):
        start=time.perf_counter();summary=jax.device_get(exe(*args));times.append(time.perf_counter()-start)
    result=dict(source_fingerprint=source_fingerprint(),source_manifest_sha256=sha256(root/'manifest.json'),device=str(jax.devices()[0]),
        parents=parents,queries=queries,replicas=replicas,horizon_steps=horizon,guidance=asdict(g),compile_seconds=cold,execution_seconds=times,
        warm_seconds=times[-1],branches=int(summary['status'].size),applied_steps=int(summary['steps'].sum()),
        outcomes={NAMES[int(k)]:int(v) for k,v in zip(*np.unique(summary['status'],return_counts=True))},
        compiled_signatures=1,limitations='Full parent x gain x replica x guidance branch workload. Timing alone is not a physical audit or deployment latency.')
    write_json(output,sanitize(result));np.savez_compressed(Path(output).with_suffix('.npz'),**summary)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','benchmark','merge']);p.add_argument('--source');p.add_argument('--output');p.add_argument('--parts',nargs='+')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--mode',choices=['fixed','learned','mixed','backup','backup_mixed','matched_learned','matched_held'],default='fixed');p.add_argument('--bundle');p.add_argument('--calibration');p.add_argument('--gain-dataset');p.add_argument('--ticks',type=int,nargs='+',default=[0,40,120,240])
    p.add_argument('--fallback-mode',choices=['fixed_set','hold_previous'],default='fixed_set')
    p.add_argument('--observation-selection',choices=['fixed_tick','episode_fraction'],default='fixed_tick');p.add_argument('--fc-bundle');p.add_argument('--fc-calibration')
    p.add_argument('--noise-clearance-weight',type=float,default=0.);p.add_argument('--terminal-transition-distance',type=float,default=0.)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--parents',type=int,default=4);p.add_argument('--queries',type=int,default=16);p.add_argument('--replicas',type=int,default=4);p.add_argument('--horizon',type=int,default=160);a=p.parse_args()
    if a.action=='prepare':prepare(a.source,a.output,a.batch,a.shard_index,a.shards,a.mode,a.bundle,a.calibration,a.gain_dataset,a.ticks,a.noise_clearance_weight,a.terminal_transition_distance,a.observation_selection,a.fc_bundle,a.fc_calibration,a.fallback_mode)
    elif a.action=='merge':merge(a.parts,a.output)
    elif a.action=='audit':audit(a.source)
    else:benchmark(a.source,a.output,a.parents,a.queries,a.replicas,a.horizon,a.noise_clearance_weight,a.terminal_transition_distance)
