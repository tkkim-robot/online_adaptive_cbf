"""Full-horizon development flight episodes, saved prefixes and physical audits.

Fixed policies here are matched OA component ablations, NOT original compared
methods. No choice/filtering by eventual outcome; all parents are retained.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import jax
from .quad2d_policy import FlightPolicy,FlightPolicyConfig,ProgressAdmissionPolicyConfig,FeasibilityTriggeredPolicyConfig,IncumbentProgressPolicyConfig,SOURCE_NAMES
from .quad2d_control import FlightConfig,flight_config_from_contract
from .quad2d_rollout import NAMES,GOAL,COLLISION,TIMEOUT
from .quad2d_audit import check_trace,check_guidance_trace
from .dataset import sha256,source_fingerprint
from .io import write_json
from .cli import sanitize


def run(source,dataset,output,bundle=None,calibration=None,mode='learned',gain=4.,steps=1600,batch=8,guidance_horizon=0,noise_clearance_weight=0.,terminal_transition_distance=0.,clearance_guard='none',motion_window=0,motion_application='forecast',record_query_statistics=False,fallback_mode='fixed_set',progress_admission=False,query_trigger='periodic',incumbent_progress=False,clipped_mixture_pilot=False,diagnostic_nearest_obstacles=None,diagnostic_cruise_speed=None,ensemble_mixture_risk=False):
    if incumbent_progress and (progress_admission or query_trigger!='periodic'):
        raise ValueError('Incumbent comparison is a separate policy experiment')
    if query_trigger not in ('periodic','previous_gain_infeasible_v1') or (query_trigger!='periodic' and progress_admission):
        raise ValueError('Unknown or incompatible query-trigger policy')
    if fallback_mode not in ('fixed_set','hold_previous'):
        raise ValueError('Unknown fallback mode')
    if fallback_mode=='hold_previous' and (mode=='fixed' or not guidance_horizon):
        raise ValueError('Held-gain fallback requires guided learned/component evaluation')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);source=Path(source);dataset=Path(dataset)
    sm=json.loads((source/'manifest.json').read_text());rows=json.loads((source/'scenes.json').read_text())
    if any('waypoint_goals' in row for row in rows):raise ValueError('Use the ordered flight runner; final-goal bypass forbidden')
    config=flight_config_from_contract(sm['config']);policy_config=FlightPolicyConfig(mode=mode,fixed_gain=(gain,gain))
    if sm['scenes_sha256']!=sha256(source/'scenes.json'):raise ValueError('Changed physical scene contract')
    dm=json.loads((dataset/'manifest.json').read_text());entry=json.loads((dataset/'index.json').read_text())[0]
    if flight_config_from_contract(dm['config'])!=config:raise ValueError('Candidate data uses a different physical contract')
    if sha256(dataset/entry['file'])!=entry['sha256']:raise ValueError('Changed gain-bank source')
    with np.load(dataset/entry['file']) as f:candidates=f['gains'][0,::dm['replicas']].copy()
    if {r['group_id'] for r in rows}&{g['group_id'] for g in dm['groups']}:raise ValueError('Episode parents overlap training/calibration')
    if len(rows)%batch:raise ValueError('Static batch must divide all parents; never drop a remainder')
    if not isinstance(guidance_horizon,int) or guidance_horizon<0:raise ValueError('Invalid guidance horizon')
    from .quad2d_guidance import GuidanceConfig,NoiseClearanceGuidanceConfig,TerminalGuidanceConfig
    guidance=GuidanceConfig(horizon=guidance_horizon) if guidance_horizon else None
    if not np.isfinite(noise_clearance_weight) or noise_clearance_weight<0:raise ValueError('Invalid noise-clearance score weight')
    if noise_clearance_weight:
        if guidance is None:raise ValueError('Noise-clearance score requires predictive guidance')
        guidance=NoiseClearanceGuidanceConfig(horizon=guidance_horizon,noise_clearance_weight=noise_clearance_weight)
    if not np.isfinite(terminal_transition_distance) or terminal_transition_distance<0:raise ValueError('Invalid terminal transition')
    if terminal_transition_distance:
        if not noise_clearance_weight:raise ValueError('Terminal guidance requires the noise-aware parent contract')
        guidance=TerminalGuidanceConfig(horizon=guidance_horizon,noise_clearance_weight=noise_clearance_weight,terminal_transition_distance=terminal_transition_distance)
    if clearance_guard not in ('none','one_step','predictive','inflated'):raise ValueError('Unknown clearance-guard mode')
    if clearance_guard!='none':
        from .quad2d_guidance import GuardedGuidanceConfig,InflatedGuidanceConfig
        if not isinstance(guidance,TerminalGuidanceConfig):raise ValueError('Clearance guard requires the terminal guidance contract')
        cls=InflatedGuidanceConfig if clearance_guard=='inflated' else GuardedGuidanceConfig
        guidance=cls(**asdict(guidance),hard_prediction_clearance=clearance_guard in ('predictive','inflated'))
    if type(motion_window) is not int or motion_window<0:raise ValueError('Invalid motion window')
    if motion_application not in ('forecast','current_and_future') or (not motion_window and motion_application!='forecast'):
        raise ValueError('Motion application requires an explicit observer window')
    if motion_window:
        from .quad2d_guidance import ForecastMotionGuidanceConfig,ObservedMotionGuidanceConfig
        if type(guidance) is not TerminalGuidanceConfig or clearance_guard!='none':
            raise ValueError('Motion forecast pilot requires unguarded terminal guidance')
        cls=ObservedMotionGuidanceConfig if motion_application=='current_and_future' else ForecastMotionGuidanceConfig
        guidance=cls(**asdict(guidance),motion_window=motion_window)
    if guidance is not None and mode in ('learned','backup'):policy_config=FlightPolicyConfig(mode=mode,fixed_gain=(gain,gain),validation_horizon=guidance.horizon)
    if fallback_mode=='hold_previous':policy_config=replace(policy_config,backup_gains=())
    if progress_admission:
        if type(guidance) is not TerminalGuidanceConfig or mode not in ('learned','backup'):
            raise ValueError('Progress admission requires the raw terminal-guidance contract')
        policy_config=ProgressAdmissionPolicyConfig(**asdict(policy_config))
    if query_trigger!='periodic':
        if type(guidance) is not TerminalGuidanceConfig:
            raise ValueError('Feasibility trigger requires raw terminal guidance')
        policy_config=FeasibilityTriggeredPolicyConfig(**asdict(policy_config))
    if incumbent_progress:
        if type(guidance) is not TerminalGuidanceConfig:
            raise ValueError('Incumbent comparison requires raw terminal guidance')
        policy_config=IncumbentProgressPolicyConfig(**asdict(policy_config))
    if clipped_mixture_pilot and mode!='learned':raise ValueError('Explicit learned mixture pilot required')
    if ensemble_mixture_risk:
        from .quad2d_policy import EnsembleRiskPolicyConfig
        if not incumbent_progress or mode!='learned':raise ValueError('Mixture risk requires guided learned incumbent comparison')
        policy_config=EnsembleRiskPolicyConfig(**asdict(policy_config))
    policy=FlightPolicy(bundle,calibration,config,policy_config,guidance,record_query_statistics,clipped_mixture_pilot,diagnostic_nearest_obstacles,diagnostic_cruise_speed)
    config=policy.config
    if clipped_mixture_pilot:
        cal=json.loads(Path(calibration).read_text())
        for path in cal['datasets']:
            groups=json.loads((Path(path)/'manifest.json').read_text())['groups']
            for field in ('group_id','seed'):
                if {r[field] for r in rows}&{r[field] for r in groups}:
                    raise ValueError('Physical mixture pilot overlaps original/additional reserved data')
    if mode=='learned' and policy.metadata['dataset_manifest_sha256']!=sha256(dataset/'manifest.json'):raise ValueError('Candidate source does not match trained model')
    manifest=dict(schema='oa_cbf_quad2d_episode_development_v1',source=str(source.resolve()),source_manifest_sha256=sha256(source/'manifest.json'),
        source_fingerprint=source_fingerprint(),config=asdict(config),policy=asdict(policy_config),steps=steps,batch=batch,device=str(jax.devices()[0]),
        gain_dataset=str(dataset.resolve()),gain_dataset_manifest_sha256=sha256(dataset/'manifest.json'),candidates=candidates.tolist(),
        model_weights_sha256=policy.metadata.get('weights_sha256'),calibration_sha256=sha256(calibration) if calibration else None,
        initial_conditions='Fresh actual six-state flight, same raw prior and innovation keys across policies',
        scope='Development learned flight / matched OA component ablations. Not authentic baseline comparison or trajectory-calibrated final evaluation.',final_test=False)
    if diagnostic_nearest_obstacles is not None:
        manifest.update(observation_neighborhood=policy.neighborhood_contract,calibration_coverage_valid=False,
            scope='Frozen-model neighborhood diagnosis only. Old calibration is a reference gate, with no coverage claim under changed observations/QP. Not a final benchmark.')
    if diagnostic_cruise_speed is not None:
        manifest.update(diagnostic_nominal_speed=policy.nominal_speed_contract,calibration_coverage_valid=False,
            scope='Frozen-model speed-only diagnosis. Existing calibration is a reference gate with no coverage claim at the changed speed. Not a final benchmark.')
    if record_query_statistics:
        manifest['query_statistics_schema']='quad2d_live_query_statistics_v1'
        if clipped_mixture_pilot:
            manifest.update(query_statistics_schema='quad2d_clipped_mixture_live_query_statistics_v1',
                risk_distribution_contract=policy.metadata['risk_distribution_contract'])
    if guidance is not None:
        manifest['predictive_guidance']=asdict(guidance)
        manifest['validation']=f'One{guidance.horizon}tickheld-gain/profile witness from each shortlisted observed-state guidance prediction; returned currentQP reused, requery every4ticks or failed held-profile prediction.' if mode!='fixed' else 'Fixed-gain predictive guidance component ablation'
        if query_trigger!='periodic':manifest['validation']='Check the previous gain with the observed QP/CBF and predictive guidance at every tick; query the learned gain pool only when that check fails.'
    write_json(root/'manifest.json',manifest)
    compile_seconds=policy.warm(candidates,batch,64,64,steps);print(json.dumps(dict(stage='warmed',compile_seconds=compile_seconds)),flush=True)
    index=[];start=time.perf_counter()
    for offset in range(0,len(rows),batch):
        group=rows[offset:offset+batch]
        x,g,o,m=(np.asarray([r[k] for r in group],bool if k=='obstacle_mask' else np.float32) for k in ('initial_state','goal','obstacles','obstacle_mask'))
        points=np.asarray([r['route']['points'] for r in group],np.float32);rm=np.asarray([r['route']['mask'] for r in group],bool)
        noise=np.asarray([r['noise'] for r in group],np.float32);ready=np.asarray([r['route']['status']=='ready' for r in group],bool)
        # Kept independent from all queried training replicas, shared by policies.
        keys=np.asarray([jax.random.PRNGKey(r['seed']+7193) for r in group])
        tick=time.perf_counter();summary,trace,truth=jax.device_get(policy.run(x,g,o,m,points,rm,noise,keys,ready,steps));seconds=time.perf_counter()-tick
        for i,row in enumerate(group):
            status=int(summary['status'][i]);count=int(summary['steps'][i]);length=max(1,min(steps,count+int(status not in (GOAL,COLLISION,8))))
            data={k:v[i,:length] for k,v in trace.items()};data.update(true_initial_state=truth['initial_state'][i],true_obstacles=truth['obstacles'][i],
                initial_observation=x[i],observed_obstacles_initial=o[i],obstacle_mask=m[i],noise=noise[i],goal=g[i],points=points[i],route_mask=rm[i],key=keys[i])
            path=root/f'episode_{offset+i:05d}.npz';np.savez_compressed(path,**data)
            sources={name:int(np.sum((data['source']==k)&data['requery'])) for k,name in enumerate(SOURCE_NAMES)}
            record=dict(group_id=row['group_id'],family=row['family'],obstacles=int(m[i].sum()),noise_scale=float(noise[i,0]/.015),
                file=path.name,sha256=sha256(path),status=NAMES[status],status_code=status,steps=count,
                min_clearance=float(summary['min_clearance'][i]),route_progress=float(summary['route_progress'][i]),
                final_state=summary['final_state'][i].tolist(),source_counts=sources,
                gate_stage_sum=data['stages'][data['requery']].sum(axis=0).tolist())
            index.append(sanitize(record))
        write_json(root/'index.json',index)
        print(json.dumps(dict(completed=len(index),total=len(rows),batch_seconds=seconds,applied_steps=int(summary['steps'].sum()),elapsed_seconds=time.perf_counter()-start)),flush=True)
    def counts(items):
        return dict(episodes=len(items),applied_steps=sum(r['steps'] for r in items),outcomes={name:sum(r['status']==name for r in items) for name in sorted({r['status'] for r in items})},
            sources={name:sum(r['source_counts'][name] for r in items) for name in SOURCE_NAMES})
    result=dict(complete=True,manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),compile_seconds=compile_seconds,
        execution_seconds=time.perf_counter()-start,aggregate=counts(index),by_family={family:counts([r for r in index if r['family']==family]) for family in sorted({r['family'] for r in index})},
        physical_audit_pending=True,compiled_signatures=len(policy.compiled),deadline_measurement=False)
    write_json(root/'summary.json',result);print(json.dumps(result),flush=True)


def audit(directory):
    root=Path(directory);manifest=json.loads((root/'manifest.json').read_text());config=flight_config_from_contract(manifest['config'])
    source=Path(manifest['source']);sm=json.loads((source/'manifest.json').read_text())
    from .quad2d_nominal_speed import reference_from_manifest
    if flight_config_from_contract(sm['config'])!=reference_from_manifest(manifest):raise ValueError('Wrong flight physical audit config')
    if sha256(source/'manifest.json')!=manifest['source_manifest_sha256'] or sm['scenes_sha256']!=sha256(source/'scenes.json'):raise ValueError('Scene source changed')
    parents=json.loads((source/'scenes.json').read_text());index=json.loads((root/'index.json').read_text());reports=[]
    trigger_audit=None
    incumbent_audit=None
    if manifest['policy'].get('incumbent_progress') and manifest['policy']['mode']=='learned':
        from .quad2d_incumbent_progress import make_auditor
        incumbent_audit=make_auditor(config,manifest['predictive_guidance'],manifest['batch'],manifest['candidates'])
    if manifest['policy'].get('query_trigger'):
        from .quad2d_policy import flight_policy_from_contract
        from .quad2d_gain_trigger import make_trigger_auditor
        flight_policy_from_contract(manifest['policy'])
        trigger_audit=make_trigger_auditor(config,manifest['predictive_guidance'],manifest['batch'])
    if [r['group_id'] for r in index]!=[r['group_id'] for r in parents]:raise ValueError('Lost, duplicated or reordered evaluation parents')
    for row,parent in zip(index,parents):
        if sha256(root/row['file'])!=row['sha256']:raise ValueError('Episode trace changed')
        with np.load(root/row['file']) as f:data={k:f[k] for k in f.files}
        for field,sourcefield in [('initial_observation','initial_state'),('observed_obstacles_initial','obstacles'),('obstacle_mask','obstacle_mask'),('noise','noise'),('goal','goal')]:
            np.testing.assert_allclose(data[field],parent[sourcefield],atol=1e-7)
        summary=dict(status=row['status_code'],steps=row['steps'],min_clearance=row['min_clearance'] if row['min_clearance'] is not None else np.inf)
        guidance=manifest.get('predictive_guidance',{});forecast_audit={};command_obstacles=None
        if 'motion_window' in guidance:
            from .quad2d_motion import check_forecast_trace
            forecast_audit=check_forecast_trace(data,guidance,config.robot.dt)
        if 'motion_application' in guidance:
            if guidance['motion_application']!='current_and_future_v1' or not forecast_audit:
                raise ValueError('Unaudited/unknown controller-observation contract')
            command_obstacles=data['forecast_obstacles']
        from .quad2d_audit import independent_residual
        residual=independent_residual
        neighborhood_audit={}
        if 'observation_neighborhood' in manifest:
            from .quad2d_neighborhood import neighborhood_contract, audit_neighborhood, numpy_neighborhood_mask
            count=manifest['observation_neighborhood']['capacity']
            if manifest['observation_neighborhood']!=neighborhood_contract(count) or manifest.get('calibration_coverage_valid') is not False:
                raise ValueError('Neighborhood diagnosis must explicitly disclaim calibration coverage')
            neighborhood_audit=audit_neighborhood(data,count)
            def residual(x,u,obs,mask,gains,config):
                return independent_residual(x,u,obs,numpy_neighborhood_mask(x,obs,mask,count),gains,config)
        result=check_trace(data,summary,data['initial_observation'],data['observed_obstacles_initial'],data['obstacle_mask'],data['noise'],data['gain'],config,command_obstacles=command_obstacles,command_residual=residual)
        result.update(forecast_audit)
        result.update(neighborhood_audit)
        final=data['state'][-1];np.testing.assert_allclose(final,row['final_state'],atol=1e-7)
        if row['status_code']==GOAL:
            assert np.linalg.norm(final[:2]-data['goal'])<=config.goal_tolerance+1e-6
            assert np.linalg.norm(final[3:5])<=config.terminal_speed+1e-6 and abs(final[2])<=config.terminal_pitch+1e-6 and abs(final[5])<=config.terminal_pitch_rate+1e-6
        if row['status_code']==TIMEOUT and row['steps']!=manifest['steps']:raise ValueError('Censored future counted as timeout')
        if row['status_code']==6 and not (data['source'][-1]==3 and not data['active'][-1]):raise ValueError('Missing recorded policy rejection')
        if 'predictive_guidance' in manifest:
            check_guidance_trace(data,manifest['predictive_guidance'],config)
            if trigger_audit is not None:result.update(trigger_audit(data))
            if incumbent_audit is not None:result.update(incumbent_audit(data))
            if manifest['policy']['mode']=='fixed':np.testing.assert_array_equal(data['gain'],np.broadcast_to(manifest['policy']['fixed_gain'],data['gain'].shape))
            else:
                candidates=np.asarray(manifest['candidates']);previous=np.array([4.,4.]);backups=np.asarray(manifest['policy']['backup_gains']).reshape(-1,2)
                for k,gain in enumerate(data['gain']):
                    if not data['active'][k] and not data['requery'][k]:continue
                    source=int(data['source'][k]);stages=data['stages'][k]
                    if source==0:
                        if manifest['policy']['mode']!='learned':raise ValueError('Learned decision in fallback-only ablation')
                        if not data['requery'][k] or min(stages[:4])<1 or stages[-1]!=0:raise ValueError('Invalid learned acceptance record')
                        pool=np.vstack((candidates,previous))
                    elif source==1:
                        if not data['requery'][k] or stages[-1]<1:raise ValueError('Invalid fallback record')
                        pool=np.vstack((backups,previous))
                    elif source==4:pool=previous[None]
                    elif source==3:
                        if data['active'][k]:raise ValueError('Applied rejected flight action')
                        continue
                    else:raise ValueError('Unknown guided adaptive source')
                    if not np.any(np.all(np.abs(pool-gain)<1e-6,axis=1)):raise ValueError('Gain outside declared candidate/fallback source')
                    if data['active'][k]:previous=gain
        reports.append(dict(group_id=row['group_id'],**result))
    report=dict(audit_passed=all(r['audit_passed'] for r in reports),episodes=len(reports),steps=sum(r['steps'] for r in reports),
        manifest_sha256=sha256(root/'manifest.json'),index_sha256=sha256(root/'index.json'),all_episodes_audited=True,rows=reports,
        auditor_source_fingerprint=source_fingerprint(),
        scope='Every saved applied six-state prefix and rejection observation; all physical moving disks, independent continuous dynamics/CBF/input/envelope and raw sensor bounds. Explicit current-motion mode uses independently reconstructed causal velocities for CBF rows. No unobserved continuation safety claim.')
    write_json(root/'independent_replay.json',sanitize(report));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if not report['audit_passed']:raise ValueError('Independent full-episode flight audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('run');r.add_argument('--source',required=True);r.add_argument('--dataset',required=True);r.add_argument('--output',required=True)
    r.add_argument('--bundle');r.add_argument('--calibration');r.add_argument('--mode',choices=['learned','fixed','backup'],default='learned')
    r.add_argument('--gain',type=float,default=4.);r.add_argument('--steps',type=int,default=1600);r.add_argument('--batch',type=int,default=8)
    r.add_argument('--guidance-horizon',type=int,default=0,help='Predictive nominal; learned mode requires exactly matched guided labels, bundle and calibration')
    r.add_argument('--noise-clearance-weight',type=float,default=0.,help='OA guidance-design pilot; changed contract requires new labels/calibration for learned mode')
    r.add_argument('--terminal-transition-distance',type=float,default=0.,help='OA terminal-distance progress ablation; requires matched new labels/calibration')
    r.add_argument('--clearance-guard',choices=['none','one_step','predictive','inflated'],default='none')
    r.add_argument('--motion-window',type=int,default=0,help='Forecast-only causal obstacle observer pilot; requires new labels before learned use')
    r.add_argument('--motion-application',choices=['forecast','current_and_future'],default='forecast',help='Explicit current-and-future measurement ablation; raw physical sensors always retained')
    r.add_argument('--record-query-statistics',action='store_true',help='Save the actual neural statistics used by each queried policy decision for trajectory calibration')
    r.add_argument('--incumbent-progress',action='store_true')
    r.add_argument('--clipped-mixture-pilot',action='store_true')
    r.add_argument('--ensemble-mixture-risk',action='store_true')
    r.add_argument('--fallback-mode',choices=['fixed_set','hold_previous'],default='fixed_set',help='Matched policy candidate: validate only the previous accepted gain when learned proposals fail; no alternative fallback gain search')
    r.add_argument('--progress-admission',action='store_true')
    r.add_argument('--query-trigger',choices=['periodic','previous_gain_infeasible_v1'],default='periodic')
    a=sub.add_parser('audit');a.add_argument('--directory',required=True)
    args=vars(p.parse_args());action=args.pop('action');run(**args) if action=='run' else audit(**args)
