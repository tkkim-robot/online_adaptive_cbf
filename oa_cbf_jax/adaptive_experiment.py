"""Shared-scenario development evaluation of the actual learned adaptive loop."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from .adaptive import DevelopmentPolicy,NonlearnedPolicy,PolicyConfig,SOURCE_NAMES,BACKUP,LEARNED
from .config import UnicycleConfig
from .unicycle_inputs import source_robot
from .closed_loop import make_closed_loop,PLANNER_FAILURE
from .cli import sanitize
from .dataset import source_fingerprint,sha256
from .io import write_json
from .predictive import PREDICTIVE_REJECTED
from .predictive_experiment import CANDIDATES
from .route_control import INADMISSIBLE
from .simulation import STATUS_NAMES
from .stochastic import STATE_BOUND_VIOLATION


def candidate_pool(queries,upper=4.,design='legacy'):
    if design=='wide':
        if queries<32 or upper<=16:raise ValueError('Wide proposal design requires >=32queries and upper>16')
        from .gain_queries import wide_pairs
        gains=candidate_pool(queries,upper)
        gains[:16]=candidate_pool(16,16.)
        gains[16:24]=wide_pairs(upper)
        return gains
    if design!='legacy':raise ValueError('Unknown candidate design')
    if not np.isfinite(upper) or upper<4.:raise ValueError('Candidate upper bound must be finite and at least four')
    if queries==3:
        if upper!=4.:raise ValueError('The original three-query comparator has a fixed gain domain')
        return np.array([[.5,.5],[1.,1.],[3.,3.]],np.float32)
    if queries<4:raise ValueError('Use three baseline queries or at least four queries')
    values=qmc.Sobol(2,scramble=True,seed=88421).random_base2(int(np.ceil(np.log2(queries))))[:queries]
    gains=np.exp(np.log(.3)+values*np.log(upper/.3)).astype(np.float32)
    count=min(queries,len(CANDIDATES));gains[:count]=CANDIDATES[:count]
    if upper>4.:
        if queries<16:raise ValueError('Expanded-domain probes require at least sixteen candidates')
        extra=np.geomspace(4.,upper,5)[1:]
        gains[count:count+4]=extra[:,None]
    return gains


def run(source,output,bundle=None,calibration=None,mode='learned',queries=64,shortlist=4,horizon=80,interval=4,
        noise_scale=0.,batch=8,steps=800,limit=None,fixed_gain=2.,reactive_reselection=False,backup_gains=(1.,2.,3.),robot_config=None,candidate_upper=4.,sensor_margin_scale=0.,margin_guidance=False,shared_clearance_budget=False,motion_observer_window=0,candidate_design='legacy',filter_obstacle_position=False,ordered_waypoints=False):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    records=json.loads((Path(source)/'scenes.json').read_text())
    if any('waypoint_goals' in r for r in records) and not ordered_waypoints:
        raise ValueError('Ordered tasks require --ordered-waypoints; direct final-goal bypass forbidden')
    if ordered_waypoints:
        from .waypoint_tasks import CONTRACT
        for r in records:
            count=r['waypoint_count'];goals=np.asarray(r['waypoint_goals']);routes=r['waypoint_routes']
            if not 1<=count<=len(goals) or goals.shape!=(CONTRACT['capacity'],2) or np.asarray(routes['points']).shape!=(CONTRACT['capacity'],64,2):
                raise ValueError('Invalid ordered task shapes or count')
            if not np.array_equal(goals[count-1],r['scene']['goal']):raise ValueError('Final waypoint mismatch')
    if limit is not None:
        development=[r for r in records if not r['scene']['scene_id'].startswith('fixture:')]
        records=development[:limit]+[r for r in records if r['scene']['scene_id'].startswith('fixture:')]
    config=PolicyConfig(mode=mode,validation_horizon=horizon,interval=interval,shortlist=shortlist,fixed_gain=(fixed_gain,fixed_gain),
                        reactive_reselection=reactive_reselection,backup_gains=tuple((g,g) for g in backup_gains),sensor_margin_scale=sensor_margin_scale,margin_guidance=margin_guidance,shared_clearance_budget=shared_clearance_budget,motion_observer_window=motion_observer_window,filter_obstacle_position=filter_obstacle_position)
    robot=UnicycleConfig(**json.loads(Path(robot_config).read_text())) if robot_config else None
    if mode in ('learned','ungated'):
        if not bundle or not calibration:raise ValueError('Learned policies require a complete calibrated bundle')
        policy=DevelopmentPolicy(bundle,calibration,config,robot=robot,allow_development=True)
    else:
        if robot is None:
            robot=UnicycleConfig(**json.loads(Path(calibration).read_text())['robot']) if calibration else source_robot(source)
        policy=NonlearnedPolicy(config,robot)
    source_robot(source,policy.robot)
    gains=candidate_pool(queries,candidate_upper,candidate_design)
    if mode in ('learned','ungated') and (np.any(gains<np.float32(policy.gain_domain['lower'])) or
                                        np.any(gains>np.float32(policy.gain_domain['upper']))):
        raise ValueError('Candidate pool exceeds the trained/calibrated gain domain; collect and calibrate a fresh compatible model')
    if queries+1<shortlist:raise ValueError('Shortlist exceeds pool')
    if ordered_waypoints and (policy.robot.goal_tolerance!=CONTRACT['arrival_tolerance'] or policy.robot.radius!=.3):
        raise ValueError('Ordered hero variant requires the declared shared physical contract')
    single=make_closed_loop(policy,steps,batch_axis='scenes',ordered_waypoints=ordered_waypoints)
    axes=(None,None,0,0,0,0,None,0,0,0,0,0)+((0,) if ordered_waypoints else ())
    runner=jax.jit(jax.vmap(single,in_axes=axes,axis_name='scenes'))
    names={**STATUS_NAMES,INADMISSIBLE:'hocbf_inadmissible',PREDICTIVE_REJECTED:'predictive_rejected',
           PLANNER_FAILURE:'planner_failure',STATE_BOUND_VIOLATION:'state_bound_violation'}
    write_json(root/'manifest.json',dict(stage='development_ordered_waypoint_closed_loop' if ordered_waypoints else 'development_adaptive_closed_loop',final_test=False,source_fingerprint=source_fingerprint(),
         policy=asdict(config),robot=asdict(policy.robot),queries=queries,candidate_upper=candidate_upper,candidate_design=candidate_design,pool=gains.tolist(),noise_scale=noise_scale,steps=steps,batch=batch,
         input_scene_sha256=sha256(Path(source)/'scenes.json'),input_scenes=str(Path(source).resolve()),
         bundle=str(Path(bundle).resolve()) if mode in ('learned','ungated') else None,
         weights_sha256=policy.predictor.metadata['weights_sha256'] if mode in ('learned','ungated') else None,
         prediction_horizon_steps=policy.metadata['horizon_steps'] if mode in ('learned','ungated') else None,
         validation_scope='Fixed-gain copied-state check over policy.validation_horizon steps; distinct from the learned prediction horizon. No recursive-feasibility or beyond-window safety guarantee.',
         calibration_sha256=sha256(calibration) if mode in ('learned','ungated') else None,calibration_interpretation=policy.metadata['interpretation'],
         conditions='Identical observed initial scenes, routes, physical prior, sensor noise stream and limits for every method. Synchronous compute; no real-delay or final-test claim.',
         backup=f'Same copied-state {horizon}-step fixed-gain validator on previous and {config.backup_gains}; explicit rejection if none survives. No unverified emergency action.'))
    if ordered_waypoints:
        manifest=json.loads((root/'manifest.json').read_text());manifest['waypoint_contract']=CONTRACT
        manifest['conditions']+=' Ordered waypoint contract: '+CONTRACT['name']
        manifest['route_progress_scope']='Local current-leg coordinate; do not sum as actual mission travel or proof of earlier visits.'
        write_json(root/'manifest.json',manifest)
    rows=[];timings=[];begin=time.perf_counter();compiled=None;compile_seconds=0.
    for offset in range(0,len(records),batch):
        chosen=records[offset:offset+batch];valid=len(chosen);chosen+= [chosen[-1]]*(batch-valid)
        arrays=[np.stack([r['scene'][key] for r in chosen]) for key in ('initial_state','goal','obstacles','obstacle_mask')]
        route_field='waypoint_routes' if ordered_waypoints else 'route'
        points=np.stack([r[route_field]['points'] for r in chosen]);masks=np.stack([r[route_field]['mask'] for r in chosen])
        if ordered_waypoints:arrays[1]=np.stack([r['waypoint_goals'] for r in chosen])
        noise=np.broadcast_to(noise_scale*np.array([.02,.03,.02,.03,.025,.01]),(batch,6)).copy()
        seeds=[int.from_bytes(hashlib.sha256(r['scene']['scene_id'].encode()).digest()[:4],'little') for r in chosen]
        keys=np.stack([np.asarray(jax.random.PRNGKey(seed)) for seed in seeds])
        ready=np.array([r['waypoint_routes']['ready'] if ordered_waypoints else r['route']['status']=='ready' for r in chosen])
        floats=lambda a:jnp.asarray(a,dtype=bool if a.dtype==bool else jnp.float32)
        inputs=(policy.params,policy.calibration,*(floats(a) for a in arrays),jnp.asarray(gains),floats(points),floats(masks),floats(noise),jnp.asarray(keys),jnp.asarray(ready))
        if ordered_waypoints:inputs+=(jnp.asarray([r['waypoint_count'] for r in chosen],jnp.int32),)
        if compiled is None:
            tick=time.perf_counter();compiled=runner.lower(*inputs).compile()
            compile_seconds=time.perf_counter()-tick
            print(json.dumps(dict(stage='compiled',seconds=compile_seconds)),flush=True)
        tick=time.perf_counter();summary,trace,truth=compiled(*inputs);jax.block_until_ready((summary,trace,truth))
        elapsed=time.perf_counter()-tick;summary,trace,truth=jax.device_get((summary,trace,truth))
        timings.append(dict(offset=offset,seconds=elapsed,includes_compile=False))
        np.savez_compressed(root/f'traces_{offset:05d}.npz',**{k:v[:valid] for k,v in trace.items()},
                            true_initial_state=truth['initial_state'][:valid],true_obstacles=truth['obstacles'][:valid],
                            scene_id=np.array([r['scene']['scene_id'] for r in chosen[:valid]]),noise=noise[:valid],key=keys[:valid])
        for local in range(valid):
            active=trace['active'][local];ticks=trace['selection_tick'][local]
            source_codes=trace['applied_source'][local][active]
            changes=np.diff(np.log(trace['gains'][local][active]),axis=0)
            row=dict(scene_id=chosen[local]['scene']['scene_id'],family=chosen[local]['scene']['family'],mode=mode,
                 status=names[int(summary.status[local])],steps=int(summary.steps[local]),min_clearance=float(summary.min_clearance[local]),
                 goal_progress=float(summary.progress[local]),final_route_coordinate=float(truth['route_progress'][local]),
                 applied_source_counts={name:int(np.sum(source_codes==code)) for code,name in SOURCE_NAMES.items()},
                 selection_attempts=int(ticks.sum()),selection_rejections=int(np.sum(ticks&~trace['selection_accepted'][local])),
                 reactive_reselections=int(np.sum(trace['reactive_reselection'][local])),
                 gate_stage_totals=trace['gate_stages'][local][ticks].sum(axis=0).tolist(),
                 gain_total_variation=float(np.linalg.norm(changes,axis=1).sum()),
                 max_physical_bound_violation=float(np.max(trace['state_bound_violation'][local][active])) if np.any(active) else None)
            if ordered_waypoints:
                row.update(waypoint_index=int(truth['waypoint_index'][local]),waypoints_visited=int(truth['waypoints_visited'][local]),
                    required_waypoints=chosen[local]['waypoint_count'],waypoint_handoffs=int(trace['waypoint_handoff'][local].sum()))
            rows.append(row)
        write_json(root/'progress.json',dict(completed_groups=offset+valid,total_groups=len(records),elapsed_seconds=time.perf_counter()-begin))
        write_json(root/'results.json',sanitize(rows))
        print(json.dumps(dict(stage='batch',offset=offset,seconds=elapsed,completed_groups=offset+valid)),flush=True)
    evaluated=[r for r in rows if not r['scene_id'].startswith('fixture:')]
    aggregate=dict(groups=len(evaluated),outcomes={s:sum(r['status']==s for r in evaluated) for s in sorted({r['status'] for r in evaluated})},
         applied_sources={name:sum(r['applied_source_counts'][name] for r in evaluated) for name in SOURCE_NAMES.values()},
         selection_attempts=sum(r['selection_attempts'] for r in evaluated),selection_rejections=sum(r['selection_rejections'] for r in evaluated))
    if runner._cache_size()!=0:raise RuntimeError('Unexpected implicit closed-loop compilation')
    write_json(root/'summary.json',dict(aggregate=aggregate,timings=timings,elapsed_seconds=time.perf_counter()-begin,
        explicit_compile_count=int(compiled is not None),compile_seconds=compile_seconds,jit_signatures=runner._cache_size()))
    print(json.dumps(dict(stage='completed',aggregate=aggregate,elapsed_seconds=time.perf_counter()-begin)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','output'):parser.add_argument('--'+name,required=True)
    for name in ('bundle','calibration','robot-config'):parser.add_argument('--'+name)
    parser.add_argument('--mode',default='learned');parser.add_argument('--noise-scale',type=float,default=0.);parser.add_argument('--fixed-gain',type=float,default=2.)
    for name,default in [('queries',64),('shortlist',4),('horizon',80),('interval',4),('batch',8),('steps',800)]:parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--reactive-reselection',action='store_true')
    parser.add_argument('--ordered-waypoints',action='store_true')
    parser.add_argument('--backup-gains',nargs='+',type=float,default=[1.,2.,3.])
    parser.add_argument('--candidate-upper',type=float,default=4.)
    parser.add_argument('--candidate-design',choices=['legacy','wide'],default='legacy')
    parser.add_argument('--sensor-margin-scale',type=float,default=0.)
    parser.add_argument('--margin-guidance',action='store_true')
    parser.add_argument('--shared-clearance-budget',action='store_true')
    parser.add_argument('--motion-observer-window',type=int,default=0)
    parser.add_argument('--filter-obstacle-position',action='store_true')
    run(**vars(parser.parse_args()))
