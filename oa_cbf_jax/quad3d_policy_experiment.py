"""Fresh V96 trajectory calibration and paired learned Quad3D development audit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing
import json
import shutil
import time
import numpy as np
import jax
import jax.numpy as jnp
from .quad3d_policy import Quad3DSelector,Quad3DPolicyConfig,GATE_SCHEMA
from .quad3d_policy_rollout import make_policy_rollout
from .quad3d_policy_audit import audit_one
from .quad3d_data_experiment import RUNTIME as DATA_RUNTIME
from .quad3d_observation_experiment import plan_parent,observation_arrays
from .quad3d_observation import BASE_NOISE
from .quad3d_control import Quad3DControlConfig
from .quad3d_diverse_experiment import FAMILIES
from .scenes import DIVERSE_FAMILIES
from .multiscale_scenes import scene as diverse_scene
from .generalization_scenes import scene as structural_scene
from .quad3d_foundation_experiment import compile_fn,to_device,measure
from .quad3d_learning_contract import read
from .models import predict_ensemble
from .quad3d_candidate_data import GAIN_BANK
from .uncertainty import conformal_threshold
from .dataset import sha256
from .io import write_json
from .cli import sanitize

RUNTIME=tuple(dict.fromkeys((*DATA_RUNTIME,'quad3d_policy.py','quad3d_policy_rollout.py','quad3d_policy_audit.py','quad3d_policy_experiment.py',
    'quad3d_learning_contract.py','quad3d_predictive_calibration.py','models.py','inference.py','uncertainty.py','quad2d_trajectory_gate.py',
    'comparison_contracts.py','metrics.py','io.py')))
MATCHED_SCHEMA='quad3d_matched_static_policy'
PAPER_SCHEMA='paper_nearest_fc_quad3d_inputs'
DENSITY_SCHEMA='quad3d_frozen_random_density_policy_v1'
INPUT_SUPPORT_SCHEMA='quad3d_frozen_input_support_policy'


def runtime_names(schema):
    extra=('nearest_fc.py','nearest_fc_qualification.py','paper_quad3d_fc_job.py') if schema==PAPER_SCHEMA else ()
    if schema==DENSITY_SCHEMA:extra=('nearest_fc.py','nearest_fc_qualification.py','quad3d_density_study.py')
    if schema==INPUT_SUPPORT_SCHEMA:extra=('nearest_fc.py','nearest_fc_qualification.py','quad3d_density_study.py','quad3d_input_support.py','quad3d_input_support_study.py','quad3d_failure_requery.py','quad3d_frozen_confirmation.py','quad2d_random_density.py')
    return tuple(dict.fromkeys((*RUNTIME,*extra)))


def model_paths(encoder,version=96):
    if version==108 and encoder=='gat':
        report=read('reports/quad3d_v107_refit.json')
        root=Path(report['bundle']).parent
        return root/'bundle',root/'prediction_calibration/prediction_fit.json'
    model_version=99 if version in (100,108) and encoder=='gat' else 95
    root=Path(f'artifacts/training/quad3d_v{model_version}')/encoder
    return root/'bundle',root/'prediction_calibration/prediction_fit.json'


def assert_disjoint_parents(new,old):
    """Seeded suites and early unseeded probes both participate in overlap checks."""
    assert not {p['id'] for p in new}&{p['id'] for p in old}
    assert not {p['seed'] for p in new if 'seed' in p}&{p['seed'] for p in old if 'seed' in p}
    def physical(p):
        active=np.asarray(p['obstacles'],float)[np.asarray(p['mask'],bool)]
        return json.dumps([np.asarray(p['x'],float).tolist(),np.asarray(p['goal'],float).tolist(),active.tolist()],separators=(',',':'))
    assert not {physical(p) for p in new}&{physical(p) for p in old}


def parent_design(version=96):
    if version not in (96,100,108):raise ValueError('Unregistered policy experiment')
    count=432 if version==96 else 576;first_seed={96:696100,100:700100,108:708100}[version]
    for i in range(count):
        seed=first_seed+i;family=FAMILIES[i%12];level=(i//12+i%12)%3
        scene=diverse_scene(seed,family) if family in DIVERSE_FAMILIES else structural_scene(seed,family,64)
        rng=np.random.default_rng(seed+11007);x=np.zeros(12);x[:2]=scene.initial_state[:2];x[2]=rng.uniform(.4,2.4)
        broad=version==108 and (i//12+i%12)%2==1
        tilt,velocity,vz,rate=(.1,.4,.2,.15) if broad else (.025,.08,.05,.025)
        x[3:5]=rng.uniform(-tilt,tilt,2);x[5]=rng.uniform(-.2,.2);x[6:8]=rng.uniform(-velocity,velocity,2);x[8]=rng.uniform(-vz,vz);x[9:12]=rng.uniform(-rate,rate,3)
        gid=f'quad3d_v{version}:{family}:{seed}'
        p=dict(id=gid,group_id=gid,index=i,seed=seed,sensor_seed=seed+1000000,family=family,
            partition='gate_calibration' if i<288 else 'policy_audit',capacity=64,x=x.tolist(),goal=np.r_[scene.goal,rng.uniform(.2,2.8)].tolist(),
            obstacles=scene.obstacles.astype(np.float64).tolist(),mask=scene.obstacle_mask.tolist(),gains=[2.]*4,noise_level=level,noise=(level*BASE_NOISE).tolist(),
            solvability='unknown; no outcome-based resampling or exclusion')
        if version==108:p.update(motion_stratum='broad' if broad else 'small',initial_gains_by_encoder=dict(gat=[4.]*4,full_fc=[2.]*4))
        yield p


def policy_configuration(m,encoder):
    if 'policy_configs' not in m:return m['policy_config']
    assert m['schema']=='quad3d_fresh_learned_policy_v108'
    return m['policy_configs'][encoder]


def policy_parent(p,m,encoder):
    if m['schema']!='quad3d_fresh_learned_policy_v108':return p
    expected=[policy_configuration(m,encoder)['initial_gain']]*4
    assert p['initial_gains_by_encoder']==dict(gat=[4.]*4,full_fc=[2.]*4)
    assert p['gains']==p['initial_gains_by_encoder']['full_fc']
    assert p['initial_gains_by_encoder'][encoder]==expected
    return dict(p,gains=expected)


def prepare(output,version=96):
    parents=list(parent_design(version));count=len(parents)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    # Check identities only, before outcomes. Earlier scene families may recur;
    # these physical parents and sensor seeds do not enter model weight fitting.
    old=read('artifacts/datasets/quad3d_v95_inputs/parents.json')
    assert not {p['seed'] for p in old}&{p['seed'] for p in parents}
    disjoint={}
    if version in (100,108):
        for area in ('datasets','experiments'):
            for prior in sorted(Path('artifacts',area).glob('quad3d_v*_inputs/parents.json')):
                if prior.parent.resolve()==root.resolve():continue
                used=read(prior);assert_disjoint_parents(parents,used)
                disjoint[str(prior)]=sha256(prior)
    write_json(root/'preplanning_parents.json',parents)
    with ProcessPoolExecutor(12,mp_context=multiprocessing.get_context('spawn')) as pool:parents=list(pool.map(plan_parent,parents))
    write_json(root/'parents.json',parents);models={}
    for encoder in ('gat','full_fc'):
        bundle,fit=model_paths(encoder,version);pc=Quad3DPolicyConfig(initial_gain=4. if version==108 and encoder=='gat' else 2.)
        selector=Quad3DSelector(bundle,fit,reference=True,config=pc)
        expected=Quad3DControlConfig(hold_guard='bernstein_v87',nominal_bias_observer='innovation_ema_v97',qp_refinement='active_faces_v99') if version in (100,108) and encoder=='gat' else Quad3DControlConfig(hold_guard='bernstein_v87')
        assert asdict(selector.robot)==asdict(expected)
        models[encoder]=dict(bundle=str(bundle.resolve()),bundle_manifest_sha256=sha256(bundle/'manifest.json'),weights_sha256=selector.metadata['weights_sha256'],
            prediction_fit=str(fit.resolve()),prediction_fit_sha256=sha256(fit),config=asdict(selector.robot))
    configuration=dict(config=asdict(Quad3DControlConfig(hold_guard='bernstein_v87'))) if version==96 else dict(
        configs={e:models[e]['config'] for e in models},prior_parent_files=disjoint,
        frozen_fc_gate=str(Path('artifacts/experiments/quad3d_v96_policy/full_fc_trajectory_gate.json').resolve()),
        frozen_fc_gate_sha256=sha256('artifacts/experiments/quad3d_v96_policy/full_fc_trajectory_gate.json'))
    if version==108:
        configuration.update(policy_configs={e:asdict(Quad3DPolicyConfig(initial_gain=4. if e=='gat' else 2.)) for e in models},storage_floor_gib=100.,
            learning_review_sha256=sha256('reports/quad3d_v107_review.json'),
            gate_reference_initial_gain=4.,motion_distribution='Balanced small/broad within each family/noise/partition; no outcome filtering')
    write_json(root/'manifest.json',dict(schema=f'quad3d_fresh_learned_policy_v{version}',parents=count,reference_parents=288,adaptive_parents=count-288,
        parents_sha256=sha256(root/'parents.json'),preplanning_sha256=sha256(root/'preplanning_parents.json'),source_files={n:sha256(Path(__file__).parent/n) for n in RUNTIME},
        **configuration,policy_config=asdict(Quad3DPolicyConfig()),steps=1600,batch=12,capacity=64,
        models=models,training_use=False,weight_fit_authorized=False,final_test=False,
        partitioning=f'Before outcomes:24reference and{(count-288)//12}adaptive physicalparents per family; balancednoise0/1/2. Same physical parents and observation tapes for both encoders.',
        gate='Model-specific family trajectory-max CS rank95, maximum across12families. V108 reference fixed4; prior reference fixed2. V100/V108 calibrate GAT only and preserve V96 FC gate. No automatic coverage claim on changed adaptive trajectories.',
        policy='Every4ticks: actual trained model scores16gains, epistemic/conditional Gaussian-tail/adverse screens, current lower-cascade admission, predicted progress ranking. Failed selected QP retries only previous gain, rechecking its full hard constraints. No physical rollout search.',
        route_statuses=dict(Counter(p['route']['status'] for p in parents))))
    print(json.dumps(dict(stage='prepared',parents=count,gate=288,adaptive=count-288)),flush=True)


def inputs(source,encoder,phase,slot,gate):
    root=Path(source);m=read(root/'manifest.json');assert m['parents_sha256']==sha256(root/'parents.json')
    assert m['source_files']=={n:sha256(Path(__file__).parent/n) for n in runtime_names(m['schema'])}
    mp=m['models'][encoder];assert mp['prediction_fit_sha256']==sha256(mp['prediction_fit']) and mp['bundle_manifest_sha256']==sha256(Path(mp['bundle'])/'manifest.json')
    selector=Quad3DSelector(mp['bundle'],mp['prediction_fit'],gate=gate,reference=phase=='gate_calibration',config=Quad3DPolicyConfig(**policy_configuration(m,encoder)))
    expected=m['configs'][encoder] if 'configs' in m else m['config']
    assert asdict(selector.robot)==expected
    if m['schema']==INPUT_SUPPORT_SCHEMA:
        from .quad3d_input_support_study import verify_source
        from .quad3d_input_support import InputSupportSelector
        parents=verify_source(root)
        saved=m['gates'][encoder]
        if phase!='policy_audit' or gate is None or Path(gate).resolve()!=Path(saved['path']).resolve() or sha256(gate)!=saved['sha256']:
            raise ValueError('Input support requires the original frozen adaptive gate')
        chosen=[p for p in parents if p['study_slot']==slot]
        if slot not in range(4) or len(chosen)!=m['slot_counts'][slot] or not chosen:
            raise ValueError('Wrong complete input-support partition')
        return m,chosen,InputSupportSelector(selector)
    if m['schema']==DENSITY_SCHEMA:
        from .quad3d_density_study import verify_source
        parents=verify_source(root,m)
        if phase!='policy_audit' or gate is None or encoder not in ('gat','nearest_fc'):
            raise ValueError('Frozen density study is deployment only')
        saved=m['gates'][encoder]
        if Path(gate).resolve()!=Path(saved['path']).resolve() or sha256(gate)!=saved['sha256']:
            raise ValueError('Density study cannot refit or change gates')
        chosen=[p for p in parents if p['study_slot']==slot]
        if slot not in range(4) or len(chosen)!=m['slot_counts'][slot] or not chosen:
            raise ValueError('Wrong complete density partition')
        return m,chosen,selector
    if m['schema']==PAPER_SCHEMA:
        from .paper_quad3d_fc_job import verify
        verify(root, m, selector.metadata)
        assert encoder=='nearest_fc'
    if m['schema']==MATCHED_SCHEMA:
        from .comparison_contracts import matched_controller_settings,physical_obstacle_scope
        matched_controller_settings(m['comparison_contracts']['gat'],m['comparison_contracts']['matched_fc'])
        assert encoder in ('gat','matched_fc') and selector.metadata['architecture']['encoder']==encoder
        assert 'configs' not in m and 'policy_configs' not in m
        for parent in read(root/'parents.json'):
            physical_obstacle_scope('quad3d',parent['obstacles'],parent['mask'])
            assert parent['gains']==[m['policy_config']['initial_gain']]*4
    if m['schema'] in ('quad3d_fresh_learned_policy_v100','quad3d_fresh_learned_policy_v108') and encoder=='full_fc':
        assert phase=='policy_audit' and gate is not None
        assert Path(gate).resolve()==Path(m['frozen_fc_gate']) and sha256(gate)==m['frozen_fc_gate_sha256']
    parents=[policy_parent(p,m,encoder) for p in read(root/'parents.json') if p['partition']==phase and (p['index']//12)%4==slot]
    assert len(parents)==m['reference_parents' if phase=='gate_calibration' else 'adaptive_parents']//4
    return m,parents,selector


def benchmark(source,encoder,phase,slot,gate,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);m,parents,selector=inputs(source,encoder,phase,slot,gate)
    args=(selector.predictor.params,*to_device(observation_arrays(parents[:12],40)))
    fn,exe,cold=compile_fn(jax.vmap(make_policy_rollout(selector,40),in_axes=(None,)+(0,)*12),args)
    result,timing=measure(exe,args,repeats=3);summary,(trace,queries)=jax.device_get(result)
    np.savez_compressed(root/'numerics.npz',status=summary['status'],steps=summary['steps'],final_state=summary['final_state'],
        control=trace['control'],gains=trace['controller_gain'],query_mean=queries['prediction_mean'][:,::4],query_variance=queries['prediction_variance'][:,::4],
        query_logits=queries['prediction_event_logits'][:,::4],query_selected=queries['selected_index'][:,::4])
    assert fn._cache_size()==0
    write_json(root/'report.json',dict(source_sha256=sha256(Path(source)/'manifest.json'),encoder=encoder,phase=phase,
        weights_sha256=selector.metadata['weights_sha256'],compile_seconds=cold,**timing,implicit_jit_cache_entries=0,numerics_sha256=sha256(root/'numerics.npz'),device=str(jax.devices()[0])))
    print((root/'report.json').read_text(),flush=True)


def replay_predictions(out,rows,selector):
    """Re-evaluate the actual stored graph on a separately compiled batch32 path."""
    norm=selector.metadata['normalization'];mean=jnp.asarray(norm['target_mean'],jnp.float32);scale=jnp.asarray(norm['target_scale'],jnp.float32)
    vs=jnp.asarray(selector.fit['variance_scale'],jnp.float32)
    def predict(params,f,mask,gains):
        raw=predict_ensemble(selector.predictor.model,params,f,mask,gains)
        return dict(prediction_mean=raw['mean']*scale+mean,prediction_variance=jnp.exp(raw['log_variance'])*scale**2*vs,prediction_event_logits=raw['event_logits'])
    bank=jnp.asarray(np.broadcast_to(selector.bank.astype(np.float32),(32,16,4)))
    fn=jax.jit(predict);exe=fn.lower(selector.predictor.params,jnp.zeros((32,66,selector.metadata['graph_features']),jnp.float32),jnp.ones((32,66),bool),bank).compile()
    maxima={k:0. for k in ('prediction_mean','prediction_variance','prediction_event_logits')};queries=0
    for row in rows:
        assert row['query_sha256']==sha256(out/row['query_file'])
        with np.load(out/row['query_file']) as z:q=dict(z)
        for i in range(0,len(q['query_tick']),32):
            n=min(32,len(q['query_tick'])-i);f=np.pad(q['features'][i:i+n],((0,32-n),(0,0),(0,0)));mask=np.pad(q['node_mask'][i:i+n],((0,32-n),(0,0)))
            pred=jax.device_get(exe(selector.predictor.params,jnp.asarray(f),jnp.asarray(mask),bank))
            for key,v in pred.items():
                replay=np.moveaxis(v[:,:n],0,1);expected=q[key][i:i+n]
                np.testing.assert_allclose(replay,expected,atol=3e-5,rtol=3e-5,err_msg=key)
                maxima[key]=max(maxima[key],float(np.max(abs(replay-expected))))
            queries+=n
    assert fn._cache_size()==0
    return dict(all_saved_query_predictions_recomputed=True,queries=queries,maximum_absolute_errors=maxima,implicit_jit_cache_entries=0)


def padded_density_batch(parents):
    if not 1<=len(parents)<=12:raise ValueError('Expected one nonempty batch of at most12 parents')
    return parents+[parents[-1]]*(12-len(parents))


def collect(source,encoder,phase,slot,gate,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);m,parents,selector=inputs(source,encoder,phase,slot,gate)
    fn=exe=None;rows=[];duration=0.;start=time.perf_counter()
    for first in range(0,len(parents),12):
        if shutil.disk_usage(out).free/2**30<m.get('storage_floor_gib',125.):raise ValueError('Learning storage reserve reached')
        pp=parents[first:first+12]
        executed=padded_density_batch(pp) if m['schema'] in (DENSITY_SCHEMA,INPUT_SUPPORT_SCHEMA) else pp
        args=(selector.predictor.params,*to_device(observation_arrays(executed,m['steps'])))
        if exe is None:fn,exe,cold=compile_fn(jax.vmap(make_policy_rollout(selector,m['steps'],failure_requery=m.get('failure_requery',False)),in_axes=(None,)+(0,)*12),args)
        tick=time.perf_counter();summary,(trace,prediction)=jax.device_get(exe(*args));duration+=time.perf_counter()-tick
        for i,p in enumerate(pp):
            count=int(summary['steps'][i]);length=min(m['steps'],count+1);d={k:v[i,:length] for k,v in trace.items()};qt=np.flatnonzero(d['requery'])
            q={k:v[i,qt] for k,v in prediction.items()};q['query_tick']=qt
            file=f'{first+i:03d}.npz';qfile=f'{first+i:03d}_queries.npz';np.savez_compressed(out/file,**d);np.savez_compressed(out/qfile,**q)
            rows.append(dict(id=p['id'],group_id=p['group_id'],family=p['family'],noise_level=p['noise_level'],file=file,query_file=qfile,sha256=sha256(out/file),query_sha256=sha256(out/qfile),
                steps=count,status=int(summary['status'][i]),final_state=summary['final_state'][i].tolist(),queries=len(qt),maximum_cs=float(q['cs_score'].max()),
                learned_queries=int(np.sum(q['selected_index']>=0)),uncertainty_fallback_queries=int(q['uncertainty_fallback'].sum()),admission_fallback_queries=int(q['admission_fallback'].sum()),
                qp_switch_fallbacks=int(d['qp_switch_fallback'].sum()),applied_gain_changes=int(np.sum(d['active']&np.any(d['controller_gain']!=d['previous_gain'],axis=-1)))))
            if m['schema']=='quad3d_fresh_learned_policy_v108':rows[-1].update(initial_gains=p['gains'],motion_stratum=p['motion_stratum'])
            if m.get('failure_requery',False):rows[-1]['failure_requeries']=int(d['failure_requery'].sum())
        progress=dict(parents_complete=len(rows),parents_total=len(parents),physical_steps=sum(r['steps'] for r in rows),queries=sum(r['queries'] for r in rows),execute_seconds=duration,elapsed_seconds=time.perf_counter()-start)
        write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert fn._cache_size()==0
    replay=replay_predictions(out,rows,selector);write_json(out/'prediction_replay.json',replay);write_json(out/'index.json',rows)
    write_json(out/'manifest.json',dict(source=str(Path(source).resolve()),source_sha256=sha256(Path(source)/'manifest.json'),source_files=m['source_files'],
        config=asdict(selector.robot),policy_config=asdict(selector.config),steps=m['steps'],transition_guard=True,adaptive_gain_trace=True,encoder=encoder,phase=phase,
        prediction_fit=m['models'][encoder]['prediction_fit'],prediction_fit_sha256=m['models'][encoder]['prediction_fit_sha256'],weights_sha256=selector.metadata['weights_sha256'],
        gate=None if gate is None else str(Path(gate).resolve()),gate_sha256=None if gate is None else sha256(gate),cs_threshold=None if gate is None else float(selector.threshold),
        index_sha256=sha256(out/'index.json'),prediction_replay_sha256=sha256(out/'prediction_replay.json'),compile_seconds=cold,execute_seconds=duration,
        implicit_jit_cache_entries=0,device=str(jax.devices()[0]),
        **(dict(input_box_admission=True,failure_requery=m.get('failure_requery',False)) if m['schema']==INPUT_SUPPORT_SCHEMA else {})))


def audit(output,workers=12):
    out=Path(output);m=read(out/'manifest.json');sm=read(Path(m['source'])/'manifest.json');parents=read(Path(m['source'])/'parents.json');by={p['id']:policy_parent(p,sm,m['encoder']) for p in parents}
    assert m.get('input_box_admission',False)==(sm['schema']==INPUT_SUPPORT_SCHEMA)
    assert m.get('failure_requery',False)==sm.get('failure_requery',False)
    assert m['source_sha256']==sha256(Path(m['source'])/'manifest.json') and m['source_files']==sm['source_files']=={n:sha256(Path(__file__).parent/n) for n in runtime_names(sm['schema'])}
    assert m['config']==(sm['configs'][m['encoder']] if 'configs' in sm else sm['config'])
    assert m['policy_config']==policy_configuration(sm,m['encoder'])
    assert m['index_sha256']==sha256(out/'index.json') and m['prediction_replay_sha256']==sha256(out/'prediction_replay.json')
    assert m['prediction_fit_sha256']==sha256(m['prediction_fit']) and read(out/'prediction_replay.json')['all_saved_query_predictions_recomputed']
    if m['gate'] is not None:assert m['gate_sha256']==sha256(m['gate'])
    rows=read(out/'index.json');results=[]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(audit_one,[(by[r['id']],r,str(out),m) for r in rows]):
            results.append(r);write_json(out/'audit_progress.json',dict(parents_audited=len(results),parents_total=len(rows),physical_steps=sum(r['steps'] for r in results)))
    write_json(out/'independent_audit.json',dict(audit_passed=True,manifest_sha256=sha256(out/'manifest.json'),index_sha256=sha256(out/'index.json'),
        physical_steps=sum(r['steps'] for r in results),rows=results,all_query_features_decisions_memories_verified=True,
        all_model_predictions_recomputed=True,all_continuous_physics_checked=True))


def gate(source,encoder,directories,output):
    sm=read(Path(source)/'manifest.json');pp=read(Path(source)/'parents.json');rows=[];proof=[]
    for directory in map(Path,directories):
        m=read(directory/'manifest.json');a=read(directory/'independent_audit.json')
        assert m['phase']=='gate_calibration' and m['encoder']==encoder and m['source_sha256']==sha256(Path(source)/'manifest.json')
        assert a['audit_passed'] and a['manifest_sha256']==sha256(directory/'manifest.json') and a['index_sha256']==sha256(directory/'index.json')
        rows.extend(a['rows']);proof.append(dict(directory=str(directory),audit_sha256=sha256(directory/'independent_audit.json')))
    expected={p['id'] for p in pp if p['partition']=='gate_calibration'};assert len(rows)==sm['reference_parents'] and {r['id'] for r in rows}==expected
    families={f:conformal_threshold([r['maximum_cs'] for r in rows if r['family']==f],.95) for f in sm.get('families',FAMILIES)}
    assert all(v['n_groups']==24 and v['status']=='calibrated' for v in families.values())
    write_json(output,dict(schema=GATE_SCHEMA,encoder=encoder,weights_sha256=sm['models'][encoder]['weights_sha256'],prediction_fit_sha256=sm['models'][encoder]['prediction_fit_sha256'],
        policy_config=policy_configuration(sm,encoder),source_manifest_sha256=sha256(Path(source)/'manifest.json'),threshold=max(v['threshold'] for v in families.values()),family_thresholds=families,
        reference_parents=len(expected),group_ids=sorted(expected),proof=proof,production_eligible=False,whole_goal_complete=False,
        limitation='Family trajectory-max CS rank on frozen fixed-gain reference; changed adaptive trajectories need independent empirical audit. This is not a physical-safety or automatic adaptive-coverage theorem.'))
    print(json.dumps(dict(stage='gate_frozen',encoder=encoder,threshold=read(output)['threshold'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','benchmark','collect','audit','gate']);p.add_argument('--source');p.add_argument('--output',required=True)
    p.add_argument('--encoder',choices=['gat','full_fc','matched_fc','nearest_fc']);p.add_argument('--phase',choices=['gate_calibration','policy_audit']);p.add_argument('--slot',type=int,default=0)
    p.add_argument('--gate');p.add_argument('--workers',type=int,default=12);p.add_argument('--directories',nargs='+');p.add_argument('--version',type=int,choices=[96,100,108],default=96);a=p.parse_args()
    if a.action=='prepare':prepare(a.output,a.version)
    elif a.action=='audit':audit(a.output,a.workers)
    elif a.action=='gate':gate(a.source,a.encoder,a.directories,a.output)
    else:dict(benchmark=benchmark,collect=collect)[a.action](a.source,a.encoder,a.phase,a.slot,a.gate,a.output)
