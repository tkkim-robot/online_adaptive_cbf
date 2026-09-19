"""Native flight BarrierNet deployment, independent audit and timing."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from .quad2d_barriernet import BarrierNet,require_x64,contract,features,nominal,constraints,deployment_qp
from .quad2d_control import FlightConfig
from .quad2d_barriernet_audit import numpy_features,numpy_nominal,numpy_network,check_deployment
from .dataset import sha256
from .io import write_json


def load_bundle(bundle):
    require_x64();root=Path(bundle);m=json.loads((root/'manifest.json').read_text())
    if m['schema']!='barriernet_quad2d_jax_v1' or m['task_contract']!=contract():raise ValueError('Wrong native flight model/task contract')
    for file,key in [('weights.msgpack','weights_sha256'),('normalization.npz','normalization_sha256')]:
        if sha256(root/file)!=m[key]:raise ValueError('Changed native flight weights/normalization')
    model=BarrierNet();template=model.init(jax.random.PRNGKey(0),jnp.zeros(25),jnp.zeros(6),jnp.zeros(2),jnp.zeros(2))['params']
    params=serialization.from_bytes(template,(root/'weights.msgpack').read_bytes())
    if not all(np.isfinite(v).all() for v in jax.tree.leaves(params)):raise ValueError('Nonfinite native weights')
    with np.load(root/'normalization.npz') as f:mean,std=f['mean'],f['std']
    if mean.shape!=(25,) or std.shape!=(25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.all(std>0):raise ValueError('Invalid native normalization')
    return m,model,params,jnp.asarray(mean),jnp.asarray(std)


def policy_function(model,params,mean,std,config=FlightConfig()):
    def policy(state,goal,obstacles,mask):
        z,ctx=features(state,goal,obstacles,mask,config.robot.radius)
        reference=nominal(state,goal,config)
        u_nom,p=model.apply({'params':params},(z-mean)/std,state,goal,reference)
        G,h=constraints(state,ctx[8:].reshape(5,7),p,config.robot.radius)
        candidate,valid,violation,raw,solver_valid,raw_violation=deployment_qp(u_nom,G,h,config)
        return dict(candidate=candidate,valid=valid,violation=violation,raw_control=raw,solver_valid=solver_valid,raw_violation=raw_violation,
            gains=p,u_nom=u_nom,reference=reference)
    return jax.jit(policy)


def make_policy(bundle):
    m,model,params,mean,std=load_bundle(bundle)
    return policy_function(model,params,mean,std),m


def audit(bundle,samples=512):
    m,model,params,mean,std=load_bundle(bundle);dataset=Path(m['dataset']);dm=json.loads((dataset/'manifest.json').read_text())
    if sha256(dataset/'manifest.json')!=m['dataset_manifest_sha256'] or sha256(dataset/'data.npz')!=dm['data_sha256']:raise ValueError('Changed native training provenance')
    da=json.loads((dataset/'independent_audit.json').read_text())
    if not da['audit_passed'] or da['manifest_sha256']!=sha256(dataset/'manifest.json'):raise ValueError('Missing native dataset audit')
    with np.load(dataset/'data.npz') as f:d=dict(f)
    train=d['z'][(d['split']==0)&d['valid']];expected_mean=train.mean(0);expected_std=train.std(0);expected_std=np.where(expected_std==0,1.,expected_std)
    np.testing.assert_array_equal(mean,expected_mean);np.testing.assert_array_equal(std,expected_std)
    pool=np.flatnonzero(d['split']==2);chosen=np.random.default_rng(904).choice(pool,min(samples,len(pool)),replace=False)
    source=Path(dm['source']);entries={e['group_id']:e for e in json.loads((source/'index.json').read_text())};by_group={}
    for row in chosen:by_group.setdefault(str(d['group_id'][row]),[]).append(int(row))
    weights=jax.device_get(params);fn=policy_function(model,params,mean,std);execute=None;details=[];maximum_p=maximum_u=maximum_kkt=0.;invalid=clipped=feasible_rejections=0
    for group,rows in by_group.items():
        e=entries[group];path=source/e['file']
        if sha256(path)!=e['sha256']:raise ValueError('Changed raw deployment audit trace')
        with np.load(path) as trace:
            states=trace['observed_state'];obs=trace['observed_obstacles'];targets=trace['route_target'];mask=trace['obstacle_mask']
            for row in rows:
                tick=int(d['tick'][row]);x=states[tick].astype(float);goal=targets[tick].astype(float);o=obs[tick].astype(float)
                z,ctx=numpy_features(x,goal,o,mask,m['radius']);np.testing.assert_allclose(z,d['z'][row],atol=1e-10,rtol=0)
                ref=numpy_nominal(x,goal);u_nom,p=numpy_network(weights,expected_mean,expected_std,z,ctx,ref)
                args=tuple(jnp.asarray(v) for v in (x,goal,o,mask))
                if execute is None:execute=fn.lower(*args).compile()
                result=jax.device_get(execute(*args))
                maximum_p=max(maximum_p,float(np.max(np.abs(result['gains']-p))));maximum_u=max(maximum_u,float(np.max(np.abs(result['u_nom']-u_nom))))
                if maximum_p>1e-10 or maximum_u>1e-10:raise ValueError('Independent native network mismatch')
                checks=check_deployment(result,ctx,p,u_nom,classify=True);maximum_kkt=max(maximum_kkt,checks['stationarity']);clipped+=int(checks['clipped'])
                invalid+=int(not checks['accepted']);feasible_rejections+=int(not result['solver_valid'] and checks['independently_feasible'])
                details.append(dict(row=row,group_id=group,tick=tick,raw_solver_valid=bool(result['solver_valid']),post_clip_valid=bool(result['valid']),actual_fp32_accepted=checks['accepted'],clipped=checks['clipped']))
    report=dict(audit_passed=True,bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),samples=len(details),groups=len(by_group),invalid_applied_commands=invalid,clipped_commands=clipped,
        feasible_raw_solver_rejections=feasible_rejections,max_gain_error=maximum_p,max_nominal_error=maximum_u,max_stationarity_error=maximum_kkt,train_only_normalization_verified=True,runtime_compilations=0,
        scope='Raw pre-action observations from disjoint development-audit parents, native NumPy network and geometry, raw QP KKT, original wrapper clip and actual FP32 recheck. No expert label at inference.',samples_detail=details)
    write_json(Path(bundle)/'independent_inference_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='samples_detail'}),flush=True)


def benchmark(bundle,output,samples=256):
    fn,m=make_policy(bundle);dm=json.loads((Path(m['dataset'])/'manifest.json').read_text());source=Path(dm['source'])
    with np.load(Path(m['dataset'])/'data.npz') as f:
        selected=np.random.default_rng(906).choice(np.flatnonzero(f['split']==2),16,replace=False);groups=f['group_id'][selected];ticks=f['tick'][selected]
    entries={e['group_id']:e for e in json.loads((source/'index.json').read_text())};args=[]
    for group,tick in zip(groups,ticks):
        e=entries[str(group)];path=source/e['file']
        if sha256(path)!=e['sha256']:raise ValueError('Changed benchmark context')
        with np.load(path) as f:args.append(tuple(jnp.asarray(v,bool if i==3 else jnp.float64) for i,v in enumerate((f['observed_state'][tick],f['route_target'][tick],f['observed_obstacles'][tick],f['obstacle_mask']))))
    start=time.perf_counter();execute=fn.lower(*args[0]).compile();jax.block_until_ready(execute(*args[0]));cold=time.perf_counter()-start;times=[]
    for i in range(samples):
        start=time.perf_counter();jax.device_get(execute(*args[i%16]));times.append(time.perf_counter()-start)
    report=dict(device=str(jax.devices()[0]),samples=samples,observations=16,cold_seconds=cold,milliseconds=(1000*np.quantile(times,[.5,.95,.99,1])).tolist(),deadline50ms_misses=int(np.sum(np.asarray(times)>.05)),runtime_compilations=0,
        weights_sha256=m['weights_sha256'],scope='AOT synchronized full native flight policy including feature selection, MLP, hard QP, clip/status;16 real development observations. Whole physical episode throughput still required for final hardware selection.')
    write_json(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','benchmark']);p.add_argument('--bundle',required=True);p.add_argument('--output');a=p.parse_args()
    audit(a.bundle) if a.action=='audit' else benchmark(a.bundle,a.output)
