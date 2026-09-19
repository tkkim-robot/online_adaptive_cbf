"""Native BarrierNet deployment with explicit status and complete-policy timing."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from .barriernet import BarrierNet,features,nominal,constraints,hard_deployment_qp,require_x64
from .dataset import sha256
from .io import write_json


def load_bundle(bundle):
    require_x64();root=Path(bundle);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['schema']!='barriernet_unicycle_jax_v1':raise ValueError('Not native BarrierNet')
    for filename,field in [('weights.msgpack','weights_sha256'),('normalization.npz','normalization_sha256')]:
        if sha256(root/filename)!=manifest[field]:raise ValueError('Changed BarrierNet bundle')
    model=BarrierNet();template=model.init(jax.random.PRNGKey(0),jnp.zeros(25),jnp.zeros(4),jnp.zeros(2),jnp.zeros(2))['params']
    params=serialization.from_bytes(template,(root/'weights.msgpack').read_bytes())
    with np.load(root/'normalization.npz') as f:mean,std=f['mean'],f['std']
    if mean.shape!=(25,) or std.shape!=(25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.all(std>0):raise ValueError('Invalid normalization')
    return manifest,model,params,jnp.asarray(mean),jnp.asarray(std)


def make_policy(bundle,radius=None):
    manifest,model,params,mean,std=load_bundle(bundle)
    radius=manifest['radius'] if radius is None else radius
    def policy(state,goal,obstacles,mask):
        z,ctx=features(state,goal,obstacles,mask,radius)
        reference=nominal(state,goal)
        u_nom,p=model.apply({'params':params},(z-mean)/std,state,goal,reference)
        G,h=constraints(state,ctx[6:].reshape(5,7),p,radius)
        control,valid,violation=hard_deployment_qp(u_nom,G,h)
        return control,valid,violation,p,u_nom
    return jax.jit(policy),manifest


def audit(bundle,samples=512):
    from .barriernet_audit import numpy_features,numpy_nominal,numpy_constraints,check_optimality
    from scipy.optimize import linprog
    manifest,_,params,mean,std=load_bundle(bundle);weights=jax.device_get(params)
    dataset=Path(manifest['dataset']);dm=json.loads((dataset/'manifest.json').read_text())
    if sha256(dataset/'manifest.json')!=manifest['dataset_manifest_sha256'] or sha256(dataset/'data.npz')!=dm['data_sha256']:raise ValueError('Changed training provenance')
    da=json.loads((dataset/'independent_audit.json').read_text())
    if not da['audit_passed'] or da['manifest_sha256']!=sha256(dataset/'manifest.json'):raise ValueError('Missing dataset audit')
    with np.load(dataset/'data.npz') as f:data={k:f[k] for k in f.files}
    # Normalization is independently recomputed from valid training parents.
    train=data['z'][(data['split']==0)&data['valid']]
    expected_mean=train.mean(0);expected_std=train.std(0);expected_std=np.where(expected_std==0,1.,expected_std)
    np.testing.assert_array_equal(mean,expected_mean);np.testing.assert_array_equal(std,expected_std)
    pool=np.flatnonzero(data['split']==2);chosen=np.random.default_rng(904).choice(pool,min(samples,len(pool)),replace=False)
    source=Path(dm['source']);provenance={p['group_id']:p for p in dm['provenance']};entries={e['visitation_file']:e for e in json.loads((source/'index.json').read_text())}
    by_group={}
    for row in chosen:by_group.setdefault(str(data['group_id'][row]),[]).append(int(row))
    fn,_=make_policy(bundle);execute=None;maximum_p=maximum_nominal=maximum_primal=maximum_kkt=0.;invalid=0;feasible_rejections=0;checked=[]
    def linear(x,name):return x@weights[name]['kernel']+weights[name]['bias']
    for group,rows in by_group.items():
        e=entries[provenance[group]['source_trace']]
        if sha256(source/e['visitation_file'])!=e['visitation_sha256']:raise ValueError('Changed raw observations')
        with np.load(source/e['visitation_file']) as trace:
            states=trace['observed_state'];obstacles=trace['raw_observed_obstacles'];mask=trace['obstacle_mask']
            for row in rows:
                k=int(data['tick'][row]);x=states[k].astype(float);obs=obstacles[k].astype(float);goal=data['ctx'][row,4:6]
                z,ctx=numpy_features(x,goal,obs,mask,manifest['radius']);reference=numpy_nominal(x,goal)
                encoded=np.maximum(linear(((z-expected_mean)/expected_std).reshape(5,5),'obs_fc1'),0.)
                encoded=np.maximum(linear(encoded,'obs_fc2'),0.);p=4/(1+np.exp(-linear(encoded,'fc_p')))
                hidden=np.maximum(linear(np.r_[encoded.mean(0),x,goal,reference],'u_fc1'),0.)
                u_nom=reference+linear(hidden,'u_out')
                args=tuple(jnp.asarray(v) for v in (x,goal,obs,mask))
                if execute is None:execute=fn.lower(*args).compile()
                u,valid,violation,jp,ju=jax.device_get(execute(*args))
                maximum_p=max(maximum_p,float(np.max(np.abs(jp-p))));maximum_nominal=max(maximum_nominal,float(np.max(np.abs(ju-u_nom))))
                if maximum_p>1e-10 or maximum_nominal>1e-10:raise ValueError('NumPy network/inference mismatch')
                A,b=numpy_constraints(ctx,p,manifest['radius'])
                if valid:
                    primal,kkt=check_optimality(u,u_nom,A,b,1+1e-6)
                    if primal>1e-5 or kkt>2e-6:raise ValueError(f'Invalid deployed optimum {primal}, {kkt}')
                    maximum_primal=max(maximum_primal,primal);maximum_kkt=max(maximum_kkt,kkt)
                else:
                    result=linprog(np.zeros(2),A_ub=A,b_ub=b,bounds=[(None,None)]*2,method='highs')
                    if result.status not in (0,2):raise ValueError('Independent deployment feasibility unresolved')
                    if result.status==0:
                        feasible_rejections+=1
                        if np.isfinite(u).all() and np.max(A@u-b)<=1e-5:raise ValueError('Rejected finite command has no independent constraint violation')
                    invalid+=1
                checked.append(dict(row=row,group_id=group,tick=k,valid=bool(valid)))
    report=dict(audit_passed=True,bundle_manifest_sha256=sha256(Path(bundle)/'manifest.json'),samples=len(checked),groups=len(by_group),invalid_solves_checked=invalid,
        feasible_solver_rejections_checked=feasible_rejections,independently_infeasible_solves_checked=invalid-feasible_rejections,
        max_gain_error=maximum_p,max_nominal_error=maximum_nominal,max_primal_violation=maximum_primal,max_stationarity_error=maximum_kkt,
        train_only_normalization_verified=True,runtime_compilations=0,source='Raw recorded pre-action observations from independent development-audit parents. Original nominal reference only; no expert label at deployment.',
        limitation='Algebraic deployment/lineage audit, not closed-loop performance or generalization evidence.',samples_detail=checked)
    write_json(Path(bundle)/'independent_inference_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='samples_detail'}),flush=True)


def benchmark(bundle,output,samples=256):
    fn,_=make_policy(bundle)
    # Fixed-capacity whole policy: feature selection, MLP, CBF and bounded QP.
    rng=np.random.default_rng(905);obs=np.zeros((64,5));obs[:,:2]=rng.uniform(-10,10,(64,2));obs[:,2]=.3
    args=tuple(jnp.asarray(v) for v in (np.array([0.,0.,.2,.3]),np.array([5.,2.]),obs,np.ones(64,bool)))
    begin=time.perf_counter();execute=fn.lower(*args).compile();jax.block_until_ready(execute(*args));cold=time.perf_counter()-begin
    times=[]
    for _ in range(samples):
        t=time.perf_counter();jax.device_get(execute(*args));times.append(time.perf_counter()-t)
    report=dict(device=str(jax.devices()[0]),samples=samples,cold_seconds=cold,milliseconds=(1000*np.quantile(times,[.5,.95,.99,1])).tolist(),
        runtime_compilations=0,deadline50ms_misses=int(np.sum(np.asarray(times)>.05)),weights_sha256=sha256(Path(bundle)/'weights.msgpack'),
        scope='Synchronized native FP64 complete policy from64 raw obstacle slots to returned command/status/per-obstacle gains; excludes physical simulator, observation transfer and real compute delay.')
    write_json(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','benchmark']);p.add_argument('--bundle',required=True);p.add_argument('--output')
    a=p.parse_args();audit(a.bundle) if a.action=='audit' else benchmark(a.bundle,a.output)
