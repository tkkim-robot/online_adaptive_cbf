"""Accuracy-gated CPU/GPU QP throughput and complete rollout timing.

Research benchmark, not evidence of superiority of the learned OA-CBF method.
Every timing synchronizes results; cold compilation is recorded separately.
"""

import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from .config import UnicycleConfig
from .controllers import solve_qp2,unicycle_cbf_qp,NativeOSQP
from .scenes import random_scene
from .simulation import rollout_fixed
from .cli import save_json,sanitize


def qpax_solve(ref,a,b):
    import qpax
    qmat=jnp.eye(2,dtype=ref.dtype)
    # Ordinary hard QP API. Elastic-QP substitution is forbidden here.
    u,s,z,y,converged,iters=qpax.solve_qp(qmat,-ref,jnp.zeros((0,2),ref.dtype),jnp.zeros((0,),ref.dtype),a,b,
                                        solver_tol=1e-5,max_iter=60)
    # qpax returns an integer 0/1, not a Boolean. Keeping it integer makes a
    # NumPy "mask" select row indices 0/1 and corrupts reference-error audits.
    return u,converged==1,jnp.max(a@u-b),iters


def timed(fn,args,repeats=30):
    start=time.perf_counter();result=fn(*args);jax.block_until_ready(result)
    cold=time.perf_counter()-start
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();result=fn(*args);jax.block_until_ready(result)
        samples.append(time.perf_counter()-start)
    return result,dict(cold_seconds=cold,p50_seconds=float(np.percentile(samples,50)),p95_seconds=float(np.percentile(samples,95)),
                       p99_seconds=float(np.percentile(samples,99)),max_seconds=max(samples),repeats=repeats)


def qp_data(count,capacity,seed=731):
    scenes=[random_scene(seed+i,capacity=capacity) for i in range(count)]
    x=jnp.asarray(np.stack([s.initial_state for s in scenes]),dtype=jnp.float32)
    goal=jnp.asarray(np.stack([s.goal for s in scenes]),dtype=jnp.float32)
    obs=jnp.asarray(np.stack([s.obstacles for s in scenes]),dtype=jnp.float32)
    masks=jnp.asarray(np.stack([s.obstacle_mask for s in scenes]))
    alpha=jnp.asarray(np.exp(np.random.default_rng(seed).uniform(np.log(.5),np.log(3),size=(count,2))),dtype=jnp.float32)
    assemble=jax.jit(jax.vmap(lambda x,g,o,m,p:unicycle_cbf_qp(x,g,o,m,p,UnicycleConfig())[:3]))
    refs,a,b=assemble(x,goal,obs,masks,alpha)
    return (refs,a,b),(x,goal,obs,masks,alpha)


def run_benchmarks(output,quick=False):
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_default_matmul_precision','highest')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    def record(r):
        records.append(sanitize(r));save_json(out/'results.json',records)
        print(json.dumps(sanitize(r)),flush=True)
    for capacity in ([8] if quick else [8,16,32]):
        host_qp,host_roll=qp_data(128,capacity)
        # Stage through host memory. Direct CUDA peer copies were observed to
        # corrupt data on this workstation; do not use them for correctness or
        # claim these timings establish multi-GPU collective performance.
        host_qp=tuple(np.array(v) for v in host_qp)
        host_roll=tuple(np.array(v) for v in host_roll)
        ref,a,b=host_qp
        native=NativeOSQP(capacity+4)
        native_controls=[];native_feasible=[];samples=[]
        for i in range(128):
            start=time.perf_counter();r=native.solve(ref[i],a[i],b[i]);samples.append(time.perf_counter()-start)
            native_controls.append(r['control'] if r['feasible'] else [np.nan,np.nan]);native_feasible.append(r['feasible'])
        record(dict(backend='native_osqp',capacity=capacity,batch=1,mode='host_input_to_host_action',
                    p50_seconds=np.median(samples),p99_seconds=np.percentile(samples,99),feasible_count=sum(native_feasible),cases=128))
        # Tiny dense QP often suits ProxQP better than sparse OSQP.
        try:
            import proxsuite
            solver=proxsuite.proxqp.dense.QP(2,0,capacity+4)
            solver.settings.eps_abs=1e-8;solver.settings.eps_rel=1e-8
            solver.init(np.eye(2),-ref[0],None,None,a[0],np.full(capacity+4,-np.inf),b[0])
            timings=[];valid=0
            for i in range(128):
                start=time.perf_counter();solver.update(g=-ref[i],C=a[i],u=b[i]);solver.solve();timings.append(time.perf_counter()-start)
                valid+=int(np.isfinite(solver.results.x).all() and np.max(a[i]@solver.results.x-b[i])<=1e-5)
            record(dict(backend='native_proxqp',capacity=capacity,batch=1,mode='host_input_to_host_action',
                        p50_seconds=np.median(timings),p99_seconds=np.percentile(timings,99),feasible_count=valid,cases=128))
        except Exception as e:
            record(dict(backend='native_proxqp',capacity=capacity,status='benchmark_error',error=repr(e)))
        for device in jax.devices()+jax.devices('cpu'):
            # CPU may be the default backend; avoid duplicate records.
            if any(r.get('device')==str(device) and r.get('capacity')==capacity for r in records):continue
            with jax.default_device(device):
                for batch in ([1,32] if quick else [1,32,128]):
                    args=tuple(jax.device_put(v[:batch],device) for v in host_qp)
                    for source,transferred in zip(host_qp,args):
                        np.testing.assert_array_equal(source[:batch],np.asarray(transferred))
                    for backend in ['enumeration','qpax_fp32','qpax_fp64']:
                        fn=jax.jit(jax.vmap(lambda r,a,b:solve_qp2(r,a,b,jnp.ones(2,r.dtype)))) if backend=='enumeration' else jax.jit(jax.vmap(qpax_solve))
                        try:
                            solver_args=tuple(v.astype(jnp.float64) for v in args) if backend=='qpax_fp64' else args
                            result,timing=timed(fn,solver_args,repeats=10 if quick else 40)
                            u,feas=(result.control,result.feasible) if backend=='enumeration' else result[:2]
                            violation=np.max(np.einsum('bmi,bi->bm',a[:batch],np.asarray(u))-b[:batch],axis=-1)
                            physical=np.asarray(feas)&np.isfinite(np.asarray(u)).all(axis=-1)&(violation<=1e-5)
                            expected=np.asarray(native_feasible[:batch])
                            both=physical&expected
                            error=np.max(np.abs(np.asarray(u)[both]-np.asarray(native_controls)[:batch][both])) if both.any() else None
                            record(dict(backend=backend,device=str(device),capacity=capacity,batch=batch,mode='device_resident',**timing,
                                        feasible_count=physical.sum(),reference_feasibility_disagreement=(physical!=expected).sum(),
                                        max_action_error_vs_osqp=error,qp_per_second=batch/timing['p50_seconds'],
                                        dtype=str(solver_args[0].dtype),accuracy_eligible=bool(np.array_equal(physical,expected) and (error is None or error<1e-3))))
                        except Exception as e:
                            record(dict(backend=backend,device=str(device),capacity=capacity,batch=batch,status='benchmark_error',error=repr(e)))
                batch=32 if not quick else 4
                args=tuple(jax.device_put(v[:batch],device) for v in host_roll)
                fn=jax.jit(jax.vmap(lambda x,g,o,m,p:rollout_fixed(x,g,o,m,p,steps=100)[0]))
                result,timing=timed(fn,args,repeats=3)
                actual_steps=int(np.asarray(result.steps).sum())
                record(dict(backend='rollout_enumeration',device=str(device),capacity=capacity,batch=batch,**timing,
                            simulated_steps=actual_steps,simulated_steps_per_second=actual_steps/timing['p50_seconds'],
                            outcomes={str(k):int(v) for k,v in zip(*np.unique(np.asarray(result.status),return_counts=True))}))
        save_json(out/'progress.json',dict(last_capacity_completed=capacity,records=len(records)))
    save_json(out/'complete.json',dict(status='completed',records=len(records),purpose='initial_numerical_backend_screen'))
