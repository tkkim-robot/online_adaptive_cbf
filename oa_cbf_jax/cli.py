"""Headless commands; use `python -m oa_cbf_jax.cli --help`."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import time

# Set before importing JAX; preserve explicit caller settings.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def sanitize(value):
    import numpy as np
    if hasattr(value,"tolist"): return sanitize(value.tolist())
    if isinstance(value,dict): return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [sanitize(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def save_json(path,value):
    from .io import write_json
    write_json(path,sanitize(value))


def doctor(output):
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .config import UnicycleConfig,config_hash
    records=[]
    for device in jax.devices():
        with jax.default_device(device):
            x=jnp.ones((256,256),dtype=jnp.float32)
            t=time.perf_counter(); y=jax.jit(lambda z:z@z)(x);y.block_until_ready()
            np.testing.assert_allclose(np.asarray(y),256)
            records.append(dict(device=str(device),kind=device.device_kind,compile_execute_seconds=time.perf_counter()-t))
    pkgs={p:importlib.metadata.version(p) for p in ['jax','jaxlib','flax','optax','numpy','scipy','qpax','osqp','proxsuite']}
    transfers=[]
    source=np.arange(2048,dtype=np.float32).reshape(32,64)
    origin=jax.device_put(source,jax.devices()[0])
    for device in jax.devices():
        direct=np.asarray(jax.device_put(origin,device))
        staged=np.asarray(jax.device_put(source.copy(),device))
        np.testing.assert_array_equal(staged,source)
        transfers.append(dict(destination=str(device),direct_exact=bool(np.array_equal(direct,source)),
                              direct_max_error=float(np.max(np.abs(direct-source))),host_staged_exact=True))
    report=dict(python=platform.python_version(),platform=platform.platform(),devices=records,packages=pkgs,
                gpu_transfers=transfers,peer_transfers_eligible=all(t['direct_exact'] for t in transfers),
                available_cpu_affinity=sorted(os.sched_getaffinity(0)),config=asdict(UnicycleConfig()),config_hash=config_hash(UnicycleConfig()),
                nvidia_smi=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,driver_version','--format=csv'],capture_output=True,text=True).stdout)
    save_json(output,report);print(json.dumps(report,indent=2),flush=True)


def smoke(output,steps):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .config import UnicycleConfig,config_hash
    from .scenes import fixture
    from .simulation import rollout_fixed,STATUS_NAMES
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ['open','offset','blocking','collision','crossing']:
        scene=fixture(name)
        for gain in [.5,1.,2.]:
            t=time.perf_counter()
            summary,trace=rollout_fixed(jnp.asarray(scene.initial_state,dtype=jnp.float32),jnp.asarray(scene.goal,dtype=jnp.float32),
                                       jnp.asarray(scene.obstacles,dtype=jnp.float32),jnp.asarray(scene.obstacle_mask),jnp.full(2,gain),steps=steps)
            summary.final_state.block_until_ready()
            elapsed=time.perf_counter()-t
            rec=sanitize(summary._asdict());rec.update(scene_id=scene.scene_id,gain=gain,status_name=STATUS_NAMES[int(summary.status)],wall_seconds=elapsed,
                                                      config_hash=config_hash(UnicycleConfig()),timing_mode='synchronous_compute_ignored')
            records.append(rec)
            np.savez_compressed(out/f'{name}_{gain}.npz',initial_state=scene.initial_state,goal=scene.goal,obstacles=scene.obstacles,
                                obstacle_mask=scene.obstacle_mask,alpha=np.full(2,gain),**{k:np.asarray(v) for k,v in trace.items()})
            print(json.dumps(rec),flush=True)
    save_json(out/'summary.json',records)
    assert all(r['status_name']=='goal_reached' for r in records if r['scene_id']=='fixture:open'), 'Open-space regression'
    assert all(r['status_name']=='collision' and r['steps']==0 for r in records if r['scene_id']=='fixture:collision'), 'Initial collision regression'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('doctor');p.add_argument('--output',default='reports/environment.json')
    p=sub.add_parser('smoke');p.add_argument('--output',default='artifacts/smoke');p.add_argument('--steps',type=int,default=400)
    p=sub.add_parser('benchmark');p.add_argument('--output',default='artifacts/benchmarks');p.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    if args.command=='doctor': doctor(args.output)
    elif args.command=='smoke': smoke(args.output,args.steps)
    elif args.command=='benchmark':
        from .benchmark import run_benchmarks
        run_benchmarks(args.output,args.quick)


if __name__=='__main__':main()
