"""Detached-job entry point: independent GPU members, host-only export, audit.

The supervisor waits on child exit notifications; it does not poll progress.
Each worker sees exactly one GPU, avoiding this host's broken peer-copy path.
"""

import argparse
import os
from pathlib import Path
import subprocess
import time

from .io import write_json


def run(dataset,output,width=96,layers=3,epochs=300,encoder='gat',seed_base=101,flight_history_invariant=False):
    if isinstance(seed_base,bool) or not isinstance(seed_base,int) or not 0<=seed_base<=2**32-4:
        raise ValueError('seed_base must allow four distinct unsigned 32-bit seeds')
    seeds=[seed_base+i for i in range(4)]
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    processes=[];logs=[];start=time.perf_counter()
    try:
        for index in range(4):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(index)
            env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
            log=(root/f'member_{index}.log').open('w',buffering=1);logs.append(log)
            command=['taskset','-c',f'{index*14}-{index*14+13}','scripts/local_python.sh','-m','oa_cbf_jax.training',
                     '--dataset',str(Path(dataset).resolve()),'--output',str(root/f'member_{index}'),
                     '--seed',str(seeds[index]),'--width',str(width),'--layers',str(layers),'--epochs',str(epochs),'--encoder',encoder]
            if flight_history_invariant:command.append('--flight-history-invariant')
            processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
        write_json(root/'supervisor.json',dict(status='training',pids=[p.pid for p in processes],width=width,layers=layers,epochs=epochs,encoder=encoder,seeds=seeds,flight_history_invariant=flight_history_invariant))
        print('Four independent GPU training members started.',flush=True)
        codes=[p.wait() for p in processes]
        if any(codes):raise RuntimeError(f'Training members failed: {codes}; inspect member logs')
    finally:
        for log in logs:log.close()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    export=['scripts/local_python.sh','-m','oa_cbf_jax.inference','export-pilot','--members',
            *(str(root/f'member_{i}') for i in range(4)),'--output',str(root/'bundle')]
    subprocess.run(export,env=env,check=True)
    evaluate=['scripts/local_python.sh','-m','oa_cbf_jax.inference','evaluate-pilot','--bundle',str(root/'bundle'),
              '--dataset',str(Path(dataset).resolve()),'--output',str(root/'evaluation')]
    subprocess.run(evaluate,env=env,check=True)
    write_json(root/'supervisor.json',dict(status='completed',elapsed_seconds=time.perf_counter()-start,seeds=seeds,
                limitation='Development training and prediction audit only; calibrated adaptive closed loop is a separate gate'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--width',type=int,default=96);parser.add_argument('--layers',type=int,default=3);parser.add_argument('--epochs',type=int,default=300)
    parser.add_argument('--encoder',choices=['gat','legacy_fc','full_fc'],default='gat')
    parser.add_argument('--seed-base',type=int,default=101,help='Four consecutive independent member seeds')
    parser.add_argument('--flight-history-invariant',action='store_true')
    run(**vars(parser.parse_args()))
