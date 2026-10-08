"""Fresh local unicycle collection and matched GAT/nearest-FC training."""


import os


import subprocess
import sys


def launch_child(job,name,args,backend,slot):
    env=os.environ.copy();env.pop('JAX_ENABLE_X64',None)
    tmp=job/'tmp';tmp.mkdir(exist_ok=True)
    env.update(JAX_PLATFORMS=backend,CUDA_VISIBLE_DEVICES=str(slot),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',TMPDIR=str(tmp.resolve()))
    with (job/(name+'.log')).open('w') as log:
        subprocess.run(['taskset','-c',f'{slot*14}-{slot*14+13}',sys.executable,'-m',*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
