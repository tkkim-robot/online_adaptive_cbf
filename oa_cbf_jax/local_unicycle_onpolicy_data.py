"""Shared acquired-state coverage for local GAT and nearest-only FC learning.

Both encoders receive the union of states visited by both frozen learned
policies and the common reference. Eight-second branches still HOLD a gain.
Full acquisition traces are retained. Branch evidence stores every outcome,
independent audit, trace digest and deterministic replay input instead of
duplicating millions of transient physical trajectory arrays on disk.
"""


import hashlib
import json
from pathlib import Path


import numpy as np


from .dataset import sha256


from .local_unicycle_dataset import validate as validate_base, kernel as label_kernel


SCHEMA='unicycle_local_shared_onpolicy_coverage_v1'
MODES=('gat','nearest_fc','fixed_reference')
FRACTIONS=(0.,1/3,2/3,1.)


def read(path):return json.loads(Path(path).read_text())


def array_digest(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps([a.dtype.str,a.shape]).encode());h.update(a.tobytes())
    return h.hexdigest()


def tree_digest(tree):
    return hashlib.sha256(json.dumps({k:array_digest(v) for k,v in sorted(tree.items())},sort_keys=True).encode()).hexdigest()


def snapshot_ticks(trace):
    """All selected times are live BEFORE their decision, including a failed QP.

    Fixed fractions cover an entire acquired trajectory without choosing
    successful scenes. Short trajectories retain explicitly correlated repeats.
    """
    query=np.asarray(trace['query'],bool);status=np.asarray(trace['status'])
    was_live=np.concatenate(([True],status[:-1]==0))
    if not np.array_equal(query,was_live&(np.arange(len(query))%4==0)):
        raise ValueError('Invalid live acquisition query cadence')
    times=np.flatnonzero(query)
    if not len(times):raise ValueError('Missing initial query')
    return times[np.rint(np.asarray(FRACTIONS)*(len(times)-1)).astype(int)]


def acquisition_errors(record):
    density=0 if record['density']=='moderate' else 1
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([record['physical_seed'],density,9209])))
    return (rng.normal(size=(1600,33,2))*record['noise']).astype(np.float32)


def future_errors(record,mode,visit,tick,current_error):
    density=0 if record['density']=='moderate' else 1
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [record['physical_seed'],density,MODES.index(mode),visit,int(tick),9221])))
    value=(rng.normal(size=(2,160,33,2))*record['noise']).astype(np.float32)
    value[:,0]=current_error
    return value


def validate_source(dataset):
    root=Path(dataset);m=read(root/'manifest.json');base=Path(m['base_dataset']);bm=validate_base(base)
    for file,key in (('manifest.json','base_manifest_sha256'),('index.json','base_index_sha256'),
        ('source.npz','base_source_sha256'),('records.json','base_records_sha256'),('reserved_groups.json','reservation_sha256')):
        if sha256(base/file)!=m[key]:raise ValueError('Changed base physical reservation: '+file)
    if (m['schema']!=SCHEMA or m['snapshot_fractions']!=list(FRACTIONS) or m['horizon_steps']!=160
        or m['modes']!=list(MODES) or m['forward_parents_used'] or m['calibration_parents_used']):
        raise ValueError('Changed on-policy collection contract')
    records=read(base/'records.json')
    if len(m['indices'])!=len(set(m['indices'])) or any(records[i]['partition'] not in ('train','validation') for i in m['indices']):
        raise ValueError('Invalid acquired parent population')
    for k,v in m['original_observation_contract'].items():
        if bm[k]!=v:raise ValueError('Changed physical observation contract')
    return m,base,records
