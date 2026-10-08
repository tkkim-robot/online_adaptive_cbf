"""Compact, exactly replayable held-gain labels with causal observer memory.

All acquisition and branch steps are physically audited. Deterministic seeds,
snapshots, every outcome and full trace hashes replace duplicated trajectory
arrays on disk. Raw-controller labels/bundles are never mixed into this dataset.
"""


from pathlib import Path
import json


import numpy as np
import jax
import jax.numpy as jnp
from .dataset import sha256


from .local_unicycle_observer import Memory,observe,reference_rollout,contract


from .local_unicycle_collection import ROBOT,K


from .obstacle_selection import nearest_obstacles
from .models import unicycle_graph

COLLECTION='local_causal_observer_compact_labels_v1'


def read(path):return json.loads(Path(path).read_text())


def graph_inputs(x,world,goal,error,memory):
    raw=x.at[:2].add(error[0]);seen=world.at[:,:2].add(error[1:])
    observed,seen=observe(memory,raw,seen)
    selected,valid,ids=nearest_obstacles(observed[:2],seen,jnp.ones(len(world),bool),K)
    features,mask=unicycle_graph(observed,goal,selected,valid,ROBOT)
    return features,mask,observed,ids


def label_kernel():
    return jax.jit(jax.vmap(lambda x,o,g,gains,errors,memory:jax.vmap(
        lambda gain,e:reference_rollout(x,o,g,e,gain,memory))(gains,errors)))


def qualified_source(root):
    root=Path(root);m=read(root/'manifest.json')
    if m.get('collection_contract')!=COLLECTION or m['controller'].get('position_observer')!=contract():raise ValueError('Unregistered observer labels')
    if m['numpy_version']!=np.__version__ or m['collector_sha256']!=sha256(__file__):raise ValueError('Use the frozen collector/RNG for replay')
    for filename,key in (('source.npz','source_sha256'),('records.json','records_sha256'),('reserved_groups.json','reservation_sha256')):
        if sha256(root/filename)!=m[key]:raise ValueError('Changed observer dataset reservation')
    records=read(root/'records.json')
    roles=('predictive_fit','predictive_audit') if m['schema']=='unicycle_local_reserved_prediction_v1' else ('train','validation')
    if len(m['indices'])!=len(set(m['indices'])) or any(records[i]['partition'] not in roles for i in m['indices']):
        raise ValueError('Forbidden or duplicate collection parent')
    return m,records


def snapshot(trace,ticks):
    ii=np.arange(4)
    return trace['before'][ii,ticks],Memory(trace['observer_prediction'][ii,ticks],trace['observer_centers'][ii,ticks],trace['observer_ready'][ii,ticks])


def branch_inputs(x,world,goal,bank,future,memory):
    expanded=np.broadcast_to(future[:,None],(4,32,2,160,33,2)).reshape(4,64,160,33,2)
    gains=np.broadcast_to(np.repeat(bank,2,axis=0),(4,64,2)).copy()
    args=(*map(jnp.asarray,(x,world,goal,gains,expanded)),jax.tree.map(jnp.asarray,memory))
    return args,gains,expanded


def validate(dataset):
    root=Path(dataset);m,records=qualified_source(root);a=read(root/'audit.json');r=read(root/'compact_replay.json')
    if m['pilot'] or not m['weight_fit_authorized'] or m['indices']!=list(range(len(records))):raise ValueError('Pilot/incomplete observer data cannot train')
    if a['manifest_sha256']!=sha256(root/'manifest.json') or a['index_sha256']!=sha256(root/'index.json') or not a['all_observer_memories_independently_checked']:
        raise ValueError('Changed observer audit')
    if r['audit_sha256']!=sha256(root/'audit.json') or not r['exact_reconstruction']:raise ValueError('Unqualified compact replay')
    partitions={}
    for record in records:partitions.setdefault(record['group_id'],set()).add(record['partition'])
    if any(len(p)!=1 for p in partitions.values()) or {next(iter(p)) for p in partitions.values()}!={'train','validation'}:
        raise ValueError('Observer parent partition leakage')
    return m
