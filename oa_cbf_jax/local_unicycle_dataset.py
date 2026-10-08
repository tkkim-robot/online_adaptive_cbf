"""Reserved, parent-grouped local-observation unicycle training data.

Same eight-second physical controller as the qualified pilot. The noisy
acquisition gain is explicitly (4,1); this is shared data acquisition, not a
change to a compared baseline. Future innovations are stored once per replica.
"""


import json
from pathlib import Path


import numpy as np
import jax


from .dataset import sha256


from .local_unicycle_collection import ROBOT,NOMINAL,K,HORIZON,rollout,graph_inputs,audit_branch
from .local_unicycle_study import make_scene,GAINS,FAMILIES as DENSE_FAMILIES
from .obstacle_selection import contract

SCHEMA='unicycle_local_observation_learning_v1'
FAMILIES=(*DENSE_FAMILIES,'scatter','wide_corridor','staggered_islands','split_gate')


def scene(seed,family,density,side):
    if family in DENSE_FAMILIES:return make_scene(seed,family,density,side)
    rng=np.random.default_rng(seed);r=rng.uniform(.18,.36,32);o=np.zeros((32,5),np.float32);o[:,2]=r
    factor=1. if density=='dense' else 1.6
    if family=='scatter':
        o[:,0]=rng.uniform(3.,3.+12*factor,32);o[:,1]=rng.uniform(-3.5,3.5,32)
    elif family=='wide_corridor':
        o[:,0]=3.+np.repeat(np.arange(16),2)*.75*factor
        o[:,1]=np.tile([-1.,1.],16)*(1.1+r)+rng.uniform(-.1,.1,32)
    elif family=='staggered_islands':
        o[:,0]=3.+np.repeat(np.arange(8),4)*2*factor+np.tile([0,.25,.5,.75],8)
        o[:,1]=np.repeat(np.tile([-1.,1.],4),4)*np.tile([.1,.4,.7,1.],8)
    elif family=='split_gate':
        o[:,0]=3.+np.repeat(np.arange(8),4)*1.4*factor
        o[:,1]=np.tile([-1.8,-.9,.9,1.8],8)+np.repeat(rng.uniform(-.4,.4,8),4)
    else:raise ValueError('Unknown family')
    o[:,1]*=side
    return o,np.array([float(o[:,0].max()+3.),0.],np.float32)


def kernel():
    return jax.jit(jax.vmap(lambda x,o,g,gains,errors,status:jax.vmap(lambda gain,e:rollout(x,o,g,gain,e,status))(gains,errors)))


def validate(dataset):
    root=Path(dataset);m=json.loads((root/'manifest.json').read_text());a=json.loads((root/'audit.json').read_text())
    if m.get('controller',{}).get('position_observer') is not None:
        from .local_unicycle_observer_data import validate as validate_observer
        return validate_observer(root)
    if (m['schema']!=SCHEMA or m['pilot'] or not m['weight_fit_authorized'] or not a['weight_fit_authorized']
        or m['horizon_steps']!=160 or m['neighborhood']!=contract(10) or m['acquisition_gain']!=[4.,1.]
        or a['manifest_sha256']!=sha256(root/'manifest.json') or a['index_sha256']!=sha256(root/'index.json')
        or not a['all_saved_bindings_checked'] or not a['all_applied_branches_audited_at_collection']):raise ValueError('Unqualified local-observation training data')
    if m.get('records_sha256')!=sha256(root/'records.json'):
        raise ValueError('Changed parent-group reservation')
    partitions={}
    for r in json.loads((root/'records.json').read_text()):partitions.setdefault(r['group_id'],set()).add(r['partition'])
    if any(len(v)!=1 for v in partitions.values()):raise ValueError('Base-parent split leakage')
    if {next(iter(v)) for v in partitions.values()}!={'train','validation'}:raise ValueError('Invalid weight-fitting partition')
    return m
