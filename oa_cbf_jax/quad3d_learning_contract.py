"""Quad3D-only training/inference lineage, split and censoring contracts."""
from pathlib import Path
import json
import numpy as np
from .dataset import sha256
from .quad3d_candidate_data import SCHEMA,REPLICAS,HORIZON,QUERY_TICKS,TARGETS,EVENTS,OBSERVER_SCHEMA,OBSERVER_QUERY_TICKS,WIDE_SCHEMA,WIDE_QUERY_TICKS,candidate_bank,candidate_domain
from .quad3d_features import SCHEMA as GRAPH_SCHEMA,FEATURES,OBSERVER_SCHEMA as OBSERVER_GRAPH,OBSERVER_FEATURES,BIAS_INDICES,BIAS_SCALE
from .quad3d_observation import SCHEMA as SENSOR_SCHEMA

CONTRACT_FIELDS=('config','sensor_schema','graph_schema','horizon_steps','capacity','snapshot_ticks','replicas','gain_bank','branch_contract','censoring')


def read(path):return json.loads(Path(path).read_text())


def registered(schema):
    if schema==SCHEMA:return GRAPH_SCHEMA,FEATURES,QUERY_TICKS,'none'
    if schema==OBSERVER_SCHEMA:return OBSERVER_GRAPH,OBSERVER_FEATURES,OBSERVER_QUERY_TICKS,'innovation_ema_v97'
    if schema==WIDE_SCHEMA:return OBSERVER_GRAPH,OBSERVER_FEATURES,WIDE_QUERY_TICKS,'innovation_ema_v97'
    raise ValueError('Unregistered Quad3D dataset schema')


def validate_arrays(d,manifest):
    graph,features,ticks,observer=registered(manifest['schema'])
    bank=candidate_bank(manifest['schema']);n=len(d['group_id']);capacity=manifest['capacity'];branches=len(bank)*REPLICAS
    shapes=dict(features=(n,capacity+2,features),node_mask=(n,capacity+2),gains=(n,branches,4),
        target=(n,branches,2),target_mask=(n,branches,2),events=(n,branches,2),event_mask=(n,branches,2),
        status=(n,branches),steps=(n,branches),recorded_physical_steps=(n,branches),initial_state=(n,12),goal=(n,3))
    if observer!='none':shapes['nominal_bias_estimate']=(n,12)
    for key,shape in shapes.items():
        if d[key].shape!=shape:raise ValueError('Quad3D shape mismatch: '+key)
        if not np.isfinite(d[key]).all():raise ValueError('Nonfinite Quad3D field: '+key)
    if d['features'].dtype!=np.float32 or d['node_mask'].dtype!=bool:raise ValueError('Wrong graph storage dtype')
    if np.any(d['features'][~d['node_mask']]) or not d['node_mask'][:,:2].all():raise ValueError('Invalid graph mask')
    if observer!='none':
        expected=d['nominal_bias_estimate'][:,BIAS_INDICES]/BIAS_SCALE
        expected=np.where(d['node_mask'][...,None],expected[:,None,:],0.).astype(np.float32)
        if not np.allclose(d['features'][...,-8:],expected,atol=2e-7,rtol=2e-7):raise ValueError('Graph disagrees with causal observer memory')
        if np.any(d['nominal_bias_estimate'][:,[0,1,2,5]]):raise ValueError('Unidentifiable observer bias')
    bank=np.repeat(bank,REPLICAS,axis=0).astype(np.float32)
    if not np.array_equal(d['gains'],np.broadcast_to(bank,d['gains'].shape)):raise ValueError('Changed candidate/replica order')
    if not np.isin(d['query_tick'],ticks).all():raise ValueError('Undeclared acquired query tick')
    status=d['status'];observed=np.isin(status,[1,4,6])
    if not np.isin(status,[1,2,3,4,5,6]).all():raise ValueError('Unterminated label branch')
    if not np.array_equal(d['target_mask'][...,0],observed) or not np.array_equal(d['event_mask'][...,0],observed):raise ValueError('Censored safety targets were changed')
    if not d['target_mask'][...,1].all() or not d['event_mask'][...,1].all():raise ValueError('Missing observed prefix/adverse targets')
    if not np.array_equal(d['events'][...,0],status==4) or not np.array_equal(d['events'][...,1],~np.isin(status,[1,6])):raise ValueError('Invalid first-event targets')
    expected=np.where(observed,-np.minimum(d['min_clearance'],.6)/.3,0.).astype(np.float32)
    if not np.array_equal(d['target'][...,0],expected):raise ValueError('Risk target disagrees with audited clearance')
    if np.any(d['steps']<0) or np.any(d['recorded_physical_steps']>HORIZON) or np.any(d['steps']>d['recorded_physical_steps']):raise ValueError('Invalid physical prefix lengths')


def validate_training_dataset(directory):
    root=Path(directory);m=read(root/'manifest.json');audit=read(root/'independent_replay.json')
    if (root/'INVALIDATED.json').exists():raise ValueError('Invalidated Quad3D dataset')
    if m.get('weight_fit_authorized') is not True or m.get('schema') not in (SCHEMA,OBSERVER_SCHEMA,WIDE_SCHEMA):raise ValueError('Authorized Quad3D learning data required')
    graph,features,ticks,observer=registered(m['schema'])
    if m['config'].get('nominal_bias_observer','none')!=observer:raise ValueError('Wrong Quad3D observer contract')
    if m['graph_schema']!=graph or m['sensor_schema']!=SENSOR_SCHEMA or m['gain_dimension']!=4 or m['graph_features']!=features:raise ValueError('Wrong Quad3D model contract')
    if m['targets']!=TARGETS or m['events']!=EVENTS or m['horizon_steps']!=HORIZON or m['replicas']!=REPLICAS or m['snapshot_ticks']!=list(ticks):raise ValueError('Changed Quad3D target/horizon semantics')
    if m['gain_bank']!=candidate_bank(m['schema']).tolist() or m['gain_domain']!=candidate_domain(m['schema']):raise ValueError('Wrong Quad3D gain bank')
    flags=('audit_passed','all_features_independently_checked','all_branch_physics_checked','all_parents_retained','all_split_groups_disjoint')
    if not all(audit.get(k) for k in flags) or audit['manifest_sha256']!=sha256(root/'manifest.json') or audit['index_sha256']!=sha256(root/'index.json'):raise ValueError('Complete bound independent Quad3D audit required')
    source=Path(m['source']);sm=read(source/'manifest.json');parents=read(source/'parents.json')
    if m['source_manifest_sha256']!=sha256(source/'manifest.json') or sm['parents_sha256']!=sha256(source/'parents.json'):raise ValueError('Wrong physical parent reservation')
    for field in (*CONTRACT_FIELDS,'groups','source_files','controller','targets','events','partitions','weight_fit_authorized'):
        if m[field]!=sm[field]:raise ValueError('Changed source semantics: '+field)
    reservation={p['group_id']:p['partition'] for p in parents}
    if len(reservation)!=m['parents']:raise ValueError('Missing/duplicate physical parents')
    if set(reservation.values())!={'train','validation','prediction_fit','prediction_audit'}:raise ValueError('Required separate fit/audit splits absent')
    seen=set();queries=set();branches=steps=0
    for entry in read(root/'index.json'):
        path=root/entry['file'];proof=read(path.parent/'audit.json')
        if entry['sha256']!=sha256(path) or entry['audit_sha256']!=sha256(path.parent/'audit.json') or proof['data_sha256']!=entry['sha256'] or not proof['audit_passed']:raise ValueError('Unbound Quad3D label shard')
        if not proof['all_features_independently_checked'] or not proof['all_branch_physics_checked']:raise ValueError('Unaudited labels/features')
        if observer!='none' and not proof.get('all_observer_memories_inherited_and_replayed'):raise ValueError('Unaudited branch observer memory')
        with np.load(path,allow_pickle=False) as z:d=dict(z)
        validate_arrays(d,m)
        for group,role,tick in zip(d['group_id'],d['partition'],d['query_tick'],strict=True):
            if reservation.get(group)!=role or (group,int(tick)) in queries:raise ValueError('Split leakage or duplicated acquired query')
            queries.add((group,int(tick)));seen.add(group)
        if entry['parent_id']!=str(d['group_id'][0]) or entry['partition']!=str(d['partition'][0]):raise ValueError('Shard reservation mismatch')
        branches+=d['status'].size;steps+=int(d['recorded_physical_steps'].sum())
    if seen!=set(reservation) or branches!=audit['branches'] or steps!=audit['physical_steps']:raise ValueError('Missing physical parents or labels')
    return m


def validate_model(metadata,manifest=None):
    graph,features,ticks,observer=registered(metadata.get('dataset_schema'))
    if metadata.get('gain_dimension')!=4 or metadata.get('graph_features')!=features:raise ValueError('Full-state four-gain Quad3D model required')
    arch=metadata['architecture'];contract=metadata.get('quad3d_contract',{})
    if arch.get('encoder') == 'nearest_fc':
        from .nearest_fc import validate_metadata
        validate_metadata(metadata)
    variants=arch.get('quad3d_history_invariant',False) or arch.get('paired_gain_quadratic',False) or arch.get('continuous_log_variance_min',-10.)!=-10. or arch.get('quad3d_obstacle_pooling',False)
    if variants and (metadata['dataset_schema']!=WIDE_SCHEMA or arch.get('encoder') not in ('gat','matched_fc')):
        raise ValueError('Quad3D feature/variance variants require the registered wide-gain GAT')
    if arch.get('encoder') not in ('gat','full_fc','matched_fc','nearest_fc') or arch.get('flight_history_invariant') or arch.get('scalar_gain_quadratic'):raise ValueError('Incompatible Quad3D encoder/feature transform')
    if contract.get('config',{}).get('nominal_bias_observer','none')!=observer:raise ValueError('Wrong Quad3D observer contract')
    if contract.get('snapshot_ticks')!=list(ticks):raise ValueError('Wrong Quad3D history contract')
    if contract.get('graph_schema')!=graph or contract.get('sensor_schema')!=SENSOR_SCHEMA:raise ValueError('Wrong Quad3D observation/graph schema')
    schema=metadata['dataset_schema']
    if contract.get('gain_bank')!=candidate_bank(schema).tolist() or contract.get('replicas')!=REPLICAS or contract.get('horizon_steps')!=HORIZON:raise ValueError('Wrong Quad3D branch contract')
    if metadata.get('gain_domain')!=candidate_domain(schema):raise ValueError('Wrong Quad3D model gain domain')
    if manifest is not None:
        for field in CONTRACT_FIELDS:
            if contract.get(field)!=manifest.get(field):raise ValueError('Changed Quad3D contract: '+field)
        for field in ('controller','targets','events','gain_domain'):
            if metadata[field]!=manifest[field]:raise ValueError('Changed Quad3D prediction semantics: '+field)
