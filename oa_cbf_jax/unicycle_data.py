"""Unicycle data functions and shared contracts."""

from dataclasses import replace

from .config import UnicycleConfig

ROBOT=UnicycleConfig(a_max=.5,w_max=.5,v_max=2.,stationary_obstacles=True)

NOMINAL=replace(ROBOT,v_max=1.5)

K=10


import json

from pathlib import Path

from .io import sha256

SCHEMA = 'local_unicycle_observed_coverage_union_v1'

def read(path):
    return json.loads(Path(path).read_text())

def members(dataset, sources):
    from .unicycle_data import expansion_validate as validate_single
    from .unicycle_coverage import SCHEMA as OBSERVED_SCHEMA
    sources = [Path(p).resolve() for p in sources]
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError('At least two distinct coverage generations required')
    proofs, index, records = [], [], None
    navigations = set()
    for generation, source in enumerate(sources):
        manifest = read(source/'manifest.json')
        if manifest['schema'] != OBSERVED_SCHEMA:
            raise ValueError('Only original observed coverage may enter a union; no nested unions')
        identity = manifest['navigation_protocol_sha256']
        if identity in navigations:
            raise ValueError('Duplicate acquisition policy generation')
        navigations.add(identity)
        proof, rows = validate_single(dataset, source)
        if records is not None and records != rows:
            raise ValueError('Coverage generations use different physical parents/roles')
        if proofs:
            for key in ('primary_manifest_sha256', 'physical_parent_count',
                        'shared_acquisition_modes', 'position_observer',
                        'horizon_seconds', 'adaptation_seconds'):
                if proof[key] != proofs[0][key]:
                    raise ValueError('Coverage generations changed '+key)
        records = rows
        proofs.append(proof)
        for entry in read(source/'index.json'):
            entry = dict(entry, generation=generation, source_dataset=str(source))
            for field in ('file', 'audit_file', 'acquisition_file'):
                entry[field] = str((source/entry[field]).resolve())
            index.append(entry)
    return proofs, records, index

def validate(dataset, additional):
    root = Path(additional).resolve()
    manifest = read(root/'manifest.json')
    if (manifest['schema'] != SCHEMA or manifest['primary_dataset'] != str(Path(dataset).resolve())
            or manifest['weight_fit_authorized'] is not True
            or manifest['validation_weight_fitting'] or manifest['forward_parents_used']):
        raise ValueError('Changed union or parent-role authorization')
    proofs, records, expected = members(dataset, manifest['sources'])
    if (proofs != manifest['member_proofs'] or read(root/'index.json') != expected
            or sha256(root/'index.json') != manifest['index_sha256']
            or manifest['generations'] != len(proofs)
            or manifest['physical_parent_count'] != len({r['group_id'] for r in records})
            or manifest['added_observations'] != sum(p['added_observations'] for p in proofs)
            or manifest['added_branches'] != sum(p['added_branches'] for p in proofs)):
        raise ValueError('Changed or partial coverage union')
    bound = {str(root/name): sha256(root/name) for name in ('manifest.json', 'index.json')}
    for proof in proofs:
        bound.update(proof['bound_files'])
    return dict(schema=SCHEMA, primary_dataset=str(Path(dataset).resolve()),
        additional_dataset=str(root), primary_manifest_sha256=proofs[0]['primary_manifest_sha256'],
        bound_files=bound, physical_parent_count=manifest['physical_parent_count'],
        added_observations=manifest['added_observations'], added_branches=manifest['added_branches'],
        shared_acquisition_modes=proofs[0]['shared_acquisition_modes'],
        coverage_generations=len(proofs), member_proofs=proofs,
        position_observer=proofs[0]['position_observer'], horizon_seconds=8., adaptation_seconds=.2,
        validation_weight_fitting=False, forward_parents_used=False,
        weighting=manifest['weighting'], retention=manifest['retention'],
        normalization='TRAIN only, base plus every retained shared observation; equal total weight per physical parent.'), records


from .unicycle_policy import FAMILIES as DENSE_FAMILIES

from .obstacle_selection import contract

DATASET_SCHEMA='unicycle_local_observation_learning_v1'

FAMILIES=(*DENSE_FAMILIES,'scatter','wide_corridor','staggered_islands','split_gate')

def dataset_validate(dataset):
    root=Path(dataset);m=json.loads((root/'manifest.json').read_text());a=json.loads((root/'audit.json').read_text())
    if m.get('controller',{}).get('position_observer') is not None:
        from .unicycle_data import observer_data_validate as validate_observer
        return validate_observer(root)
    if (m['schema']!=DATASET_SCHEMA or m['pilot'] or not m['weight_fit_authorized'] or not a['weight_fit_authorized']
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


import hashlib


import numpy as np


ONPOLICY_DATA_SCHEMA='unicycle_local_shared_onpolicy_coverage_v1'

MODES=('gat','nearest_fc','fixed_reference')

FRACTIONS=(0.,1/3,2/3,1.)

def onpolicy_data_read(path):return json.loads(Path(path).read_text())

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
    root=Path(dataset);m=onpolicy_data_read(root/'manifest.json');base=Path(m['base_dataset']);bm=dataset_validate(base)
    for file,key in (('manifest.json','base_manifest_sha256'),('index.json','base_index_sha256'),
        ('source.npz','base_source_sha256'),('records.json','base_records_sha256'),('reserved_groups.json','reservation_sha256')):
        if sha256(base/file)!=m[key]:raise ValueError('Changed base physical reservation: '+file)
    if (m['schema']!=ONPOLICY_DATA_SCHEMA or m['snapshot_fractions']!=list(FRACTIONS) or m['horizon_steps']!=160
        or m['modes']!=list(MODES) or m['forward_parents_used'] or m['calibration_parents_used']):
        raise ValueError('Changed on-policy collection contract')
    records=onpolicy_data_read(base/'records.json')
    if len(m['indices'])!=len(set(m['indices'])) or any(records[i]['partition'] not in ('train','validation') for i in m['indices']):
        raise ValueError('Invalid acquired parent population')
    for k,v in m['original_observation_contract'].items():
        if bm[k]!=v:raise ValueError('Changed physical observation contract')
    return m,base,records


from collections import Counter


from .io import load_dataset

EXPANSION_SCHEMA='local_unicycle_shared_parent_expansion_v1'

def expansion_read(path):return json.loads(Path(path).read_text())

def coverage_source(extra):
    """Dispatch only explicitly registered, sensor-matched observations."""
    from .unicycle_coverage import validate_source as validate_observed_source, SCHEMA as OBSERVED_SCHEMA
    manifest=expansion_read(Path(extra)/'manifest.json')
    if manifest['schema']==OBSERVED_SCHEMA:
        m,base,records=validate_observed_source(extra)
    elif manifest['schema']=='unicycle_local_ordered_shared_coverage_v1':
        from .unicycle_data import ordered_coverage_validate_source as validate_ordered_source
        m,base,records=validate_ordered_source(extra)
        proof=expansion_read(Path(extra)/'independent_completion_review.json')
        if (proof['status']!='passed' or proof['audit_sha256']!=sha256(Path(extra)/'audit.json')
                or proof['manifest_sha256']!=sha256(Path(extra)/'manifest.json')
                or not proof['every_outcome_target_mask_and_parent_role_reproduced']
                or not proof['original_acquisition_byte_identity'] or proof['forward_parents_used']):
            raise ValueError('Ordered coverage requires independent complete-label review')
    elif manifest['schema']==ONPOLICY_DATA_SCHEMA:
        m,base,records=validate_source(extra)
    else:raise ValueError('Unknown shared-policy coverage schema')
    expected=expansion_read(base/'manifest.json')['controller'].get('position_observer')
    if m.get('position_observer')!=expected:
        raise ValueError('Cannot mix raw and filtered acquired observations')
    return m,base,records

def expansion_validate(dataset, additional):
    primary=Path(dataset).resolve(); extra=Path(additional).resolve()
    from .unicycle_data import SCHEMA as UNION_SCHEMA, validate as validate_union
    if expansion_read(extra/'manifest.json')['schema']==UNION_SCHEMA:
        return validate_union(primary,extra)
    m,base,records=coverage_source(extra)
    if (base.resolve()!=primary or m['pilot']
        or m['weight_fit_authorized'] is not True or m['validation_weight_fitting']
        or m['indices']!=list(range(len(records)))):
        raise ValueError('Complete same-parent TRAIN/validation expansion required; pilot and holdouts forbidden')
    audit=expansion_read(extra/'audit.json'); replay=expansion_read(extra/'independent_compact_replay.json')
    for file,key in (('manifest.json','manifest_sha256'),('index.json','index_sha256'),('acquisitions.json','acquisitions_sha256')):
        if sha256(extra/file)!=audit[key]:raise ValueError('Changed shared collection evidence')
    if (not audit['all_branches_audited'] or not audit['all_outcomes_retained']
        or not audit['weight_fit_authorized'] or audit['forward_parents_used']
        or audit['observations']!=len(records)*12 or audit['branches']!=len(records)*12*64
        or replay['audit_sha256']!=sha256(extra/'audit.json')
        or replay['manifest_sha256']!=sha256(extra/'manifest.json')
        or not replay['exact_reconstruction'] or len(replay['replayed'])!=6
        or not all(r['exact_results_and_full_traces'] and r['independent_physical_reaudit'] for r in replay['replayed'])):
        raise ValueError('Complete audited collection and exact replay required')
    if m.get('position_observer') is not None:
        if not audit.get('all_observer_memories_audited') or not all(r.get('exact_acquisition_and_memory') for r in replay['replayed']):
            raise ValueError('Observed expansion requires copied-memory audits and acquisition replay')
    files=[extra/name for name in ('manifest.json','index.json','audit.json','independent_compact_replay.json','acquisitions.json')]
    proof=dict(schema=EXPANSION_SCHEMA,primary_dataset=str(primary),additional_dataset=str(extra),
        primary_manifest_sha256=sha256(primary/'manifest.json'),
        bound_files={str(p):sha256(p) for p in files},
        physical_parent_count=len({r['group_id'] for r in records}),
        added_observations=audit['observations'],added_branches=audit['branches'],
        shared_acquisition_modes=m['modes'],horizon_seconds=8.,adaptation_seconds=.2,
        validation_weight_fitting=False,forward_parents_used=False,
        weighting='Same physical-parent bootstrap shared across original and added LIVE observations.',
        normalization='TRAIN only, original plus added observations; equal total weight per physical parent.')
    if m.get('position_observer') is not None:
        proof['position_observer']=m['position_observer']
    if 'candidate_bank_contract' in m:
        from .unicycle_policy import validate_bank
        if validate_bank(primary)!=m['candidate_bank_contract']:
            raise ValueError('Cannot mix different candidate banks in shared coverage')
        proof['candidate_bank_contract']=m['candidate_bank_contract']
        proof['source_coverage']=m['source_coverage']
        path=extra/'independent_completion_review.json'
        proof['bound_files'][str(path)]=sha256(path)
    return proof,records

def concatenate(original, extra, partition, expected_groups):
    """Strict roles/shapes; genuine live failures are never filtered."""
    if partition not in ('train','validation'):raise ValueError('Weight fitting and selection roles only')
    groups=set(expected_groups)
    for data in (original,extra):
        if set(data['group_id'])!=groups or not np.all(data['partition']==partition):
            raise ValueError('Physical-parent role mismatch or held-out leakage')
    if set(original)!=set(extra):raise ValueError('Different raw observation fields')
    if np.any(extra['prior_status']!=0):raise ValueError('An acquired query was already terminal')
    for key in original:
        a,b=original[key],extra[key]
        if a.shape[1:]!=b.shape[1:] or (a.dtype!=b.dtype and not (a.dtype.kind==b.dtype.kind=='U')):
            raise ValueError('Changed observation shape/dtype: '+key)
    return {k:np.concatenate((original[k],extra[k]),axis=0) for k in original}

def extend(dataset, additional, train, validation):
    proof,records=expansion_validate(dataset,additional); result=[]
    for part,original in (('train',train),('validation',validation)):
        extra=load_dataset(additional,part)
        expected=[r['group_id'] for r in records if r['partition']==part]
        counts=Counter(extra['group_id'])
        generations=proof.get('coverage_generations',1)
        if counts!={g:n*12*generations for g,n in Counter(expected).items()}:
            raise ValueError('Incomplete actor/snapshot coverage for a reserved parent')
        s=extra['status']; valid=np.stack((np.isin(s,(1,2,4)),np.isin(s,(1,4))),axis=-1)
        np.testing.assert_array_equal(extra['target_mask'],valid)
        target=np.where(valid,np.stack((-np.minimum(extra['min_clearance'],.6)/.3,extra['progress']/12),axis=-1),0.).astype(np.float32)
        np.testing.assert_array_equal(extra['target'],target)
        np.testing.assert_array_equal(extra['events'],np.stack((s==2,np.isin(s,(3,5,8))),axis=-1))
        if not extra['event_mask'].all():raise ValueError('An observed failure was censored')
        if not np.isfinite(extra['features']).all():raise ValueError('Nonfinite acquired input')
        result.append(concatenate(original,extra,part,expected))
    if set(result[0]['group_id']) & set(result[1]['group_id']):raise ValueError('TRAIN/validation parent leakage')
    return *result,proof


import jax

import jax.numpy as jnp

from .obstacle_selection import nearest_obstacles

from .models import unicycle_graph

COLLECTION='local_causal_observer_compact_labels_v1'

def observer_data_read(path):return json.loads(Path(path).read_text())

def graph_inputs(x,world,goal,error,memory):
    from .unicycle_policy import observe
    raw=x.at[:2].add(error[0]);seen=world.at[:,:2].add(error[1:])
    observed,seen=observe(memory,raw,seen)
    selected,valid,ids=nearest_obstacles(observed[:2],seen,jnp.ones(len(world),bool),K)
    features,mask=unicycle_graph(observed,goal,selected,valid,ROBOT)
    return features,mask,observed,ids

def label_kernel():
    from .unicycle_policy import reference_rollout
    return jax.jit(jax.vmap(lambda x,o,g,gains,errors,memory:jax.vmap(
        lambda gain,e:reference_rollout(x,o,g,e,gain,memory))(gains,errors)))

def qualified_source(root):
    from .unicycle_policy import contract as observer_data_contract
    root=Path(root);m=observer_data_read(root/'manifest.json')
    if m.get('collection_contract')!=COLLECTION or m['controller'].get('position_observer')!=observer_data_contract():raise ValueError('Unregistered observer labels')
    if m['numpy_version']!=np.__version__ or m['collector_sha256']!=sha256(__file__):raise ValueError('Use the frozen collector/RNG for replay')
    for filename,key in (('source.npz','source_sha256'),('records.json','records_sha256'),('reserved_groups.json','reservation_sha256')):
        if sha256(root/filename)!=m[key]:raise ValueError('Changed observer dataset reservation')
    records=observer_data_read(root/'records.json')
    roles=('predictive_fit','predictive_audit') if m['schema']=='unicycle_local_reserved_prediction_v1' else ('train','validation')
    if len(m['indices'])!=len(set(m['indices'])) or any(records[i]['partition'] not in roles for i in m['indices']):
        raise ValueError('Forbidden or duplicate collection parent')
    return m,records

def snapshot(trace,ticks):
    from .unicycle_policy import Memory
    ii=np.arange(4)
    return trace['before'][ii,ticks],Memory(trace['observer_prediction'][ii,ticks],trace['observer_centers'][ii,ticks],trace['observer_ready'][ii,ticks])

def branch_inputs(x,world,goal,bank,future,memory):
    expanded=np.broadcast_to(future[:,None],(4,32,2,160,33,2)).reshape(4,64,160,33,2)
    gains=np.broadcast_to(np.repeat(bank,2,axis=0),(4,64,2)).copy()
    args=(*map(jnp.asarray,(x,world,goal,gains,expanded)),jax.tree.map(jnp.asarray,memory))
    return args,gains,expanded

def observer_data_validate(dataset):
    root=Path(dataset);m,records=qualified_source(root);a=observer_data_read(root/'audit.json');r=observer_data_read(root/'compact_replay.json')
    if m['pilot'] or not m['weight_fit_authorized'] or m['indices']!=list(range(len(records))):raise ValueError('Pilot/incomplete observer data cannot train')
    if a['manifest_sha256']!=sha256(root/'manifest.json') or a['index_sha256']!=sha256(root/'index.json') or not a['all_observer_memories_independently_checked']:
        raise ValueError('Changed observer audit')
    if r['audit_sha256']!=sha256(root/'audit.json') or not r['exact_reconstruction']:raise ValueError('Unqualified compact replay')
    partitions={}
    for record in records:partitions.setdefault(record['group_id'],set()).add(record['partition'])
    if any(len(p)!=1 for p in partitions.values()) or {next(iter(p)) for p in partitions.values()}!={'train','validation'}:
        raise ValueError('Observer parent partition leakage')
    return m


from scipy.stats import qmc

def candidate_bank(original, seed):
    """Canonicalize the original bank, then fill duplicates without outcomes."""
    from .unicycle_policy import GAINS
    stream = np.exp(np.log(.5) + qmc.Sobol(2, scramble=True, seed=seed)
                    .random_base2(6) * np.log(16)).astype(np.float32)
    expected = np.concatenate((GAINS, stream[:16]))
    if original.dtype != np.float32 or not np.array_equal(original, expected):
        raise ValueError('Expected the original registered 32-gain Sobol bank')
    bank, seen = [], set()
    for pair in np.concatenate((original, stream[16:])):
        ordered = tuple(sorted(map(float, pair), reverse=True))
        if ordered not in seen:
            seen.add(ordered)
            bank.append(ordered)
        if len(bank) == 32:
            break
    if len(bank) != 32:
        raise ValueError('Insufficient unique ordered candidates')
    return np.asarray(bank, np.float32)

def bank_contract(original, bank, seed):
    return dict(schema='unicycle_ordered_unique_candidates_v1', seed=seed,
        rule='Stable unique descending original pairs, then next unique descending pairs from the same scrambled log-Sobol stream starting at index16.',
        original_candidates=32, canonical_original_candidates=27, candidates=32,
        original_bank=original.tolist(), bank=bank.tolist(),
        outcome_dependent_candidate_selection=False, gain_bounds=[.5, 8.],
        held_gain_seconds=8, adaptation_interval_seconds=.2,
        original_collector_unchanged=True, physical_controller_unchanged=True,
        requires_new_labels_training_and_reserved_calibration=True)

def selected_indices(records, pilot):
    if not pilot:
        return list(range(len(records)))
    parents = set()
    for family in sorted({r['family'] for r in records}):
        eligible = sorted({r['group_id'] for r in records
                           if r['family'] == family and r['partition'] == 'train'})
        if len(eligible) < 2:
            raise ValueError('Two TRAIN parents per family required')
        parents.update(eligible[:2])
    return [i for i, r in enumerate(records) if r['group_id'] in parents]


from . import unicycle_coverage as original

ORDERED_COVERAGE_SCHEMA = 'unicycle_local_ordered_shared_coverage_v1'

def ordered_coverage_read(path):
    return json.loads(Path(path).read_text())

def ordered_coverage_validate_source(dataset):
    from .unicycle_policy import contract as observer_data_contract
    root = Path(dataset)
    m = ordered_coverage_read(root/'manifest.json')
    if m['schema'] != ORDERED_COVERAGE_SCHEMA or m['collector_sha256'] != sha256(__file__):
        raise ValueError('Unregistered ordered coverage or changed collector')
    base, source = Path(m['base_dataset']), Path(m['source_coverage'])
    bm = observer_data_validate(base)
    old, acquisition_base, records = original.validate_source(source)
    for directory, bindings in [(base, m['bound_base_files']), (source, m['bound_source_files'])]:
        for name, digest in bindings.items():
            if sha256(directory/name) != digest:
                raise ValueError('Changed relabel source: '+str(directory/name))
    if (m['acquisition_base'] != str(acquisition_base.resolve())
        or m['acquisition_base_manifest_sha256'] != sha256(acquisition_base/'manifest.json')
        or bm['base_dataset'] != str(acquisition_base.resolve())
        or m['candidate_bank_contract'] != bm['candidate_bank_contract']
        or m['original_collector_sha256'] != old['collector_sha256']
        or m['models'] != old['models'] or m['position_observer'] != observer_data_contract()
        or m['indices'] != selected_indices(records, m['pilot'])
        or m['forward_parents_used'] or m['calibration_parents_used']
        or not m['acquisition_policy_and_bank_unchanged'] or not m['label_bank_only_changed']):
        raise ValueError('Changed acquisition/label/split contract')
    for key in ['modes', 'snapshot_fractions', 'horizon_steps', 'held_gain_seconds',
                'adaptation_interval_seconds', 'replicas', 'queries', 'numpy_version']:
        if m[key] != old[key]:
            raise ValueError('Changed shared visitation contract: '+key)
    return m, base, records
