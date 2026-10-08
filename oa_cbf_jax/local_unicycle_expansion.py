"""Append shared-policy observations without changing physical-parent roles."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .dataset import load_dataset, sha256
from .local_unicycle_onpolicy_data import validate_source as validate_raw_source, SCHEMA as DATA_SCHEMA

SCHEMA='local_unicycle_shared_parent_expansion_v1'


def read(path):return json.loads(Path(path).read_text())


def coverage_source(extra):
    """Dispatch only explicitly registered, sensor-matched observations."""
    from .local_unicycle_observed_coverage import validate_source as validate_observed_source, SCHEMA as OBSERVED_SCHEMA
    manifest=read(Path(extra)/'manifest.json')
    if manifest['schema']==OBSERVED_SCHEMA:
        m,base,records=validate_observed_source(extra)
    elif manifest['schema']=='unicycle_local_ordered_shared_coverage_v1':
        from .local_unicycle_ordered_coverage import validate_source as validate_ordered_source
        m,base,records=validate_ordered_source(extra)
        proof=read(Path(extra)/'independent_completion_review.json')
        if (proof['status']!='passed' or proof['audit_sha256']!=sha256(Path(extra)/'audit.json')
                or proof['manifest_sha256']!=sha256(Path(extra)/'manifest.json')
                or not proof['every_outcome_target_mask_and_parent_role_reproduced']
                or not proof['original_acquisition_byte_identity'] or proof['forward_parents_used']):
            raise ValueError('Ordered coverage requires independent complete-label review')
    elif manifest['schema']==DATA_SCHEMA:
        m,base,records=validate_raw_source(extra)
    else:raise ValueError('Unknown shared-policy coverage schema')
    expected=read(base/'manifest.json')['controller'].get('position_observer')
    if m.get('position_observer')!=expected:
        raise ValueError('Cannot mix raw and filtered acquired observations')
    return m,base,records


def validate(dataset, additional):
    primary=Path(dataset).resolve(); extra=Path(additional).resolve()
    from .local_unicycle_coverage_union import SCHEMA as UNION_SCHEMA, validate as validate_union
    if read(extra/'manifest.json')['schema']==UNION_SCHEMA:
        return validate_union(primary,extra)
    m,base,records=coverage_source(extra)
    if (base.resolve()!=primary or m['pilot']
        or m['weight_fit_authorized'] is not True or m['validation_weight_fitting']
        or m['indices']!=list(range(len(records)))):
        raise ValueError('Complete same-parent TRAIN/validation expansion required; pilot and holdouts forbidden')
    audit=read(extra/'audit.json'); replay=read(extra/'independent_compact_replay.json')
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
    proof=dict(schema=SCHEMA,primary_dataset=str(primary),additional_dataset=str(extra),
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
        from .local_unicycle_candidates import validate_bank
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
    proof,records=validate(dataset,additional); result=[]
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
