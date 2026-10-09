"""Quad2d training functions and shared contracts."""

from pathlib import Path

import hashlib

import jax

import jax.numpy as jnp

import numpy as np

from .io import load_dataset, sha256

from .quad2d_static_inputs import read

KEYS=('features','node_mask','gains','target','target_mask','events','event_mask')

PREFIX='reflected_'

def check_training_rows(original,reflected,expected):
    if (original['group_id'].tolist()!=expected or reflected['group_id'].tolist()!=expected
            or len(set(expected))!=len(expected)):
        raise ValueError('Reflection must preserve unique TRAIN parent order')
    for data in (original,reflected):
        if not np.all(data['partition']=='train'):
            raise ValueError('Only TRAIN orientations may enter fitting')
    np.testing.assert_array_equal(original['gains'],reflected['gains'])
    for key in KEYS:
        if original[key].shape!=reflected[key].shape or original[key].dtype!=reflected[key].dtype:
            raise ValueError('Changed paired feature/target layout: '+key)

def load_pair(original_dataset,reflected_dataset,original):
    a,b=map(Path,(original_dataset,reflected_dataset));ma,mb=read(a/'manifest.json'),read(b/'manifest.json')
    if any(m.get('weight_fit_authorized') is not True or m.get('final_test') is not False
            or m['schema']!='oa_cbf_quad2d_guided_hurdle_v1' for m in (ma,mb)):
        raise ValueError('Audited TRAIN-authorized flight data required')
    for key in ('groups','config','controller','queries','replicas','horizon_steps','targets','events',
            'graph_features','graph_schema','capacity','route_capacity','gain_domain'):
        if ma[key]!=mb[key]:raise ValueError('Different paired learning contract: '+key)
    for path in (a,b):
        audit=read(path/'independent_replay.json');done=read(path/'complete.json')
        for flag in ('audit_passed','all_collision_bound_branches_audited',
                'all_observed_graph_features_independently_checked','all_guidance_approvals_checked',
                'all_physical_task_target_inputs_checked'):
            if audit.get(flag) is not True:raise ValueError('Incomplete paired physical audit')
        for key,file in [('manifest_sha256','manifest.json'),('index_sha256','index.json')]:
            if done[key]!=sha256(path/file) or audit[key]!=done[key]:
                raise ValueError('Changed paired labels')
    report=read(b.parent/'review.json')
    if (not report['actual_new_labels_not_inherited'] or not report['original_parent_partitions_preserved']
            or report['dataset']!=str(b.resolve()) or report['manifest_sha256']!=sha256(b/'manifest.json')
            or report['index_sha256']!=sha256(b/'index.json')
            or report['audit_sha256']!=sha256(b/'independent_replay.json')):
        raise ValueError('Unbound actual reflected outcome review')
    spec=read(b.parent/'protocol.json')
    if spec['original_dataset']!=str(a.resolve()):raise ValueError('Different original donor dataset')
    for path,digest in spec['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed reflection donor binding')
    reflected=load_dataset(b,'train')
    expected=[g['group_id'] for g in ma['groups'] if g['partition']=='train']
    check_training_rows(original,reflected,expected)
    paths=[p/f for p in (a,b) for f in ('manifest.json','index.json','independent_replay.json','complete.json')]
    paths += [b.parent/'review.json',b.parent/'protocol.json',b.parent/'qualification.json']
    proof=dict(schema='quad2d_actual_paired_orientation_v1',original_dataset=str(a.resolve()),
        reflected_dataset=str(b.resolve()),bindings={str(p.resolve()):sha256(p) for p in paths},
        independent_training_parents=len(expected),orientations_per_parent=2,
        training_parent_ids_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
        sampling='Original parent bootstrap, one Bernoulli(0.5) orientation per sampled parent and optimizer update; all gain/replica branches share that orientation and its ACTUAL targets/masks/events.',
        normalization='Unchanged original TRAIN-only target units, shared between encoders.',
        weighting='Unchanged original TRAIN gain-opportunity weights, shared between encoders.',
        validation='Original validation only for checkpoint selection; reflected validation is secondary, never independent added parents.',
        calibration='Original reserved parents excluded from fitting; no reflected calibration enters fitting.',
        seed_rule='fold_in(PRNGKey(member_seed+37001), optimizer_step)',copied_targets=False)
    return reflected,proof

def pack(original,reflected):
    if set(original)!=set(KEYS) or set(reflected)!=set(KEYS):
        raise ValueError('Paired augmentation only supports the original flight heads')
    return dict(original,**{PREFIX+k:reflected[k] for k in KEYS})

def orientation_mask(count,seed,step):
    return jax.random.bernoulli(jax.random.fold_in(jax.random.PRNGKey(seed),step),.5,(count,))

def select_orientation(batch,seed,step):
    mirrored=orientation_mask(batch['features'].shape[0],seed,step)
    return {key:jnp.where(mirrored.reshape((-1,)+(1,)*(batch[key].ndim-1)),
                         batch[PREFIX+key],batch[key]) for key in KEYS}


from collections import Counter


def disjoint_reservations(primary, additional):
    for manifest in (primary, additional):
        if (manifest.get('weight_fit_authorized') is not True or manifest.get('final_test') is not False
                or manifest.get('data_role') != 'training'):
            raise ValueError('Only reserved training datasets may supply extra TRAIN rows')
        groups = manifest['groups']
        if len({r['group_id'] for r in groups}) != len(groups) or len({r['seed'] for r in groups}) != len(groups):
            raise ValueError('Duplicate reserved parent/seed')
        if any(r['partition'] not in ('train','validation','development_calibration') for r in groups):
            raise ValueError('Unknown parent role')
    for field in ('group_id','seed'):
        if {r[field] for r in primary['groups']} & {r[field] for r in additional['groups']}:
            raise ValueError('Additional data overlaps an existing training or held-out parent: '+field)
    for key in ('schema','config','controller','queries','replicas','horizon_steps','targets','events',
                'gain_domain','graph_schema','graph_features','capacity','route_capacity'):
        if primary[key] != additional[key]:
            raise ValueError('Changed extra-data contract: '+key)

def concatenate_train(primary, additional, expected_primary, expected_additional):
    for data, expected in ((primary,expected_primary),(additional,expected_additional)):
        if data['group_id'].tolist() != expected or not np.all(data['partition']=='train'):
            raise ValueError('Exact reserved TRAIN order required; holdouts cannot be appended')
    if len(set(expected_primary+expected_additional)) != len(expected_primary+expected_additional):
        raise ValueError('Repeated TRAIN identity')
    if set(primary) != set(additional):
        raise ValueError('Different raw training array fields')
    for key in primary:
        a,b = primary[key],additional[key]
        if a.shape[1:] != b.shape[1:] or (a.dtype != b.dtype and not (a.dtype.kind == b.dtype.kind == 'U')):
            raise ValueError('Changed TRAIN array shape or dtype: '+key)
    return {k:np.concatenate((primary[k],additional[k]),axis=0) for k in primary}

def extend(dataset, extra_dataset, extra_reflection, original, original_reflected):
    from .quad2d_training import validate_training_dataset
    a,b,c = map(Path,(dataset,extra_dataset,extra_reflection))
    ma,mb = read(a/'manifest.json'),read(b/'manifest.json')
    disjoint_reservations(ma,mb)
    validate_training_dataset(b,b.parent/'data_review.json',dict(ma,dataset=str(a.resolve())))
    report = read(b.parent/'review.json')
    if (report.get('schema') != 'quad2d_shared_policy_training_collection_v1'
            or not report.get('all_context_selections_recomputed')
            or not report.get('all_parent_failures_retained')
            or report['data_review_sha256'] != sha256(b.parent/'data_review.json')):
        raise ValueError('Reviewed, outcome-independent shared policy collection required')
    new = load_dataset(b,'train')
    reflected,pair = load_pair(b,c,new)
    expected_a = [r['group_id'] for r in ma['groups'] if r['partition']=='train']
    expected_b = [r['group_id'] for r in mb['groups'] if r['partition']=='train']
    raw = concatenate_train(original,new,expected_a,expected_b)
    mirrored = concatenate_train(original_reflected,reflected,expected_a,expected_b)
    check_training_rows(raw,mirrored,expected_a+expected_b)
    paths = [p/f for p in (a,b,c) for f in ('manifest.json','index.json','independent_replay.json','complete.json')]
    paths += [b.parent/'data_review.json',b.parent/'review.json',b.parent/'protocol.json',
              c.parent/'review.json',c.parent/'protocol.json',c.parent/'qualification.json']
    expected = expected_a+expected_b
    proof = dict(schema='quad2d_shared_train_expansion_v1',primary_dataset=str(a.resolve()),
        additional_dataset=str(b.resolve()),additional_manifest_sha256=sha256(b/'manifest.json'),
        additional_reflection=str(c.resolve()),additional_pair=pair,
        bindings={str(p.resolve()):sha256(p) for p in paths},
        primary_train_parents=len(expected_a),additional_train_parents=len(expected_b),
        independent_train_parents=len(expected),orientations_per_parent=2,
        training_parent_ids_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
        additional_partitions=dict(Counter(r['partition'] for r in mb['groups'])),
        normalization='Original primary TRAIN only; unchanged units for both encoders.',
        selection='Original primary validation only; additional validation reported separately.',
        calibration='Neither primary nor additional calibration parents enter weight fitting.',
        weighting='Same gain-opportunity rule recomputed on the shared TRAIN union, followed by same-seed parent bootstrap.',
        sampling='Shared TRAIN union; independent Bernoulli0.5 original/actually simulated reflection per parent per update.',
        all_roles_disjoint=True,models_share_labels=True)
    return raw,mirrored,proof


def append_contexts(dataset, reflection, raw, mirrored, previous):
    from .quad2d_training import validate_training_dataset
    dataset, reflection = map(Path, (dataset, reflection))
    primary, additional = [Path(previous[k]) for k in ('primary_dataset','additional_dataset')]
    manifests = [read(p/'manifest.json') for p in (primary, additional, dataset)]
    for i in range(3):
        for j in range(i+1, 3):
            disjoint_reservations(manifests[i], manifests[j])
    if 'action_context' in previous:
        raise ValueError('Action-context expansion cannot be appended twice')
    for path, digest in previous['bindings'].items():
        if sha256(path) != digest:
            raise ValueError('Changed existing TRAIN union')
    validate_training_dataset(dataset, dataset.parent/'data_review.json',
        dict(manifests[0], dataset=str(primary.resolve())))
    report = read(dataset.parent/'review.json')
    proof = read(dataset.parent/'independent_completion_review.json')
    protocol = read(dataset.parent/'protocol.json')
    if (protocol['schema'] != 'quad2d_observed_action_training_collection_v1'
            or not report['every_failed_parent_retained']
            or not report['all_selection_rules_independently_reconstructed']
            or report['data_review_sha256'] != sha256(dataset.parent/'data_review.json')
            or proof['status'] != 'passed' or proof['review_sha256'] != sha256(dataset.parent/'review.json')):
        raise ValueError('Reviewed, outcome-independent action observations required')
    reflected_review = read(reflection.parent/'independent_completion_review.json')
    if (reflected_review['status'] != 'passed'
            or reflected_review['review_sha256'] != sha256(reflection.parent/'review.json')):
        raise ValueError('Independent reflected-label completion audit required')
    old_ids = [r['group_id'] for m in manifests[:2] for r in m['groups'] if r['partition']=='train']
    if (len(old_ids) != previous['independent_train_parents']
            or hashlib.sha256('\n'.join(old_ids).encode()).hexdigest() != previous['training_parent_ids_sha256']):
        raise ValueError('Changed retained TRAIN identities')
    new = load_dataset(dataset, 'train'); reflected, pair = load_pair(dataset, reflection, new)
    new_ids = [r['group_id'] for r in manifests[2]['groups'] if r['partition']=='train']
    joined = concatenate_train(raw, new, old_ids, new_ids)
    joined_mirror = concatenate_train(mirrored, reflected, old_ids, new_ids)
    ids = old_ids+new_ids
    check_training_rows(joined, joined_mirror, ids)
    paths = [p/f for p in (dataset, reflection) for f in ('manifest.json','index.json','complete.json','independent_replay.json')]
    paths += [dataset.parent/f for f in ('review.json','protocol.json','data_review.json','independent_completion_review.json')]
    paths += [reflection.parent/f for f in ('review.json','protocol.json','qualification.json','independent_completion_review.json')]
    addition = dict(dataset=str(dataset.resolve()),manifest_sha256=sha256(dataset/'manifest.json'),
        reflection=str(reflection.resolve()),pair=pair,train_parents=len(new_ids),
        partitions=dict(Counter(r['partition'] for r in manifests[2]['groups'])))
    expanded = dict(previous, action_context=addition, previous_expansion=previous,
        independent_train_parents=len(ids),
        training_parent_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        bindings={**previous['bindings'], **{str(p.resolve()):sha256(p) for p in paths}},
        weighting='Same gain-opportunity weights recomputed on the complete shared TRAIN union; full-parent ensemble sampling.',
        calibration='All three original calibration reservations excluded from weight fitting.')
    return joined, joined_mirror, expanded


from dataclasses import asdict


from .comparison_contracts import physical_obstacle_scope


from .quad2d_control import FlightConfig

from .quad2d_static_inputs import seeded_fields as multiscale_fields

from .quad2d_static_inputs import FAMILIES as LAYOUT_FAMILIES, static_topologies_seeded_fields as layout_fields

from .scenes import DIVERSE_FAMILIES

SCHEMA = 'quad2d_static_coverage_reservation'

FAMILIES = tuple(DIVERSE_FAMILIES) + tuple(LAYOUT_FAMILIES)

FRESH_ROLE = 'fresh_predictive_calibration'

def role_contract(role, seed, groups):
    result=dict(data_role=role,training_use=role in ('pilot','training'),
        weight_fit_authorized=role=='training',final_test=False)
    if role==FRESH_ROLE:
        result['calibration_reservation']=dict(schema='quad2d_coverage_fresh_calibration_v1',
            seed=seed,parents=groups,selection='All prespecified parents retained, no outcome filtering.',
            partition_interpretation='Legacy storage tags only; every parent is reserved from weight fitting.',
            intended_use='Fresh prediction fit/gate/audit on disjoint parents; no trajectory coverage or final promotion.')
    return result

def fields(seed, index, groups):
    if groups < 256 or groups % 32 or not 0 <= index < groups:
        raise ValueError('Balanced sixteen-family source with at least256 parents required')
    generator = multiscale_fields if index < groups // 2 else layout_fields
    result = generator(seed, index)
    result.pop('partition', None)
    result['coverage_stratum'] = 'multiscale' if index < groups // 2 else 'layout'
    return result

def partitions(groups, seed, role):
    if role not in ('pilot', 'training', 'comparison', FRESH_ROLE) or groups < 256 or groups % 32:
        raise ValueError('Known coverage role with balanced families required')
    if role == 'comparison':
        # At least32 reference parents per family keeps the existing95% finite
        # sample rank attainable. The remaining two thirds are forward trials.
        if groups < 1536 or groups % 96:
            raise ValueError('Comparison needs at least32 reference parents per family')
        return ['trajectory_calibration' if (i//8)%3 == 0 else 'controller_validation'
                for i in range(groups)]
    rng = np.random.default_rng(seed)
    result = [None] * groups
    for stratum in range(2):
        for family in range(8):
            indices = np.arange(stratum * groups//2 + family, (stratum+1)*groups//2, 8)
            for rank, index in enumerate(rng.permutation(indices)):
                result[index] = ('train' if rank < int(.7*len(indices)) else
                    'validation' if rank < int(.85*len(indices)) else 'development_calibration')
    if role == 'training' and Counter(result)['development_calibration'] < 200:
        raise ValueError('Training requires at least200 independent calibration parents')
    return result

def verify(directory):
    root = Path(directory)
    reservation, manifest, rows = [read(root/n) for n in ('reservation.json','manifest.json','scenes.json')]
    if (reservation['schema'] != SCHEMA or manifest['schema'] != SCHEMA
            or reservation['manifest_sha256'] != sha256(root/'manifest.json')
            or reservation['scenes_sha256'] != sha256(root/'scenes.json')
            or manifest['scenes_sha256'] != reservation['scenes_sha256']):
        raise ValueError('Changed coverage reservation')
    role, groups, seed = reservation['role'], manifest['groups'], manifest['seed']
    expected_parts = partitions(groups, seed, role)
    if (manifest['config'] != asdict(FlightConfig(stationary_obstacles=True))
            or manifest['families'] != list(FAMILIES) or len(rows) != groups
            or any(manifest.get(k)!=v for k,v in role_contract(role,seed,groups).items())):
        raise ValueError('Changed physical scope or reserved data role')
    for i, row in enumerate(rows):
        if any(row[k] != v for k,v in fields(seed,i,groups).items()) or row['partition'] != expected_parts[i]:
            raise ValueError('Changed prespecified parent, family or partition')
        physical_obstacle_scope('quad2d',row['obstacles'],row['obstacle_mask'])
    if len({r['group_id'] for r in rows}) != groups or len({r['seed'] for r in rows}) != groups:
        raise ValueError('Duplicate coverage parent')
    counts = dict(Counter(expected_parts))
    if reservation['partitions'] != counts:raise ValueError('Changed split counts')
    return dict(source=str(root.resolve()),role=role,parents=groups,partitions=counts,
        families=dict(Counter(r['family'] for r in rows)),
        manifest_sha256=sha256(root/'manifest.json'),scenes_sha256=sha256(root/'scenes.json'),
        seeded_geometry_static_truth_and_splits_verified=True,
        routes=dict(Counter(r['route']['status'] for r in rows)),benchmark_complete=False)


import json


from .uncertainty import conformal_threshold

GAIN_IMPROVEMENT_SCHEMA = 'quad2d_paired_progress_admission_v1'

SCALE_FLOOR = 1e-3

def difference_statistics(means, variances, anchor, xp=np):
    """Ensemble axis first; a shared anchor preserves memberwise differences."""
    delta = means-means[:, anchor:anchor+1]
    center = xp.mean(delta, axis=0)
    scale = xp.sqrt(xp.mean(variances+variances[:, anchor:anchor+1], axis=0)
                    +xp.var(delta, axis=0)+SCALE_FLOOR**2)
    return center, scale

def admission(means, variances, quantile):
    """Last pool entry is the exact previous gain; all others need evidence."""
    center, scale = difference_statistics(means, variances, means.shape[1]-1, jnp)
    lower = center-quantile*scale
    accepted = jnp.isfinite(lower)&(lower>0)
    return accepted.at[-1].set(True), lower

def validate(info):
    """Recompute the finite-sample rank from bound development-only evidence."""
    c = info.get('gain_improvement', {})
    if (c.get('schema') != GAIN_IMPROVEMENT_SCHEMA or c.get('scale_floor') != SCALE_FLOOR
            or c.get('weights_sha256') != info['weights_sha256']):
        raise ValueError('Matched progress improvement calibration required')
    for file,digest in c['frozen_files'].items():
        if sha256(file) != digest: raise ValueError('Changed progress calibration evidence')
    bases = [file for file in c['frozen_files'] if Path(file).name=='calibration.json']
    if len(bases)!=1:raise ValueError('One frozen original predictive calibration required')
    base = json.loads(Path(bases[0]).read_text())
    for key in ('weights_sha256','dataset_manifest_sha256','targets','events','robot','controller',
                'variance_scale','event_calibration','gain_domain','horizon_steps'):
        if info[key]!=base[key]:raise ValueError('Changed original progress prediction transform')
    if (c['gate_group_ids']!=base['group_ids']['gate']
            or c['audit_group_ids']!=base['group_ids']['audit']):
        raise ValueError('Changed reserved improvement calibration/audit parents')
    if sha256(c['scores_file']) != c['scores_sha256']: raise ValueError('Changed paired calibration scores')
    with np.load(c['scores_file'], allow_pickle=False) as z:
        ids = z['group_id'].tolist(); positions = {g:i for i,g in enumerate(ids)}
        expected = [g for rows in base['group_ids'].values() for g in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
            raise ValueError('Changed complete development calibration inventory')
        scores = np.max((z['prediction_difference']-z['actual_difference'])/z['scale'], axis=-1)
        np.testing.assert_allclose(scores, z['group_score'], rtol=0, atol=1e-12)
    q = conformal_threshold(scores[[positions[g] for g in c['gate_group_ids']]], c['coverage'])
    if q != c['gate'] or c['threshold'] != max(0., q['threshold']):
        raise ValueError('Changed paired progress threshold')
    return c


from copy import deepcopy


from .quad2d_control import flight_config_from_contract

from .quad2d_static_inputs import verify as static_learning_verify

def prediction_contract(manifest, reference):
    old = read(Path(reference['dataset']) / 'manifest.json')
    expected = deepcopy(reference['controller'])
    expected['config'] = asdict(flight_config_from_contract(expected['config']))
    expected['config']['stationary_obstacles'] = True
    if manifest['schema'] != 'oa_cbf_quad2d_guided_hurdle_v1' or manifest['controller'] != expected:
        raise ValueError('Unexpected terminal-guidance/physical change beyond stationary truth')
    if manifest['config'] != expected['config']:
        raise ValueError('Dataset and controller physics differ')
    for key in ('targets', 'events', 'gain_domain', 'graph_features'):
        if manifest[key] != reference[key]:
            raise ValueError('Changed matched prediction contract: ' + key)
    for key in ('horizon_steps', 'queries', 'replicas', 'capacity', 'route_capacity'):
        if manifest[key] != old[key]:
            raise ValueError('Changed reference sampling/shape: ' + key)
    return True

def validate_training_dataset(dataset, review_path, reference):
    root = Path(dataset)
    checked = read(review_path)
    if checked.get('pilot') is not False or checked.get('weight_fit_authorized') is not True:
        raise ValueError('Pilot/comparison data cannot authorize matched weight fitting')
    for field, file in [('manifest_sha256', 'manifest.json'), ('index_sha256', 'index.json'),
                        ('audit_sha256', 'independent_replay.json')]:
        if checked[field] != sha256(root / file):
            raise ValueError('Changed reviewed learning dataset')
    if (not checked.get('static_source_and_all_saved_truth_verified')
            or not checked.get('matched_prediction_contract_verified')):
        raise ValueError('Incomplete static source/prediction review')
    source, raw = Path(checked['source']), Path(checked['raw_source'])
    if (checked['source_manifest_sha256'] != sha256(source / 'manifest.json')
            or checked['source_audit_sha256'] != sha256(source / 'independent_replay.json')
            or checked['reservation_sha256'] != sha256(raw / 'reservation.json')):
        raise ValueError('Changed reviewed acquisition/reservation')
    reservation = static_learning_verify(raw)
    manifest = read(root / 'manifest.json')
    if (reservation['role'] != 'training' or manifest.get('weight_fit_authorized') is not True
            or reservation['partitions'] != checked['partitions']
            or checked['partitions'].get('development_calibration', 0) < 200):
        raise ValueError('Unreserved or insufficient independent training/calibration data')
    prediction_contract(manifest, reference)
    for entry in read(root / 'index.json'):
        if sha256(root / entry['file']) != entry['sha256']:
            raise ValueError('Changed training arrays')
    return manifest


def gain_opportunity_weights(data, replicas):
    """Equal expected mass for opportunity/other parents before bootstrapping.

    A candidate is empirically nonadverse only if all saved replicas reach the
    goal or the horizon. An opportunity improves observed progress by >0.01
    over gain[4,4], or avoids that reference's adverse outcome. This is an
    offline training stratum, not a safety certificate or runtime oracle.
    Every parent and branch remains present. Validation/calibration stay
    unweighted; their outcomes never enter these weights.
    """
    n=len(data['group_id']);gains=np.asarray(data['gains'])
    if (n==0 or len(set(data['group_id'].tolist()))!=n or replicas<1
            or gains.ndim!=3 or gains.shape[0]!=n or gains.shape[-1]!=2
            or gains.shape[1]%replicas):
        raise ValueError('One independent two-gain training query per parent required')
    q=gains.shape[1]//replicas;bank=gains.reshape(n,q,replicas,2)
    if not np.array_equal(bank,np.broadcast_to(bank[:,:,:1],bank.shape)):
        raise ValueError('Paired replicas must use the same gain')
    reference=np.all(bank[:,:,0]==4.,axis=-1)
    if not np.all(reference.sum(-1)==1):raise ValueError('Exactly one gain[4,4] reference required')
    progress=np.asarray(data['target'])[...,1]
    if (progress.shape!=(n,q*replicas) or not np.isfinite(progress).all()
            or not np.asarray(data['target_mask'])[...,1].all()):
        raise ValueError('Complete observed-prefix progress labels required')
    status=np.asarray(data['status'])
    if status.shape!=progress.shape or not np.isin(status,np.arange(1,9)).all():
        raise ValueError('Complete terminal branch outcomes required')
    safe=np.isin(status,[1,4]).reshape(n,q,replicas).all(-1)
    actual=progress.reshape(n,q,replicas).mean(-1);ref=reference.argmax(-1);rows=np.arange(n)
    useful=(safe&((actual-actual[rows,ref,None]>.01)|~safe[rows,ref,None])).any(-1)
    positive=int(useful.sum());negative=n-positive
    if not positive or not negative:raise ValueError('Both training opportunity strata required')
    weights=np.where(useful,.5*n/positive,.5*n/negative).astype(np.float32)
    return weights,dict(schema='quad2d_observed_gain_opportunity_weight_v1',parents=n,
        opportunity_parents=positive,other_parents=negative,reference_gain=[4.,4.],
        physical_progress_difference=.01,replicas=replicas,
        opportunity_weight=float(weights[useful][0]),other_weight=float(weights[~useful][0]),
        opportunity_group_ids=data['group_id'][useful].tolist(),
        sampling='Equal expected stratum mass before the original parent bootstrap; all branches share their parent weight.',
        validation_and_calibration='Original unweighted held-out partitions; no opportunity-based filtering.',
        runtime_use=False)
