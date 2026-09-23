"""Reviewed union of fixed-gain and frozen learned-policy bicycle observations.

All existing labels remain immutable. Both encoders receive this same view;
parent identities, split roles and reserved calibration queries do not change.
"""
from collections import Counter
import copy
from pathlib import Path

import numpy as np

from .bicycle_experiment import read
from .dataset import sha256, source_fingerprint
from .io import write_json

MODE = 'balanced_fixed_and_shared_frozen_policy'
CONTRACT = 'bicycle_shared_observation_union_v1'


def bound_files(directory, names):
    root = Path(directory).resolve()
    return {str(root/name): sha256(root/name) for name in names}


def check_bounds(files):
    for path, digest in files.items():
        if sha256(path) != digest: raise ValueError('Changed union component: '+str(path))


def index_rows(base, parts, parents):
    """Canonical source-index view; no duplicate weighting via renamed parents."""
    base = Path(base).resolve(); rows = []; seen = set()
    sources = [('fixed', base, read(base/'index.json'))]
    for part in parts:
        root = Path(part['directory']).resolve()
        sources.append(('learned', root, read(root/'index.json')))
    for origin, root, entries in sources:
        for e in entries:
            identity, tick = e['group_id'], e['query_tick']
            if identity not in parents: raise ValueError('Unreserved new physical parent')
            parent = parents[identity]
            encoder = 'fixed' if origin == 'fixed' else e['acquisition_encoder']
            if origin == 'learned':
                q = e['query']
                if (encoder not in ('gat','matched_fc') or parent['partition'] not in ('train','validation')
                        or parent['calibration_role'] != 'none' or q['partition'] != parent['partition']
                        or q['group_id'] != identity or q['tick'] != tick or q['encoder'] != encoder):
                    raise ValueError('Learned observation changes original role or acquisition identity')
            key = encoder, identity, tick
            if key in seen: raise ValueError('Duplicate source/parent/time query')
            seen.add(key)
            rows.append(dict(file=str((root/e['file']).resolve()), sha256=e['sha256'],
                groups=e['groups'], branches=e['branches'], observed_steps=e['observed_steps'],
                group_id=identity, query_tick=tick, query_origin=encoder))
    if {e['group_id'] for e in rows} != set(parents):
        raise ValueError('Missing original physical parent')
    return rows


def assemble(base, parts, output, visitation_review, label_pilot_review):
    """Authorize only the complete, already audited union for a new shared fit."""
    from .bicycle_data import validate_training_dataset as validate_base
    from .bicycle_label_validation import verify_part
    from .bicycle_policy_labels import validate_visitation, worker_queries
    base, root = Path(base).resolve(), Path(output).resolve()
    original = validate_base(base)
    if original['acquisition_mode'] != 'balanced8' or original['horizon_steps'] != 40:
        raise ValueError('Original dense fixed-gain dataset required')
    if original['snapshot_ticks'] != list(range(0,320,10)) or original['replicas'] != 2:
        raise ValueError('Unchanged dense cadence and paired gain labels required')
    checked = read(label_pilot_review)
    if checked.get('full_physical_storage_query_split_and_aggregation_bindings_verified') is not True:
        raise ValueError('Independent shared-label pilot review required')
    proofs = [verify_part(p['directory']) for p in parts]
    contract = None; actual = []; binding = bound_files(base, ('manifest.json','index.json','independent_replay.json','complete.json'))
    for path in (visitation_review, label_pilot_review): binding[str(Path(path).resolve())] = sha256(path)
    for k, proof in enumerate(proofs):
        p = Path(proof['directory']); pm = read(p/'manifest.json')
        if (pm.get('query_selection') != 'dense10' or pm['limit'] != 0 or pm['shards'] != 4 or pm['shard'] != k
                or pm['config'] != original['config'] or pm['controller'] != original['controller']
                or Path(pm['review']).resolve() != Path(visitation_review).resolve()):
            raise ValueError('Incomplete or changed dense learned-policy label contract')
        if contract is None:
            contract = validate_visitation(pm['visitation'], pm['review'], 'dense10')
            pilot_path = Path(pm['visitation']).parent/'bicycle_shared_policy_label_pilot'/'report.json'
            if checked['report_sha256'] != sha256(pilot_path):
                raise ValueError('Changed independently reviewed pilot result')
            binding[str(pilot_path)] = sha256(pilot_path)
        if read(p/'selected_queries.json') != worker_queries(contract,k,4,0):
            raise ValueError('Dropped or duplicated dense observation')
        actual.extend(read(p/'selected_queries.json'))
        binding.update(bound_files(p, ('manifest.json','index.json','trace_index.json','selected_queries.json','independent_replay.json','summary.json')))
    if len(proofs) != 4 or len(actual) != len(contract['queries']):
        raise ValueError('Four complete shared observation parts required')
    source = Path(original['source']); parents = {p['group_id']:p for p in read(source/'scenes.json')}
    if any(parents.get(identity) != parent for identity,parent in contract['parents'].items()):
        raise ValueError('Visitation changed the original full-dataset parents')
    rows = index_rows(base, proofs, parents)
    # Do not copy or rewrite any source data; references bind immutable bytes.
    root.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(original)
    manifest.update(stage='shared_fixed_and_learned_observation_training', acquisition_mode=MODE,
        snapshot_ticks=list(range(0,1600,10)), acquisition_steps=1600, source_fingerprint=source_fingerprint(),
        shared_observation_union=dict(schema=CONTRACT, base=str(base), parts=proofs, bound_inputs=binding,
            visitation=str(Path(read(Path(proofs[0]['directory'])/'manifest.json')['visitation']).resolve()),
            visitation_review=str(Path(visitation_review).resolve()), pilot_review=str(Path(label_pilot_review).resolve()),
            original_fixed_acquisition_steps=320, learned_acquisition_steps=1600,
            learned_query_rule='Every10recorded ticks plus actual last query; both policies, every reserved parent and failure.',
            learned_parents=160, learned_histories=320, learned_queries=len(actual),
            unchanged_prediction_calibration=True, parent_identity='Original physical parent across fixed/OA/FC histories; never rename or duplicate as independent parents.'),
        conditioning='Original fixed-acquisition pool plus shared frozen OA/FC visits. Exact causal observations retained. Latent states/biases only initialize offline labels. All simulation contracts identical.',
        weight_fit_authorized=True, training_use=True, final_test=False, production_eligible=False)
    write_json(root/'manifest.json',manifest); write_json(root/'index.json',rows)
    old = read(base/'independent_replay.json')
    audit = dict(audit_passed=True, manifest_sha256=sha256(root/'manifest.json'), index_sha256=sha256(root/'index.json'),
        all_physical_prefixes_replayed=True, all_original_observed_rows_checked=True, all_graphs_independently_checked=True,
        all_acquired_history_bindings_checked=True, all_recorded_margin_bounds_checked=True,
        all_component_audits_and_union_lineage_verified=True, base=str(base), parts=proofs,
        parents=len(parents), queries=len(rows), branches=sum(e['branches'] for e in rows),
        physical_steps=old['physical_steps']+sum(p['physical_steps'] for p in proofs),
        feasible_qp_rejections=0, unchanged_calibration_parents_and_queries=True,
        note='Aggregates previously independently replayed components; no new physical replay claimed during assembly.')
    write_json(root/'independent_replay.json',audit)
    write_json(root/'complete.json',dict(status='completed',audit_passed=True,
        manifest_sha256=sha256(root/'manifest.json'), index_sha256=sha256(root/'index.json'), audit_sha256=sha256(root/'independent_replay.json')))
    validate_union(root)
    return audit


def validate_union(directory):
    from .bicycle_data import validate_training_dataset as validate_base
    from .bicycle_label_validation import verify_part
    from .bicycle_policy_labels import validate_visitation, worker_queries
    root = Path(directory).resolve(); m = read(root/'manifest.json'); a = read(root/'independent_replay.json'); done = read(root/'complete.json')
    union = m['shared_observation_union']
    if (m['acquisition_mode'] != MODE or union['schema'] != CONTRACT or m.get('weight_fit_authorized') is not True
            or m.get('training_use') is not True or m.get('final_test') is not False
            or done['status'] != 'completed' or not done['audit_passed'] or not a['audit_passed']
            or not a['all_component_audits_and_union_lineage_verified']):
        raise ValueError('Complete authorized shared observation union required')
    for k,f in (('manifest_sha256','manifest.json'), ('index_sha256','index.json')):
        if a[k] != sha256(root/f) or done[k] != sha256(root/f): raise ValueError('Changed union evidence')
    if done['audit_sha256'] != sha256(root/'independent_replay.json'): raise ValueError('Changed union replay')
    check_bounds(union['bound_inputs'])
    base = Path(union['base'])
    if base == root: raise ValueError('Recursive shared dataset')
    bm = read(base/'manifest.json')
    if bm['acquisition_mode'] != 'balanced8': raise ValueError('Original fixed-gain base required')
    validate_base(base)
    for key in ('schema','config','sensor_schema','graph_schema','horizon_steps','capacity','replicas','controller',
                'gain_domain','targets','events','graph_features','gain_dimension','queries','source','source_manifest_sha256','groups'):
        if m[key] != bm[key]: raise ValueError('Changed shared prediction/source contract: '+key)
    if m['snapshot_ticks'] != list(range(0,1600,10)) or m['acquisition_steps'] != 1600:
        raise ValueError('Changed learned observation cadence/budget')
    contract = validate_visitation(union['visitation'],union['visitation_review'],'dense10')
    parents = {p['group_id']:p for p in read(Path(bm['source'])/'scenes.json')}
    if any(parents.get(g) != p for g,p in contract['parents'].items()): raise ValueError('Changed original parent roles/geometry')
    if len(union['parts']) != 4: raise ValueError('Incomplete shared branches')
    for k, expected in enumerate(union['parts']):
        proof = verify_part(expected['directory'])
        if proof != expected: raise ValueError('Changed independently audited observation part')
        p = Path(proof['directory']); pm = read(p/'manifest.json')
        if (pm['query_selection'] != 'dense10' or pm['shard'] != k or pm['shards'] != 4 or pm['limit'] != 0
                or pm['config'] != bm['config'] or pm['controller'] != bm['controller']
                or pm['bound_inputs'] != contract['bound_inputs']
                or read(p/'selected_queries.json') != worker_queries(contract,k,4,0)):
            raise ValueError('Different shared learned-query coverage or physical contract')
    expected = index_rows(base,union['parts'],parents); rows = read(root/'index.json')
    if rows != expected: raise ValueError('Changed, missing or duplicated union query')
    learned = [e for e in rows if e['query_origin'] != 'fixed']
    if (union['learned_queries'] != len(learned) or union['learned_parents'] != len(contract['parents'])
            or union['learned_histories'] != len({(e['query_origin'],e['group_id']) for e in learned})
            or union.get('unchanged_prediction_calibration') is not True
            or a.get('unchanged_calibration_parents_and_queries') is not True):
        raise ValueError('Changed shared-coverage or calibration declaration')
    expected_steps = read(base/'independent_replay.json')['physical_steps']+sum(p['physical_steps'] for p in union['parts'])
    if (a['parents'] != len(parents) or a['queries'] != len(rows) or a['branches'] != sum(e['branches'] for e in rows)
            or a['physical_steps'] != expected_steps or a['feasible_qp_rejections'] != 0):
        raise ValueError('Changed union denominator')
    for e in rows:
        if sha256(e['file']) != e['sha256']: raise ValueError('Changed physical label bytes')
    return m


def training_reference(previous, manifest, dataset, digest):
    from .bicycle_learning_contracts import prediction_contract
    before = copy.deepcopy(manifest)
    before['acquisition_mode'] = previous['bicycle_contract']['acquisition_mode']
    before['snapshot_ticks'] = previous['bicycle_contract']['snapshot_ticks']
    prediction_contract(before,previous)
    if (manifest['acquisition_mode'] != MODE or manifest['horizon_steps'] != 40 or manifest['replicas'] != 2
            or manifest['snapshot_ticks'] != list(range(0,1600,10)) or manifest['acquisition_steps'] != 1600):
        raise ValueError('Only shared observation coverage may change')
    result = copy.deepcopy(previous)
    result.update(dataset=str(Path(dataset).resolve()),dataset_manifest_sha256=digest)
    for key in ('acquisition_mode','snapshot_ticks'): result['bicycle_contract'][key] = manifest[key]
    for key in ('normalization','group_bootstrap','parent_sampling','safety_opportunity_sampling','source_fingerprint','device','stage'):
        result.pop(key,None)
    prediction_contract(manifest,result)
    return result
