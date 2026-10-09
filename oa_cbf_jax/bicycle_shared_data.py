"""Shared bicycle shared data implementation."""

from pathlib import Path

from .bicycle_control import read

from .io import sha256

MODE = 'balanced_fixed_and_shared_frozen_policy'

CONTRACT = 'bicycle_shared_observation_union_v1'

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

def validate_union(directory):
    from .bicycle_data import validate_training_dataset as validate_base
    from .bicycle_data import verify_part
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
