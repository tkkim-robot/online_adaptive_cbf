"""Shared bicycle policy labels implementation."""

from collections import Counter, defaultdict

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import get_context

from pathlib import Path

import jax

import numpy as np

from .bicycle_control import read, control_config

from .bicycle_trace_storage import open_trace, verify_index_dependencies

from .io import sha256

from .io import write_json

SCHEMA = 'bicycle_shared_learned_visitation_labels_v1'

BANK = np.geomspace(.5, 8, 8).astype(np.float32)

HORIZON = 40

REPLICAS = 2

def select_queries(queries):
    """First, middle, last *available* grid query, without outcome filtering."""
    grouped = defaultdict(list)
    for q in queries:
        if q['partition'] not in ('train', 'validation') or q['encoder'] not in ('gat', 'matched_fc'):
            raise ValueError('Only reserved TRAIN/validation OA/FC observations allowed')
        grouped[q['encoder'], q['group_id']].append(q)
    selected = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda q: q['tick'])
        if (len({q['tick'] for q in rows}) != len(rows) or rows[0]['tick'] != 0
                or len({(q['partition'], q['trace'], q['trace_sha256']) for q in rows}) != 1):
            raise ValueError('Duplicated query or inconsistent acquired trajectory')
        selected.extend(dict(rows[i]) for i in sorted({0, len(rows)//2, len(rows)-1}))
    roles = {}
    for q in selected:
        old = roles.setdefault(q['group_id'], q['partition'])
        if old != q['partition']:
            raise ValueError('Physical parent split changed across policy histories')
    encoders = {e: {q['group_id'] for q in selected if q['encoder'] == e} for e in ('gat', 'matched_fc')}
    if encoders['gat'] != encoders['matched_fc']:
        raise ValueError('Both encoders must contribute every same physical parent')
    return selected

def dense_queries(trace_map, stride=10):
    """Uniform recorded observations plus actual last query, including failures."""
    if type(stride) is not int or stride < 1:
        raise ValueError('Positive integer observation cadence required')
    selected = []
    for path, (encoder, entry, parent) in trace_map.items():
        if parent['partition'] not in ('train', 'validation') or parent['calibration_role'] != 'none':
            raise ValueError('Dense visits cannot add reserved calibration or benchmark parents')
        if sha256(path) != entry['sha256']:
            raise ValueError('Changed independently audited trajectory')
        with np.load(path) as z:
            active, previous_gain = z['active'], z['previous_gain']
        length = len(active)
        if length != entry['queries'] or length < 1 or previous_gain.shape != (length,):
            raise ValueError('Incomplete acquired physical history')
        for tick in sorted(set(range(0, length, stride)) | {length-1}):
            selected.append(dict(encoder=encoder, group_id=parent['group_id'], partition=parent['partition'],
                trace=str(path), trace_sha256=entry['sha256'], tick=tick, last_query=tick==length-1,
                previous_gain=float(previous_gain[tick]), acquisition_active=bool(active[tick])))
    selected.sort(key=lambda q: (q['encoder'], q['group_id'], q['tick']))
    # Reuse the whole-parent/split/duplicate guard, not its sparse subset.
    select_queries(selected)
    return selected

def validate_visitation(directory, review, selection='sparse3'):
    root = Path(directory).resolve(); report = read(root/'report.json'); proof = read(review)
    if (proof.get('source_selection_roles_physical_models_and_query_bindings_verified') is not True
            or proof['report_sha256'] != sha256(root/'report.json')
            or report['query_index_sha256'] != sha256(root/'query_index.json')
            or report['training_performed'] or report['benchmark_parents_used']):
        raise ValueError('Independently reviewed, unmodified reserved visitation required')
    from .bicycle_data import validate_source
    parents = None; config = None; controller = None; bound = {}
    for file in (Path(review), root/'report.json', root/'query_index.json'):
        bound[str(file.resolve())] = sha256(file)
    for file, digest in report['frozen_files'].items():
        if sha256(file) != digest: raise ValueError('Changed frozen acquisition input')
        bound[file] = digest
    trace_map = {}
    for encoder, path in report['sources'].items():
        sm, rows = validate_source(path)
        if parents is not None and rows != parents: raise ValueError('Unmatched physical parents')
        parents = rows
        for file in ('manifest.json', 'scenes.json'):
            p = Path(path)/file; bound[str(p)] = sha256(p)
        for part in report['parts'][encoder]:
            p = Path(part['directory']); manifest = read(p/'manifest.json'); audit = read(p/'independent_replay.json')
            for field, file in (('manifest_sha256', 'manifest.json'), ('index_sha256', 'index.json'), ('audit_sha256', 'independent_replay.json')):
                if part[field] != sha256(p/file): raise ValueError('Changed acquired trace audit')
                bound[str(p/file)] = part[field]
            if (audit['manifest_sha256'] != part['manifest_sha256'] or audit['index_sha256'] != part['index_sha256']
                    or manifest['phase'] != 'training_acquisition' or manifest['smoke_only']):
                raise ValueError('Unbound full acquired trajectory')
            if config is not None and (config != manifest['config'] or controller != manifest['controller']):
                raise ValueError('Different OA/FC physical guidance')
            config, controller = manifest['config'], manifest['controller']
            for entry in read(p/'index.json'):
                parent = rows[entry['source_index']]
                trace_map[str((p/entry['file']).resolve())] = (encoder, entry, parent)
    queries = read(root/'query_index.json')
    for q in queries:
        encoder, entry, parent = trace_map[q['trace']]
        if (q['encoder'] != encoder or q['group_id'] != entry['group_id'] or q['group_id'] != parent['group_id']
                or q['partition'] != parent['partition'] or q['trace_sha256'] != entry['sha256']
                or not 0 <= q['tick'] < entry['queries']):
            raise ValueError('Query differs from reviewed physical acquisition')
    if selection == 'sparse3': selected = select_queries(queries)
    elif selection == 'dense10': selected = dense_queries(trace_map)
    else: raise ValueError('Unknown shared observation selection')
    if (len(parents) != 160 or Counter(p['partition'] for p in parents) != {'train': 128, 'validation': 32}
            or {q['group_id'] for q in selected} != {p['group_id'] for p in parents}):
        raise ValueError('All160 original reserved parents must remain')
    return dict(queries=selected, parents={p['group_id']: p for p in parents}, config=config,
                controller=controller, bound_inputs=bound)

def worker_queries(contract, shard, shards, limit):
    if shards < 1 or not 0 <= shard < shards or limit < 0: raise ValueError('Invalid shard/limit')
    # Both policy histories of the same parent stay in one worker; no dropped parents.
    ids = sorted(contract['parents'])
    selected = {p for i, p in enumerate(ids) if i % shards == shard}
    rows = [q for q in contract['queries'] if q['group_id'] in selected]
    return rows[:limit] if limit else rows

def audit(directory, workers=12):
    from .bicycle_data import audit_one
    from .bicycle_data import targets
    from .bicycle_observed_audit import check_graph, check_route_progress
    root = Path(directory); m = read(root/'manifest.json')
    if m['schema'] != SCHEMA or m['horizon_steps'] != HORIZON or m['replicas'] != REPLICAS:
        raise ValueError('Unexpected shared policy label contract')
    contract = validate_visitation(m['visitation'], m['review'], m.get('query_selection', 'sparse3')); c = control_config(m['config'])
    if contract['bound_inputs'] != m['bound_inputs'] or contract['controller'] != m['controller'] or contract['config'] != m['config']:
        raise ValueError('Changed shared physical/source contract')
    selected = worker_queries(contract, m['shard'], m['shards'], m['limit'])
    entries = read(root/'index.json'); traces = read(root/'trace_index.json')
    if read(root/'selected_queries.json') != selected or [e['query'] for e in entries] != selected:
        raise ValueError('Dropped, altered or duplicated selected observation')
    if [t for e in entries for t in e['traces']] != traces: raise ValueError('Inconsistent branch index')
    storage = verify_index_dependencies(root, traces)
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as pool:
        results = []
        for result in pool.map(audit_one, [(root, t, m['config'], True) for t in traces], chunksize=4):
            results.append(result)
            if len(results) % 256 == 0 or len(results) == len(traces):
                print(dict(stage='independent_physical_replay', checked=len(results), total=len(traces)), flush=True)
    by_file = {r['file']: r for r in results}; rounding = 0; cached_path = None
    for e, q in zip(entries, selected, strict=True):
        if sha256(root/e['file']) != e['sha256']: raise ValueError('Changed training label')
        with np.load(root/e['file']) as z: d = {k: z[k] for k in z}
        if cached_path != q['trace']:
            if sha256(q['trace']) != q['trace_sha256']: raise ValueError('Changed reviewed physical trajectory')
            with np.load(q['trace']) as z: a = {k: z[k] for k in z}
            cached_path = q['trace']
        tick = q['tick']; parent = contract['parents'][q['group_id']]
        for field, value in (('group_id', q['group_id']), ('partition', parent['partition']), ('query_tick', tick),
                             ('acquisition_encoder', q['encoder']), ('calibration_role', 'none')):
            np.testing.assert_array_equal(d[field], [value])
        np.testing.assert_array_equal(d['gains'][0, :, 0], np.repeat(BANK, REPLICAS))
        np.testing.assert_array_equal(d['previous_gain'][0], a['previous_gain'][tick])
        np.testing.assert_array_equal(d['previous_control'][0], a['previous_control'][tick])
        for datafield, tracefield in (('initial_state', 'state_before'), ('observed_state', 'observed_state'), ('observed_obstacles', 'observed_obstacles'), ('cursor', 'cursor_before')):
            np.testing.assert_array_equal(d[datafield][0], a[tracefield][tick])
        moving = a['obstacles'].copy(); moving[:, :2] += tick*c.robot.dt*moving[:, 3:5]
        np.testing.assert_array_equal(d['obstacles'][0], moving)
        for datafield, field in (('goal','goal'), ('obstacle_mask','mask'), ('points','points'), ('route_mask','route_mask'), ('noise','noise')):
            np.testing.assert_array_equal(d[datafield][0], a[field])
        check_graph(d['features'][0], d['node_mask'][0], (d['observed_state'][0], d['goal'][0], d['observed_obstacles'][0],
            d['obstacle_mask'][0], d['points'][0], d['route_mask'][0], d['noise'][0], c, float(d['cursor'][0]), d['previous_control'][0], float(d['previous_gain'][0])))
        for k, value in targets({k: d[k][0] for k in ('status','min_clearance','route_progress')}, HORIZON, c).items():
            np.testing.assert_array_equal(d[k][0], value)
        if len(e['traces']) != 16 or e['branches'] != 16: raise ValueError('Missing candidate or replica')
        keys = np.array(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(parent['seed']+3), tick), REPLICAS))
        innovations = {}
        for i, tr in enumerate(e['traces']):
            if tr['acquisition_file'] != q['trace'] or tr['acquisition_sha256'] != q['trace_sha256'] or tr['candidate'] != i:
                raise ValueError('Changed branch acquisition lineage')
            with open_trace(root/tr['file'], tr['sha256']) as b:
                for field, value in (('initial', a['state_before'][tick]), ('first_x', a['observed_state'][tick]),
                                     ('first_o', a['observed_obstacles'][tick]), ('cursor', a['cursor_before'][tick]),
                                     ('obstacles', moving), ('alpha', BANK[i//REPLICAS]), ('key', keys[i % REPLICAS])):
                    np.testing.assert_array_equal(b[field], value)
                for field in ('bias_x','bias_o','mask','goal','noise','points','route_mask','ready'):
                    np.testing.assert_array_equal(b[field], a[field])
                if int(b['horizon']) != HORIZON or int(b['expected_steps']) != int(d['steps'][0, i]) or int(b['final_status']) != int(d['status'][0, i]):
                    raise ValueError('Wrong label termination/horizon')
                np.testing.assert_array_equal(d['final_state'][0, i], b['state'][-1])
                np.testing.assert_array_equal(d['final_cursor'][0, i], b['route_progress'][-1])
                np.testing.assert_allclose(d['min_clearance'][0, i], by_file[tr['file']]['min_clearance'], atol=1e-4, rtol=0)
                rounding += check_route_progress(d['route_progress'][0, i], b['initial'].astype(np.float32)[:2], b['state'][-1].astype(np.float32)[:2],
                    b['points'], b['route_mask'], float(b['cursor']), float(b['route_progress'][-1]))
                replica = i % REPLICAS; innovation = (b['innovation_x'], b['innovation_o'])
                if replica in innovations:
                    length = min(len(innovations[replica][0]), len(innovation[0]))
                    for x, y in zip(innovations[replica], innovation): np.testing.assert_array_equal(x[:length], y[:length])
                if replica not in innovations or len(innovation[0]) > len(innovations[replica][0]):
                    innovations[replica] = innovation
    result = dict(audit_passed=True, manifest_sha256=sha256(root/'manifest.json'), index_sha256=sha256(root/'index.json'),
        trace_index_sha256=sha256(root/'trace_index.json'), selected_queries_sha256=sha256(root/'selected_queries.json'),
        all_physical_prefixes_replayed=True, all_original_observed_rows_checked=True, all_graphs_independently_checked=True,
        all_acquired_history_bindings_checked=True, all_recorded_margin_bounds_checked=True, trace_storage_verification=storage,
        queries=len(entries), branches=len(traces), parents=len({q['group_id'] for q in selected}),
        physical_steps=sum(r['steps'] for r in results), feasible_qp_rejections=sum(r['feasible_qp_rejected'] for r in results),
        route_progress_rounding_checks=int(rounding), rows=results)
    write_json(root/'independent_replay.json', result)
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)
