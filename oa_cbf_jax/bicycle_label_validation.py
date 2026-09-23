"""Validate complete offline labels and their independent physical replay."""
from pathlib import Path
from .bicycle_experiment import read
from .bicycle_policy_labels import SCHEMA, HORIZON, REPLICAS
from .dataset import sha256

def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def verify_part(path):
    p = Path(path); m = read(p/'manifest.json'); a = read(p/'independent_replay.json'); s = read(p/'summary.json')
    flags = ('audit_passed', 'all_physical_prefixes_replayed', 'all_original_observed_rows_checked',
             'all_graphs_independently_checked', 'all_acquired_history_bindings_checked', 'all_recorded_margin_bounds_checked')
    if (m['schema'] != SCHEMA or m['horizon_steps'] != HORIZON or m['replicas'] != REPLICAS
            or not all(a.get(k) is True for k in flags) or a['feasible_qp_rejections']
            or not a['trace_storage_verification']['all_shared_dependencies_verified']
            or not s['complete'] or s['compiled_signatures'] != 2 or s['implicit_jit_cache_entries'] != 0):
        raise ValueError('Incomplete or failed physical/compilation audit')
    for field, file in (('manifest_sha256','manifest.json'), ('index_sha256','index.json'),
                        ('trace_index_sha256','trace_index.json'), ('selected_queries_sha256','selected_queries.json')):
        if a[field] != sha256(p/file): raise ValueError('Changed audited evidence')
    for k in ('queries','branches','physical_steps','parents'):
        if a[k] != s[k]: raise ValueError('Different collection/audit totals')
    from .bicycle_trace_storage import verify_index_dependencies
    verify_index_dependencies(p, read(p/'trace_index.json'))
    return dict(directory=str(p.resolve()), parents=a['parents'], queries=a['queries'], branches=a['branches'],
        physical_steps=a['physical_steps'], bytes=directory_bytes(p), manifest_sha256=sha256(p/'manifest.json'),
        index_sha256=sha256(p/'index.json'), audit_sha256=sha256(p/'independent_replay.json'))
