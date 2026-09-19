"""Strict observed-history dataset identity and immutable multipart provenance."""
from dataclasses import asdict
import json
from pathlib import Path

from .dataset import sha256, source_fingerprint
from .io import write_json
from .quad2d_history import SCHEMA, GRAPH_SCHEMA
from .quad2d_control import FlightConfig
from .quad2d_guidance import ObservedMotionGuidanceConfig
from .quad2d_task_targets import contract
from .quad2d_relabel import TARGETS


def validate_manifest(m):
    c = asdict(FlightConfig())
    g = json.loads(json.dumps(asdict(ObservedMotionGuidanceConfig(noise_clearance_weight=1.))))
    controller = m.get('controller', {})
    if (m.get('schema') != SCHEMA or m.get('graph_schema') != GRAPH_SCHEMA
            or m.get('graph_features') != 50 or m.get('config') != c
            or controller.get('config') != c or controller.get('dynamics') != 'Quad2D'
            or controller.get('predictive_guidance') != g
            or controller.get('graph_schema') != GRAPH_SCHEMA
            or controller.get('initial_gain') != [4., 4.]
            or controller.get('observation_context') != 'raw sensor plus two-anchor causal velocity history; no true state/bias in graph'
            or controller.get('performance_target') != contract('terminal_task')
            or m.get('targets') != [TARGETS[0], contract('terminal_task')['target']]
            or m.get('events') != ['collision_first', 'any_adverse_termination']):
        raise ValueError('Reviewed observed-history graph/controller/target contract required')
    if (m.get('queries') != 32 or m.get('replicas', 0) < 1 or m.get('horizon_steps', 0) < 1
            or m.get('observation_ticks') != [0, 40, 120, 240, 400, 640]
            or m.get('capacity') != 64 or m.get('route_capacity') != 64
            or m.get('gain_domain') != dict(lower=.5, upper=8.) or m.get('final_test') is not False):
        raise ValueError('Unsupported history data shape, split or gain contract')


def _bound(root, record, fields):
    for key, name in fields:
        if record.get(key) != sha256(root / name):
            raise ValueError(f'Changed history provenance: {root / name}')


def validate_dataset(directory, allow_merge=True):
    """Validate existing audits and all referenced bytes, without redoing physics.

    Multipart views copy no arrays and change no partition: their index points
    to the exact independently audited part shards. Physical audit scope stays
    that of each part (one branch/parent plus every collision/envelope branch).
    """
    root = Path(directory)
    if (root / 'INVALIDATED.json').exists():
        raise ValueError('Invalidated history dataset')
    m = json.loads((root / 'manifest.json').read_text())
    validate_manifest(m)
    audit = json.loads((root / 'independent_replay.json').read_text())
    _bound(root, audit, [('manifest_sha256', 'manifest.json'), ('index_sha256', 'index.json')])
    required = ('audit_passed', 'all_histories_and_graphs_checked', 'all_physical_initial_states_preserved',
                'all_collision_bound_branches_audited')
    if not all(audit.get(key) is True for key in required):
        raise ValueError('Complete independent observed-history audit required')
    index = json.loads((root / 'index.json').read_text())
    ids = [r['group_id'] for r in m['groups']]
    if len(ids) != len(set(ids)) or audit['parents'] != len(ids):
        raise ValueError('Duplicate or incomplete history parents')
    if m.get('collection_parts'):
        if not allow_merge:
            raise ValueError('Nested history merges are not supported')
        groups, entries = [], []
        for part in m['collection_parts']:
            path = Path(part['directory'])
            _bound(path, part, [('manifest_sha256', 'manifest.json'), ('index_sha256', 'index.json'),
                                ('audit_sha256', 'independent_replay.json')])
            pm, _ = validate_dataset(path, allow_merge=False)
            if shared_contract(pm) != shared_contract(m):
                raise ValueError('Mixed history collection contracts')
            groups.extend(pm['groups'])
            entries.extend(absolute_index(path))
        if m['groups'] != groups or index != entries:
            raise ValueError('Changed merged history parent order or shard ancestry')
    else:
        source = Path(m['source'])
        _bound(source, m, [('source_manifest_sha256', 'manifest.json'), ('source_index_sha256', 'index.json'),
                           ('source_audit_sha256', 'independent_replay.json')])
        sm = json.loads((source / 'manifest.json').read_text())
        sa = json.loads((source / 'independent_replay.json').read_text())
        _bound(source, sa, [('manifest_sha256', 'manifest.json'), ('index_sha256', 'index.json')])
        if not sa.get('audit_passed'):
            raise ValueError('Unaudited history acquisition')
        original = Path(sm['source'])
        _bound(original, sm, [('source_manifest_sha256', 'manifest.json')])
        om = json.loads((original / 'manifest.json').read_text())
        _bound(original, om, [('scenes_sha256', 'scenes.json')])
        if om.get('training_use') is not True:
            raise ValueError('Nontraining history acquisition')
        parents = json.loads((original / 'scenes.json').read_text())
        if m['groups'] != [{k: p[k] for k in ('group_id', 'family', 'seed', 'partition')} for p in parents]:
            raise ValueError('Changed original history parent partitions')
        for row in json.loads((source / 'index.json').read_text()):
            _bound(source, row, [('sha256', row['file'])])
        for entry in index:
            _bound(root, entry, [('sha256', entry['file'])])
            for trace in entry['traces']:
                _bound(root, trace, [('sha256', trace['file'])])
    return m, audit


def shared_contract(m):
    return {k: m[k] for k in ('schema', 'config', 'capacity', 'route_capacity', 'graph_schema', 'graph_features',
            'queries', 'replicas', 'horizon_steps', 'observation_ticks', 'frozen_gain_bank', 'gain_domain',
            'targets', 'events', 'controller', 'replica_semantics', 'censoring')}


def absolute_index(directory):
    root = Path(directory).resolve()
    return [dict(entry, file=str(root / entry['file']),
                 traces=[dict(t, file=str(root / t['file'])) for t in entry['traces']],
                 collection_directory=str(root))
            for entry in json.loads((root / 'index.json').read_text())]


def merge(parts, output):
    reviewed = [(Path(p).resolve(), *validate_dataset(p, allow_merge=False)) for p in parts]
    if not reviewed:
        raise ValueError('Empty history merge')
    contract_fields = shared_contract(reviewed[0][1])
    if any(shared_contract(m) != contract_fields for _, m, _ in reviewed):
        raise ValueError('Mixed history collection contracts')
    groups = [g for _, m, _ in reviewed for g in m['groups']]
    if len({g['group_id'] for g in groups}) != len(groups):
        raise ValueError('Repeated history parent across parts')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    provenance = [dict(directory=str(p), manifest_sha256=sha256(p / 'manifest.json'),
        index_sha256=sha256(p / 'index.json'), audit_sha256=sha256(p / 'independent_replay.json')) for p, _, _ in reviewed]
    m = dict(contract_fields, groups=groups, collection_parts=provenance, stage='observed_history_learning_pilot',
             source_fingerprint=source_fingerprint(), final_test=False, production_eligible=False,
             limitations='Immutable concatenation of audited history parts. Development fixed-gain labels; not adaptive trajectory calibration.')
    write_json(root / 'manifest.json', m)
    write_json(root / 'index.json', [e for p, _, _ in reviewed for e in absolute_index(p)])
    report = dict(audit_passed=True, parents=len(groups), manifest_sha256=sha256(root / 'manifest.json'),
        index_sha256=sha256(root / 'index.json'), collection_parts=provenance,
        all_histories_and_graphs_checked=True, all_physical_initial_states_preserved=True,
        all_collision_bound_branches_audited=True, source_fingerprint=source_fingerprint(),
        scope='Part audit composition with every referenced byte checked, identical shards and original parent partition/order. Physical trace scope is unchanged from parts; no new physical integration.')
    for key in ('branches', 'steps', 'physical_traces', 'collision_bound_branches'):
        report[key] = sum(a[key] for _, _, a in reviewed)
    write_json(root / 'independent_replay.json', report)
    validate_dataset(root)
    write_json(root / 'complete.json', dict(status='completed', audit_passed=True, **{k: report[k] for k in ('manifest_sha256', 'index_sha256')}))
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--parts', nargs='+', required=True); p.add_argument('--output', required=True)
    print(json.dumps(merge(**vars(p.parse_args()))), flush=True)
