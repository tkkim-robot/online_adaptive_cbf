"""Stopping calibration on a shared union of reserved policy visits.

This is a controlled calibration experiment: no neural weights, variance,
collision calibration, gain-selection rules or physical controller are changed.
The fixed-policy and both adaptive-policy populations have identical parent
coverage. Fit and audit identities retain their original reserved roles.
"""


import json
from pathlib import Path


from .dataset import sha256


from .local_unicycle_policy_calibration_data import validate_source
from .local_unicycle_prediction_attribution import check_bindings

SCHEMA = 'local_unicycle_shared_policy_stop_calibration_v1'


def read(path):
    return json.loads(Path(path).read_text())


def collection_contract(collection):
    root = Path(collection)
    m, base, records = validate_source(root)
    audit = read(root / 'audit.json')
    proof = read(root / 'independent_completion_review.json')
    if (m['pilot'] or not m['predictive_calibration_fit_authorized']
            or m['variants_per_policy'] != 1024 or m['selected_parent_count'] != 256
            or m['held_gain_seconds'] != 8 or m['adaptation_interval_seconds'] != .2
            or m['replicas'] != 2 or m['queries'] != 32
            or not proof['independently_reproduced'] or proof['pilot']
            or proof['audit_sha256'] != sha256(root / 'audit.json')
            or proof['observations'] != 16384 or proof['branches'] != 1048576
            or proof['physical_parents'] != 256
            or proof['roles'] != {'predictive_fit': 8192, 'predictive_audit': 8192}
            or audit['index_sha256'] != sha256(root / 'index.json')
            or audit['manifest_sha256'] != sha256(root / 'manifest.json')):
        raise ValueError('Complete, independently reviewed reserved policy collection required')
    paths = [root / n for n in ('manifest.json', 'index.json', 'audit.json',
                               'independent_completion_review.json', 'independent_compact_replay.json')]
    paths += [base / n for n in ('manifest.json', 'index.json', 'audit.json', 'records.json', 'reserved_groups.json')]
    return m, base, records, {str(p.resolve()): sha256(p) for p in paths}


def validate_supplement(info):
    """Optional runtime extension; historical calibrations take the old path."""
    proof = info['policy_visited_stop_calibration']
    if (proof['schema'] != SCHEMA or proof['weight_fitting'] or proof['forward_parents_used']
            or proof['changed_parameters'] != ['stop_temperature', 'stop_bias']):
        raise ValueError('Unknown policy-visited calibration extension')
    check_bindings(proof['bound_files'])
    _, base, _, _ = collection_contract(proof['collection'])
    if str(base.resolve()) != info['calibration_dataset']:
        raise ValueError('Different predictive reservation')
    previous = read(proof['previous_calibration'])
    unchanged = ('bundle', 'bundle_manifest_sha256', 'weights_sha256', 'dataset',
                 'dataset_manifest_sha256', 'calibration_dataset', 'calibration_manifest_sha256',
                 'calibration_audit_sha256', 'local_unicycle_contract', 'variance_scale',
                 'controller', 'targets', 'events', 'gain_domain', 'qualification',
                 'qualification_sha256', 'group_ids')
    if (any(info[k] != previous[k] for k in unchanged)
            or info['event_calibration'][0] != previous['event_calibration'][0]):
        raise ValueError('Stop-only refit changed frozen model or calibration fields')
    report = read(proof['report'])
    if (report['event_calibration'] != info['event_calibration']
            or report['variance_scale'] != info['variance_scale']
            or report['fit_groups'] != info['group_ids']['fit']
            or report['audit_groups'] != info['group_ids']['audit']
            or set(report['fit_groups']) & set(report['audit_groups'])):
        raise ValueError('Changed stop fit or parent roles')
