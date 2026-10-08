"""Bind fresh static terminal-guided labels to a matched encoder refit."""


from copy import deepcopy
from dataclasses import asdict
from pathlib import Path


from .dataset import sha256


from .quad2d_control import flight_config_from_contract
from .quad2d_static_inputs import read, verify


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
    reservation = verify(raw)
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
