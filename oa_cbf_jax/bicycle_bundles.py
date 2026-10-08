"""Load matched bicycle bundles and their bound calibration evidence."""
from pathlib import Path
from .bicycle_experiment import read
from .comparison_contracts import matched_training_settings
from .dataset import sha256
ENCODERS = ('gat', 'matched_fc')


def models(training, completion):
    root = Path(training)
    done = read(completion)
    from .bicycle_variance_calibration import RUNTIME_SCHEMA, runtime_models
    if done.get('schema') == RUNTIME_SCHEMA:
        original = Path(done['base_training_complete'])
        if (sha256(original) != done['base_training_complete_sha256']
                or read(original).get('schema') == RUNTIME_SCHEMA):
            raise ValueError('Original completed training required for calibration variant')
        return runtime_models(training,done,models(training,original))
    if (done.get('status') != 'completed' or done.get('matched_training_settings_verified') is not True
            or Path(done['output']).resolve() != root.resolve()):
        raise ValueError('Complete matched training is required before policy qualification')
    for member in range(4):
        matched_training_settings(read(root/f'gat/member_{member}/settings.json'),
                                  read(root/f'matched_fc/member_{member}/settings.json'))
    result = {}
    for encoder in ENCODERS:
        path = root/encoder
        bundle, fit = path/'bundle_fp64', path/'prediction_calibration/prediction_fit.json'
        metadata, calibration = read(bundle/'manifest.json'), read(fit)
        if metadata.get('offline_wide_gain_pilot'):
            from .bicycle_candidate_calibration import validate_fitted_model
            validate_fitted_model(calibration,bundle)
        finished = read(path/'prediction_calibration/complete.json')
        if (metadata['architecture']['encoder'] != encoder or metadata['architecture']['compute_dtype'] != 'float64'
                or metadata['weights_sha256'] != sha256(bundle/'weights.msgpack')
                or calibration['bundle_manifest_sha256'] != sha256(bundle/'manifest.json')
                or calibration['weights_sha256'] != metadata['weights_sha256']
                or finished['status'] != 'completed' or finished['prediction_fit_sha256'] != sha256(fit)
                or finished['prediction_audit_sha256'] != sha256(path/'prediction_calibration/prediction_audit.json')):
            raise ValueError('Changed model or reserved prediction fit')
        for field in ('controller', 'bicycle_contract', 'targets', 'events', 'gain_domain', 'dataset_manifest_sha256'):
            if calibration[field] != metadata[field]:
                raise ValueError('Prediction fit/model semantics differ: '+field)
        if calibration.get('bicycle_task_progress_contract')!=metadata.get('bicycle_task_progress_contract'):
            raise ValueError('Prediction fit/model task-progress contracts differ')
        result[encoder] = dict(bundle=str(bundle.resolve()), fit=str(fit.resolve()), metadata=metadata,
            calibration=calibration, weights_sha256=sha256(bundle/'weights.msgpack'),
            bundle_manifest_sha256=sha256(bundle/'manifest.json'), fit_sha256=sha256(fit))
    for field in ('controller', 'bicycle_contract', 'targets', 'events', 'gain_domain',
                  'dataset_manifest_sha256', 'normalization', 'graph_features', 'gain_dimension'):
        if result['gat']['metadata'][field] != result['matched_fc']['metadata'][field]:
            raise ValueError('Unmatched encoder prediction treatment: '+field)
    for field in ('group_ids', 'candidates', 'prediction_batch', 'prediction_candidates',
                  'dataset_index_sha256', 'dataset_audit_sha256', 'source_manifest_sha256', 'event_budget_statistic'):
        if result['gat']['calibration'][field] != result['matched_fc']['calibration'][field]:
            raise ValueError('Unmatched reserved calibration protocol: '+field)
    if result['gat']['metadata'].get('bicycle_task_progress_contract')!=result['matched_fc']['metadata'].get('bicycle_task_progress_contract'):
        raise ValueError('Unmatched encoder task-progress treatment')
    return result
