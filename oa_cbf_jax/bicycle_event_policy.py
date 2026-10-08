"""Reviewed research-policy binding for frozen-encoder event-readout refits."""
from pathlib import Path


from .bicycle_experiment import read
from .dataset import sha256


def validate_fit(fit, bundle):
    """Authenticate the completed calibration, without rewriting frozen fits.

    This permits research inference. A separate trajectory gate remains required
    for adaptive navigation; neither this binding nor calibration grants safety.
    """
    from .bicycle_event_bundle import validate
    bundle=Path(bundle).resolve();meta=read(bundle/'manifest.json')
    contract=validate(bundle,meta);name=contract['method'];root=bundle.parent.parent
    recorded=root/name/'prediction_fit.json';report=read(root/'report.json')
    proof=read(root/'independent_completion_review.json');frozen=read(root/'both_fits_frozen.json')
    flags=('review_passed','all_bundle_refit_and_label_bindings_checked',
           'all_roles_and_saved_query_orders_checked','all_density_noise_scores_independently_recomputed',
           'all_moment_scales_and_event_fit_losses_recomputed','frozen_nonadverse_predictions_exact_on_audit')
    if (not all(proof.get(k) is True for k in flags)
            or proof['report_sha256']!=sha256(root/'report.json')
            or proof['both_fits_frozen_sha256']!=sha256(root/'both_fits_frozen.json')
            or report['both_fits_frozen_sha256']!=sha256(root/'both_fits_frozen.json')
            or not report['audit_read_after_both_fits_frozen'] or report['network_training_performed']):
        raise ValueError('Completed independent readout calibration review required')
    if (set(frozen)!={'gat','nearest_fc'} or fit!=read(recorded)
            or any(sha256(root/n/'prediction_fit.json')!=h for n,h in frozen.items())):
        raise ValueError('Changed frozen readout prediction fit')
    protocol=read(root/'protocol.json');qualification=read(root/'numerical_qualification.json')
    if (report['protocol_sha256']!=sha256(root/'protocol.json')
            or qualification['protocol_sha256']!=report['protocol_sha256']
            or report['numerical_qualification_sha256']!=sha256(root/'numerical_qualification.json')
            or fit['runtime_qualification_sha256']!=report['numerical_qualification_sha256']):
        raise ValueError('Changed readout numerical/calibration protocol')
    for path,digest in protocol['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed readout calibration evidence')
    q=qualification[name]
    if (not qualification['qualification_passed'] or q['backend']!='cpu' or q['calibration_parents_used']
            or q['queries']!=1787 or not all(q[k] for k in ('all_observed_graphs_checked',
               'all_nonadverse_outputs_exact','all_refit_logits_reproduced','all_frozen_weights_exact'))
            or q['prediction_sha256']!=sha256(root/name/'qualification_predictions.npz')
            or q['bundle_manifest_sha256']!=sha256(bundle/'manifest.json')
            or q['weights_sha256']!=meta['weights_sha256'] or fit['event_readout_refit']!=contract
            or fit['bundle_manifest_sha256']!=q['bundle_manifest_sha256']
            or fit['weights_sha256']!=q['weights_sha256'] or not fit['dense_readout_calibration']
            or fit['runtime_ready'] or fit['trajectory_gate_ready'] or fit['production_eligible']):
        raise ValueError('Unqualified or changed readout inference bundle')
    roles=fit['group_ids'];a=set(roles['prediction_fit']);b=set(roles['prediction_audit'])
    if len(a)!=64 or len(b)!=64 or a&b:raise ValueError('Readout calibration roles overlap')
    return dict(calibration_root=str(root),method=name,review_sha256=sha256(root/'independent_completion_review.json'))


def validate_gate(path, fit, bundle):
    """Bind an adaptive refit to its independently reviewed reference gate."""
    path=Path(path).resolve();root=path.parent.parent
    proof=read(root/'independent_completion_review.json');report=read(root/'report.json');gate=read(path)
    name=read(Path(bundle)/'manifest.json')['event_readout_refit']['method']
    flags=('review_passed','all_source_fit_and_model_bindings_checked',
           'all_parent_roles_and_disjointness_checked','all_physical_replay_bindings_checked',
           'all_raw_maxima_and_family_ranks_recomputed','all_reference_actions_fixed',
           'identical_reference_physics_across_models')
    if (not all(proof.get(k) is True for k in flags) or proof['report_sha256']!=sha256(root/'report.json')
            or proof['protocol_sha256']!=sha256(root/'inputs/protocol.json')
            or report['protocol_sha256']!=proof['protocol_sha256']
            or path!=Path(report['methods'][name]['gate']).resolve()
            or proof['methods'][name]['gate_sha256']!=sha256(path)
            or report['methods'][name]['gate_sha256']!=sha256(path)
            or gate['runtime_guidance']!=fit['runtime_guidance']
            or gate['weights_sha256']!=fit['weights_sha256']
            or gate['prediction_fit_sha256']!=sha256(Path(bundle).parent/'prediction_fit.json')):
        raise ValueError('Changed or unreviewed readout trajectory gate')
    return gate
