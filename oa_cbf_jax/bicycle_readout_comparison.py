"""Matched full-cohort comparison of original and refitted bicycle policies.

The original readouts, prediction fits and gates are restored together; this is
an ablation of the whole refit/calibration package, not the readout alone. No
weights, gates, physical settings, scenes or denominators are optimized here.
"""


from pathlib import Path


from .bicycle_experiment import read


from .dataset import sha256


SCHEMA='bicycle_original_readout_matched_navigation'


def validate_evidence(root):
    root=Path(root);proof=read(root/'independent_completion_review.json')
    flags=('review_passed','all_source_and_frozen_policy_bindings_checked',
           'new_parent_identifiers_and_seeds_disjoint','all_complete_physical_replay_bindings_checked',
           'all_raw_status_and_online_decision_counts_checked','all_strata_independently_recomputed',
           'paired_interval_independently_recomputed')
    if (not all(proof.get(k) is True for k in flags) or proof['comparison_sha256']!=sha256(root/'comparison.json')
            or proof['outcomes_sha256']!=sha256(root/'outcomes.json') or proof['protocol_sha256']!=sha256(root/'inputs/protocol.json')):
        raise ValueError('Completed independently reviewed navigation evidence required')
    return read(root/'inputs/protocol.json')


def validate_source(source):
    source=Path(source);m=read(source/'manifest.json');parents=read(source/'scenes.json')
    if (m.get('schema')!=SCHEMA or m.get('groups')!=768 or m.get('data_role')!='frozen_policy_margin_recovery_development'
            or m.get('training_use') is not False or m.get('weight_fit_authorized') is not False or m.get('final_test') is not False
            or sha256(m['reservation'])!=m['reservation_sha256'] or sha256(source/'scenes.json')!=m['scenes_sha256']):
        raise ValueError('Changed matched original-policy reservation')
    s=read(m['reservation']);evidence=Path(s['evidence']);refit=validate_evidence(evidence);info=s['models'][m['encoder']]
    if (len(parents)!=768 or sha256(source/'scenes.json')!=sha256(evidence/'inputs/scenes.json')
            or m['phase_order']!=read(Path(refit['models'][m['encoder']]['source'])/'manifest.json')['phase_order']
            or any(m[k]!=s[k] for k in ('config','controller','runtime_guidance'))
            or m['policy_config']!=info['policy_config'] or m['weights_sha256']!=sha256(Path(info['bundle'])/'weights.msgpack')
            or m['prediction_fit_sha256']!=sha256(info['prediction_fit']) or m['gate_sha256']!=sha256(info['gate'])):
        raise ValueError('Unequal matched parent or original policy')
    for path,digest in s['bindings'].items():
        if sha256(path)!=digest:raise ValueError('Changed matched comparison dependency')
    if any(p['partition']!='policy_audit' or p['calibration_role']!='none' or p['weight_fit_allowed'] for p in parents):
        raise ValueError('Matched navigation parents cannot fit weights or gates')
    return m,parents
