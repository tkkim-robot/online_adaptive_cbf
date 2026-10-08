"""Paired, unfiltered evaluation of necessary motor-box gain admission.

Preserves every scene, route, sensor seed, model, gate, score and query cadence
from the completed density comparison. Only the additional admission rule is
changed, identically for GAT and nearest-obstacle FC. This is development data.
"""


from pathlib import Path


from .dataset import sha256


from .quad3d_learning_contract import read
from . import quad3d_density_study as density

ORIGIN=Path('artifacts/experiments/quad3d_large_random_density_comparison')


SUPPORTED=Path('artifacts/experiments/quad3d_input_support_navigation_gpu_qualified')


def verify_source(root):
    from .quad3d_policy_experiment import INPUT_SUPPORT_SCHEMA
    root=Path(root);s=read(root/'manifest.json');origin=Path(s['origin'])
    if s.get('prospective_confirmation',False):
        from .quad3d_frozen_confirmation import verify_source as verify_confirmation
        return verify_confirmation(root,s)
    if s['schema']!=INPUT_SUPPORT_SCHEMA or s['training_use'] or s['weight_fit_authorized'] or s['final_test']:
        raise ValueError('Wrong frozen development scope')
    if not s['input_box_admission'] or s['reference_parents']!=0:raise ValueError('Wrong admission variant')
    if type(s.get('failure_requery',False)) is not bool:raise ValueError('Wrong failed-QP query mode')
    if s.get('failure_requery',False):
        if s['previous_comparison']!=str(SUPPORTED.resolve()):raise ValueError('Wrong paired reference')
        for name in ('comparison.json','outcomes.json','independent_completion_review.json'):
            if str((SUPPORTED/name).resolve()) not in s['evidence_bindings']:raise ValueError('Unbound preceding comparison')
    if origin.resolve()!=(ORIGIN/('pilot' if s['pilot'] else 'inputs')).resolve():raise ValueError('Wrong origin')
    if sha256(origin/'manifest.json')!=s['origin_manifest_sha256']:raise ValueError('Changed origin')
    prior=read(origin/'manifest.json');parents=density.verify_source(origin)
    for k in ('config','policy_config','models','gates','steps','capacity','batch','adaptive_parents',
              'slot_counts','pilot','pilot_indices','parents_sha256','preplanning_sha256','frozen_files'):
        if s[k]!=prior[k]:raise ValueError('Changed retained setting: '+k)
    for file,key in (('parents.json','parents_sha256'),('preplanning_parents.json','preplanning_sha256')):
        if sha256(root/file)!=s[key]:raise ValueError('Changed physical parents')
    for p,h in s['evidence_bindings'].items():
        if sha256(p)!=h:raise ValueError('Changed prior evidence: '+p)
    return parents
