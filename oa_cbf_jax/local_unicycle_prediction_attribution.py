"""Attribute a frozen-policy regression on identical observed reference visits.

This is an offline diagnostic, not a deployable hybrid policy. It replaces one
physical-unit prediction component at a time in either direction between two
reviewed ensembles. Every selected gain is evaluated using already collected,
independently audited eight-second labels. No fitting or forward benchmark use.
"""


import json
from pathlib import Path


from .dataset import sha256


def read(path):
    return json.loads(Path(path).read_text())


def reviewed_navigation(path):
    path = Path(path)
    spec, done = read(path/'protocol.json'), read(path/'complete.json')
    root = Path(done['output'])
    proof = read(root/'independent_completion_review.json')
    if (read(path/'job.json')['status'] != 'completed' or done['status'] != 'completed'
            or not proof['independently_reproduced']
            or done['review_sha256'] != sha256(root/'review.json')
            or proof['review_sha256'] != done['review_sha256']):
        raise ValueError('Complete independently reviewed navigation required')
    bound = {str(p.resolve()): sha256(p) for p in (
        path/'protocol.json', path/'complete.json', root/'review.json',
        root/'independent_completion_review.json')}
    return spec, bound


def check_bindings(bindings):
    for path, digest in bindings.items():
        if sha256(path) != digest:
            raise ValueError(f'Changed diagnostic source: {path}')
