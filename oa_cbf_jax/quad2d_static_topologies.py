"""Reserve fresh static topology parents without fitting or outcome filtering.

The eight templates predate this comparison and have historical development use.
Only their geometry is reused: physical velocities, including latent noisy truth,
are stationary. These are fresh forward parents, not untouched final families.
"""


import hashlib


import numpy as np


from .quad2d_ood_scenes import FAMILIES as TEMPLATES, geometry


FAMILIES = ('alternating_gates', 'zigzag_channel', 'offset_rooms',
            'nested_open_boxes', 'interleaved_rows', 'three_lanes',
            'narrow_gate', 'large_disks')


def seeded_fields(seed, index):
    local = seed*100000+index
    x, goal, obstacles, mask = geometry(local, TEMPLATES[index % 8])
    obstacles[:, 3:5] = 0.
    rng = np.random.default_rng(local+233)
    noise = float(rng.choice([0., .5, 1., 2.]))*np.array(
        [.015, .01, .015, .015, .02, .02, .008], np.float32)
    fingerprint = hashlib.sha256(b''.join(a.tobytes() for a in
        (x, goal, obstacles, mask, noise))).hexdigest()
    family = FAMILIES[index % 8]
    return dict(group_id=f'quad2d_static_topology:{family}:{local}',
        family=family, template=TEMPLATES[index % 8], seed=local,
        partition='frozen_forward_topology', initial_state=x.tolist(), goal=goal.tolist(),
        obstacles=obstacles.tolist(), obstacle_mask=mask.tolist(), noise=noise.tolist(),
        solvability='unknown', scene_fingerprint=fingerprint)
