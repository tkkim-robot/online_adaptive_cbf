"""Join frozen OA, single-obstacle FC and original native evidence by parent.

This does not run, fit, select or censor physical trials. Old full-scene FC
results remain in their original reports and never enter the paper FC column.
"""


import numpy as np


def key(row):
    parent = row.get('scene_id', row.get('group_id', row.get('id')))
    if parent is None: raise ValueError('Missing physical parent identity')
    # Unicycle repeats each parent under four paired sensor-noise conditions.
    return (parent, float(row['noise_scale'])) if 'scene_id' in row else (parent,)


def success(row):
    return row.get('audited_status', row['status']) in (1, 'goal_reached')


def paired(left, right, seed=270925, samples=10000):
    a={key(r):r for r in left};b={key(r):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):
        raise ValueError('Missing, duplicate or unmatched trial denominator')
    if any(a[k]['family']!=b[k]['family'] for k in a):raise ValueError('Changed paired family')
    keys=sorted(a); delta=np.array([int(success(a[k]))-int(success(b[k])) for k in keys])
    # Resample whole parents within family, retaining every noise condition.
    clusters={}
    for k,d in zip(keys,delta):
        clusters.setdefault((a[k]['family'],k[0]),[]).append(int(d))
    rng=np.random.default_rng(seed);numerator=np.zeros(samples);denominator=np.zeros(samples)
    for family in sorted({x[0] for x in clusters}):
        values=[v for (f,_),v in clusters.items() if f==family]
        sums=np.array([sum(v) for v in values]);sizes=np.array([len(v) for v in values])
        indices=rng.integers(0,len(values),size=(samples,len(values)))
        numerator+=sums[indices].sum(1);denominator+=sizes[indices].sum(1)
    return dict(success_difference_pp=float(delta.mean()*100),
        family_stratified_parent_bootstrap95_pp=np.percentile(100*numerator/denominator,[2.5,97.5]).tolist(),
        oa_only=int((delta==1).sum()),comparator_only=int((delta==-1).sum()),
        parent_clusters=len(clusters),paired_trials=len(keys),bootstrap_seed=seed,bootstrap_samples=samples,
        scope='Descriptive paired uncertainty on reviewed development cohorts; not an untouched confirmatory test.')
