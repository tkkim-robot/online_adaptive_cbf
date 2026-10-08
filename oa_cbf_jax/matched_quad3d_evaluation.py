"""Matched static-scene Quad3D calibration and all-default-baseline evaluation."""


from collections import Counter


import numpy as np


def summarize(rows, steps):
    return dict(trials=len(rows), successes=sum(r['audited_status']==1 for r in rows),
        success_rate=float(np.mean([r['audited_status']==1 for r in rows])),
        collisions=sum(r['physical_collision'] for r in rows),
        unsafe_rate=float(np.mean([bool(r['physical_collision'] or r['envelope_exit']) for r in rows])),
        envelope_exits=sum(r['envelope_exit'] for r in rows),
        statuses=dict(Counter(str(r['audited_status']) for r in rows)),
        gain_changes=sum(r.get('applied_gain_changes',0) for r in rows),
        episodes_with_gain_changes=sum(r.get('applied_gain_changes',0)>0 for r in rows),
        accepted_learned_queries=sum(r.get('learned_queries',0) for r in rows),
        failure_capped_completion_seconds=float(np.mean([r['steps']*.05 if r['audited_status']==1 else steps*.05 for r in rows])))
