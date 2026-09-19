"""Predeclared gain exploration for collection, never a deployed gain selector."""
from pathlib import Path
import numpy as np
from .quad3d_candidate_data import GAIN_BANK
from .quad3d_observation_audit import audit_parent
from .quad3d_control import control_config
from .quad3d_observation import unit_tape,numpy_observe,numpy_obstacles
from .quad3d_audit import independent_obstacle_values,independent_envelope_values


def schedule(seed,exploratory,initial_gain=2.,bank=GAIN_BANK):
    bank=np.asarray(bank,np.float64);gains=np.full((8,4),initial_gain,np.float64)
    if exploratory:
        rng=np.random.default_rng(np.random.SeedSequence([seed,9801]))
        gains[1:]=bank[rng.integers(0,len(bank),7)]
    return gains


def audit_exploration(arguments):
    parent,row,directory,manifest=arguments
    result=audit_parent(arguments)
    c=control_config(manifest['config'])
    with np.load(Path(directory)/row['file']) as z:d=dict(z)
    assert len(set(parent['gains']))==1
    expected=schedule(parent['seed'],parent['exploratory'],parent['gains'][0],manifest.get('gain_bank',GAIN_BANK))
    np.testing.assert_array_equal(parent['gain_schedule'],expected)
    # Gain proposals depend only on the predeclared seed and elapsed tick.
    x=np.asarray(parent['x']);o=np.asarray(parent['obstacles']);mask=np.asarray(parent['mask'])
    noise=np.asarray(parent['noise']);gain=np.asarray(parent['gains']);u=np.zeros(4)
    bx,bo,ix,io=unit_tape(parent['sensor_seed'],manifest['steps'],len(mask))
    for k in range(len(d['active'])):
        np.testing.assert_array_equal(d['previous_gain'][k],gain)
        np.testing.assert_array_equal(d['previous_control'][k],u)
        proposal=expected[k//200] if k%200==0 else gain
        np.testing.assert_array_equal(d['scheduled_gain'][k],proposal)
        true_o=o.copy();true_o[:,:2]+=k*c.robot.dt*o[:,3:5]
        seen,so=numpy_observe(d['state'][k],true_o,mask,bx,bo,noise,ix[k],io[k])
        controlled=numpy_obstacles(so,mask,noise)
        psi,_=independent_obstacle_values(seen,np.zeros(4),controlled,mask,proposal,c)
        domain,_=independent_envelope_values(seen,np.zeros(4),c)
        np.testing.assert_allclose(d['attempted_psi'][k],psi,atol=1e-8,rtol=1e-10)
        np.testing.assert_allclose(d['attempted_domain'][k],domain,atol=1e-8,rtol=1e-10)
        valid=bool(d['attempted_feasible'][k] and psi>=-c.qp_tolerance and domain>=-c.qp_tolerance)
        retry=not valid and np.any(proposal!=gain)
        assert retry==d['qp_switch_fallback'][k]
        np.testing.assert_array_equal(d['controller_gain'][k],gain if retry else proposal)
        if not retry:np.testing.assert_array_equal(d['proposed'][k],d['attempted_control'][k])
        if d['active'][k]:u=d['control'][k];gain=d['controller_gain'][k]
    result.update(exploration_history_verified=True,
        applied_gain_changes=int(np.sum(d['active']&np.any(d['controller_gain']!=d['previous_gain'],axis=-1))),
        qp_switch_fallbacks=int(d['qp_switch_fallback'].sum()))
    return result
