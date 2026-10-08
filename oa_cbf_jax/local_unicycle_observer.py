"""Causal position filtering for the local STATIC-obstacle sensor model.

The observer receives only present measurements, previous estimates and applied
controls. Tracks have stable identities. Heading, speed and radii retain the
existing exact-measurement assumption. A fixed innovation weight does not remove
constant bias or provide a bound for Gaussian errors. No uncertainty/safety
guarantee is inferred from the variance reduction.
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from .local_unicycle_collection import ROBOT,K
from .local_unicycle_policy import qp_control,empty_decision
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .obstacle_selection import nearest_obstacles

WEIGHT=.25


class Memory(NamedTuple):
    predicted_position:jax.Array
    obstacle_centers:jax.Array
    ready:jax.Array


def initialize(raw_state,raw_world):
    # No physical truth is accepted, including at initialization.
    return Memory(jnp.zeros_like(raw_state[:2]),jnp.zeros_like(raw_world[:,:2]),jnp.bool_(False))


def observe(memory,raw_state,raw_world):
    position=jnp.where(memory.ready,memory.predicted_position+WEIGHT*(raw_state[:2]-memory.predicted_position),raw_state[:2])
    centers=jnp.where(memory.ready,memory.obstacle_centers+WEIGHT*(raw_world[:,:2]-memory.obstacle_centers),raw_world[:,:2])
    return raw_state.at[:2].set(position),raw_world.at[:,:2].set(centers)


def advance(observed,seen,control,applied):
    prediction,_=integrate_unicycle(observed,control,ROBOT.dt,ROBOT.integration_substeps)
    return Memory(jnp.where(applied,prediction[:2],observed[:2]),seen[:,:2],jnp.bool_(True))


def reference_rollout(initial,world,goal,errors,gain,memory=None,prior_status=0):
    """Fixed-gain diagnostic; learned bundles require new observer-aware labels.

    Copy memory BEFORE the current reading when branching from an acquired
    state; never reset it or process that same reading twice.
    """
    def tick(carry,inputs):
        x,status,done,minimum,mem=carry;k,error=inputs
        raw=x.at[:2].add(error[0]);raw_world=world.at[:,:2].add(error[1:])
        observed,seen=observe(mem,raw,raw_world)
        rows,mask,ids=nearest_obstacles(observed[:2],seen,jnp.ones(len(world),bool),K)
        qp,admissible=qp_control(observed,goal,rows,mask,gain)
        active=(status==0)&qp.feasible&admissible;u=jnp.where(active,qp.control,jnp.zeros(2,jnp.float32))
        y,sub=integrate_unicycle(x,u,ROBOT.dt,ROBOT.integration_substeps)
        starts=jnp.concatenate((x[None],sub[:-1]))
        clear=jnp.min(jax.vmap(lambda a,b:swept_disk_clearance(a,b,world,jnp.ones(len(world),bool),ROBOT.radius,0.,0.))(starts,sub))
        bounds=jnp.max(jnp.maximum(-sub[:,3],sub[:,3]-ROBOT.v_max))
        reached=(jnp.linalg.norm(y[:2]-goal)<=ROBOT.goal_tolerance)&(jnp.abs(y[3])<=.2)
        ns=jnp.where((status==0)&~qp.feasible,3,status);ns=jnp.where((status==0)&~admissible,5,ns)
        ns=jnp.where(active&reached,1,ns);ns=jnp.where(active&(bounds>ROBOT.qp_tolerance),8,ns)
        ns=jnp.where(active&(clear<=0.),2,ns)
        state=jnp.where(active,y,x);minimum=jnp.minimum(minimum,jnp.where(active,clear,jnp.inf))
        record=dict(before=x,state=state,observed=observed,observed_world=seen,control=u,active=active,status=ns,
            selected_ids=ids,feasible=qp.feasible,admissible=admissible,clearance=jnp.where(active,clear,0.),gain=gain,
            query=(status==0)&(k%4==0),observer_prediction=mem.predicted_position,
            observer_centers=mem.obstacle_centers,observer_ready=mem.ready,**empty_decision(2))
        return (state,ns,done+active.astype(jnp.int32),minimum,advance(observed,seen,u,active)),record
    if memory is None:memory=initialize(initial,world)
    start=(initial,jnp.asarray(prior_status,jnp.int32),jnp.int32(0),
        jnp.min(signed_clearance(initial[:2],world,jnp.ones(len(world),bool),ROBOT.radius)),memory)
    final,trace=jax.lax.scan(tick,start,(jnp.arange(len(errors)),errors))
    state,status,steps,clearance,_=final
    return dict(final_state=state,status=jnp.where(status==0,4,status),steps=steps,min_clearance=clearance,
        progress=jnp.linalg.norm(goal-initial[:2])-jnp.linalg.norm(goal-state[:2])),trace


def audit_observations(world,errors,trace,memory=None):
    """Independent recurrence and exact-input Gauss quadrature, not JAX replay.

    Audit each transition using recorded rounded previous observations; no
    cumulative tolerance growth. Existing2e-6 observation tolerance retained.
    """
    x=np.asarray(trace['before'],np.float32);u=np.asarray(trace['control'],np.float32)
    raw=x.copy();raw[:,:2]+=errors[:,0]
    worlds=np.broadcast_to(np.asarray(world,np.float32),(len(x),*world.shape)).copy();worlds[:,:,:2]+=errors[:,1:]
    observed=np.asarray(trace['observed'],np.float32);seen=np.asarray(trace['observed_world'],np.float32)
    # Heading/speed/radius/velocity are current measurements, never estimates.
    np.testing.assert_array_equal(observed[:,2:],raw[:,2:]);np.testing.assert_array_equal(seen[:,:,2:],worlds[:,:,2:])
    nodes,weights=np.polynomial.legendre.leggauss(16);tt=(nodes+1)*ROBOT.dt/2
    v=observed[:,3,None].astype(float)+u[:,0,None]*tt;theta=observed[:,2,None].astype(float)+u[:,1,None]*tt
    displacement=ROBOT.dt/2*np.column_stack((np.sum(weights*v*np.cos(theta),-1),np.sum(weights*v*np.sin(theta),-1)))
    # Compare to the independent unrounded integral. Rounding this reference
    # to FP32 before comparison can spuriously introduce a full-ULP jump when
    # two accurate integrators straddle a rounding midpoint (e.g. x>32m).
    predicted=observed[:,:2].astype(float)+displacement
    predicted=np.where(np.asarray(trace['active'])[:,None],predicted,observed[:,:2])
    prior_position=np.zeros(2,np.float32) if memory is None else np.asarray(memory.predicted_position,np.float32)
    prior_centers=np.zeros_like(world[:,:2]) if memory is None else np.asarray(memory.obstacle_centers,np.float32)
    ready=np.r_[False if memory is None else bool(memory.ready),np.ones(len(x)-1,bool)]
    predicted=np.concatenate((prior_position[None],predicted[:-1]));centers=np.concatenate((prior_centers[None],seen[:-1,:,:2]))
    np.testing.assert_allclose(trace['observer_prediction'],predicted,atol=2e-6,rtol=0)
    # The next operation consumed the recorded, rounded prediction. Verify
    # that operation separately after checking its predictor independently.
    rounded_prediction=np.asarray(trace['observer_prediction'],np.float32)
    expect_position=np.where(ready[:,None],rounded_prediction+WEIGHT*(raw[:,:2]-rounded_prediction),raw[:,:2])
    expect_centers=np.where(ready[:,None,None],centers+WEIGHT*(worlds[:,:,:2]-centers),worlds[:,:,:2])
    np.testing.assert_allclose(observed[:,:2],expect_position,atol=2e-6,rtol=0)
    np.testing.assert_allclose(seen[:,:,:2],expect_centers,atol=2e-6,rtol=0)
    np.testing.assert_array_equal(trace['observer_centers'],centers);np.testing.assert_array_equal(trace['observer_ready'],ready)
    return dict(maximum_observer_error=float(max(np.max(np.abs(observed[:,:2]-expect_position)),np.max(np.abs(seen[:,:,:2]-expect_centers)))),
        maximum_prediction_error=float(np.max(np.abs(rounded_prediction-predicted))),
        weight=WEIGHT,independent_observer_audit_passed=True)


def contract():
    return dict(schema='local_static_causal_position_filter_v1',measurement_weight=WEIGHT,
        initialization='First raw readings; zero hidden truth initialization.',
        ego_prediction='Previous filtered position, measured heading/speed, actual applied input and existing dynamics.',
        obstacles='Static tracks with persistent identities; same causal preprocessing for both encoders and QPs.',
        observations='Only noisy positions filtered; existing exact heading/speed/radii retained.',
        limitations='No constant-bias removal, process/motion/identity-error guarantee or Gaussian hard error bound.',
        gain_adaptation_seconds=.2,held_label_seconds=8,guards_changed=False)
