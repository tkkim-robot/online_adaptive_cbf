"""Headless fixed-gain simulation with recorded applied controls and events.

This synchronous engine explicitly excludes wall-clock compute delay; a separate
delay-realistic evaluator is required before real-time safety claims.
"""

from functools import partial

from typing import NamedTuple

import jax

import jax.numpy as jnp

from .config import UnicycleConfig

from .controllers import control_unicycle

from .dynamics import integrate_unicycle, signed_clearance, swept_disk_clearance

RUNNING, GOAL, COLLISION, INFEASIBLE, TIMEOUT = range(5)

STATUS_NAMES = {RUNNING:"running",GOAL:"goal_reached",COLLISION:"collision",INFEASIBLE:"solver_infeasible",TIMEOUT:"timeout"}

class Summary(NamedTuple):
    final_state: jax.Array
    status: jax.Array
    steps: jax.Array
    min_clearance: jax.Array
    min_psi1: jax.Array
    progress: jax.Array
    worst_qp_violation: jax.Array

@partial(jax.jit, static_argnames=("config","steps"))
def rollout_fixed(x0, goal, obstacles, mask, alpha, config=UnicycleConfig(), steps=400):
    initial_clearance = jnp.min(signed_clearance(x0[:2],obstacles,mask,config.radius))
    initial_status = jnp.where(initial_clearance<=0,COLLISION,RUNNING)
    def tick(carry, k):
        x, status, count, clearance, psi_min, violation_max = carry
        active = status == RUNNING
        now = k * config.dt
        current_obs = obstacles.at[:,:2].set(obstacles[:,:2]+now*obstacles[:,3:5])
        qp, _, psi = control_unicycle(x,goal,current_obs,mask,alpha,config)
        # A failed solve terminates explicitly; no nominal/emergency action is
        # silently substituted and counted as a successful CBF control step.
        can_step = active & qp.feasible
        u = jnp.where(can_step,qp.control,jnp.zeros(2,dtype=x.dtype))
        y, sub = integrate_unicycle(x,u,config.dt,config.integration_substeps)
        starts = jnp.concatenate((x[None,:],sub[:-1]))
        times = now + jnp.arange(config.integration_substeps)*config.dt/config.integration_substeps
        swept = jax.vmap(lambda a,b,t: swept_disk_clearance(a,b,obstacles,mask,config.radius,t,t+config.dt/config.integration_substeps))(starts,sub,times)
        step_clearance = jnp.min(swept)
        collided = can_step & (step_clearance<=0)
        reached = can_step & (jnp.linalg.norm(y[:2]-goal)<=config.goal_tolerance) & (jnp.abs(y[3])<=.2)
        status = jnp.where(active & ~qp.feasible,INFEASIBLE,status)
        status = jnp.where(reached,GOAL,status)
        status = jnp.where(collided,COLLISION,status)
        new_x = jnp.where(can_step,y,x)
        count = count + can_step.astype(jnp.int32)
        clearance = jnp.minimum(clearance,jnp.where(can_step,step_clearance,jnp.inf))
        psi_min = jnp.minimum(psi_min,jnp.where(active,psi,jnp.inf))
        violation_max = jnp.maximum(violation_max,jnp.where(can_step,qp.max_violation,-jnp.inf))
        trace = dict(state=new_x,control=u,active=can_step,status=status,clearance=jnp.where(can_step,step_clearance,jnp.nan),
                     qp_violation=jnp.where(can_step,qp.max_violation,jnp.nan),psi1=jnp.where(active,psi,jnp.nan))
        return (new_x,status,count,clearance,psi_min,violation_max),trace
    initial = (x0,initial_status,jnp.int32(0),initial_clearance,jnp.asarray(jnp.inf,x0.dtype),jnp.asarray(-jnp.inf,x0.dtype))
    (x,status,count,clearance,psi_min,violation_max),trace = jax.lax.scan(tick,initial,jnp.arange(steps))
    status = jnp.where(status==RUNNING,TIMEOUT,status)
    progress = jnp.linalg.norm(x0[:2]-goal)-jnp.linalg.norm(x[:2]-goal)
    return Summary(x,status,count,clearance,psi_min,progress,violation_max),trace
