"""Common noisy physical environment for learned and nonlearned controllers.

The controller receives only sensed values. All methods share the same physical
prior, noise innovations, integration, route and limits. This initial evaluator
is synchronous; full control computation delay is a separate required gate.
"""

import jax
import jax.numpy as jnp
from .adaptive import Decision,REJECTED
from .dynamics import integrate_unicycle,signed_clearance,swept_disk_clearance
from .predictive import PREDICTIVE_REJECTED
from .route_control import route_control,INADMISSIBLE
from .routing import physical_route_coordinate
from .simulation import Summary,RUNNING,GOAL,COLLISION,INFEASIBLE,TIMEOUT
from .stochastic import conditioned_sensor_model,STATE_BOUND_VIOLATION
from .sensor_margin import clearance_inflation
from .motion_observer import initialize as initialize_observer,update as update_observer,effective_noise
from .position_observer import initialize as position_initialize,update as position_update

PLANNER_FAILURE=7


def make_closed_loop(policy,steps=800,batch_axis=None,ordered_waypoints=False):
    """Run one continuous physical episode, optionally through ordered goals.

    Ordered inputs use goal[L,2], points[L,C,2], route_mask[L,C] and
    route_ready[L]. Intermediate handoff uses only the current sensor reading
    and its known error bound. Physical state, noise, observers, previous action
    and gains are never restarted. Only the local route cursor changes at a leg
    boundary, which always triggers a fresh policy decision.
    """
    robot=policy.robot;config=policy.config;select=policy.selector
    def run(params,calibration,x0,goal,obstacles,mask,candidates,points,route_mask,noise,key,route_ready=True,waypoint_count=None):
        if ordered_waypoints:
            goals,all_points,all_masks,all_ready=goal,points,route_mask,jnp.asarray(route_ready)
            total=jnp.int32(goals.shape[0]) if waypoint_count is None else jnp.asarray(waypoint_count,jnp.int32)
            final_goal=goals[total-1]
            goal,points,route_mask,route_ready=goals[0],all_points[0],all_masks[0],all_ready[0]
        else:final_goal=goal
        uncertainty=clearance_inflation(noise,config.sensor_margin_scale)
        x0,truth_obs,x_bias,obs_bias,x_scale,obs_scale,innovations=conditioned_sensor_model(x0,obstacles,mask,noise,key,robot,steps)
        initial_clearance=jnp.min(signed_clearance(x0[:2],truth_obs,mask,robot.radius))
        at_goal=(jnp.linalg.norm(x0[:2]-goal)<=robot.goal_tolerance)&(jnp.abs(x0[3])<=.2)
        if ordered_waypoints:at_goal=at_goal&(total==1)
        status=jnp.where(jnp.logical_not(route_ready),PLANNER_FAILURE,jnp.where(initial_clearance<=0,COLLISION,jnp.where(at_goal,GOAL,RUNNING)))
        initial_gain=jnp.asarray(config.fixed_gain,x0.dtype)
        empty=Decision(initial_gain,jnp.asarray(True),jnp.int32(REJECTED),jnp.zeros(7,jnp.int32),jnp.int32(0),jnp.int32(0),
                       jnp.asarray(jnp.nan,x0.dtype),jnp.asarray(jnp.nan,x0.dtype),jnp.full(2,jnp.nan))
        def tick(carry,inputs):
            x,status,count,clearance,psi_min,violation_max,cursor,gain,previous_control,last_decision,observer,position,leg=carry
            k,innovation=inputs;active=status==RUNNING;now=k*robot.dt
            sensed_x=x-x_bias+.15*x_scale*innovation[:4]
            handoff=jnp.asarray(False)
            current_goal,current_points,current_mask=goal,points,route_mask
            if ordered_waypoints:
                # The triangle inequality guarantees physical arrival under the
                # stated componentwise sensor bounds. No latent position input.
                observed_arrival=(jnp.linalg.norm(sensed_x[:2]-goals[leg])+jnp.sqrt(2.)*1.15*noise[0]<=robot.goal_tolerance)&(jnp.abs(sensed_x[3])+1.15*noise[2]<=.2)
                handoff=active&observed_arrival&(leg<total-1)
                leg=leg+handoff.astype(jnp.int32)
                cursor=jnp.where(handoff,jnp.asarray(0.,cursor.dtype),cursor)
                current_goal,current_points,current_mask=goals[leg],all_points[leg],all_masks[leg]
                status=jnp.where(active&~all_ready[leg],PLANNER_FAILURE,status)
                active=status==RUNNING
            physical_obs=truth_obs.at[:,:2].set(truth_obs[:,:2]+now*truth_obs[:,3:5])
            sensed_obs=physical_obs-obs_bias+.15*obs_scale*innovation[4:].reshape(obstacles.shape)
            raw_sensed_obs=sensed_obs
            feature_noise=noise
            bad=jnp.zeros_like(mask)
            if config.motion_observer_window:
                observer,sensed_obs,bounds,lag,bad=update_observer(observer,raw_sensed_obs,mask,noise,robot.dt,config.motion_observer_window)
                feature_noise=effective_noise(noise,bounds,mask)
            if config.filter_obstacle_position:
                position,sensed_obs,position_bad=position_update(position,sensed_obs,sensed_obs[:,3:5],
                    1.15*feature_noise[4],mask,noise,robot.dt)
                bad=bad|jnp.any(position_bad,axis=-1)
            propose=lambda _:select(params,calibration,sensed_x,current_goal,sensed_obs,mask,candidates,current_points,current_mask,cursor,gain,previous_control,feature_noise)
            reactive=jnp.asarray(False)
            if config.reactive_reselection:
                held_qp,held_h,held_psi,_,_,_=route_control(sensed_x,current_goal,sensed_obs,mask,gain,current_points,current_mask,cursor,robot,1.15*noise[2],uncertainty,config.margin_guidance,config.shared_clearance_budget)
                reactive=active&(~held_qp.feasible|(held_h < -robot.qp_tolerance)|(held_psi < -robot.qp_tolerance))
            # In a named batch, one scalar "any event" predicate preserves the
            # actual conditional under vmap. Recompute the batch only when one
            # scene needs it, and discard proposals for every non-triggering
            # scene. This does not change the per-scene policy/update schedule.
            reselection=reactive|handoff
            batch_event=(jax.lax.pmax(reselection.astype(jnp.int32),batch_axis)>0) if batch_axis is not None else reselection
            batch_active=(jax.lax.pmax(active.astype(jnp.int32),batch_axis)>0) if batch_axis is not None else active
            def between_updates(_):
                proposal=jax.lax.cond(batch_event,propose,lambda _:last_decision,operand=None)
                return jax.tree.map(lambda new,old:jnp.where(reselection,new,old),proposal,last_decision)
            decision=jax.lax.cond((k%config.interval==0)&batch_active,propose,
                between_updates,operand=None)
            qp,h,psi,proposed_cursor,remaining,target=route_control(sensed_x,current_goal,sensed_obs,mask,decision.gains,current_points,current_mask,cursor,robot,1.15*noise[2],uncertainty,config.margin_guidance,config.shared_clearance_budget)
            admissible=(h>=-robot.qp_tolerance)&(psi>=-robot.qp_tolerance)
            can_step=active&decision.accepted&qp.feasible&admissible
            u=jnp.where(can_step,qp.control,jnp.zeros(2,x.dtype))
            y,sub=integrate_unicycle(x,u,robot.dt,robot.integration_substeps)
            starts=jnp.concatenate((x[None],sub[:-1]));times=now+jnp.arange(robot.integration_substeps)*robot.dt/robot.integration_substeps
            clear=jnp.min(jax.vmap(lambda a,b,t:swept_disk_clearance(a,b,truth_obs,mask,robot.radius,t,
                              t+robot.dt/robot.integration_substeps))(starts,sub,times))
            bound=jnp.maximum(-y[3],y[3]-robot.v_max)
            reached=can_step&(jnp.linalg.norm(y[:2]-current_goal)<=robot.goal_tolerance)&(jnp.abs(y[3])<=.2)
            if ordered_waypoints:reached=reached&(leg==total-1)
            status=jnp.where(active&~decision.accepted,PREDICTIVE_REJECTED,status)
            status=jnp.where(active&decision.accepted&~qp.feasible,INFEASIBLE,status)
            status=jnp.where(active&decision.accepted&~admissible,INADMISSIBLE,status)
            status=jnp.where(reached,GOAL,status)
            status=jnp.where(can_step&(bound>robot.qp_tolerance),STATE_BOUND_VIOLATION,status)
            status=jnp.where(can_step&(clear<=0),COLLISION,status)
            new_x=jnp.where(can_step,y,x);cursor=jnp.where(can_step,proposed_cursor,cursor)
            gain=jnp.where(can_step,decision.gains,gain);previous_control=jnp.where(can_step,u,previous_control)
            committed=jax.tree.map(lambda new,old:jnp.where(can_step,new,old),decision,last_decision)
            count+=can_step.astype(jnp.int32)
            clearance=jnp.minimum(clearance,jnp.where(can_step,clear,jnp.inf))
            psi_min=jnp.minimum(psi_min,jnp.where(active&decision.accepted,psi,jnp.inf))
            violation_max=jnp.maximum(violation_max,jnp.where(can_step,qp.max_violation,-jnp.inf))
            trace=dict(state=new_x,control=u,active=can_step,status=status,gains=gain,route_progress=cursor,
                       observed_state=sensed_x,observed_obstacles=sensed_obs,raw_observed_obstacles=raw_sensed_obs,
                       effective_noise=feature_noise,observer_inconsistent=bad,route_target=target,route_remaining=remaining,
                       clearance=jnp.where(can_step,clear,jnp.nan),state_bound_violation=jnp.where(can_step,bound,jnp.nan),
                       qp_violation=jnp.where(can_step,qp.max_violation,jnp.nan),psi1=jnp.where(active&decision.accepted,psi,jnp.nan),
                       selection_tick=active&((k%config.interval==0)|reselection),reactive_reselection=reactive&(k%config.interval!=0),selection_accepted=decision.accepted,
                       proposed_source=decision.source,applied_source=jnp.where(can_step,decision.source,REJECTED),
                       gate_stages=decision.stages,primary_valid=decision.primary_valid,backup_valid=decision.backup_valid,
                       selected_cvar=decision.risk,selected_cs=decision.disagreement,selected_event_probability=decision.event_probability)
            if ordered_waypoints:
                trace.update(waypoint_index=leg,waypoint_handoff=handoff,waypoints_visited=leg+(status==GOAL).astype(jnp.int32),mission_goal=current_goal)
            return (new_x,status,count,clearance,psi_min,violation_max,cursor,gain,previous_control,committed,observer,position,leg),trace
        initial=(x0,status,jnp.int32(0),initial_clearance,jnp.asarray(jnp.inf,x0.dtype),jnp.asarray(-jnp.inf,x0.dtype),
                 jnp.asarray(0.,x0.dtype),initial_gain,jnp.zeros(2,x0.dtype),empty,initialize_observer(obstacles),position_initialize(obstacles) if config.filter_obstacle_position else None,jnp.int32(0))
        (x,status,count,clearance,psi,violation,cursor,_,_,_,_,_,leg),trace=jax.lax.scan(tick,initial,(jnp.arange(steps),innovations))
        status=jnp.where(status==RUNNING,TIMEOUT,status)
        summary=Summary(x,status,count,clearance,psi,jnp.linalg.norm(x0[:2]-final_goal)-jnp.linalg.norm(x[:2]-final_goal),violation)
        if ordered_waypoints:
            points,route_mask=all_points[leg],all_masks[leg]
        truth=dict(initial_state=x0,obstacles=truth_obs,x_bias=x_bias,obs_bias=obs_bias,route_progress=physical_route_coordinate(x[:2],points,route_mask,cursor))
        if ordered_waypoints:
            truth.update(waypoint_index=leg,waypoints_visited=leg+(status==GOAL).astype(jnp.int32))
        return summary,trace,truth
    return run
