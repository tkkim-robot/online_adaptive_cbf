"""Observable geometric uncertainty compensation, independent of robot physics.

The synthetic sensor prior has coordinate bounds and 15% innovations. Triangle
inequality bounds the current clearance error by sqrt(2)*(robot_xy+obs_xy)
+ radius_error, all multiplied by 1.15. Applying this to a CBF radius compensates
current geometry; it is NOT a proof for noisy derivatives or sampled dynamics.
"""

import math
import jax.numpy as jnp


def controller_contract(sensor_margin_scale=0.,margin_guidance=False,shared_clearance_budget=False,motion_observer_window=0,filter_obstacle_position=False):
    value=float(sensor_margin_scale)
    if not math.isfinite(value) or value<0:raise ValueError('Sensor margin scale must be finite and nonnegative')
    if not isinstance(margin_guidance,bool):raise ValueError('Margin guidance must be boolean')
    if not isinstance(shared_clearance_budget,bool):raise ValueError('Shared clearance budget must be boolean')
    if shared_clearance_budget and not margin_guidance:raise ValueError('Shared clearance budget requires margin guidance')
    if isinstance(motion_observer_window,bool) or not isinstance(motion_observer_window,int) or not 0<=motion_observer_window<=512:
        raise ValueError('Motion observer window must be an integer between zero and512')
    if not isinstance(filter_obstacle_position,bool) or (filter_obstacle_position and not motion_observer_window):
        raise ValueError('Position filtering requires a motion observer and boolean opt-in')
    result=dict(sensor_margin_scale=value,margin_guidance=margin_guidance,shared_clearance_budget=shared_clearance_budget,motion_observer_window=motion_observer_window)
    # Preserve legacy manifest identities when this optional feature is absent.
    if filter_obstacle_position:result['filter_obstacle_position']=True
    return result


def clearance_inflation(noise,scale=1.):
    return scale*1.15*(jnp.sqrt(jnp.asarray(2.,noise.dtype))*(noise[0]+noise[3])+noise[5])


def require_matching_controller(model,calibration,scale,margin_guidance=False,shared_clearance_budget=False,motion_observer_window=0,filter_obstacle_position=False):
    expected=controller_contract(scale,margin_guidance,shared_clearance_budget,motion_observer_window,filter_obstacle_position)
    if controller_contract(**model.get('controller',{}))!=expected or controller_contract(**calibration.get('controller',{}))!=expected:
        raise ValueError('Controller/model/calibration mismatch; collect and calibrate matching targets')
