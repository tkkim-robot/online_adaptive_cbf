"""Observation-only clearance lower bound for one held rotor command.

The reference center follows observed position/velocity and constant initial
acceleration. Lipschitz continuity of the thrust direction bounds its error
under the actual nonlinear pitch dynamics. This is a next-command check, not
recursive feasibility, a future feedback tube, or safety after rejection.
"""
import jax.numpy as jnp
from .quad2d_control import FlightConfig


NUMERICAL_ALLOWANCE = 2e-5


def current_clearance_uncertainty(noise):
    n=jnp.asarray(noise,jnp.float64)
    return 1.15*(jnp.sqrt(jnp.float64(2.))*(n[0]+n[4])+n[6])


def command_clearance_bound(x,u,obs,mask,noise,config=FlightConfig()):
    """Conservative disk separation through dt for a constant actual input.

    For time t, center error <= initial position + relative velocity error*t
    + F/m*[pitch_error*t²/2 + (|pitch_rate|+rate_error)*t³/6
    + |angular_acceleration|*t⁴/24]. Radius error is added separately.
    Each quadratic-center chord subtracts |initial acceleration|*subdt²/8.
    The fixed numerical allowance covers the tested FP32 physical replay
    tolerance; it is an engineering allowance, not an interval solver proof.
    """
    c=config.robot;x=jnp.asarray(x,jnp.float64);u=jnp.asarray(u,jnp.float64)
    obs=jnp.asarray(obs,jnp.float64);n=jnp.asarray(noise,jnp.float64)
    total=jnp.sum(jnp.abs(u))/c.mass
    acceleration=jnp.array([-jnp.sin(x[2])*jnp.sum(u)/c.mass,
                            jnp.cos(x[2])*jnp.sum(u)/c.mass-c.gravity])
    angular=jnp.abs(c.arm/c.inertia*(u[0]-u[1]))
    times=jnp.arange(c.integration_substeps+1,dtype=jnp.float64)*c.dt/c.integration_substeps
    centers=x[:2]+times[:,None]*x[3:5]+.5*times[:,None]**2*acceleration
    relative=centers[:,None,:]-obs[None,:,:2]-times[:,None,None]*obs[None,:,3:5]
    start=relative[:-1];delta=relative[1:]-start
    fraction=jnp.clip(-jnp.sum(start*delta,axis=-1)/jnp.maximum(jnp.sum(delta*delta,axis=-1),1e-24),0.,1.)
    distance=jnp.linalg.norm(start+fraction[...,None]*delta,axis=-1)-c.radius-obs[None,:,2]
    t=times[1:]
    error=current_clearance_uncertainty(n)+1.15*jnp.sqrt(jnp.float64(2.))*(n[2]+n[5])*t
    error+=total*(1.15*n[1]*t**2/2+(jnp.abs(x[5])+1.15*n[3])*t**3/6+angular*t**4/24)
    sag=jnp.linalg.norm(acceleration)*(c.dt/c.integration_substeps)**2/8
    lower=jnp.min(jnp.where(mask[None,:],distance-error[:,None]-sag-NUMERICAL_ALLOWANCE,jnp.inf))
    nominal=jnp.min(jnp.where(mask[None,:],distance,jnp.inf))
    return dict(lower_clearance=lower,nominal_clearance=nominal,uncertainty=error[-1],curvature_allowance=sag)


def numpy_command_clearance_bounds(x,u,obs,mask,noise,config=FlightConfig()):
    """Independent vectorized NumPy reconstruction for saved applied inputs."""
    import numpy as np
    c=config.robot;x=np.asarray(x,float);u=np.asarray(u,float);obs=np.asarray(obs,float);n=np.asarray(noise,float)
    times=np.linspace(0,c.dt,c.integration_substeps+1);t=times[1:]
    total=np.abs(u).sum(axis=-1)/c.mass
    acceleration=np.column_stack((-np.sin(x[:,2])*u.sum(axis=-1)/c.mass,
                                   np.cos(x[:,2])*u.sum(axis=-1)/c.mass-c.gravity))
    centers=x[:,None,:2]+times[None,:,None]*x[:,None,3:5]+.5*times[None,:,None]**2*acceleration[:,None,:]
    relative=centers[:,:,None,:]-obs[:,None,:,:2]-times[None,:,None,None]*obs[:,None,:,3:5]
    start=relative[:,:-1];delta=relative[:,1:]-start
    fraction=np.clip(-np.sum(start*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),1e-24),0,1)
    distance=np.linalg.norm(start+fraction[...,None]*delta,axis=-1)-c.radius-obs[:,None,:,2]
    error=1.15*(np.sqrt(2)*(n[0]+n[4])+n[6])+1.15*np.sqrt(2)*(n[2]+n[5])*t[None,:]
    angular=np.abs(c.arm/c.inertia*(u[:,0]-u[:,1]))
    error=error+total[:,None]*(1.15*n[1]*t[None,:]**2/2+(np.abs(x[:,5])+1.15*n[3])[:,None]*t[None,:]**3/6+angular[:,None]*t[None,:]**4/24)
    sag=np.linalg.norm(acceleration,axis=-1)*(c.dt/c.integration_substeps)**2/8
    return dict(lower_clearance=np.min(np.where(mask[None,None,:],distance-error[:,:,None]-sag[:,None,None]-NUMERICAL_ALLOWANCE,np.inf),axis=(1,2)),
        nominal_clearance=np.min(np.where(mask[None,None,:],distance,np.inf),axis=(1,2)),uncertainty=error[:,-1],curvature_allowance=sag)


def check_guard_trace(data,guidance,config=FlightConfig(),noise=None,mask=None,gains=None):
    import numpy as np
    if guidance['clearance_guard']!='one_step_v1' or type(guidance['hard_prediction_clearance']) is not bool:
        raise ValueError('Unknown clearance-guard contract')
    active=np.asarray(data['active'],bool);noise=np.asarray(data.get('noise') if noise is None else noise,float)
    mask=np.asarray(data.get('obstacle_mask') if mask is None else mask,bool)
    if mask.shape!=(data['observed_obstacles'].shape[1],):raise ValueError('Missing observed obstacle mask for guard audit')
    expected=numpy_command_clearance_bounds(data['observed_state'][active],data['control'][active],
        data['observed_obstacles'][active],mask,noise,config)
    for key,value in expected.items():
        np.testing.assert_allclose(data['guidance_guard_'+key][active],value,atol=2e-9,rtol=2e-9)
    if not np.all(expected['lower_clearance']>0):raise ValueError('Applied command failed observation clearance bound')
    floor=1.15*(np.sqrt(2)*(noise[0]+noise[4])+noise[6]) if guidance['hard_prediction_clearance'] else 0.
    np.testing.assert_allclose(data['guidance_guard_prediction_minimum'][active],floor,atol=2e-9,rtol=2e-9)
    if not np.all(data['guidance_predicted_clearance'][active]>floor):raise ValueError('Applied profile failed declared predictive clearance filter')
    if 'cbf_clearance_inflation' in guidance:
        if guidance['cbf_clearance_inflation']!='current_position_radius_v1':raise ValueError('Unknown inflated CBF contract')
        inflation=1.15*(np.sqrt(2)*(noise[0]+noise[4])+noise[6])
        np.testing.assert_allclose(data['guidance_cbf_clearance_inflation'][active],inflation,atol=2e-9,rtol=2e-9)
        x=np.asarray(data['observed_state'][active],float);obs=np.asarray(data['observed_obstacles'][active],float)
        u=np.asarray(data['control'][active],float);gains=np.asarray(data.get('gain') if gains is None else gains,float);c=config.robot
        if gains.shape==(2,):gains=np.broadcast_to(gains,(len(active),2))
        if gains.shape!=(len(active),2):raise ValueError('Missing applied gains for inflated CBF audit')
        gains=gains[active]
        delta=x[:,None,:2]-obs[:,:,:2];relative_velocity=x[:,None,3:5]-obs[:,:,3:5]
        h=np.sum(delta**2,axis=-1)-(c.radius+c.clearance_buffer+obs[:,:,2]+inflation)**2
        hd=2*np.sum(delta*relative_velocity,axis=-1);psi=hd+gains[:,0,None]*h
        acceleration=np.column_stack((-np.sin(x[:,2])*u.sum(axis=-1)/c.mass,np.cos(x[:,2])*u.sum(axis=-1)/c.mass-c.gravity))
        hdd=2*np.sum(relative_velocity**2,axis=-1)+2*np.sum(delta*acceleration[:,None,:],axis=-1)
        residual=hdd+gains.sum(axis=-1)[:,None]*hd+gains.prod(axis=-1)[:,None]*h-c.cbf_margin
        hmin=np.min(np.where(mask[None,:],h,np.inf),axis=-1);pmin=np.min(np.where(mask[None,:],psi,np.inf),axis=-1)
        np.testing.assert_allclose(data['h'][active],hmin,atol=3e-6,rtol=2e-6)
        np.testing.assert_allclose(data['psi1'][active],pmin,atol=3e-6,rtol=2e-6)
        if np.any(hmin<-2e-5) or np.any(pmin<-2e-5) or np.any(residual[:,mask]<-2e-5):
            raise ValueError('Applied command fails inflated current CBF or hierarchy')
