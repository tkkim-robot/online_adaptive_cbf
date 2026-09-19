"""Pinned planar-flight optimal-decay QP, with OSQP library defaults.

The repository controller has one obstacle row, two unrestricted independent
coefficients, and a fixed original nominal. No OA guidance or rescue is added.
"""
from dataclasses import asdict
import time
import numpy as np
from .quad2d_control import FlightConfig


def nominal(x, target, config=FlightConfig()):
    """Direct NumPy transcription of Quad2D.nominal_input's default gains."""
    x=np.asarray(x,float);target=np.asarray(target,float);c=config.robot
    acceleration=np.array([3*(target[0]-x[0])-.5*x[3],.1*(target[1]-x[1])-.5*x[4]+c.gravity])
    thrust=c.mass*np.linalg.norm(acceleration);desired=-np.arctan2(*acceleration)
    error=np.arctan2(np.sin(desired-x[2]),np.cos(desired-x[2]))
    torque=np.clip(.05*error-.05*x[5],-1.,1.)
    # The legacy nominal uses robot radius as its rotor lever arm. Shared
    # FlightConfig has equal radius/arm; no substitute gain tuning is applied.
    return np.clip(.5*np.array([thrust+torque/c.radius,thrust-torque/c.radius]),c.force_min,c.force_max)


def nearest(x,obs,mask):
    """Tracking's default Quad2D full-angle nearest-center selection."""
    if not np.any(mask):return -1
    return int(np.argmin(np.where(mask,np.linalg.norm(obs[:,:2]-x[:2],axis=1),np.inf)))


def problem(x,target,obs,mask,config=FlightConfig(),shared_buffer=True):
    x=np.asarray(x,float);obs=np.asarray(obs,float);mask=np.asarray(mask,bool);c=config.robot
    selected=nearest(x,obs,mask);A=np.zeros((5,4));b=np.zeros(5)
    if selected>=0:
        o=obs[selected];d=x[:2]-o[:2];v=x[3:5]
        h=d@d-1.01*(o[2]+c.radius+(c.clearance_buffer if shared_buffer else 0.))**2
        hd=2*d@v;drift=2*v@v-2*c.gravity*d[1]
        collective=2*(-d[0]*np.sin(x[2])+d[1]*np.cos(x[2]))/c.mass
        A[0]=[-collective,-collective,-hd,-.25*h];b[0]=drift
    A[1:,:2]=[[1,0],[-1,0],[0,1],[0,-1]];b[1:]=[c.force_max,-c.force_min,c.force_max,-c.force_min]
    return np.r_[nominal(x,target,config),1.,1.],A,b,selected


def contract(config=FlightConfig(),shared_buffer=True):
    import osqp
    return dict(method='optimal_decay_cbf_qp',schema='quad2d_native_odqp_v1',alpha1=.5,alpha2=.5,omega_reference=[1.,1.],penalties=[1e4,1e4],
        omega_bounds='unrestricted independent coefficients; not a product of effective class-K gains',
        objective='sum((u-u_ref)^2)+1e4*(omega1-1)^2+1e4*(omega2-1)^2',
        barrier='hddot+(alpha1+alpha2)*omega1*hdot+alpha1*alpha2*omega2*h>=0',beta=1.01,
        obstacles='Single nearest observed center among every active obstacle; no velocity term, no field-of-view truncation for Quad2D',
        nominal='Original Quad2D.nominal_input defaults: kpx3,kdx.5,kpz.1,kdz.5,kptheta.05,kdtheta.05; original torque/rotor clipping',
        solver='OSQP',solver_version=osqp.__version__,solver_options=dict(verbose=False),
        initialization='New default OSQP instance per episode; default warm starting within episode; no retries, scaling or tolerance overrides',
        shared_buffer=shared_buffer,config=asdict(config),
        task_adapter='Common observed route target, shared robot/sensor/physical plant/arrival/envelope termination. Shared buffer optionally added to barrier radius. Original single-obstacle API receives one vector (legacy tracking currently passes an incompatible multi-obstacle array). No added flight-envelope CBF or OA fallback.',
        acceptance='Solver solved/solved-inaccurate plus independently checked original QP rows and actual stored FP32 actuator command, tolerance1e-5. Rejected future censored.')


class Quad2DODQP:
    def __init__(self,config=FlightConfig(),shared_buffer=True):
        import osqp
        from scipy import sparse
        self.config=config;self.shared_buffer=shared_buffer;self.weights=np.array([1.,1.,1e4,1e4]);self.solver=osqp.OSQP();self.initialized=False;self.setup_seconds=0.

    def solve(self,x,target,obs,mask):
        ref,A,b,selected=problem(x,target,obs,mask,self.config,self.shared_buffer)
        if not all(np.isfinite(v).all() for v in [ref,A,b]):raise ValueError('Nonfinite original OD-QP inputs')
        if not self.initialized:
            from scipy import sparse
            start=time.perf_counter()
            # Default solver scaling must see the real first QP. Retain explicit
            # zeros in the CSC pattern so every later coefficient can update.
            pattern=sparse.csc_matrix((A.ravel(order='F'),np.tile(np.arange(5),4),np.arange(0,21,5)),shape=(5,4))
            self.solver.setup(P=sparse.diags(2*self.weights,format='csc'),q=-2*self.weights*ref,A=pattern,l=np.full(5,-np.inf),u=b,verbose=False)
            self.setup_seconds=time.perf_counter()-start;self.initialized=True
        else:self.solver.update(q=-2*self.weights*ref,Ax=A.ravel(order='F'),u=b)
        start=time.perf_counter();result=self.solver.solve(raise_error=False)
        solution=np.full(4,np.nan) if result.x is None else np.asarray(result.x).copy()
        success=bool(result.info.status_val in (1,2) and np.isfinite(solution).all())
        raw_violation=float(np.max(A@solution-b)) if np.isfinite(solution).all() else np.inf
        applied=solution.copy();applied[:2]=solution[:2].astype(np.float32)
        stored_violation=float(np.max(A@applied-b)) if np.isfinite(applied).all() else np.inf
        return dict(solution=solution,reference=ref,selected_obstacle=selected,solver_success=success,solver_status=result.info.status,status_value=int(result.info.status_val),
            iterations=int(result.info.iter),solve_seconds=time.perf_counter()-start,primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res),
            raw_violation=raw_violation,stored_violation=stored_violation,feasible=bool(success and max(raw_violation,stored_violation)<=self.config.robot.qp_tolerance))
