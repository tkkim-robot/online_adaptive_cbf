"""Default repository discrete MPC formulations on the shared unicycle plant.

CasADi/IPOPT is an independent CPU comparator, with no solver parameter search.
JAX remains the learned OA-CBF training/inference/simulation path. These gains
are dimensionless discrete gains; they are not the continuous OA-CBF gains.
"""
from dataclasses import asdict
import time
import numpy as np
from .config import UnicycleConfig


DEFAULTS={'fixed_low':(.01,.01),'fixed_high':(.35,.35),'optimal_decay':(.01,.01)}


def numpy_euler(x,u,dt):
    x=np.asarray(x,float);u=np.asarray(u,float)
    return x+dt*np.asarray([x[3]*np.cos(x[2]),x[3]*np.sin(x[2]),u[1],u[0]])


def numpy_barrier(x,u,obs,gains,omega,robot,buffer=None):
    """Independent NumPy version of the two repeated-input Euler differences."""
    obs=np.asarray(obs,float);a=np.asarray(gains)*np.asarray(omega)
    radius=robot.radius+(robot.clearance_buffer if buffer is None else buffer)
    x1=numpy_euler(x,u,robot.dt);x2=numpy_euler(x1,u,robot.dt)
    h=lambda y:np.sum((y[:2]-obs[:,:2])**2,axis=-1)-1.01*(radius+obs[:,2])**2
    h0,h1,h2=h(x),h(x1),h(x2)
    return h2-2*h1+h0+a.sum()*(h1-h0)+a.prod()*h0


class DiscreteMPC:
    def __init__(self,capacity,method='fixed_low',robot=UnicycleConfig(),*,shared_contract=True):
        import casadi as ca
        if method not in DEFAULTS or isinstance(capacity,bool) or not isinstance(capacity,int) or capacity<1:
            raise ValueError('Unknown discrete MPC method/capacity')
        self.capacity=capacity;self.method=method;self.robot=robot;self.horizon=10
        self.gains=np.asarray(DEFAULTS[method]);self.decay=method=='optimal_decay';self.shared_contract=shared_contract
        self.last_omega=np.zeros(2);H=self.horizon;C=capacity
        X=ca.SX.sym('x',4,H+1);U=ca.SX.sym('u',2,H)
        W=ca.SX.sym('omega',2,H) if self.decay else ca.DM.ones(2,H)
        initial=ca.SX.sym('initial',4);goal=ca.SX.sym('goal',2);obs=ca.SX.sym('obs',C,5)
        active=ca.SX.sym('active',C);previous=ca.SX.sym('previous',2);speed_error=ca.SX.sym('speed_error')
        parameters=ca.vertcat(initial,goal,ca.vec(obs),active,previous,speed_error)
        variables=ca.vertcat(ca.vec(X),ca.vec(U),ca.vec(W)) if self.decay else ca.vertcat(ca.vec(X),ca.vec(U))
        goal_state=ca.vertcat(goal,0,0);Q=ca.DM([50,50,.01,30])
        equality=[X[:,0]-initial];inequality=[];cost=0
        def step(x,u):return x+robot.dt*ca.vertcat(x[3]*ca.cos(x[2]),x[3]*ca.sin(x[2]),u[1],u[0])
        def barrier(x,u,omega):
            radius=robot.radius+(robot.clearance_buffer if shared_contract else 0.)
            h=lambda y:(obs[:,0]-y[0])**2+(obs[:,1]-y[1])**2-1.01*(obs[:,2]+radius)**2
            x1=step(x,u);x2=step(x1,u);h0,h1,h2=h(x),h(x1),h(x2)
            a1,a2=self.gains[0]*omega[0],self.gains[1]*omega[1]
            return h2-2*h1+h0+(a1+a2)*(h1-h0)+a1*a2*h0
        for k in range(H):
            error=X[:,k]-goal_state;cost+=ca.dot(Q,error**2)
            if self.decay:
                # Pinned OptimalDecayMPCCBF calls set_rterm twice. do-mpc5.1.2
                # replaces the first custom term: its effective default is
                # the two omega penalties only, not an extra control cost.
                cost+=10*ca.sumsqr(W[:,k]-1)
            else:
                last=previous if k==0 else U[:,k-1]
                cost+=.5*ca.sumsqr(U[:,k]-last)
            equality.append(X[:,k+1]-step(X[:,k],U[:,k]))
            inequality.append(active*barrier(X[:,k],U[:,k],W[:,k])+(1-active))
            lower=initial[3]-ca.fmax(0.,initial[3]-speed_error) if shared_contract else -robot.v_max
            upper=robot.v_max+initial[3]-ca.fmin(robot.v_max,initial[3]+speed_error) if shared_contract else robot.v_max
            inequality.append(ca.vertcat(X[3,k+1]-lower,upper-X[3,k+1]))
        cost+=ca.dot(Q,(X[:,H]-goal_state)**2)
        constraints=ca.vertcat(*equality,*inequality);self.equalities=4*(H+1)
        self.lbg=np.r_[np.zeros(self.equalities),np.zeros(int(constraints.numel())-self.equalities)]
        self.ubg=np.r_[np.zeros(self.equalities),np.full(int(constraints.numel())-self.equalities,np.inf)]
        self.lbx=np.full(int(variables.numel()),-np.inf);self.ubx=-self.lbx.copy()
        lower_u=np.tile([-robot.a_max,-robot.w_max],H);self.lbx[4*(H+1):4*(H+1)+2*H]=lower_u
        self.ubx[4*(H+1):4*(H+1)+2*H]=-lower_u
        self.solver_options={'ipopt.print_level':0,'print_time':False}
        start=time.perf_counter()
        self.solver=ca.nlpsol('discrete_mpc','ipopt',dict(x=variables,p=parameters,f=cost,g=constraints),self.solver_options)
        self.setup_seconds=time.perf_counter()-start
        self.expression=ca.Function('discrete_mpc_expressions',[variables,parameters],[cost,constraints])
        sx=ca.SX.sym('state',4);su=ca.SX.sym('control',2);sw=ca.SX.sym('omega',2)
        self.barrier_expression=ca.Function('discrete_barrier',[sx,su,obs,sw],[barrier(sx,su,sw)])

    def contract(self):
        return dict(method=self.method,horizon=self.horizon,gains=self.gains.tolist(),Q=[50,50,.01,30],
            input_cost='10*(omega1-1)^2+10*(omega2-1)^2; pinned effective custom rterm' if self.decay else '.5*sum((u[k]-u[k-1])^2)',
            omega_bounds='unrestricted' if self.decay else 'fixed1',obstacle_capacity=self.capacity,
            barrier='h[k+2]-2*h[k+1]+h[k]+(alpha1*omega1+alpha2*omega2)*(h[k+1]-h[k])+alpha1*alpha2*omega1*omega2*h[k]',
            predictor='Euler, same control repeated in the two-step barrier difference, observed obstacle centers held fixed across horizon as in the pinned repository. All active rows included; inactive rows1>=0.',
            radius_beta=1.01,shared_contract=self.shared_contract,robot=asdict(self.robot),solver='CasADi3.7.2/IPOPT',solver_options=self.solver_options,
            adaptation='All-obstacle masked capacity and shared0..v_max speed contract; future speed tightened by the sensed speed interval intersected with physical bounds. Shared CBF clearance_buffer added to robot radius. Initial measurement is not projected. These declared common-contract changes are not a literal five-obstacle legacy run.',
            initialization='Repeat current sensed state and previous applied command, as repository set_initial_guess; omega starts0 at each new episode, then previous applied omega. No multiplier warm-start or solver tuning.')

    def pack(self,x,goal,obs,mask,previous,speed_error):
        x=np.asarray(x,float);goal=np.asarray(goal,float);obs=np.asarray(obs,float);mask=np.asarray(mask,bool);previous=np.asarray(previous,float)
        if x.shape!=(4,) or goal.shape!=(2,) or obs.shape!=(self.capacity,5) or mask.shape!=(self.capacity,) or previous.shape!=(2,):
            raise ValueError('Discrete MPC input shape mismatch')
        packed=np.r_[x,goal,obs.ravel(order='F'),mask.astype(float),previous,speed_error]
        if not np.isfinite(packed).all() or speed_error<0 or speed_error>=self.robot.v_max/2:raise ValueError('Invalid MPC observation/uncertainty')
        return packed

    def solve(self,x,goal,obs,mask,previous,speed_error=0.):
        H=self.horizon;p=self.pack(x,goal,obs,mask,previous,speed_error)
        guess=np.r_[np.tile(np.asarray(x,float),H+1),np.tile(previous,H)]
        if self.decay:guess=np.r_[guess,np.tile(self.last_omega,H)]
        start=time.perf_counter();answer=self.solver(x0=guess,p=p,lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
        elapsed=time.perf_counter()-start;stats=self.solver.stats();v=np.asarray(answer['x']).reshape(-1)
        X=v[:4*(H+1)].reshape(H+1,4);U=v[4*(H+1):4*(H+1)+2*H].reshape(H,2)
        W=v[-2*H:].reshape(H,2) if self.decay else np.ones((H,2))
        finite=np.isfinite(v).all();eq=np.inf;ineq=np.inf
        if finite:
            eq=max(float(np.max(np.abs(X[0]-x))),max(float(np.max(np.abs(X[k+1]-numpy_euler(X[k],U[k],self.robot.dt)))) for k in range(H)))
            residuals=np.stack([numpy_barrier(X[k],U[k],obs,self.gains,W[k],self.robot,buffer=None if self.shared_contract else 0.) for k in range(H)])
            cbf=float(np.max(-residuals[:,np.asarray(mask,bool)],initial=-np.inf))
            low=x[3]-max(0.,x[3]-speed_error) if self.shared_contract else -self.robot.v_max
            high=self.robot.v_max+x[3]-min(self.robot.v_max,x[3]+speed_error) if self.shared_contract else self.robot.v_max
            ineq=max(cbf,float(np.max(low-X[1:,3])),float(np.max(X[1:,3]-high)),float(np.max(np.abs(U)-[self.robot.a_max,self.robot.w_max])))
        feasible=bool(stats.get('success',False) and finite and eq<=self.robot.qp_tolerance and ineq<=self.robot.qp_tolerance)
        return dict(control=U[0],omega=W[0],states=X,controls=U,omegas=W,feasible=feasible,
            solver_success=bool(stats.get('success',False)),
            solver_status=str(stats.get('return_status')),iterations=int(stats.get('iter_count',-1)),solve_seconds=elapsed,
            max_equality_error=eq,max_constraint_violation=ineq,objective=float(answer['f']))
