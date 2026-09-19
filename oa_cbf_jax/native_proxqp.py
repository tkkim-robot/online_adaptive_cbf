"""Persistent ProxQP adapter for diagonal QPs with scaled decision variables.

y=sqrt(weights)*u is an invertible change of variables, not a relaxation.
Original-unit residuals are checked independently after converting back.
"""

import numpy as np


class NativeProxQP:
    def __init__(self,n_rows,weights=(1.,1.),tolerance=1e-9):
        import proxsuite
        self.api=proxsuite.proxqp;self.weights=np.asarray(weights,float)
        if self.weights.ndim!=1 or not np.isfinite(self.weights).all() or np.any(self.weights<=0):
            raise ValueError('QP weights must be finite and positive')
        self.scale=np.sqrt(self.weights);self.n_variables=len(self.weights);self.n_rows=n_rows
        self.solver=self.api.dense.QP(self.n_variables,0,n_rows)
        self.solver.settings.eps_abs=tolerance;self.solver.settings.eps_rel=0.
        self.solver.settings.eps_primal_inf=1e-12;self.solver.settings.eps_dual_inf=1e-12
        self.solver.settings.max_iter=10000
        self.solver.settings.verbose=False
        self.solver.init(np.eye(self.n_variables),np.zeros(self.n_variables),None,None,
                         np.zeros((n_rows,self.n_variables)),np.full(n_rows,-np.inf),np.ones(n_rows))
        self.previous_solved=False

    def solve(self,reference,a,b):
        reference,a,b=map(lambda v:np.asarray(v,float),(reference,a,b))
        if reference.shape!=(self.n_variables,) or a.shape!=(self.n_rows,self.n_variables) or b.shape!=(self.n_rows,):
            raise ValueError('QP shape changed after setup')
        if not all(np.isfinite(v).all() for v in (reference,a,b)):
            return dict(control=None,feasible=False,violation=float('inf'),status='nonfinite_input',iterations=0)
        violation=float(np.max(a@reference-b))
        if violation<=0:
            return dict(control=reference.copy(),feasible=True,violation=violation,status='solved_reference',iterations=0)
        # ProxQP 0.7.3 can retain an infeasibility status/internal iterate across
        # updates. A clean initialization is cheap for these 3/4-variable QPs
        # and is necessary even when a previously rejected scene becomes empty.
        self.solver.cleanup()
        self.solver.settings.initial_guess=self.api.InitialGuess.NO_INITIAL_GUESS
        self.solver.init(np.eye(self.n_variables),-self.scale*reference,None,None,
                         a/self.scale,np.full(self.n_rows,-np.inf),b)
        self.solver.solve();result=self.solver.results
        x=np.asarray(result.x).copy()/self.scale
        solved=result.info.status==self.api.QPSolverOutput.PROXQP_SOLVED
        finite=np.isfinite(x).all();violation=float(np.max(a@x-b)) if finite else float('inf')
        self.previous_solved=bool(solved and finite and violation<=1e-5)
        return dict(control=x if finite else None,feasible=self.previous_solved,violation=violation,
                    status=str(result.info.status),iterations=int(result.info.iter))
