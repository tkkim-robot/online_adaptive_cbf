"""Small diagonal QPs solved by a centered, scaled Clarabel problem.

The variable change z=sqrt(weights)*(u-reference) is invertible. The original
constraints and objective are preserved, with independent original-unit checks.
"""

import numpy as np


class NativeClarabel:
    def __init__(self,n_rows,weights=(1.,1.),tolerance=1e-9):
        import clarabel
        from scipy import sparse
        self.api=clarabel;self.sparse=sparse;self.weights=np.asarray(weights,float)
        if self.weights.ndim!=1 or not np.isfinite(self.weights).all() or np.any(self.weights<=0):
            raise ValueError('QP weights must be finite and positive')
        self.scale=np.sqrt(self.weights);self.n_variables=len(self.weights);self.n_rows=n_rows
        self.hessian=sparse.eye(self.n_variables,format='csc');self.linear=np.zeros(self.n_variables)
        self.cones=[clarabel.NonnegativeConeT(n_rows)]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.tol_gap_abs=tolerance;settings.tol_gap_rel=tolerance;settings.tol_feas=tolerance
        self.settings=settings

    def solve(self,reference,a,b):
        reference,a,b=map(lambda value:np.asarray(value,float),(reference,a,b))
        if reference.shape!=(self.n_variables,) or a.shape!=(self.n_rows,self.n_variables) or b.shape!=(self.n_rows,):
            raise ValueError('QP shape changed after setup')
        if not all(np.isfinite(value).all() for value in (reference,a,b)):
            return dict(control=None,feasible=False,violation=float('inf'),status='nonfinite_input',iterations=0)
        violation=float(np.max(a@reference-b))
        if violation<=0:
            return dict(control=reference.copy(),feasible=True,violation=violation,status='solved_reference',iterations=0)
        solver=self.api.DefaultSolver(self.hessian,self.linear,self.sparse.csc_matrix(a/self.scale),b-a@reference,
                                      self.cones,self.settings)
        result=solver.solve();x=reference+np.asarray(result.x)/self.scale
        finite=np.isfinite(x).all();violation=float(np.max(a@x-b)) if finite else float('inf')
        solved=result.status==self.api.SolverStatus.Solved
        return dict(control=x if finite else None,feasible=bool(solved and finite and violation<=1e-5),
                    violation=violation,status=str(result.status),iterations=int(result.iterations))
