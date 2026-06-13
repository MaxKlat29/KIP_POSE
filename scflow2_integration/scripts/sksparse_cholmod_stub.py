# Drop-in replacement for the subset of scikit-sparse / CHOLMOD that SCFlow2 uses,
# backed by scipy's SuperLU (no system SuiteSparse needed). SCFlow2's RAFT-3D
# GridSmoother (models/utils/raft_3d_basic_blocks.py) calls:
#     fac  = cholmod.analyze_AAt(A, ordering_method='best')   # symbolic
#     chol = fac.cholesky_AAt(A)                               # numeric
#     x    = chol(b)                                           # solve (A A^T) x = b
# We solve the SPD system (A A^T + beta I) x = b exactly via scipy.sparse splu.
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class CholmodError(Exception):
    pass


class _Solver:
    """Callable factor: __call__(b) solves (A A^T + beta I) x = b."""
    def __init__(self, lu):
        self._lu = lu

    def __call__(self, b):
        b = np.asarray(b)
        x = self._lu.solve(b.astype(np.float64))
        return x.astype(b.dtype, copy=False)

    # API aliases used by some sksparse callers
    def solve_A(self, b):
        return self(b)

    def __getattr__(self, _name):  # tolerate unused factor methods
        raise AttributeError(_name)


class Factor:
    def __init__(self, ordering_method="default", beta=1e-6):
        self.ordering_method = ordering_method
        self.beta = beta

    def cholesky_AAt(self, A, beta=None):
        beta = self.beta if beta is None else beta
        A = A.tocsr()
        AAt = (A @ A.transpose()).tocsc()
        n = AAt.shape[0]
        if beta:
            AAt = (AAt + beta * sp.identity(n, format="csc")).tocsc()
        try:
            lu = splu(AAt)
        except RuntimeError as e:
            raise CholmodError(str(e))
        return _Solver(lu)

    # in-place variants just delegate
    def cholesky_AAt_inplace(self, A, beta=None):
        return self.cholesky_AAt(A, beta)


def analyze_AAt(A, mode="auto", ordering_method="default", use_long=None):
    return Factor(ordering_method=ordering_method)


def analyze(A, mode="auto", ordering_method="default", use_long=None):
    return Factor(ordering_method=ordering_method)


def cholesky_AAt(A, beta=0, mode="auto", ordering_method="default", use_long=None):
    return Factor(ordering_method=ordering_method, beta=beta).cholesky_AAt(A, beta)


def cholesky(A, beta=0, mode="auto", ordering_method="default", use_long=None):
    # Cholesky of A itself: solve (A + beta I) x = b
    A = A.tocsc()
    n = A.shape[0]
    if beta:
        A = (A + beta * sp.identity(n, format="csc")).tocsc()
    return _Solver(splu(A))
