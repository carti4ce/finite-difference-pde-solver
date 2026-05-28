"""Linear solver adapter utilities."""
from __future__ import annotations

from typing import Any
import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    _HAS_SCIPY = True
except Exception:
    sparse = None
    spsolve = None
    _HAS_SCIPY = False


class DenseSolver:
    """Dense NumPy solver fallback. Converts sparse matrices to dense arrays."""

    def solve(self, A: Any, b: np.ndarray) -> np.ndarray:
        # Accept either dense ndarray or scipy sparse matrix
        if _HAS_SCIPY and sparse is not None and sparse.issparse(A):
            A = A.toarray()
        return np.linalg.solve(A, b)


class LinearSolver:
    """Adapter that uses SciPy sparse direct solver when available,
    otherwise falls back to `DenseSolver` using NumPy.
    """

    def __init__(self):
        self._dense = DenseSolver()

    def solve(self, A: Any, b: np.ndarray) -> np.ndarray:
        if _HAS_SCIPY and spsolve is not None:
            try:
                return spsolve(A, b)
            except Exception:
                # Fall back to dense
                return self._dense.solve(A, b)
        # No SciPy: use dense solver
        return self._dense.solve(A, b)
