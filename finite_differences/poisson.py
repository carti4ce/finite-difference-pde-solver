"""Poisson solver utilities."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .operators import laplacian_5pt


def assemble_rhs_from_source(grid, source_func):
    """Evaluate source function at cell centers and return flattened RHS vector."""
    X, Y = grid.mesh()
    if grid.ny == 1:
        f = source_func(X)
        return f.copy()
    f = source_func(X, Y)
    return f.ravel(order="C").copy()


def solve_poisson(grid, source_func, bc=None, solver=None):
    """Solve Poisson equation L u = f on a Cartesian cell-centered grid.

    - `source_func` should be callable returning f(x, y) at cell centers.
    - `bc` is currently ignored except that Dirichlet zero BC is assumed at boundaries.
    - `solver` should implement `.solve(A, b)` and return solution vector.

    Returns a NumPy array of shape (nx, ny) with the interior solution.
    """
    nx, ny = grid.nx, grid.ny
    N = nx * ny
    A = laplacian_5pt(grid)
    b = assemble_rhs_from_source(grid, source_func)
    if solver is None:
        # default to SciPy direct solver
        from scipy.sparse.linalg import spsolve

        u_flat = spsolve(A, b)
    else:
        u_flat = solver.solve(A, b)

    if ny == 1:
        return u_flat
    return u_flat.reshape((nx, ny), order="C")
