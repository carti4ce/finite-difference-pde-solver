"""Poisson / Laplace solvers built on the term-based `PDE` layer."""
from __future__ import annotations

import numpy as np

from .bc import Dirichlet
from .pde import PDE
from .terms import DiffusionTerm, SourceTerm


def solve_poisson(grid, source_func, bc=None, solver=None):
    """Solve the Poisson equation ``laplacian(u) = f`` on a node-centered grid.

    Written as the steady problem ``0 = laplacian(u) - f``.

    Args:
        source_func: callable returning ``f(x)`` (1D) or ``f(x, y)`` (2D) at the
            interior node coordinates.
        bc: a boundary condition (defaults to homogeneous `Dirichlet`). Nonzero
            Dirichlet data is honored via the assembled boundary lift.
        solver: object with ``.solve(A, b)`` (defaults to `LinearSolver`).

    Returns:
        Interior solution: shape ``(nx,)`` in 1D or ``(nx, ny)`` in 2D.
    """
    bc = bc or Dirichlet(0.0)

    def neg_source(*coords):
        return -np.asarray(source_func(*coords), dtype=float)

    pde = PDE(grid, [DiffusionTerm(1.0), SourceTerm(neg_source)], bc, time_order=0)
    return pde.solve_steady(solver=solver)
