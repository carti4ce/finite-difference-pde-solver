"""Time integrators: explicit and implicit schemes (placeholders)."""
from __future__ import annotations

import numpy as np


def explicit_euler(u, rhs, dt):
    """One-step explicit Euler: u^{n+1} = u^n + dt * rhs(u^n)."""
    return u + dt * rhs(u)


def rk2(u, rhs, dt):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    return u + dt * k2


def implicit_euler(u_field, assemble_A, dt, alpha, solver):
    """Perform one implicit Euler step for heat equation.

    - `u_field` is a `Field` instance whose interior represents u^n.
    - `assemble_A(grid)` should return the Laplacian matrix `L` (scipy sparse)
    - we solve (I - dt * alpha * L) u^{n+1} = u^n
    """
    grid = u_field.grid
    L = assemble_A(grid)
    from scipy import sparse
    N = grid.size
    I = sparse.identity(N, format="csr")
    A = I - dt * alpha * L
    b = u_field.flat_interior()
    u_new_flat = solver.solve(A, b)
    # reshape back into interior and set
    if grid.ny == 1:
        u_field.set_interior(u_new_flat)
    else:
        u_field.set_interior(u_new_flat.reshape((grid.nx, grid.ny), order="C"))
    return u_field


def crank_nicolson(u_field, assemble_A, dt, alpha, solver):
    """Perform one Crank-Nicolson step for heat equation.

    (I - dt/2 * alpha L) u^{n+1} = (I + dt/2 * alpha L) u^n
    """
    grid = u_field.grid
    L = assemble_A(grid)
    from scipy import sparse
    N = grid.size
    I = sparse.identity(N, format="csr")
    A = I - 0.5 * dt * alpha * L
    B = I + 0.5 * dt * alpha * L
    u_n = u_field.flat_interior()
    b = B.dot(u_n)
    u_new_flat = solver.solve(A, b)
    if grid.ny == 1:
        u_field.set_interior(u_new_flat)
    else:
        u_field.set_interior(u_new_flat.reshape((grid.nx, grid.ny), order="C"))
    return u_field
