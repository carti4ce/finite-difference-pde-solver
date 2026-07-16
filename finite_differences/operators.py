"""Finite-difference operators.

Two consistent views of the same node-centered discretization:

* **Ghost-cell stencils** (`laplacian`, `gradient`, `identity`) act on a `Field`
  whose ghost cells have already been filled by a boundary condition. They
  return interior-shaped arrays and support nonlinear / explicit evaluation.
* **Stencil matrices** (`laplacian_stencil_matrix`, `gradient_stencil_matrix`,
  `identity_stencil_matrix`) return the *boundary-condition-independent* linear
  stencil as a sparse matrix mapping the **padded** array (interior + one ghost
  layer, row-major) to the interior output. Combined with a boundary
  condition's prolongation ``P`` (padded <- interior) the interior operator is
  ``L = S @ P`` (see `PDE.assemble` and `bc` prolongations).

Both views come from the same coefficients, so they agree; the test suite
asserts ``L @ u + b == spatial_rhs(u)`` for every boundary condition.

The padded array for a grid uses one ghost layer on each side: size
``nx + 2`` in 1D and ``(nx + 2, ny + 2)`` in 2D, flattened row-major
(``i_pad * (ny + 2) + j_pad``), matching a ``Field`` with ``ng=1``.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Ghost-cell stencils
# ---------------------------------------------------------------------------

def identity(field):
    """Return a copy of the interior (the identity operator)."""
    return field.interior.copy()


def laplacian(field):
    """Discrete Laplacian via ghost cells. Returns an interior-shaped array.

    Assumes ``field.data`` contains ghost cells of width ``field.ng`` that have
    been filled by a boundary condition. Works for 1D and 2D.
    """
    ng = field.ng
    dx = field.grid.dx
    nx = field.grid.nx
    f = field.data
    if field.grid.ny == 1:
        interior = f[ng:ng + nx]
        return (f[ng + 1:ng + 1 + nx] - 2.0 * interior + f[ng - 1:ng - 1 + nx]) / (dx * dx)

    dy = field.grid.dy
    ny = field.grid.ny
    interior = f[ng:ng + nx, ng:ng + ny]
    lap_x = (f[ng + 1:ng + 1 + nx, ng:ng + ny] - 2.0 * interior + f[ng - 1:ng - 1 + nx, ng:ng + ny]) / (dx * dx)
    lap_y = (f[ng:ng + nx, ng + 1:ng + 1 + ny] - 2.0 * interior + f[ng:ng + nx, ng - 1:ng - 1 + ny]) / (dy * dy)
    return lap_x + lap_y


def gradient(field, axis=0, scheme="central", upwind_sign=1.0):
    """First derivative along ``axis`` via ghost cells. Returns interior array.

    * ``scheme="central"``: 2nd-order central difference.
    * ``scheme="upwind"``: 1st-order upwind; ``upwind_sign >= 0`` uses a
      backward difference, otherwise forward. (For an advection velocity ``v``
      the upwind direction is ``sign(v)``.)
    * ``scheme="forward"`` / ``"backward"``: one-sided differences.
    """
    ng = field.ng
    nx = field.grid.nx
    f = field.data
    if field.grid.ny == 1:
        if axis != 0:
            raise ValueError("1D field only has axis 0")
        h = field.grid.dx
        center = f[ng:ng + nx]
        left = f[ng - 1:ng - 1 + nx]
        right = f[ng + 1:ng + 1 + nx]
    else:
        ny = field.grid.ny
        if axis == 0:
            h = field.grid.dx
            center = f[ng:ng + nx, ng:ng + ny]
            left = f[ng - 1:ng - 1 + nx, ng:ng + ny]
            right = f[ng + 1:ng + 1 + nx, ng:ng + ny]
        elif axis == 1:
            h = field.grid.dy
            center = f[ng:ng + nx, ng:ng + ny]
            left = f[ng:ng + nx, ng - 1:ng - 1 + ny]
            right = f[ng:ng + nx, ng + 1:ng + 1 + ny]
        else:
            raise ValueError(f"invalid axis {axis} for 2D field")

    if scheme == "central":
        return (right - left) / (2.0 * h)
    if scheme == "forward":
        return (right - center) / h
    if scheme == "backward":
        return (center - left) / h
    if scheme == "upwind":
        if upwind_sign >= 0:
            return (center - left) / h
        return (right - center) / h
    raise ValueError(f"unknown scheme {scheme!r}")


# ---------------------------------------------------------------------------
# 1D padded stencils: sparse maps  (n interior outputs) <- (n + 2 padded inputs)
# ---------------------------------------------------------------------------

def _pad_identity_1d(n):
    """Pick the interior out of the padded array: row i -> padded index i+1."""
    rows = np.arange(n)
    cols = np.arange(n) + 1
    return sparse.csr_matrix((np.ones(n), (rows, cols)), shape=(n, n + 2))


def _pad_second_derivative_1d(n, h):
    """[1, -2, 1] / h^2 acting on the padded array."""
    rows = np.repeat(np.arange(n), 3)
    cols = np.concatenate([np.arange(n) + k for k in (0, 1, 2)]).reshape(3, n).T.ravel()
    vals = np.tile(np.array([1.0, -2.0, 1.0]) / (h * h), n)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n + 2))


def _pad_first_derivative_1d(n, h, scheme, upwind_sign=1.0):
    """First-derivative stencil acting on the padded array (row i at padded i+1)."""
    if scheme == "central":
        stencil = {0: -1.0 / (2.0 * h), 2: 1.0 / (2.0 * h)}
    elif scheme == "forward" or (scheme == "upwind" and upwind_sign < 0):
        stencil = {1: -1.0 / h, 2: 1.0 / h}
    elif scheme == "backward" or (scheme == "upwind" and upwind_sign >= 0):
        stencil = {0: -1.0 / h, 1: 1.0 / h}
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    rows, cols, vals = [], [], []
    for offset, v in stencil.items():
        rows.extend(range(n))
        cols.extend(np.arange(n) + offset)
        vals.extend([v] * n)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n + 2))


# ---------------------------------------------------------------------------
# Dimension-aware stencil matrices (padded flat -> interior flat)
# ---------------------------------------------------------------------------

def padded_size(grid):
    """Number of entries in the one-ghost-layer padded array for ``grid``."""
    if grid.ny == 1:
        return grid.nx + 2
    return (grid.nx + 2) * (grid.ny + 2)


def identity_stencil_matrix(grid):
    if grid.ny == 1:
        return _pad_identity_1d(grid.nx)
    return sparse.kron(_pad_identity_1d(grid.nx), _pad_identity_1d(grid.ny), format="csr")


def laplacian_stencil_matrix(grid):
    nx, ny = grid.nx, grid.ny
    if ny == 1:
        return _pad_second_derivative_1d(nx, grid.dx)
    Sxx = _pad_second_derivative_1d(nx, grid.dx)
    Syy = _pad_second_derivative_1d(ny, grid.dy)
    Ex = _pad_identity_1d(nx)
    Ey = _pad_identity_1d(ny)
    return (sparse.kron(Sxx, Ey) + sparse.kron(Ex, Syy)).tocsr()


def gradient_stencil_matrix(grid, axis=0, scheme="central", upwind_sign=1.0):
    nx, ny = grid.nx, grid.ny
    if ny == 1:
        if axis != 0:
            raise ValueError("1D grid only has axis 0")
        return _pad_first_derivative_1d(nx, grid.dx, scheme, upwind_sign)
    if axis == 0:
        Gx = _pad_first_derivative_1d(nx, grid.dx, scheme, upwind_sign)
        return sparse.kron(Gx, _pad_identity_1d(ny), format="csr")
    if axis == 1:
        Gy = _pad_first_derivative_1d(ny, grid.dy, scheme, upwind_sign)
        return sparse.kron(_pad_identity_1d(nx), Gy, format="csr")
    raise ValueError(f"invalid axis {axis} for 2D grid")
