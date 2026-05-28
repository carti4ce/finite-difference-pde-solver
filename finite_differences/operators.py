"""Finite-difference operators and assembly utilities."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def central_diff_x(field, dx):
    """Second-order central difference in x for interior cells."""
    f = field.interior
    if f.ndim == 1:
        return (f[2:] - f[:-2]) / (2.0 * dx)
    return (f[2:, :] - f[:-2, :]) / (2.0 * dx)


def laplacian_5pt(grid):
    """Assemble a 5-point Laplacian on a regular 2D grid (cell-centered).

    Returns a SciPy sparse matrix of shape (N,N) where N = nx*ny.
    """
    nx, ny = grid.nx, grid.ny
    dx = grid.dx
    dy = grid.dy
    N = nx * ny
    if ny == 1:
        # 1D second derivative
        main = -2.0 * np.ones(nx)
        off = 1.0 * np.ones(nx - 1)
        A = sparse.diags([off, main, off], offsets=[-1, 0, 1], shape=(nx, nx))
        return A / (dx * dx)

    rx = 1.0 / (dx * dx)
    ry = 1.0 / (dy * dy)
    rows = []
    cols = []
    data = []
    def add_entry(i, j, val):
        rows.append(i)
        cols.append(j)
        data.append(val)

    # Use row-major (C) flattening: flat_index = i*ny + j
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            diag = -2.0 * (rx + ry)
            add_entry(idx, idx, diag)
            # left (i-1, j)
            if i - 1 >= 0:
                add_entry(idx, (i - 1) * ny + j, rx)
            # right (i+1, j)
            if i + 1 < nx:
                add_entry(idx, (i + 1) * ny + j, rx)
            # down (i, j-1)
            if j - 1 >= 0:
                add_entry(idx, i * ny + (j - 1), ry)
            # up (i, j+1)
            if j + 1 < ny:
                add_entry(idx, i * ny + (j + 1), ry)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A


def laplacian(field):
    """Compute discrete Laplacian using ghost cells. Returns interior-shaped array.

    Assumes `field.data` contains ghost cells of width `field.ng`.
    Works for 1D and 2D with central differences.
    """
    ng = field.ng
    dx = field.grid.dx
    nx = field.grid.nx
    if field.grid.ny == 1:
        f = field.data
        interior = f[ng:ng + nx]
        lap = (f[ng + 1 : ng + 1 + nx] - 2.0 * interior + f[ng - 1 : ng - 1 + nx]) / (dx * dx)
        return lap

    dy = field.grid.dy
    f = field.data
    ny = field.grid.ny
    interior = f[ng : ng + nx, ng : ng + ny]
    lap_x = (f[ng + 1 : ng + 1 + nx, ng : ng + ny] - 2.0 * interior + f[ng - 1 : ng - 1 + nx, ng : ng + ny]) / (dx * dx)
    lap_y = (f[ng : ng + nx, ng + 1 : ng + 1 + ny] - 2.0 * interior + f[ng : ng + nx, ng - 1 : ng - 1 + ny]) / (dy * dy)
    lap = lap_x + lap_y
    return lap
