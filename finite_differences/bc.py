"""Boundary condition abstractions.

Each boundary condition can do two things:

* **apply(field)** fills a `Field`'s ghost cells (used by the explicit /
  nonlinear method-of-lines path). This is the full, affine ghost fill.
* **prolongation(grid)** returns the sparse *linear* map ``P`` from the interior
  unknowns to the padded array (interior + one ghost layer), used to assemble
  the interior operator ``L = S @ P`` for steady solves and implicit stepping.
  The constant part of the ghost fill (Dirichlet values, Neumann fluxes) is not
  in ``P``; it is recovered by the `PDE` as ``b = f(0)`` via ``apply``.

The two are kept consistent so that ``L @ u + b == spatial_rhs(u)`` for any
interior ``u`` (asserted in the tests).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse


def _prolongation_1d(n, kind):
    """Linear interior->padded map (shape (n+2, n)) for a single axis.

    Interior node i maps to padded index i+1. Ghost rows 0 and n+1 depend on
    the boundary kind:

    * ``"dirichlet"``: ghosts are constants -> zero rows (no interior coupling).
    * ``"neumann"``: ghost = nearest interior node (zero-gradient linear part).
    * ``"periodic"``: ghost = interior node on the opposite side (wrap).
    """
    rows = list(range(1, n + 1))
    cols = list(range(n))
    data = [1.0] * n
    if kind == "dirichlet":
        pass
    elif kind == "neumann":
        rows += [0, n + 1]
        cols += [0, n - 1]
        data += [1.0, 1.0]
    elif kind == "periodic":
        rows += [0, n + 1]
        cols += [n - 1, 0]
        data += [1.0, 1.0]
    else:
        raise ValueError(f"unknown boundary kind {kind!r}")
    return sparse.csr_matrix((data, (rows, cols)), shape=(n + 2, n))


def _prolongation(grid, kind):
    """Interior->padded prolongation for the whole grid (1D or 2D)."""
    if grid.ny == 1:
        return _prolongation_1d(grid.nx, kind)
    Px = _prolongation_1d(grid.nx, kind)
    Py = _prolongation_1d(grid.ny, kind)
    return sparse.kron(Px, Py, format="csr")


class BC(ABC):
    """Base class for boundary conditions."""

    #: single-axis prolongation kind ("dirichlet" / "neumann" / "periodic")
    kind = None

    @abstractmethod
    def apply(self, field):
        """Apply BC to ghost cells of `field` (a `Field` instance)."""

    def prolongation(self, grid):
        """Return the sparse linear interior->padded map for matrix assembly."""
        if self.kind is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support matrix assembly; use "
                "explicit integration"
            )
        return _prolongation(grid, self.kind)


class Dirichlet(BC):
    """Fixed value on every boundary."""

    kind = "dirichlet"

    def __init__(self, value=0.0):
        self.value = value

    def apply(self, field):
        ng = field.ng
        if field.grid.ny == 1:
            field.data[:ng] = self.value
            field.data[-ng:] = self.value
            return
        field.data[:ng, :] = self.value
        field.data[-ng:, :] = self.value
        field.data[:, :ng] = self.value
        field.data[:, -ng:] = self.value


class DirichletBox(BC):
    """Dirichlet condition with a (constant) value per edge.

    1D uses ``left``/``right``; 2D adds ``bottom``/``top`` (the ``y`` edges).
    The corner ghost cells are not read by the 5-point stencils, so overlap at
    the corners is harmless. The linear prolongation is identical to `Dirichlet`
    (all ghost values are constants); only the constant lift differs.
    """

    kind = "dirichlet"

    def __init__(self, left=0.0, right=0.0, bottom=0.0, top=0.0):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def apply(self, field):
        ng = field.ng
        if field.grid.ny == 1:
            field.data[:ng] = self.left
            field.data[-ng:] = self.right
            return
        field.data[:ng, :] = self.left
        field.data[-ng:, :] = self.right
        field.data[:, :ng] = self.bottom
        field.data[:, -ng:] = self.top


class Neumann(BC):
    """Prescribed gradient on the boundary (defaults to zero, i.e. insulated).

    ``derivative`` is the coordinate derivative applied at every boundary:
    ``du/dx`` at the x-edges and ``du/dy`` at the y-edges. A one-sided ghost
    value reproduces it: the left ghost is ``u_first - derivative*h`` and the
    right ghost is ``u_last + derivative*h`` (and likewise per axis in 2D).
    ``derivative=0`` is a plain zero-gradient copy.

    Note: the nonzero-derivative offset is written for a single ghost layer
    (``ng=1``, the default). Zero-gradient works for any ``ng``.
    """

    kind = "neumann"

    def __init__(self, derivative=0.0):
        self.derivative = derivative

    def apply(self, field):
        ng = field.ng
        d = self.derivative
        if field.grid.ny == 1:
            dx = field.grid.dx
            field.data[:ng] = field.data[ng] - d * dx
            field.data[-ng:] = field.data[-ng - 1] + d * dx
            return
        dx = field.grid.dx
        dy = field.grid.dy
        # x-edges (du/dx); full columns are fine, corners are never read.
        field.data[:ng, :] = field.data[ng:ng + 1, :] - d * dx
        field.data[-ng:, :] = field.data[-ng - 1:-ng, :] + d * dx
        # y-edges (du/dy)
        field.data[:, :ng] = field.data[:, ng:ng + 1] - d * dy
        field.data[:, -ng:] = field.data[:, -ng - 1:-ng] + d * dy


class Periodic(BC):
    """Periodic wrap: each ghost mirrors the interior node on the far side."""

    kind = "periodic"

    def apply(self, field):
        ng = field.ng
        if field.grid.ny == 1:
            field.data[:ng] = field.data[-2 * ng:-ng]
            field.data[-ng:] = field.data[ng:2 * ng]
            return
        field.data[:ng, ng:-ng] = field.data[-2 * ng:-ng, ng:-ng]
        field.data[-ng:, ng:-ng] = field.data[ng:2 * ng, ng:-ng]
        field.data[:, :ng] = field.data[:, -2 * ng:-ng]
        field.data[:, -ng:] = field.data[:, ng:2 * ng]
