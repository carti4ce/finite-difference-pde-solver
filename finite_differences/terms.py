"""Spatial terms that compose the right-hand side ``f`` of a PDE.

A PDE is written as one of

    0     = f(u, u_x, ...)      (steady)
    u_t   = f(u, u_x, ...)      (parabolic / advection-reaction-diffusion)
    u_tt  = f(u, u_x, ...)      (hyperbolic / wave)

where ``f`` is the sum of the terms defined here. Each term knows how to
*evaluate* its spatial contribution on a `Field` (ghost cells filled by a
boundary condition). Linear terms additionally expose a sparse ``matrix`` form
so the PDE can be assembled for steady solves and implicit time stepping.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

from . import operators as ops


class SpatialTerm(ABC):
    """Base class for a spatial contribution to ``f``.

    Attributes:
        linear: True if the term is an affine function of ``u`` and therefore
            supports :meth:`matrix` assembly.
    """

    linear = False

    @abstractmethod
    def evaluate(self, field):
        """Return this term's contribution to ``f`` as an interior-shaped array.

        ``field`` must already have its ghost cells filled by a boundary
        condition (the owning `PDE` does this before summing terms).
        """

    def stencil_matrix(self, grid):
        """Return the term's boundary-condition-independent stencil matrix.

        Maps the padded array (interior + one ghost layer, row-major) to the
        interior output. The owning `PDE` combines it with the boundary
        condition's prolongation ``P`` to form the interior operator ``S @ P``
        and carries the constant/boundary part in a separate rhs vector. Nonlinear
        terms have no stencil-matrix form.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is nonlinear and has no stencil-matrix form"
        )


class DiffusionTerm(SpatialTerm):
    """``coeff * laplacian(u)``  (isotropic diffusion)."""

    linear = True

    def __init__(self, coeff=1.0):
        self.coeff = coeff

    def evaluate(self, field):
        return self.coeff * ops.laplacian(field)

    def stencil_matrix(self, grid):
        return self.coeff * ops.laplacian_stencil_matrix(grid)


class AdvectionTerm(SpatialTerm):
    """``-v . grad(u)``  (linear advection with constant velocity ``v``).

    ``coeff`` is the velocity: a scalar in 1D, or a scalar / ``(vx, vy)`` in 2D
    (a scalar is applied to both axes). ``scheme`` is ``"upwind"`` (stable,
    1st order) or ``"central"`` (2nd order, can oscillate).
    """

    linear = True

    def __init__(self, coeff, scheme="upwind"):
        self.coeff = coeff
        self.scheme = scheme

    def _velocity(self, grid):
        if np.isscalar(self.coeff):
            v = float(self.coeff)
            return (v,) if grid.ny == 1 else (v, v)
        comps = tuple(float(c) for c in self.coeff)
        if grid.ny == 1:
            return comps[:1]
        return comps

    def evaluate(self, field):
        out = np.zeros(field.interior.shape)
        for axis, v in enumerate(self._velocity(field.grid)):
            if v == 0.0:
                continue
            g = ops.gradient(field, axis=axis, scheme=self.scheme,
                             upwind_sign=1.0 if v > 0 else -1.0)
            out = out - v * g
        return out

    def stencil_matrix(self, grid):
        S = sparse.csr_matrix((grid.size, ops.padded_size(grid)))
        for axis, v in enumerate(self._velocity(grid)):
            if v == 0.0:
                continue
            D = ops.gradient_stencil_matrix(grid, axis=axis, scheme=self.scheme,
                                            upwind_sign=1.0 if v > 0 else -1.0)
            S = S - v * D
        return S.tocsr()


class ReactionTerm(SpatialTerm):
    """``coeff * u``  (linear reaction / decay-growth)."""

    linear = True

    def __init__(self, coeff=1.0):
        self.coeff = coeff

    def evaluate(self, field):
        return self.coeff * field.interior

    def stencil_matrix(self, grid):
        return self.coeff * ops.identity_stencil_matrix(grid)


class SourceTerm(SpatialTerm):
    """``s(x[, y])``  (forcing / source, independent of ``u``).

    ``func`` is called as ``func(X)`` in 1D or ``func(X, Y)`` in 2D with the
    interior node coordinates, and may also return a scalar.
    """

    linear = True

    def __init__(self, func):
        self.func = func

    def _values(self, grid):
        X, Y = grid.mesh()
        s = self.func(X) if Y is None else self.func(X, Y)
        target = (grid.nx,) if grid.ny == 1 else (grid.nx, grid.ny)
        return np.broadcast_to(np.asarray(s, dtype=float), target)

    def evaluate(self, field):
        return self._values(field.grid).copy()

    def stencil_matrix(self, grid):
        # Contributes only to the right-hand side, not the operator.
        return sparse.csr_matrix((grid.size, ops.padded_size(grid)))


class FunctionTerm(SpatialTerm):
    """Generic nonlinear pointwise term ``g(u, x[, y])``.

    ``func`` is called as ``func(u_interior, X, Y)`` (``Y`` is ``None`` in 1D)
    and must return an interior-shaped array. Use for nonlinear reactions such
    as Fisher-KPP ``r*u*(1-u)``.
    """

    linear = False

    def __init__(self, func):
        self.func = func

    def evaluate(self, field):
        X, Y = field.grid.mesh()
        return np.asarray(self.func(field.interior, X, Y), dtype=float)
