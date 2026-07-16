"""PDE assembly and solution via the method of lines.

A `PDE` bundles a `Grid`, a list of spatial `terms`, a boundary condition, and
a ``time_order``:

* ``time_order=0``:  ``0 = f(u)``      -> steady solve ``L u = -b``
* ``time_order=1``:  ``u_t = f(u)``    -> integrate ``u' = f(u)``
* ``time_order=2``:  ``u_tt = f(u)``   -> integrate ``u'' = f(u)`` (wave)

Explicit integration only needs each term's ``evaluate`` and therefore supports
nonlinear terms. Steady solves and implicit integration assemble the linear
operator ``(L, b)`` and require every term to be linear.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .field import Field
from .solvers import LinearSolver
from . import operators as ops
from . import time_integrators as ti


class PDE:
    def __init__(self, grid, terms, bc, time_order=1):
        if time_order not in (0, 1, 2):
            raise ValueError("time_order must be 0, 1, or 2")
        self.grid = grid
        self.terms = list(terms)
        self.bc = bc
        self.time_order = int(time_order)

    # -- basic geometry helpers ------------------------------------------
    @property
    def is_linear(self):
        return all(getattr(t, "linear", False) for t in self.terms)

    def _interior_shape(self):
        g = self.grid
        return (g.nx,) if g.ny == 1 else (g.nx, g.ny)

    # -- right-hand side -------------------------------------------------
    def spatial_rhs(self, field):
        """Evaluate ``f(u)`` on ``field``, returning an interior-shaped array.

        Fills the field's ghost cells from the boundary condition first, then
        sums every term's contribution.
        """
        self.bc.apply(field)
        total = np.zeros(self._interior_shape())
        for term in self.terms:
            total = total + term.evaluate(field)
        return total

    def _make_rhs(self):
        """Return a closure ``rhs(u_flat) -> du/dt_flat`` for explicit stepping."""
        shape = self._interior_shape()
        field = Field(self.grid, ng=1)

        def rhs(u_flat):
            field.set_interior(u_flat.reshape(shape))
            return self.spatial_rhs(field).ravel(order="C")

        return rhs

    def assemble(self):
        """Assemble the linear operator ``(L, b)`` with ``f(u) = L u + b``.

        The interior operator is ``L = S @ P`` where ``S`` is the sum of the
        terms' boundary-condition-independent stencil matrices (padded ->
        interior) and ``P`` is the boundary condition's prolongation (interior
        -> padded, linear part). ``b`` carries the constant part: source terms
        plus the boundary lift from (possibly nonzero) Dirichlet/Neumann data,
        obtained as ``f(0)``. Raises if any term is nonlinear.
        """
        if not self.is_linear:
            raise ValueError(
                "assemble() requires all-linear terms; use explicit integration "
                "(scheme='rk4'/'rk2'/'euler') for nonlinear PDEs"
            )
        S = sparse.csr_matrix((self.grid.size, ops.padded_size(self.grid)))
        for term in self.terms:
            S = S + term.stencil_matrix(self.grid).tocsr()
        P = self.bc.prolongation(self.grid)
        L = (S @ P).tocsr()
        # b = f(0): with zero interior, every linear term reduces to its
        # constant part (boundary lift + sources).
        zero = Field(self.grid, ng=1)
        b = self.spatial_rhs(zero).ravel(order="C").copy()
        return L, b

    # -- steady solve ----------------------------------------------------
    def solve_steady(self, solver=None):
        """Solve ``0 = f(u)`` (``L u = -b``). Returns an interior-shaped array.

        Raises:
            numpy.linalg.LinAlgError: if the assembled operator is singular
                (e.g. pure `Neumann` or `Periodic` boundaries, where the
                solution is only defined up to an additive constant), instead
                of silently returning NaNs or garbage.
        """
        L, b = self.assemble()
        solver = solver or LinearSolver()
        u = solver.solve(L, -b)
        # spsolve signals a singular matrix only with a MatrixRankWarning and
        # NaNs; near-singular systems return garbage. Verify the solve.
        if not np.all(np.isfinite(u)):
            raise np.linalg.LinAlgError(
                "steady solve produced non-finite values: the operator is "
                "singular. The boundary conditions must pin the solution level "
                "(pure Neumann/Periodic leave it defined only up to a constant); "
                "use a Dirichlet condition on at least part of the boundary."
            )
        residual = np.linalg.norm(L @ u + b)
        scale = max(np.linalg.norm(b), 1e-30)
        if residual > 1e-6 * scale:
            raise np.linalg.LinAlgError(
                f"steady solve failed: relative residual {residual / scale:.2e}. "
                "The operator is singular or severely ill-conditioned; check that "
                "the boundary conditions make the problem well-posed."
            )
        return u.reshape(self._interior_shape())

    # -- time integration ------------------------------------------------
    def integrate(self, dt, steps, u0, scheme="rk4", solver=None,
                  record_every=1, v0=None):
        """Integrate the time-dependent PDE.

        Args:
            dt: time step.
            steps: number of steps to take.
            u0: initial condition (interior-shaped array).
            scheme: ``"euler"``, ``"rk2"``, ``"rk4"`` (explicit; any terms), or
                ``"implicit_euler"`` / ``"crank_nicolson"`` (linear only,
                ``time_order=1``).
            v0: initial ``u_t`` for ``time_order=2`` (defaults to zeros).
            record_every: store every n-th step in the returned history.

        Returns:
            ``(times, history)`` where ``history[k]`` is interior-shaped.
        """
        if self.time_order == 0:
            raise ValueError("time_order=0 is steady; call solve_steady()")
        if self.time_order == 1:
            return self._integrate_first_order(dt, steps, u0, scheme, solver, record_every)
        return self._integrate_second_order(dt, steps, u0, v0, record_every)

    def _integrate_first_order(self, dt, steps, u0, scheme, solver, record_every):
        shape = self._interior_shape()
        u = np.asarray(u0, dtype=float).reshape(shape).ravel(order="C").copy()
        step = self._first_order_stepper(scheme, dt, solver)

        times = [0.0]
        history = [u.reshape(shape).copy()]
        for n in range(steps):
            u = step(u)
            # the final state is always recorded, even off the record_every beat
            if (n + 1) % record_every == 0 or (n + 1) == steps:
                times.append((n + 1) * dt)
                history.append(u.reshape(shape).copy())
        return np.array(times), np.asarray(history)

    def _first_order_stepper(self, scheme, dt, solver):
        if scheme in ti.EXPLICIT_STEPPERS:
            rhs = self._make_rhs()
            fn = ti.EXPLICIT_STEPPERS[scheme]
            return lambda u: fn(u, rhs, dt)

        if scheme in ("implicit_euler", "crank_nicolson", "cn"):
            L, b = self.assemble()
            solver = solver or LinearSolver()
            I = sparse.identity(self.grid.size, format="csr")
            if scheme == "implicit_euler":
                A = (I - dt * L).tocsr()
                return lambda u: solver.solve(A, u + dt * b)
            A = (I - 0.5 * dt * L).tocsr()
            B = (I + 0.5 * dt * L).tocsr()
            return lambda u: solver.solve(A, B.dot(u) + dt * b)

        raise ValueError(f"unknown scheme {scheme!r}")

    def _integrate_second_order(self, dt, steps, u0, v0, record_every):
        """Velocity-Verlet integration of ``u'' = f(u)`` (works for nonlinear f)."""
        shape = self._interior_shape()
        rhs = self._make_rhs()
        u = np.asarray(u0, dtype=float).reshape(shape).ravel(order="C").copy()
        if v0 is None:
            v = np.zeros_like(u)
        else:
            v = np.asarray(v0, dtype=float).reshape(shape).ravel(order="C").copy()

        a = rhs(u)
        times = [0.0]
        history = [u.reshape(shape).copy()]
        for n in range(steps):
            v = v + 0.5 * dt * a
            u = u + dt * v
            a = rhs(u)
            v = v + 0.5 * dt * a
            # the final state is always recorded, even off the record_every beat
            if (n + 1) % record_every == 0 or (n + 1) == steps:
                times.append((n + 1) * dt)
                history.append(u.reshape(shape).copy())
        return np.array(times), np.asarray(history)
