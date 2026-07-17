"""Translate a validated `SolveRequest` into a `finite_differences.PDE` run.

Kept separate from the FastAPI route so it can be unit tested without an HTTP
client, and so `expr.py`'s safe evaluator is the only place user expressions
ever touch code.
"""
from __future__ import annotations

import numpy as np

from finite_differences import (
    Grid, PDE,
    DiffusionTerm, AdvectionTerm, ReactionTerm, SourceTerm, FunctionTerm,
)
from finite_differences.bc import Dirichlet, DirichletBox, Neumann, Periodic

from expr import ExpressionError, evaluate_expression
from schemas import SolveRequest, SolveResponse


class SolveError(ValueError):
    """A request was well-formed but could not be solved (bad expression,
    singular boundary conditions, etc). Maps to an HTTP 400."""


def _build_grid(spec):
    return Grid(spec.nx, spec.ny, lx=spec.lx, ly=spec.ly)


def _build_term(spec, grid):
    if spec.type == "diffusion":
        return DiffusionTerm(spec.coeff)
    if spec.type == "advection":
        return AdvectionTerm(spec.velocity, scheme=spec.scheme)
    if spec.type == "reaction":
        return ReactionTerm(spec.coeff)
    if spec.type == "source":
        return SourceTerm(_source_func(spec.expression, grid))
    if spec.type == "function":
        return FunctionTerm(_function_func(spec.expression))
    raise SolveError(f"unknown term type {spec.type!r}")


def _source_func(expression, grid):
    def source(X, Y=None):
        variables = {"x": X} if Y is None else {"x": X, "y": Y}
        try:
            return np.broadcast_to(
                np.asarray(evaluate_expression(expression, variables), dtype=float), X.shape
            )
        except ExpressionError as exc:
            raise SolveError(f"source expression error: {exc}") from exc
    return source


def _function_func(expression):
    def func(u, X, Y=None):
        variables = {"u": u, "x": X} if Y is None else {"u": u, "x": X, "y": Y}
        try:
            return np.broadcast_to(
                np.asarray(evaluate_expression(expression, variables), dtype=float), u.shape
            )
        except ExpressionError as exc:
            raise SolveError(f"function term expression error: {exc}") from exc
    return func


def _build_bc(spec):
    if spec.type == "dirichlet":
        return Dirichlet(spec.value)
    if spec.type == "dirichlet_box":
        return DirichletBox(left=spec.left, right=spec.right,
                            bottom=spec.bottom, top=spec.top)
    if spec.type == "neumann":
        return Neumann(spec.derivative)
    if spec.type == "periodic":
        return Periodic()
    raise SolveError(f"unknown boundary condition type {spec.type!r}")


def _build_initial_condition(spec, grid):
    X, Y = grid.mesh()
    dimension = "1D" if grid.ny == 1 else "2D"
    shape = (grid.nx,) if dimension == "1D" else (grid.nx, grid.ny)

    if spec.shape == "uniform":
        return np.full(shape, spec.intensity)

    if spec.shape == "gaussian":
        if dimension == "1D":
            center = spec.center[0] if spec.center else grid.lx / 2.0
            spread = spec.spread or grid.lx * 0.05
            return spec.intensity * np.exp(-((X - center) ** 2) / (2 * spread ** 2))
        cx, cy = spec.center if spec.center else (grid.lx / 2.0, grid.ly / 2.0)
        spread = spec.spread or min(grid.lx, grid.ly) * 0.05
        r_sq = (X - cx) ** 2 + (Y - cy) ** 2
        return spec.intensity * np.exp(-r_sq / (2 * spread ** 2))

    if spec.shape == "sine_wave":
        if dimension == "1D":
            return spec.intensity * np.sin(np.pi * X / grid.lx)
        return spec.intensity * np.sin(np.pi * X / grid.lx) * np.sin(np.pi * Y / grid.ly)

    if spec.shape == "random":
        rng = np.random.default_rng(spec.seed)
        return spec.intensity * rng.uniform(0, 1, shape)

    if spec.shape == "expression":
        variables = {"x": X} if dimension == "1D" else {"x": X, "y": Y}
        try:
            return np.broadcast_to(
                np.asarray(evaluate_expression(spec.expression, variables), dtype=float), shape
            ).copy()
        except ExpressionError as exc:
            raise SolveError(f"initial condition expression error: {exc}") from exc

    raise SolveError(f"unknown initial condition shape {spec.shape!r}")


def run_solve(request: SolveRequest) -> SolveResponse:
    grid = _build_grid(request.grid)
    terms = [_build_term(t, grid) for t in request.terms]
    bc = _build_bc(request.bc)
    u0 = _build_initial_condition(request.initial_condition, grid)

    pde = PDE(grid, terms, bc, time_order=request.time_order)

    if request.time_order == 0:
        try:
            u = pde.solve_steady()
        except np.linalg.LinAlgError as exc:
            raise SolveError(str(exc)) from exc
        times = [0.0]
        history = u[np.newaxis, ...]
    else:
        integ = request.integration
        v0 = None
        if request.time_order == 2 and integ.v0_expression:
            X, Y = grid.mesh()
            dimension = "1D" if grid.ny == 1 else "2D"
            variables = {"x": X} if dimension == "1D" else {"x": X, "y": Y}
            try:
                v0 = evaluate_expression(integ.v0_expression, variables)
            except ExpressionError as exc:
                raise SolveError(f"initial velocity expression error: {exc}") from exc
        try:
            times, history = pde.integrate(
                dt=integ.dt, steps=integ.steps, u0=u0, scheme=integ.scheme,
                record_every=integ.record_every, v0=v0,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise SolveError(str(exc)) from exc

    return SolveResponse(
        dimension=request.grid.dimension,
        nx=grid.nx,
        ny=grid.ny if request.grid.dimension == "2D" else 1,
        dx=grid.dx,
        dy=grid.dy if request.grid.dimension == "2D" else None,
        lx=grid.lx,
        ly=grid.ly,
        times=[float(t) for t in np.asarray(times).ravel()],
        history=np.asarray(history).tolist(),
        vmin=float(np.min(history)),
        vmax=float(np.max(history)),
    )
