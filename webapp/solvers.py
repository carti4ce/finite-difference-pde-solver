"""Webapp solver entry points, built on the term-based `PDE` layer."""
import numpy as np

from finite_differences import Grid
from finite_differences.bc import Dirichlet, DirichletBox
from finite_differences.pde import PDE
from finite_differences.terms import DiffusionTerm


def compute_stable_dt(dx, dy, alpha):
    """Compute a stable explicit time step (kept for callers that want a hint).

    For 1D: dt <= dx^2 / (2*alpha). For 2D: dt <= 1 / (2*alpha*(1/dx^2 + 1/dy^2)).
    Returns 80% of the limit for a safety margin.
    """
    if dy is None or dy == 0:
        dt_max = dx**2 / (2.0 * alpha)
    else:
        inv_r_squared = (1.0 / dx**2) + (1.0 / dy**2)
        dt_max = 1.0 / (2.0 * alpha * inv_r_squared)
    return 0.8 * dt_max


def solve_heat_equation(
    dimension,
    grid_bounds,
    initial_condition_array,
    alpha=0.1,
    dt=None,
    num_steps=50,
    bc_value=0.0,
):
    """Solve the heat equation ``u_t = alpha * laplacian(u)`` (Crank-Nicolson).

    Returns ``(time_array, solution_history)`` where ``solution_history`` has
    shape ``(num_steps+1, nx)`` in 1D or ``(num_steps+1, nx, ny)`` in 2D.
    """
    if dimension == "1D":
        x_min, x_max = grid_bounds
        nx = initial_condition_array.shape[0]
        grid = Grid(nx, 1, lx=x_max - x_min, ly=1.0)
    else:  # 2D
        (x_min, x_max), (y_min, y_max) = grid_bounds
        nx, ny = initial_condition_array.shape
        grid = Grid(nx, ny, lx=x_max - x_min, ly=y_max - y_min)

    if dt is None:
        dt = compute_stable_dt(grid.dx, grid.dy, alpha)

    pde = PDE(grid, [DiffusionTerm(alpha)], Dirichlet(bc_value), time_order=1)
    times, history = pde.integrate(
        dt=dt, steps=num_steps, u0=initial_condition_array, scheme="crank_nicolson"
    )
    return times, history


def solve_laplace_equation(
    dimension,
    grid_bounds,
    boundary_conditions,
):
    """Solve Laplace's equation ``laplacian(u) = 0`` with per-edge Dirichlet data.

    ``boundary_conditions`` keys: ``x_left``/``x_right`` (1D) plus
    ``y_bottom``/``y_top`` (2D). Returns ``(nx,)`` in 1D or ``(nx, ny)`` in 2D.
    """
    if dimension == "1D":
        x_min, x_max = grid_bounds
        nx = 50
        grid = Grid(nx, 1, lx=x_max - x_min, ly=1.0)
        bc = DirichletBox(
            left=boundary_conditions.get("x_left", 0.0),
            right=boundary_conditions.get("x_right", 0.0),
        )
    else:  # 2D
        (x_min, x_max), (y_min, y_max) = grid_bounds
        nx = ny = 50
        grid = Grid(nx, ny, lx=x_max - x_min, ly=y_max - y_min)
        bc = DirichletBox(
            left=boundary_conditions.get("x_left", 0.0),
            right=boundary_conditions.get("x_right", 0.0),
            bottom=boundary_conditions.get("y_bottom", 0.0),
            top=boundary_conditions.get("y_top", 0.0),
        )

    pde = PDE(grid, [DiffusionTerm(1.0)], bc, time_order=0)
    return pde.solve_steady()
