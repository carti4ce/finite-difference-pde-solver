import numpy as np

from finite_differences.grid import Grid
from finite_differences.field import Field
from finite_differences.bc import Dirichlet
from finite_differences.operators import laplacian
from finite_differences.operators import laplacian_5pt
from finite_differences.time_integrators import crank_nicolson
from finite_differences.solvers import LinearSolver


def analytic_u_1d(x, t, alpha):
    return np.sin(np.pi * x) * np.exp(-alpha * (np.pi ** 2) * t)


def source_free(*args, **kwargs):
    return 0.0


def test_crank_nicolson_1d():
    nx = 128
    alpha = 0.01
    T = 0.01
    dt = 1e-4
    steps = int(T / dt)
    grid = Grid(nx, 1, lx=1.0)
    f = Field(grid, ng=1)
    x, _ = grid.mesh()
    u0 = np.sin(np.pi * x)
    f.set_interior(u0)
    bc = Dirichlet(0.0)
    solver = LinearSolver()
    for n in range(steps):
        bc.apply(f)
        f = crank_nicolson(f, laplacian_5pt, dt, alpha, solver)
    bc.apply(f)
    u_num = f.interior
    u_exact = analytic_u_1d(x, T, alpha)
    err = np.linalg.norm(u_num - u_exact)
    assert err < 5e-3


def test_explicit_euler_1d_convergence():
    nx = 128
    alpha = 0.01
    grid = Grid(nx, 1, lx=1.0)
    dx = grid.dx
    # stability limit for 1D FTCS: dt <= dx^2/(2*alpha)
    dt = 0.4 * dx * dx / (2.0 * alpha)
    T = dt * 200
    steps = int(T / dt)
    f = Field(grid, ng=1)
    x, _ = grid.mesh()
    u0 = np.sin(np.pi * x)
    f.set_interior(u0)
    bc = Dirichlet(0.0)
    for n in range(steps):
        bc.apply(f)
        lap = laplacian(f)
        u_new = f.interior + dt * alpha * lap
        f.set_interior(u_new)
    bc.apply(f)
    u_num = f.interior
    u_exact = analytic_u_1d(x, T, alpha)
    err = np.linalg.norm(u_num - u_exact)
    assert err < 1e-2
