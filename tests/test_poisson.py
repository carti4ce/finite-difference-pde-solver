import numpy as np

from finite_differences.grid import Grid
from finite_differences.poisson import solve_poisson


def analytic_u(X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def source_f(X, Y):
    # Laplace u = -2*pi^2 * sin(pi x) sin(pi y)
    return -2.0 * (np.pi ** 2) * analytic_u(X, Y)


def test_poisson_manufactured_solution():
    nx = 32
    ny = 32
    grid = Grid(nx, ny, lx=1.0, ly=1.0)
    u_num = solve_poisson(grid, lambda X, Y: source_f(X, Y))
    X, Y = grid.mesh()
    u_exact = analytic_u(X, Y)
    err = np.linalg.norm(u_num - u_exact.ravel(order="C") if u_num.ndim==1 else u_num - u_exact)
    dx = grid.dx
    # L2 error for second-order scheme on Nx points typically scales ~ O(h)
    # Acceptable tolerance proportional to grid spacing
    assert err < 0.5 * dx
