import numpy as np

from finite_differences import Grid, PDE, DiffusionTerm
from finite_differences.bc import Dirichlet


def analytic_u_1d(x, t, alpha):
    return np.sin(np.pi * x) * np.exp(-alpha * (np.pi ** 2) * t)


def test_crank_nicolson_1d():
    nx = 128
    alpha = 0.01
    T = 0.01
    dt = 1e-4
    steps = int(round(T / dt))
    grid = Grid(nx, 1, lx=1.0)
    x, _ = grid.mesh()
    pde = PDE(grid, [DiffusionTerm(alpha)], Dirichlet(0.0), time_order=1)
    _, hist = pde.integrate(dt=dt, steps=steps, u0=np.sin(np.pi * x), scheme="crank_nicolson")
    err = np.linalg.norm(hist[-1] - analytic_u_1d(x, T, alpha))
    assert err < 5e-3


def test_explicit_euler_1d_convergence():
    nx = 128
    alpha = 0.01
    grid = Grid(nx, 1, lx=1.0)
    dx = grid.dx
    dt = 0.4 * dx * dx / (2.0 * alpha)  # under the FTCS stability limit
    steps = 200
    T = dt * steps
    x, _ = grid.mesh()
    pde = PDE(grid, [DiffusionTerm(alpha)], Dirichlet(0.0), time_order=1)
    _, hist = pde.integrate(dt=dt, steps=steps, u0=np.sin(np.pi * x), scheme="euler")
    err = np.linalg.norm(hist[-1] - analytic_u_1d(x, T, alpha))
    assert err < 1e-2


def test_nonzero_dirichlet_reaches_boundary_value():
    """Nonzero Dirichlet data must actually drive the interior (regression)."""
    grid = Grid(24, 1, lx=1.0)
    x, _ = grid.mesh()
    pde = PDE(grid, [DiffusionTerm(0.5)], Dirichlet(3.0), time_order=1)
    _, hist = pde.integrate(dt=1e-3, steps=5000, u0=np.zeros_like(x), scheme="crank_nicolson")
    # steady state of the heat equation with u=3 on both ends is u==3 everywhere
    assert np.allclose(hist[-1], 3.0, atol=1e-3)
