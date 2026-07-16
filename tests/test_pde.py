import numpy as np
import pytest

from finite_differences import (
    Grid, PDE,
    DiffusionTerm, AdvectionTerm, ReactionTerm, SourceTerm, FunctionTerm,
)
from finite_differences.bc import Dirichlet, DirichletBox


def test_steady_diffusion_reaction_1d():
    """u'' - u = 0, u(0)=1, u(1)=0 has closed form sinh((1-x))/sinh(1)."""
    grid = Grid(64, 1, lx=1.0)
    pde = PDE(grid, [DiffusionTerm(1.0), ReactionTerm(-1.0)],
              DirichletBox(left=1.0, right=0.0), time_order=0)
    u = pde.solve_steady()
    x, _ = grid.mesh()
    exact = np.sinh(1.0 - x) / np.sinh(1.0)
    assert np.linalg.norm(u - exact) < 1e-2


def test_advection_translates_profile():
    """Constant-velocity advection moves a bump to the right at speed v."""
    grid = Grid(200, 1, lx=2.0)
    x, _ = grid.mesh()
    u0 = np.exp(-((x - 0.5) ** 2) / 0.01)
    v = 1.0
    pde = PDE(grid, [AdvectionTerm(v, scheme="upwind")], Dirichlet(0.0), time_order=1)
    dt = 0.5 * grid.dx
    steps = 100
    _, hist = pde.integrate(dt=dt, steps=steps, u0=u0, scheme="rk4")
    shift = x[np.argmax(hist[-1])] - x[np.argmax(hist[0])]
    assert abs(shift - v * dt * steps) < 5 * grid.dx


def test_wave_standing_mode():
    """u_tt = c^2 u_xx with u=sin(pi x): exact u(x,t)=sin(pi x) cos(pi c t)."""
    grid = Grid(100, 1, lx=1.0)
    c = 1.0
    x, _ = grid.mesh()
    pde = PDE(grid, [DiffusionTerm(c ** 2)], Dirichlet(0.0), time_order=2)
    T = 0.5
    dt = 1e-3
    _, hist = pde.integrate(dt=dt, steps=int(round(T / dt)), u0=np.sin(np.pi * x))
    exact = np.sin(np.pi * x) * np.cos(np.pi * c * T)
    assert np.linalg.norm(hist[-1] - exact) < 5e-3


def test_nonlinear_reaction_bounded_growth():
    """Fisher-KPP: small seed grows but stays bounded by the carrying capacity."""
    grid = Grid(80, 1, lx=10.0)
    x, _ = grid.mesh()
    u0 = 0.01 * np.exp(-((x - 5.0) ** 2))
    pde = PDE(
        grid,
        [DiffusionTerm(1.0), FunctionTerm(lambda u, X, Y: u * (1.0 - u))],
        Dirichlet(0.0),
        time_order=1,
    )
    _, hist = pde.integrate(dt=1e-3, steps=2000, u0=u0, scheme="rk4")
    assert hist[-1].max() > hist[0].max()
    assert hist[-1].max() <= 1.0 + 1e-6


def test_source_term_steady_1d():
    """0 = u'' + 2  with u(0)=u(1)=0  ->  u = x(1-x)."""
    grid = Grid(50, 1, lx=1.0)
    pde = PDE(grid, [DiffusionTerm(1.0), SourceTerm(lambda X: 2.0 * np.ones_like(X))],
              Dirichlet(0.0), time_order=0)
    u = pde.solve_steady()
    x, _ = grid.mesh()
    exact = x * (1.0 - x)
    assert np.linalg.norm(u - exact) < 1e-10


def test_nonlinear_assemble_rejected():
    grid = Grid(10, 1, lx=1.0)
    pde = PDE(grid, [FunctionTerm(lambda u, X, Y: u * u)], Dirichlet(0.0), time_order=1)
    with pytest.raises(ValueError):
        pde.assemble()


def test_poisson_2d_manufactured():
    """-2*pi^2 sin(pi x) sin(pi y) source recovers sin(pi x) sin(pi y)."""
    grid = Grid(32, 32, lx=1.0, ly=1.0)
    X, Y = grid.mesh()
    exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    pde = PDE(
        grid,
        [DiffusionTerm(1.0), SourceTerm(lambda X, Y: -(-2 * np.pi ** 2) * np.sin(np.pi * X) * np.sin(np.pi * Y))],
        Dirichlet(0.0),
        time_order=0,
    )
    u = pde.solve_steady()
    assert np.linalg.norm(u - exact) < 0.5 * grid.dx


def test_laplace_2d_edge_values():
    """Laplace with one hot edge: interior stays within the edge data (max
    principle) and is warm near that edge, cold near the opposite one."""
    grid = Grid(40, 40, lx=1.0, ly=1.0)
    pde = PDE(grid, [DiffusionTerm(1.0)],
              DirichletBox(left=1.0, right=0.0, bottom=0.0, top=0.0), time_order=0)
    u = pde.solve_steady()
    assert u.min() >= -1e-9 and u.max() <= 1.0 + 1e-9   # discrete maximum principle
    assert u[0, :].mean() > 0.5      # near the hot left edge
    assert u[-1, :].mean() < 0.05    # near the cold right edge


def test_implicit_euler_matches_crank_nicolson_at_steady():
    """Both implicit schemes reach the same (correct) steady state."""
    grid = Grid(32, 1, lx=1.0)
    x, _ = grid.mesh()
    u0 = np.zeros_like(x)
    results = {}
    for scheme in ("implicit_euler", "crank_nicolson"):
        pde = PDE(grid, [DiffusionTerm(0.4)], Dirichlet(2.0), time_order=1)
        _, hist = pde.integrate(dt=2e-3, steps=6000, u0=u0, scheme=scheme)
        results[scheme] = hist[-1]
    assert np.allclose(results["implicit_euler"], 2.0, atol=1e-3)
    assert np.allclose(results["implicit_euler"], results["crank_nicolson"], atol=1e-3)


def test_singular_steady_solve_raises():
    """Pure Neumann/Periodic steady problems are singular (solution defined
    only up to a constant); solve_steady must raise, not return NaNs."""
    from finite_differences.bc import Neumann, Periodic
    import warnings

    grid = Grid(20, 1, lx=1.0)
    for bc in (Neumann(0.0), Periodic()):
        pde = PDE(grid, [DiffusionTerm(1.0), SourceTerm(lambda x: np.sin(np.pi * x))],
                  bc, time_order=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scipy MatrixRankWarning
            with pytest.raises(np.linalg.LinAlgError):
                pde.solve_steady()


def test_wellposed_steady_solve_passes_residual_check():
    """The residual guard must not reject a healthy solve."""
    grid = Grid(64, 1, lx=1.0)
    pde = PDE(grid, [DiffusionTerm(1.0), SourceTerm(lambda x: np.sin(np.pi * x))],
              Dirichlet(0.0), time_order=0)
    u = pde.solve_steady()
    assert np.all(np.isfinite(u))


@pytest.mark.parametrize("time_order", [1, 2])
def test_record_every_always_records_final_state(time_order):
    """steps not divisible by record_every must still record the final step."""
    grid = Grid(16, 1, lx=1.0)
    u0 = np.sin(np.pi * grid.x)
    pde = PDE(grid, [DiffusionTerm(0.1)], Dirichlet(0.0), time_order=time_order)
    dt, steps = 1e-3, 100
    times, hist = pde.integrate(dt=dt, steps=steps, u0=u0, record_every=30)
    assert times[-1] == pytest.approx(steps * dt)
    assert list(times) == pytest.approx([0.0, 0.03, 0.06, 0.09, 0.1])
    assert len(hist) == len(times)


def test_record_every_exact_multiple_no_duplicate():
    grid = Grid(16, 1, lx=1.0)
    u0 = np.sin(np.pi * grid.x)
    pde = PDE(grid, [DiffusionTerm(0.1)], Dirichlet(0.0), time_order=1)
    times, hist = pde.integrate(dt=1e-3, steps=100, u0=u0, record_every=50)
    assert list(times) == pytest.approx([0.0, 0.05, 0.1])
    assert len(hist) == 3


def test_2d_heat_explicit_matches_implicit():
    """Explicit RK4 and Crank-Nicolson agree on a short 2D diffusion run."""
    grid = Grid(16, 16, lx=1.0, ly=1.0)
    X, Y = grid.mesh()
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    dt = 2e-4
    steps = 50
    hists = {}
    for scheme in ("rk4", "crank_nicolson"):
        pde = PDE(grid, [DiffusionTerm(0.1)], Dirichlet(0.0), time_order=1)
        _, hist = pde.integrate(dt=dt, steps=steps, u0=u0, scheme=scheme)
        hists[scheme] = hist[-1]
    assert np.linalg.norm(hists["rk4"] - hists["crank_nicolson"]) < 1e-3
