"""Boundary condition tests: ghost fills, matrix/ghost consistency, and physics."""
import numpy as np
import pytest

from finite_differences import (
    Grid, Field, PDE, DiffusionTerm, AdvectionTerm, ReactionTerm, SourceTerm,
)
from finite_differences.bc import Dirichlet, DirichletBox, Neumann, Periodic
from finite_differences import operators as ops


# ---------------------------------------------------------------------------
# apply(): ghost cells get the right values
# ---------------------------------------------------------------------------

def test_dirichlet_apply_1d():
    grid = Grid(5, 1, lx=1.0)
    f = Field(grid, ng=1)
    f.set_interior(np.arange(1, 6, dtype=float))
    Dirichlet(9.0).apply(f)
    assert f.data[0] == 9.0 and f.data[-1] == 9.0


def test_dirichletbox_apply_2d_edges():
    grid = Grid(4, 3, lx=1.0, ly=1.0)
    f = Field(grid, ng=1)
    f.set_interior(np.ones((4, 3)))
    DirichletBox(left=1.0, right=2.0, bottom=3.0, top=4.0).apply(f)
    assert np.all(f.data[0, 1:-1] == 1.0)    # left interior-y ghosts
    assert np.all(f.data[-1, 1:-1] == 2.0)   # right
    assert np.all(f.data[1:-1, 0] == 3.0)    # bottom
    assert np.all(f.data[1:-1, -1] == 4.0)   # top


def test_neumann_zero_gradient_copies_interior():
    grid = Grid(5, 1, lx=1.0)
    f = Field(grid, ng=1)
    f.set_interior(np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
    Neumann(0.0).apply(f)
    assert f.data[0] == f.data[1]       # left ghost == first interior
    assert f.data[-1] == f.data[-2]     # right ghost == last interior


def test_neumann_reproduces_linear_profile():
    """For u = g*x (u''=0) the correct Neumann(g) ghost gives zero Laplacian."""
    grid = Grid(30, 1, lx=2.0)
    g = 1.7
    f = Field(grid, ng=1)
    f.set_interior(g * grid.x)
    Neumann(g).apply(f)
    assert np.max(np.abs(ops.laplacian(f))) < 1e-10


def test_periodic_wraps():
    grid = Grid(5, 1, lx=1.0)
    f = Field(grid, ng=1)
    f.set_interior(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    Periodic().apply(f)
    assert f.data[0] == 5.0    # left ghost == last interior
    assert f.data[-1] == 1.0   # right ghost == first interior


# ---------------------------------------------------------------------------
# Matrix/ghost consistency: L @ u + b == spatial_rhs(u) for every BC
# ---------------------------------------------------------------------------

BCS = [
    Dirichlet(0.7),
    DirichletBox(0.2, -0.5, 1.1, 0.3),
    Neumann(0.0),
    Neumann(0.4),
    Periodic(),
]

TERM_SETS = [
    [DiffusionTerm(0.9)],
    [DiffusionTerm(0.5), ReactionTerm(-1.3)],
    [AdvectionTerm((0.8, -0.6), scheme="upwind")],
    [AdvectionTerm(0.7, scheme="central")],
    [DiffusionTerm(0.3), AdvectionTerm((0.5, 0.4), scheme="upwind"),
     SourceTerm(lambda *a: 2.0 + 0.0 * a[0])],
]


@pytest.mark.parametrize("grid", [Grid(15, 1, lx=1.0), Grid(6, 5, lx=1.0, ly=1.3)])
@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("terms", TERM_SETS)
def test_assembled_operator_matches_ghost_rhs(grid, bc, terms):
    pde = PDE(grid, terms, bc, time_order=1)
    L, b = pde.assemble()
    rng = np.random.default_rng(3)
    f = Field(grid, ng=1)
    v = rng.standard_normal(f.interior.shape)
    f.set_interior(v)
    direct = pde.spatial_rhs(f).ravel(order="C")
    via_matrix = L @ v.ravel(order="C") + b
    assert np.allclose(direct, via_matrix, atol=1e-9)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def test_neumann_insulated_conserves_and_flattens():
    """Zero-flux heat: total heat is conserved and the profile relaxes to a
    (spatially uniform) mean."""
    grid = Grid(40, 1, lx=1.0)
    x = grid.x
    u0 = np.exp(-((x - 0.5) ** 2) / 0.02)
    pde = PDE(grid, [DiffusionTerm(0.05)], Neumann(0.0), time_order=1)
    _, hist = pde.integrate(dt=1e-3, steps=5000, u0=u0, scheme="crank_nicolson")
    assert abs(hist[-1].sum() - hist[0].sum()) < 1e-8 * abs(hist[0].sum())
    assert (hist[-1].max() - hist[-1].min()) < 1e-3


def test_periodic_diffusion_conserves_total():
    grid = Grid(50, 1, lx=1.0)
    x = grid.x
    u0 = np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x) + 1.0
    pde = PDE(grid, [DiffusionTerm(0.02)], Periodic(), time_order=1)
    _, hist = pde.integrate(dt=1e-3, steps=2000, u0=u0, scheme="crank_nicolson")
    assert abs(hist[-1].sum() - hist[0].sum()) < 1e-9 * abs(hist[0].sum())


def test_periodic_advection_central_is_nondissipative():
    grid = Grid(100, 1, lx=1.0)
    x = grid.x
    u0 = np.sin(2 * np.pi * x)
    pde = PDE(grid, [AdvectionTerm(0.5, scheme="central")], Periodic(), time_order=1)
    _, hist = pde.integrate(dt=5e-4, steps=1000, u0=u0, scheme="rk4")
    assert abs(hist[-1].sum() - hist[0].sum()) < 1e-9
    ratio = np.linalg.norm(hist[-1]) / np.linalg.norm(hist[0])
    assert abs(ratio - 1.0) < 1e-2


def test_periodic_prolongation_is_circulant_row_sums_zero():
    """The periodic Laplacian has zero row sums (it annihilates constants)."""
    grid = Grid(12, 1, lx=1.0)
    pde = PDE(grid, [DiffusionTerm(1.0)], Periodic(), time_order=1)
    L, b = pde.assemble()
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 0.0, atol=1e-10)
    assert np.allclose(b, 0.0, atol=1e-12)


def test_matrix_assembly_rejects_bc_without_prolongation():
    class CustomBC(Dirichlet):
        kind = None  # opts out of matrix assembly

    grid = Grid(8, 1, lx=1.0)
    pde = PDE(grid, [DiffusionTerm(1.0)], CustomBC(0.0), time_order=1)
    with pytest.raises(NotImplementedError):
        pde.assemble()
