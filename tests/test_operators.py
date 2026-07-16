"""Tests that the sparse stencil matrices reproduce the ghost-cell stencils.

For a padded array ``p`` (interior + one ghost layer), the stencil matrix ``S``
must satisfy ``S @ p.ravel() == stencil(field)`` where ``field`` is built from
the same padded data. This is the foundation of the ``L = S @ P`` assembly.
"""
import numpy as np
import pytest

from finite_differences.grid import Grid
from finite_differences.field import Field
from finite_differences import operators as ops


GRIDS = [Grid(9, 1, lx=1.3), Grid(6, 5, lx=1.0, ly=2.0)]


def _random_padded_field(grid, rng):
    """A Field with random interior *and* random ghost cells (ng=1)."""
    f = Field(grid, ng=1)
    f.data[...] = rng.standard_normal(f.data.shape)
    return f


@pytest.mark.parametrize("grid", GRIDS)
def test_laplacian_matrix_matches_stencil(grid):
    rng = np.random.default_rng(0)
    f = _random_padded_field(grid, rng)
    via_stencil = ops.laplacian(f).ravel(order="C")
    via_matrix = ops.laplacian_stencil_matrix(grid) @ f.data.ravel(order="C")
    assert np.allclose(via_stencil, via_matrix, atol=1e-10)


@pytest.mark.parametrize("grid", GRIDS)
def test_identity_matrix_matches_stencil(grid):
    rng = np.random.default_rng(1)
    f = _random_padded_field(grid, rng)
    via_stencil = ops.identity(f).ravel(order="C")
    via_matrix = ops.identity_stencil_matrix(grid) @ f.data.ravel(order="C")
    assert np.allclose(via_stencil, via_matrix, atol=1e-10)


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("scheme,sign", [
    ("central", 1.0), ("upwind", 1.0), ("upwind", -1.0),
    ("forward", 1.0), ("backward", 1.0),
])
def test_gradient_matrix_matches_stencil(grid, scheme, sign):
    rng = np.random.default_rng(2)
    f = _random_padded_field(grid, rng)
    for axis in range(1 if grid.ny == 1 else 2):
        via_stencil = ops.gradient(f, axis=axis, scheme=scheme, upwind_sign=sign).ravel(order="C")
        via_matrix = ops.gradient_stencil_matrix(
            grid, axis=axis, scheme=scheme, upwind_sign=sign) @ f.data.ravel(order="C")
        assert np.allclose(via_stencil, via_matrix, atol=1e-10)


def test_padded_size():
    assert ops.padded_size(Grid(9, 1)) == 11
    assert ops.padded_size(Grid(6, 5)) == 8 * 7


def test_gradient_stencil_orders():
    """Central difference is 2nd order; upwind is 1st order (sanity on a smooth f)."""
    errs = {}
    for n in (40, 80):
        grid = Grid(n, 1, lx=1.0)
        f = Field(grid, ng=1)
        x = grid.x
        f.set_interior(np.sin(x))
        # fill ghosts with the exact function so boundary error doesn't dominate
        f.data[0] = np.sin(0.0)
        f.data[-1] = np.sin((n + 1) * grid.dx)
        d_central = ops.gradient(f, scheme="central")
        errs[n] = np.max(np.abs(d_central - np.cos(x)))
    # halving h should cut the central-difference error by ~4x
    assert errs[80] < errs[40] / 3.5
