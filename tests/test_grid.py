import pytest
from finite_differences import Grid, Field


def test_grid_and_field_shapes():
    g = Grid(10, 1, lx=1.0)
    f = Field(g, ng=2)
    assert f.data.shape[0] == 10 + 4
    assert f.interior.shape[0] == 10

    g2 = Grid(8, 6, lx=1.0, ly=1.0)
    f2 = Field(g2, ng=1)
    assert f2.data.shape == (8 + 2, 6 + 2)
    assert f2.interior.shape == (8, 6)


def test_apply_dirichlet_bc():
    g = Grid(6, 6, lx=1.0, ly=1.0)
    f = Field(g, ng=1)
    import numpy as np
    arr = np.ones((6, 6)) * 2.0
    f.set_interior(arr)
    from finite_differences.bc import Dirichlet
    bc = Dirichlet(0.0)
    f.apply_bc(bc)
    # ghost corner should be 0.0
    assert f.data[0, 0] == 0.0
