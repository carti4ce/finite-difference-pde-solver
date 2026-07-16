"""Simple 2D heat equation example (one step of setup + integration)."""
from finite_differences import Grid, PDE, DiffusionTerm
from finite_differences.bc import Dirichlet

import numpy as np


def main():
    grid = Grid(50, 50, lx=1.0, ly=1.0)
    X, Y = grid.mesh()
    u0 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.02)
    pde = PDE(grid, [DiffusionTerm(1e-3)], Dirichlet(0.0), time_order=1)
    times, history = pde.integrate(dt=1e-3, steps=10, u0=u0, scheme="crank_nicolson")
    print("Integrated", len(times) - 1, "steps. Final interior shape:", history[-1].shape)


if __name__ == '__main__':
    main()
