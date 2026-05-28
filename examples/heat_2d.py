"""Simple heat equation example (setup only)."""
from finite_differences import Grid, Field
from finite_differences.bc import Dirichlet
from finite_differences.time_integrators import explicit_euler

import numpy as np

def main():
    grid = Grid(50, 50, lx=1.0, ly=1.0)
    f = Field(grid, ng=1)
    # initial condition: gaussian
    X, Y = grid.mesh()
    u0 = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.02)
    f.set_interior(u0)
    bc = Dirichlet(0.0)
    bc.apply(f)
    print("Setup complete. interior shape:", f.interior.shape)

if __name__ == '__main__':
    main()
