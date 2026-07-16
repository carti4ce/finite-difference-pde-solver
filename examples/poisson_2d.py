"""2D Poisson example using the term-based PDE layer."""
import numpy as np

from finite_differences import Grid
from finite_differences.poisson import solve_poisson


def main():
    grid = Grid(32, 32, lx=1.0, ly=1.0)

    def source(X, Y):
        return -2.0 * (np.pi ** 2) * np.sin(np.pi * X) * np.sin(np.pi * Y)

    u = solve_poisson(grid, source)
    X, Y = grid.mesh()
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    print("Solved Poisson. Solution shape:", u.shape)
    print("L2 error vs exact:", np.linalg.norm(u - u_exact))


if __name__ == '__main__':
    main()
