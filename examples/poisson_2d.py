"""Poisson example (skeleton)"""
from finite_differences import Grid, Field
from finite_differences.operators import laplacian_5pt

def main():
    grid = Grid(32, 32, lx=1.0, ly=1.0)
    A = laplacian_5pt(grid)
    print("Assembled operator shape:", A.shape)

if __name__ == '__main__':
    main()
