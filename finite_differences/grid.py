"""Grid utilities for structured Cartesian domains (1D and 2D)."""
from __future__ import annotations

import numpy as np
from typing import Tuple


class Grid:
    """Structured Cartesian grid.

    Supports 1D and 2D grids. Use `ny=1` for 1D.
    """

    def __init__(self, nx: int, ny: int = 1, lx: float = 1.0, ly: float = 1.0):
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be >= 1")
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        # Use node-centered interior grid points for Dirichlet problems:
        # spacing h = L/(n+1), interior nodes at i*h for i=1..n
        self.dx = self.lx / float(self.nx + 1)
        self.dy = self.ly / float(self.ny + 1) if self.ny > 1 else None

        self.spacing = {'x': self.dx} if ny == 1 else {'x': self.dx, 'y': self.dy}
        # node-centered coordinates (exclude physical boundary points)
        self.x = (np.arange(1, self.nx + 1)) * self.dx
        if self.ny > 1:
            self.y = (np.arange(1, self.ny + 1)) * self.dy
        else:
            self.y = None

    def axes(self) -> Tuple[str]:
        return list(self.spacing.keys())

    def shape(self) -> Tuple[int, int]:
        return (self.nx, self.ny)

    def mesh(self):
        """Return meshgrid arrays (X, Y). For 1D, returns (X, None)."""
        if self.ny == 1:
            return self.x, None
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        return X, Y

    def index_to_coord(self, i: int, j: int = 0) -> Tuple[float, float]:
        """Map integer cell index to cell-centered coordinates."""
        x = (i + 0.5) * self.dx
        y = (j + 0.5) * self.dy if self.ny > 1 else 0.0
        return x, y

    @property
    def size(self) -> int:
        """Total number of cells (nx*ny)."""
        return self.nx * self.ny

    def ij_to_index(self, i: int, j: int = 0) -> int:
        """Convert 2D cell indices to flat index (row-major: i + j*nx)."""
        return int(i * self.ny + j)

    def index_to_ij(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (i,j)."""
        j = int(idx % self.ny)
        i = int(idx // self.ny)
        return i, j
