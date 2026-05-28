"""Field container for discrete scalar/vector fields on a Grid."""
from __future__ import annotations

import numpy as np
from typing import Tuple


class Field:
    """Stores field data with optional ghost cells.
    
    Data layout: full array includes ghost cells. Use `ng` to control ghost width.
    """

    def __init__(self, grid, ng: int = 1, dtype=float):
        self.grid = grid
        self.ng = int(ng)
        self.dtype = dtype
        nx, ny = grid.nx, grid.ny
        if ny == 1:
            self._shape = (nx + 2 * self.ng,)
            self.data = np.zeros(self._shape, dtype=self.dtype)
        else:
            self._shape = (nx + 2 * self.ng, ny + 2 * self.ng)
            self.data = np.zeros(self._shape, dtype=self.dtype)

    @property
    def interior(self):
        """Return the interior view (without ghost cells)."""
        ng = self.ng
        if self.grid.ny == 1:
            return self.data[ng:-ng]
        return self.data[ng:-ng, ng:-ng]

    def set_interior(self, arr: np.ndarray):
        arr = np.asarray(arr)
        if self.grid.ny == 1 and arr.ndim == 1:
            self.data[self.ng:-self.ng] = arr
            return
        self.data[self.ng:-self.ng, self.ng:-self.ng] = arr

    def apply_bc(self, bc):
        """Apply a boundary condition object to this field's ghost cells."""
        bc.apply(self)

    def flat_interior(self):
        """Return a flattened copy of the interior cells in row-major order."""
        arr = self.interior
        if arr.ndim == 1:
            return arr.copy()
        return arr.ravel(order="C").copy()

    def copy(self) -> "Field":
        f = Field(self.grid, ng=self.ng, dtype=self.dtype)
        f.data[...] = self.data
        return f
