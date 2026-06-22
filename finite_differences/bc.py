"""Boundary condition abstractions."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BC(ABC):
    """Base class for boundary conditions."""

    @abstractmethod
    def apply(self, field):
        """Apply BC to ghost cells of `field` (a `Field` instance)."""


class Dirichlet(BC):
    def __init__(self, value):
        self.value = value

    def apply(self, field):
        ng = field.ng + 1
        if field.grid.ny == 1:
            field.data[:ng] = self.value
            field.data[-ng:] = self.value
            return
        # left/right (ghost columns)
        field.data[:ng, ng:-ng] = self.value
        field.data[-ng:, ng:-ng] = self.value
        # bottom/top (ghost rows) including corners
        field.data[:, :ng] = self.value
        field.data[:, -ng:] = self.value


class Neumann(BC):
    def __init__(self, derivative=0.0):
        self.derivative = derivative

    def apply(self, field):
        # Simplified: copy interior to ghost (zero-gradient) or offset by derivative*dx
        ng = field.ng
        if field.grid.ny == 1:
            # left ghost = first interior value, right ghost = last interior value
            field.data[:ng] = field.data[ng]
            field.data[-ng:] = field.data[-ng-1]
            return
        # left ghost columns <- first interior column
        field.data[:ng, ng:-ng] = field.data[ng:ng+1, ng:-ng]
        # right ghost columns <- last interior column
        field.data[-ng:, ng:-ng] = field.data[-ng-1:-ng, ng:-ng]
        # bottom ghost rows <- first interior row
        field.data[:, :ng] = field.data[:, ng:ng+1]
        # top ghost rows <- last interior row
        field.data[:, -ng:] = field.data[:, -ng-1:-ng]


class Periodic(BC):
    def apply(self, field):
        ng = field.ng
        if field.grid.ny == 1:
            field.data[:ng] = field.data[-2*ng:-ng]
            field.data[-ng:] = field.data[ng:2*ng]
            return
        # left/right
        field.data[:ng, ng:-ng] = field.data[-2*ng:-ng, ng:-ng]
        field.data[-ng:, ng:-ng] = field.data[ng:2*ng, ng:-ng]
        # bottom/top
        field.data[:, :ng] = field.data[:, -2*ng:-ng]
        field.data[:, -ng:] = field.data[:, ng:2*ng]
