"""finite_differences package.

Term-based finite-difference PDE solver. Build a spatial operator from
composable `terms`, wrap it in a `PDE` with a boundary condition and a time
order, then solve (steady) or integrate (method of lines).
"""

from .grid import Grid
from .field import Field
from .bc import BC, Dirichlet, DirichletBox, Neumann, Periodic
from .pde import PDE
from .terms import (
    SpatialTerm,
    DiffusionTerm,
    AdvectionTerm,
    ReactionTerm,
    SourceTerm,
    FunctionTerm,
)

__all__ = [
    "Grid",
    "Field",
    "BC",
    "Dirichlet",
    "DirichletBox",
    "Neumann",
    "Periodic",
    "PDE",
    "SpatialTerm",
    "DiffusionTerm",
    "AdvectionTerm",
    "ReactionTerm",
    "SourceTerm",
    "FunctionTerm",
]
