"""finite_differences package

Basic package exports.
"""

from .grid import Grid
from .field import Field
from .bc import BC, Dirichlet, Neumann, Periodic

__all__ = ["Grid", "Field", "BC", "Dirichlet", "Neumann", "Periodic"]
