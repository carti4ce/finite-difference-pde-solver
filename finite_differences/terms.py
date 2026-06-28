from abc import ABC, abstractmethod
import numpy as np

class SpatialTerm(ABC):

    @abstractmethod
    def apply(self, L, grid):
        """Apply contributing terms to sparse matrix L"""

class DiffusionTerm(SpatialTerm):

    def __init__(self, D):
        self.D = D

    def apply(self, L, grid):
        return
    
class AdvectionTerm(SpatialTerm):

    def __init__(self, a, scheme="upwind"):
        self.a = a
        self.scheme = scheme        # Can be 'upwind' or 'central_difference'

    def apply(self, L, grid):
        return
    
class ReactionTerm(SpatialTerm):

    def __init__(self, c):
        self.c = c

    def apply(self, L, grid):
        return