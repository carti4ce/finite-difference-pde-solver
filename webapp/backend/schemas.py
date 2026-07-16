"""Request/response models for the PDE solve API.

Numeric limits (grid size, step count, frame count) are enforced here so a
request can't force the server into a multi-gigabyte response or a
minutes-long solve — this is a public HTTP endpoint, not a trusted script.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

MAX_N = 256
MAX_STEPS = 20_000
MAX_FRAMES = 600


class GridSpec(BaseModel):
    dimension: Literal["1D", "2D"]
    nx: int = Field(ge=3, le=MAX_N)
    ny: int = Field(default=1, ge=1, le=MAX_N)
    lx: float = Field(default=1.0, gt=0)
    ly: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_ny(self):
        if self.dimension == "2D" and self.ny < 3:
            raise ValueError("ny must be >= 3 for a 2D grid")
        if self.dimension == "1D":
            self.ny = 1
        return self


# --- spatial terms ----------------------------------------------------------

class DiffusionTermSpec(BaseModel):
    type: Literal["diffusion"]
    coeff: float = 1.0


class AdvectionTermSpec(BaseModel):
    type: Literal["advection"]
    velocity: Union[float, tuple[float, float]] = 1.0
    scheme: Literal["upwind", "central"] = "upwind"


class ReactionTermSpec(BaseModel):
    type: Literal["reaction"]
    coeff: float = 1.0


class SourceTermSpec(BaseModel):
    type: Literal["source"]
    expression: str = Field(min_length=1, max_length=500)


class FunctionTermSpec(BaseModel):
    type: Literal["function"]
    expression: str = Field(min_length=1, max_length=500)


TermSpec = Annotated[
    Union[DiffusionTermSpec, AdvectionTermSpec, ReactionTermSpec,
          SourceTermSpec, FunctionTermSpec],
    Field(discriminator="type"),
]


# --- boundary conditions ------------------------------------------------

class DirichletBCSpec(BaseModel):
    type: Literal["dirichlet"]
    value: float = 0.0


class DirichletBoxBCSpec(BaseModel):
    type: Literal["dirichlet_box"]
    left: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    top: float = 0.0


class NeumannBCSpec(BaseModel):
    type: Literal["neumann"]
    derivative: float = 0.0


class PeriodicBCSpec(BaseModel):
    type: Literal["periodic"]


BCSpec = Annotated[
    Union[DirichletBCSpec, DirichletBoxBCSpec, NeumannBCSpec, PeriodicBCSpec],
    Field(discriminator="type"),
]


# --- initial condition -------------------------------------------------

class InitialConditionSpec(BaseModel):
    shape: Literal["uniform", "gaussian", "sine_wave", "random", "expression"] = "gaussian"
    intensity: float = 1.0
    center: tuple[float, ...] | None = None
    spread: float | None = Field(default=None, gt=0)
    seed: int | None = None
    expression: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _check_expression(self):
        if self.shape == "expression" and not self.expression:
            raise ValueError("expression is required when shape='expression'")
        return self


# --- time integration ----------------------------------------------------

class IntegrationSpec(BaseModel):
    scheme: Literal["euler", "rk2", "rk4", "implicit_euler", "crank_nicolson"] = "rk4"
    dt: float = Field(gt=0)
    steps: int = Field(ge=1, le=MAX_STEPS)
    record_every: int = Field(default=1, ge=1)
    v0_expression: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _check_frame_count(self):
        num_frames = self.steps // self.record_every + 1
        if num_frames > MAX_FRAMES:
            raise ValueError(
                f"requested {num_frames} frames (steps/record_every), max is "
                f"{MAX_FRAMES}; increase record_every or reduce steps"
            )
        return self


# --- top-level request/response -------------------------------------------

class SolveRequest(BaseModel):
    grid: GridSpec
    time_order: Literal[0, 1, 2] = 1
    terms: list[TermSpec] = Field(min_length=1, max_length=8)
    bc: BCSpec
    initial_condition: InitialConditionSpec = Field(default_factory=InitialConditionSpec)
    integration: IntegrationSpec | None = None

    @model_validator(mode="after")
    def _check_integration_present(self):
        if self.time_order != 0 and self.integration is None:
            raise ValueError("integration settings are required when time_order != 0")
        return self


class SolveResponse(BaseModel):
    dimension: Literal["1D", "2D"]
    nx: int
    ny: int
    dx: float
    dy: float | None
    lx: float
    ly: float
    times: list[float]
    # 1D: history[frame][i]; 2D: history[frame][i][j]
    history: list
    vmin: float
    vmax: float
