# Finite Differences PDE solver

A modular finite-difference PDE solver for 1D and 2D Cartesian domains. A PDE is written as one of

```
0    = f(u, u_x, ...)      (steady)
u_t  = f(u, u_x, ...)      (parabolic / advection-reaction-diffusion)
u_tt = f(u, u_x, ...)      (hyperbolic / wave)
```

where the spatial right-hand side f is a sum of composable terms (diffusion, advection,
reaction, sources, or arbitrary nonlinear functions). Steady problems are solved directly;
time-dependent ones by the **method of lines** with explicit or implicit integrators.

## Features

- Structured 1D and 2D Cartesian grids with node-centered coordinates
- Ghost-cell field storage for clean boundary condition handling
- Composable spatial terms: DiffusionTerm, AdvectionTerm (upwind or central),
  ReactionTerm, SourceTerm, and FunctionTerm for nonlinear right-hand sides
- Boundary conditions: Dirichlet, DirichletBox (per-edge values), Neumann
  (including nonzero flux), and Periodic — consistent across the explicit,
  implicit, and steady solver paths
- Time integrators: explicit Euler, RK2, RK4, implicit Euler, Crank-Nicolson,
  and velocity Verlet for second-order (wave) problems
- Static and animated plotting utilities (GIF export)

## Example gallery (2-D heat equation)

### Crank-Nicolson scheme
![2D Heat: Crank-Nicolson](examples/example_plots/heat2d_CN_animation.gif)
### Explicit Euler scheme
![2D Heat: Explicit Euler](examples/example_plots/heat2d_explicit_animation.gif)

## Usage

### Heat equation (implicit, insulated boundaries)

```python
import numpy as np
from finite_differences import Grid, PDE, DiffusionTerm, Neumann

grid = Grid(100, 1, lx=1.0)
u0 = np.exp(-((grid.x - 0.5) ** 2) / 0.01)

pde = PDE(grid, [DiffusionTerm(0.05)], Neumann(0.0), time_order=1)
times, history = pde.integrate(dt=1e-3, steps=2000, u0=u0, scheme="crank_nicolson")
```

With insulated (zero-flux) boundaries the pulse relaxes to its mean and total heat
is conserved. Swap in `Dirichlet(0.0)` to let it drain out through cold walls instead.

### Steady Poisson equation

```python
from finite_differences import Grid, PDE, DiffusionTerm, SourceTerm, Dirichlet

grid = Grid(48, 48, lx=1.0, ly=1.0)
source = lambda X, Y: 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

pde = PDE(grid, [DiffusionTerm(1.0), SourceTerm(source)], Dirichlet(0.0), time_order=0)
u = pde.solve_steady()
```

`time_order=0` solves `0 = f(u)` directly. Nonzero Dirichlet and Neumann data are
honored via the assembled boundary lift. Note the boundary conditions must pin the
solution level (a pure-Neumann or periodic steady problem is singular and raises an error).

### 2D advection–diffusion

```python
from finite_differences import Grid, PDE, DiffusionTerm, AdvectionTerm, Dirichlet

grid = Grid(48, 48, lx=1.0, ly=1.0)
X, Y = grid.mesh()
u0 = np.exp(-((X - 0.25) ** 2 + (Y - 0.25) ** 2) / 0.005)

pde = PDE(grid, [DiffusionTerm(2e-3), AdvectionTerm((1.0, 0.8), scheme="upwind")],
          Dirichlet(0.0), time_order=1)
times, history = pde.integrate(dt=2e-3, steps=250, u0=u0, scheme="crank_nicolson")
```

Multiple linear terms assemble into a single sparse operator, so implicit stepping
works far above the explicit stability limit.

### Wave equation

```python
pde = PDE(grid, [DiffusionTerm(c**2)], Dirichlet(0.0), time_order=2)
times, history = pde.integrate(dt=5e-4, steps=2000, u0=u0)   # velocity Verlet; v0 optional
```

### Nonlinear terms (Fisher–KPP)

```python
from finite_differences import FunctionTerm

grid = Grid(300, 1, lx=60.0)
u0 = np.where(np.abs(grid.x - 30.0) < 1.0, 1.0, 0.0)   # small seed population

terms = [DiffusionTerm(1.0), FunctionTerm(lambda u, X, Y: u * (1.0 - u))]
pde = PDE(grid, terms, Dirichlet(0.0), time_order=1)
times, history = pde.integrate(dt=5e-3, steps=2000, u0=u0, scheme="rk4")
```

The seed grows to the carrying capacity and spreads as a traveling front. Nonlinear
terms are integrated explicitly (`euler`, `rk2`, `rk4`), so the diffusive CFL limit
on `dt` applies (see below); steady solves and implicit schemes require all-linear terms.

For a worked tour of all of the above with plots and animations, see
[`examples/feature_tour.ipynb`](examples/feature_tour.ipynb).

## Package layout

| Module | Purpose |
|---|---|
| `finite_differences.grid` | `Grid`: node-centered 1D/2D Cartesian grids |
| `finite_differences.field` | `Field`: grid data with ghost cells |
| `finite_differences.terms` | Composable spatial terms making up `f(u)` |
| `finite_differences.bc` | Boundary conditions (ghost fills + matrix prolongations) |
| `finite_differences.pde` | `PDE`: assembly, steady solves, method-of-lines integration |
| `finite_differences.operators` | Ghost-cell stencils and sparse stencil matrices |
| `finite_differences.time_integrators` | Explicit steppers (Euler, RK2, RK4) |
| `finite_differences.solvers` | Sparse/dense linear solver adapters |
| `finite_differences.utils` | Static and animated plotting helpers |

## Example gallery (2-D heat equation)

### Crank-Nicolson scheme
![2D Heat: Crank-Nicolson](examples/example_plots/heat2d_CN_animation.gif)
### Explicit Euler scheme
![2D Heat: Explicit Euler](examples/example_plots/heat2d_explicit_animation.gif)

## On stability

Explicit time integration of the heat equation requires the CFL stability condition:

```
dt <= dx^2 / (2 * alpha)                          (1D)
dt <= 1 / (2 * alpha * (1/dx^2 + 1/dy^2))         (2D)
```

whereas the implicit Euler and Crank-Nicolson schemes are unconditionally stable.
For advection, upwind differencing is robust but adds numerical diffusion; central
differencing is second-order accurate but can oscillate on under-resolved fronts.
