# Finite Differences PDE solver

This project provides a modular finite-difference solver for PDEs in 1D and 2D.

## Features:

- Structured 1D and 2D Cartesian grids with node-centered coordinates
- Ghost-cell-based field storage for clean boundary condition handling
- Supported Boundary conditions: Dirichlet, Neumann, and Periodic
- Finite difference operators: central differences, 5-point Laplacian
- Time integrators: Explicit Euler, RK2, Implicit Euler, Crank-Nicolson
- Poisson equation solver
- Animated 2D visualisation utilities (GIF export)
- Jupyter notebooks for interactive demos

## Examples (2-D Heat Equation)

### Crank-Nicolson Scheme
![2D Heat: Crank-Nicolson](examples/example_plots/heat2d_CN_animation.gif)
### Explicit Euler Scheme
![2D Heat: Explicit Euler](examples/example_plots/heat2d_explicit_animation.gif)

## Quick start

Install dependencies (recommended into a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run examples in `examples/` or open the notebooks in `notebooks/`.

Running demos

Create a virtualenv and install dependencies (see `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the heat demo (explicit):

```bash
python3 examples/run_demo.py --mode explicit --nx 64 --ny 64 --steps 200
```

Run the Crank-Nicolson demo:

```bash
python3 examples/run_demo.py --mode cn --nx 64 --ny 64 --steps 50
```

## Notes:
- Implicit runs require `scipy` (sparse solvers). The explicit demo only requires `numpy` and `matplotlib`.
- Adjust `--dt` and `--alpha` to explore stability and diffusion rates.

## On Stability:
When using explicit time integration for the heat equation, the time step size must satisfy the following CFL stability condition:
    dt <= dx^2 / (2 * alpha)     (1D)
    dt <= 1 / (2 * alpha * (1/dx^2 + 1/dy^2))     (2D)
whereas the Crank-Nicolson Scheme is unconditionally stable.