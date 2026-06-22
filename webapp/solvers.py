import numpy as np
from finite_differences import Grid, Field
from finite_differences.bc import Dirichlet, Neumann
from finite_differences.operators import laplacian_5pt
from finite_differences.time_integrators import crank_nicolson
from finite_differences.solvers import LinearSolver
from finite_differences.poisson import solve_poisson


def compute_stable_dt(dx, dy, alpha):
    """Compute stable time step for explicit Euler using CFL condition.

    For 1D: dt <= dx^2 / (2*alpha)
    For 2D: dt <= 1 / (2*alpha*(1/dx^2 + 1/dy^2))
    We return 80% of the limit for safety margin.
    """
    if dy is None or dy == 0:
        # 1D case
        dt_max = dx**2 / (2.0 * alpha)
    else:
        # 2D case
        inv_r_squared = (1.0 / dx**2) + (1.0 / dy**2)
        dt_max = 1.0 / (2.0 * alpha * inv_r_squared)
    return 0.8 * dt_max


def solve_heat_equation(
    dimension,
    grid_bounds,
    initial_condition_array,
    alpha=0.1,
    dt=None,
    num_steps=50,
    bc_value=0.0,
):
    """Solve heat equation using Crank-Nicolson (unconditionally stable).

    Args:
        dimension: "1D" or "2D"
        grid_bounds: tuple (x_min, x_max) for 1D or ((x_min, x_max), (y_min, y_max)) for 2D
        initial_condition_array: numpy array matching grid interior shape
        alpha: thermal diffusivity (default 0.1)
        dt: time step (computed automatically if None)
        num_steps: number of time steps
        bc_value: Dirichlet boundary condition value (default 0.0)

    Returns:
        tuple: (time_array, solution_history)
            - time_array: 1D array of time values
            - solution_history: array of shape (num_steps+1, nx) for 1D or (num_steps+1, nx, ny) for 2D
    """
    if dimension == "1D":
        x_min, x_max = grid_bounds
        nx = initial_condition_array.shape[0]
        ny = 1
        grid = Grid(nx, ny, lx=x_max - x_min, ly=1.0)
    else:  # 2D
        (x_min, x_max), (y_min, y_max) = grid_bounds
        nx, ny = initial_condition_array.shape
        grid = Grid(nx, ny, lx=x_max - x_min, ly=y_max - y_min)

    dx = grid.dx
    dy = grid.dy

    if dt is None:
        dt = compute_stable_dt(dx, dy, alpha)

    # Initialize field
    f = Field(grid, ng=1)
    f.set_interior(initial_condition_array)

    # Boundary condition
    bc = Dirichlet(bc_value)

    # Linear solver for implicit steps
    solver = LinearSolver()

    # Record solution history
    solution_history = [f.interior.copy()]
    time_array = np.array([0.0])

    bc.apply(f)

    # Time stepping
    for step in range(num_steps):
        f = crank_nicolson(f, laplacian_5pt, dt, alpha, solver)
        bc.apply(f)
        solution_history.append(f.interior.copy())
        time_array = np.append(time_array, (step + 1) * dt)

    # Return stacked history
    solution_array = np.asarray(solution_history)

    return time_array, solution_array


def solve_laplace_equation(
    dimension,
    grid_bounds,
    boundary_conditions,
):
    """Solve Laplace equation (Poisson with RHS=0).

    Args:
        dimension: "1D" or "2D"
        grid_bounds: tuple (x_min, x_max) for 1D or ((x_min, x_max), (y_min, y_max)) for 2D
        boundary_conditions: dict with keys
            - For 1D: {"x_left": value, "x_right": value}
            - For 2D: {"x_left": val, "x_right": val, "y_bottom": val, "y_top": val}

    Returns:
        solution: numpy array of shape (nx,) for 1D or (nx, ny) for 2D
    """
    if dimension == "1D":
        x_min, x_max = grid_bounds
        nx = 50  # Default grid resolution for Laplace
        ny = 1
        grid = Grid(nx, ny, lx=x_max - x_min, ly=1.0)

        # Simple 1D Laplace solver: d^2u/dx^2 = 0 => linear interpolation
        x_left_val = boundary_conditions.get("x_left", 0.0)
        x_right_val = boundary_conditions.get("x_right", 0.0)
        x = np.linspace(x_min, x_max, nx)
        solution = x_left_val + (x_right_val - x_left_val) * (x - x_min) / (x_max - x_min)
        return solution

    else:  # 2D
        (x_min, x_max), (y_min, y_max) = grid_bounds
        nx = 50  # Default grid resolution
        ny = 50
        grid = Grid(nx, ny, lx=x_max - x_min, ly=y_max - y_min)

        # Use Poisson solver with zero RHS
        def zero_source(X, Y):
            return np.zeros_like(X)

        bc = Dirichlet(0.0)  # Not directly used by poisson solver, but kept for consistency
        solver = LinearSolver()

        solution = solve_poisson(grid, zero_source, bc=bc, solver=solver)

        # Apply boundary conditions to solution
        x_left = boundary_conditions.get("x_left", 0.0)
        x_right = boundary_conditions.get("x_right", 0.0)
        y_bottom = boundary_conditions.get("y_bottom", 0.0)
        y_top = boundary_conditions.get("y_top", 0.0)

        # For Laplace, enforce BCs directly (linear solution)
        for i in range(nx):
            # x-direction interpolation
            alpha_x = i / (nx - 1) if nx > 1 else 0.0
            for j in range(ny):
                # y-direction interpolation
                alpha_y = j / (ny - 1) if ny > 1 else 0.0
                solution[i, j] = (
                    (1 - alpha_x) * (1 - alpha_y) * y_bottom +
                    (1 - alpha_x) * alpha_y * y_top +
                    alpha_x * (1 - alpha_y) * x_left +
                    alpha_x * alpha_y * x_right
                )

        return solution
