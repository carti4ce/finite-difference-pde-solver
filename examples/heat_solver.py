"""Heat equation demo: explicit Euler and Crank-Nicolson options."""
from finite_differences import Grid, Field
from finite_differences.bc import Dirichlet
from finite_differences.operators import laplacian, laplacian_5pt
from finite_differences.time_integrators import explicit_euler, crank_nicolson
from finite_differences.solvers import LinearSolver
from finite_differences.utils import quick_plot_2d, animated_plot_2d
import matplotlib.pyplot as plt
import time

import numpy as np


def rhs_heat(field, alpha):
    # apply BCs must be done before calling
    lap = laplacian(field)
    return alpha * lap


def run_explicit(nx=64, ny=64, alpha=1e-3, dt=1e-4, steps=100):
    grid = Grid(nx, ny, lx=1.0, ly=1.0)
    f = Field(grid, ng=1)
    X, Y = grid.mesh()
    r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    u0 = np.exp(-((r - 0.3)**2) / 0.005)
    f.set_interior(u0)
    bc = Dirichlet(0.0)
    interior_history = []
    for n in range(steps):
        bc.apply(f)
        interior_history.append(f.interior.copy())
        u_interior = f.interior
        u_new = explicit_euler(u_interior, lambda u: rhs_heat(f, alpha), dt)
        # write back
        f.set_interior(u_new)
    bc.apply(f)
    return np.asarray(interior_history)


def run_cn(nx=64, ny=64, alpha=1e-3, dt=1e-3, steps=50):
    grid = Grid(nx, ny, lx=1.0, ly=1.0)
    f = Field(grid, ng=1)
    X, Y = grid.mesh()
    r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    u0 = np.exp(-((r - 0.3)**2) / 0.005)
    f.set_interior(u0)
    bc = Dirichlet(0.0)
    solver = LinearSolver()
    interior_history = []
    for _ in range(steps):
        bc.apply(f)
        f = crank_nicolson(f, laplacian_5pt, dt, alpha, solver)
        interior_history.append(f.interior.copy())
    bc.apply(f)
    return np.asarray(interior_history)


if __name__ == '__main__':

    # Show plots after creation
    show_plots = False

    nx = ny = 64
    alpha = 1
    dt = 5e-5
    steps = 250

    # Run Crank Nicholson
    print(f"\nRunning Crank-Nicholson scheme on 2-D heat equation with parameters: nx={nx}, ny={ny}, alpha={alpha}, dt={dt}, steps={steps}.")
    start_time = time.perf_counter()
    cn_history = run_cn(nx, ny, alpha=alpha, dt=dt, steps=steps)
    end_time = time.perf_counter()

    print(f"\nSolving time for Crank-Nicholson: {end_time - start_time}")
    print("Creating animation...\n")

    anim, fig = animated_plot_2d(
        cn_history,
        title='2-D Heat: Crank Nicholson Scheme',
        xlabel='x',
        ylabel='y',
        cbarlabel='Temperature',
        interval_ms=25
    )
    anim.save(
        "examples/example_plots/heat2d_CN_animation.gif",
        writer="ffmpeg",
        fps=1000 // 25
    )
    if show_plots: plt.show()

    print("\nAnimation created, saved in examples/example_plots/heat2d_CN_animation.gif")

    # Run Explicit Euler
    print(f"Now running explicit Euler scheme on 2-D heat equation with parameters: nx={nx}, ny={ny}, alpha={alpha}, dt={dt}, steps={steps}.")
    start_time2 = time.perf_counter()
    euler_history = run_explicit(nx, ny, alpha=alpha, dt=dt, steps=steps)
    end_time2 = time.perf_counter()

    print(f"\nSolving time for explicit Euler: {end_time2 - start_time2}")
    print("Creating animation...\n")

    anim, fig = animated_plot_2d(
        euler_history,
        title='2-D Heat: Explicit Euler Scheme',
        xlabel='x',
        ylabel='y',
        cbarlabel='Temperature',
        interval_ms=25
    )
    anim.save(
        "examples/example_plots/heat2d_explicit_animation.gif",
        writer="ffmpeg",
        fps=1000 // 25
    )
    if show_plots: plt.show()

    print("\nAnimation created, saved in examples/example_plots/heat2d_explicit_animation.gif")
