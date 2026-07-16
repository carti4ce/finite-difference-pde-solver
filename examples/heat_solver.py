"""Heat equation demo: explicit Euler and Crank-Nicolson options."""
from finite_differences import Grid, PDE, DiffusionTerm
from finite_differences.bc import Dirichlet
from finite_differences.utils import quick_plot_2d, animated_plot_2d
import matplotlib.pyplot as plt
import time

import numpy as np


def _initial_ring(grid):
    X, Y = grid.mesh()
    r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    return np.exp(-((r - 0.3)**2) / 0.005)


def run_explicit(nx=64, ny=64, alpha=1e-3, dt=1e-4, steps=100):
    grid = Grid(nx, ny, lx=1.0, ly=1.0)
    pde = PDE(grid, [DiffusionTerm(alpha)], Dirichlet(0.0), time_order=1)
    _, history = pde.integrate(dt=dt, steps=steps, u0=_initial_ring(grid), scheme="euler")
    return history


def run_cn(nx=64, ny=64, alpha=1e-3, dt=1e-3, steps=50):
    grid = Grid(nx, ny, lx=1.0, ly=1.0)
    pde = PDE(grid, [DiffusionTerm(alpha)], Dirichlet(0.0), time_order=1)
    _, history = pde.integrate(dt=dt, steps=steps, u0=_initial_ring(grid), scheme="crank_nicolson")
    return history


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
