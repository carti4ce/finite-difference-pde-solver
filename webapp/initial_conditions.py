"""Initial condition factory for PDE solutions."""

import numpy as np


def create_initial_condition(
    grid_shape,
    grid_bounds,
    shape="gaussian",
    intensity=1.0,
    center=None,
    spread=None,
    seed=None,
    dimension="2D",
):
    """Create initial condition array for PDE solvers.

    Args:
        grid_shape: tuple (nx,) for 1D or (nx, ny) for 2D
        grid_bounds: tuple (x_min, x_max) for 1D or ((x_min, x_max), (y_min, y_max)) for 2D
        shape: "uniform", "gaussian", "sine_wave", "random"
        intensity: amplitude multiplier
        center: (x_c,) for 1D or (x_c, y_c) for 2D (default: domain center)
        spread: width parameter for Gaussian (default: 5% of domain)
        seed: random seed for reproducibility
        dimension: "1D" or "2D"

    Returns:
        numpy array of shape grid_shape
    """
    if seed is not None:
        np.random.seed(seed)

    if dimension == "1D":
        nx = grid_shape[0]
        x_min, x_max = grid_bounds
        x = np.linspace(x_min, x_max, nx)

        if shape == "uniform":
            return np.full(nx, intensity)

        elif shape == "gaussian":
            if center is None:
                center = (x_min + x_max) / 2.0
            if spread is None:
                spread = (x_max - x_min) * 0.05
            return intensity * np.exp(-((x - center) ** 2) / (2 * spread ** 2))

        elif shape == "sine_wave":
            return intensity * np.sin(np.pi * (x - x_min) / (x_max - x_min))

        elif shape == "random":
            return intensity * np.random.uniform(0, 1, nx)

    else:  # 2D
        nx, ny = grid_shape
        (x_min, x_max), (y_min, y_max) = grid_bounds
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny),
            indexing="ij",
        )

        if shape == "uniform":
            return np.full((nx, ny), intensity)

        elif shape == "gaussian":
            if center is None:
                center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
            if spread is None:
                spread = min(x_max - x_min, y_max - y_min) * 0.05
            r_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2
            return intensity * np.exp(-r_sq / (2 * spread ** 2))

        elif shape == "sine_wave":
            return intensity * np.sin(np.pi * (X - x_min) / (x_max - x_min)) * np.sin(
                np.pi * (Y - y_min) / (y_max - y_min)
            )

        elif shape == "random":
            return intensity * np.random.uniform(0, 1, (nx, ny))

    raise ValueError(f"Unknown shape: {shape}")
