"""Utility helpers: small plotting and I/O helpers."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def _x_axis_1d(n, extent):
    """X coordinates for 1-D plots; defaults to [0, 1] when extent is None."""
    x0, x1 = (0.0, 1.0) if extent is None else (extent[0], extent[1])
    return np.linspace(x0, x1, num=n, endpoint=True)


def _padded_ylim(vmin, vmax, pad=0.1):
    """Y limits padded by `pad` of the data range, or None if either end is missing.

    Handles constant data (zero range) and negative values.
    """
    if vmin is None or vmax is None:
        return None
    span = vmax - vmin
    if span <= 0:
        span = max(abs(vmax), 1.0)
    return (vmin - pad * span, vmax + pad * span)


def _frame_title(title, frame, num_frames):
    """Per-frame animation title; omits the prefix when no title was given."""
    base = f"Step {frame} of {num_frames - 1}"
    return f"{title} | {base}" if title else base


def quick_plot(arr, dimension, vmin_global=None, vmax_global=None,
               title=None, xlabel=None, ylabel=None, cbarlabel=None, cmap='viridis', extent=None):
    """
        General static plot function

        Acts as an entry point and calls appropriate quick plot function based on dimension parameter.
        Args:
            arr: array containing snapshot of solution to be plotted; is either 1 or 2 dimensional.
            dimension: either "1D" or "2D"; determines which quickplot function is used
            vmin_global, vmax_global: specifies the range of function values attained, used to maintain consistency in axis limits.
                In 1-D, this corresponds to y axis limits (both must be given to take effect). In 2-D, the colorbar limits.
            extent: contains an array specifying the input intervals. In 1-D, this is the x limits. In 2-D, both the x and y limits.
                Of the form [xlim1, xlim2, ylim1, ylim2]. If None, the 1-D x axis defaults to [0, 1].
    """
    if dimension == "1D":
        return quick_plot_1d(arr, vmin_global=vmin_global, vmax_global=vmax_global, title=title, xlabel=xlabel, ylabel=ylabel, extent=extent)
    elif dimension == "2D":
        return quick_plot_2d(arr, vmin_global=vmin_global, vmax_global=vmax_global, title=title, xlabel=xlabel, ylabel=ylabel,
                             cbarlabel=cbarlabel, cmap=cmap, extent=extent)
    else:
        raise ValueError(f"Invalid Dimension: {dimension}")
    


def quick_plot_1d(arr, vmin_global=None, vmax_global=None, title=None, xlabel=None, ylabel=None, extent=None):
    """
        Static plot function for 1-D data

        Generates and returns a 1-D plot containing the data in arr. Uses extent and len(arr) to generate x axis. vmin and vmax determine
        y axis.
    """

    # Create X axis
    Xs = _x_axis_1d(len(arr), extent)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Xs, arr)

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    # Fix the y limits (with a small margin) when a full range is provided
    ylim = _padded_ylim(vmin_global, vmax_global)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax



def quick_plot_2d(arr, vmin_global=None, vmax_global=None, title=None, xlabel=None, ylabel=None, cbarlabel=None, cmap='viridis', extent=None):
    """
        Static plot function for 2-D data

        Generates and returns a 2-D color plot containing solution data.
        Args:
            vmin_global, vmax_global: Used to fix colorbar range
            extent: Used to bound x and y axes based on domain
    """
    fig, ax = plt.subplots(figsize=(8,6))
    img = ax.imshow(
        arr.T, 
        origin="lower", 
        cmap=cmap,
        vmin=vmin_global,
        vmax=vmax_global,
        extent=extent
    )

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if cbarlabel: fig.colorbar(img, ax=ax, label=cbarlabel)
    
    return fig, ax



def animated_plot(arr, dimension, title=None, xlabel=None, ylabel=None, cbarlabel=None, interval_ms=50, 
                  cmap='viridis', verbose=False, extent=None):
    if dimension == "1D":
        return animated_plot_1d(arr, title=title, xlabel=xlabel, ylabel=ylabel, interval_ms=interval_ms, verbose=verbose, extent=extent)
    elif dimension == "2D":
        return animated_plot_2d(arr, title=title, xlabel=xlabel, ylabel=ylabel, cbarlabel=cbarlabel, interval_ms=interval_ms, cmap=cmap,
                                verbose=verbose, extent=extent)
    else:
        raise ValueError(f"Invalid Dimension: {dimension}")
    


def animated_plot_1d(arr, title=None, xlabel=None, ylabel=None, interval_ms=50, verbose=False, extent=None):
    """
        Animated plot function for 1-D data

        Generates and returns a matplotlib animation of arr evolving over time, plotted as a 1-D line.
        Uses extent and arr.shape[1] to generate the x axis. Y axis limits are fixed using the global
        min/max across all frames so the axis doesn't rescale mid-animation.
    """

    num_frames = arr.shape[0]

    if num_frames <= 1:
        print("Error: Need at least 2 time steps for animation.")
        return None, None

    total_duration_seconds = (interval_ms * num_frames) / 1000

    if verbose:
        print(f"Animation calculated for {num_frames} frames over {total_duration_seconds} seconds.")
        print(f"Interval set to {interval_ms:.2f} ms per frame.")

    # Create X axis
    Xs = _x_axis_1d(arr.shape[1], extent)

    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot(Xs, arr[0])

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    # Fix the y limits so the axis doesn't rescale mid-animation
    ax.set_ylim(_padded_ylim(arr.min(), arr.max()))

    title_obj = ax.set_title(_frame_title(title, 0, num_frames))

    def update(frame):
        line.set_ydata(arr[frame])
        title_obj.set_text(_frame_title(title, frame, num_frames))
        return line, title_obj

    anim = FuncAnimation(fig,
                         update,
                         frames=num_frames,
                         interval=interval_ms,
                         blit=True,
                         repeat=True)

    return anim, fig



def animated_plot_2d(arr, title=None, xlabel=None, ylabel=None, cbarlabel=None, interval_ms=50, cmap='viridis', verbose=False, extent=None):

    num_frames = arr.shape[0]

    if num_frames <= 1:
        print("Error: Need at least 2 time steps for animation.")
        return None, None
    
    total_duration_seconds = (interval_ms * num_frames) / 1000

    if verbose:
        print(f"Animation calculated for {num_frames} frames over {total_duration_seconds} seconds.")
        print(f"Interval set to {interval_ms:.2f} ms per frame.")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Set range for colorbar to prevent it from adjusting mid-animation
    vmin_global = arr.min()
    vmax_global = arr.max()

    arr_T = arr.transpose(0, 2, 1)

    im = ax.imshow(
        arr_T[0],
        origin='lower',
        cmap=cmap,
        vmin=vmin_global,
        vmax=vmax_global,
        extent=extent
    )
    
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if cbarlabel: fig.colorbar(im, ax=ax, label=cbarlabel)

    title_obj = ax.set_title(_frame_title(title, 0, num_frames))

    def update(frame):
        im.set_array(arr_T[frame])
        title_obj.set_text(_frame_title(title, frame, num_frames))
        return im, title_obj

    anim = FuncAnimation(fig,
                         update,
                         frames=num_frames,
                         interval=interval_ms,
                         blit=True,
                         repeat=True)

    return anim,fig