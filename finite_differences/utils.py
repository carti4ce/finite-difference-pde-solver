"""Utility helpers: small plotting and I/O helpers."""
from __future__ import annotations
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from matplotlib.animation import FuncAnimation
import numpy as np

# Plot style library
# import scienceplots
# plt.style.use(['science', 'notebook'])

def quick_plot(arr, dimension, vmin_global=None, vmax_global=None,
               title=None, xlabel=None, ylabel=None, cbarlabel=None, cmap='viridis', extent=None):
    if dimension == "1D":
        return quick_plot_1d(arr, title=title, xlabel=xlabel, ylabel=ylabel, extent=extent)
    elif dimension == "2D":
        return quick_plot_2d(arr, vmin_global=vmin_global, vmax_global=vmax_global, title=title, xlabel=xlabel, ylabel=ylabel,
                             cbarlabel=cbarlabel, cmap=cmap, extent=extent)
    else:
        raise ValueError(f"Invalid Dimension: {dimension}")

def quick_plot_1d(arr, title=None, xlabel=None, ylabel=None, extent=None):
    return

def quick_plot_2d(arr, vmin_global=None, vmax_global=None, title=None, xlabel=None, ylabel=None, cbarlabel=None, cmap='viridis', extent=None):
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

    im = ax.imshow(
        arr[0].T,
        origin='lower',
        cmap=cmap,
        vmin=vmin_global,
        vmax=vmax_global,
        extent=extent
    )
    
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(f'{title} | Step {0} of {num_frames - 1}')
    if cbarlabel: fig.colorbar(im, ax=ax, label=cbarlabel)

    title_obj = ax.set_title(f'{title} | Step {1} of {num_frames}')

    def update(frame):
        new_data = arr[frame].T
        im.set_array(new_data)
        title_obj.set_text(f'{title} | Step {frame} of {num_frames}')
        return im, title_obj

    anim = FuncAnimation(fig,
                         update,
                         frames=num_frames,
                         interval=interval_ms,
                         blit=True,
                         repeat=True)

    return anim,fig