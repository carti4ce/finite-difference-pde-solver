"""Utility helpers: small plotting and I/O helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from matplotlib.animation import FuncAnimation
import numpy as np

# Plot style library
# import scienceplots
# plt.style.use(['science', 'notebook'])


def quick_plot_2d(arr, title=None, xlabel=None, ylabel=None, cbarlabel=None, cmap='viridis'):
    img = plt.imshow(arr.T, origin="lower", cmap=cmap)
    cbar = plt.colorbar(img)

    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if cbarlabel: cbar.set_label(cbarlabel, rotation=270, labelpad=15)
    
    plt.show()


def animated_plot_2d(arr, title=None, xlabel=None, ylabel=None, cbarlabel=None, interval_ms=50, cmap='viridis'):

    num_frames = arr.shape[0]

    if num_frames <= 1:
        print("Error: Need at least 2 time steps for animation.")
        return None, None
    
    total_duration_seconds = (interval_ms * num_frames) / 1000

    print(f"Animation calculated for {num_frames} frames over {total_duration_seconds} seconds.")
    print(f"Interval set to {interval_ms:.2f} ms per frame.")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Set range for colorbar to prevent it from adjusting mid-animation
    vmin_global = arr.min()
    vmax_global = arr.max()

    im = ax.imshow(arr[0],
                   origin='lower',
                   cmap=cmap,
                   vmin=vmin_global,
                   vmax=vmax_global)
    
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(f'{title} | Step {0} of {num_frames - 1}')
    if cbarlabel: fig.colorbar(im, ax=ax, label=cbarlabel)

    def update(frame):
        new_data = arr[frame]
        im.set_array(new_data)
        ax.set_title(f'{title} | Step {frame} of {num_frames - 1}')
        return im,

    anim = FuncAnimation(fig,
                         update,
                         frames=num_frames,
                         interval=interval_ms,
                         blit=False,  # CRITICAL
                         repeat=True)

    return anim,fig