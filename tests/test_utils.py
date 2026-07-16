"""Tests for the plotting helpers (regression tests for former crash cases)."""
import matplotlib
matplotlib.use("Agg")  # headless backend; must precede pyplot use in utils

import numpy as np
import pytest
import matplotlib.pyplot as plt

from finite_differences.utils import (
    quick_plot, quick_plot_1d, animated_plot, animated_plot_1d, animated_plot_2d,
    _padded_ylim, _x_axis_1d, _frame_title,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_padded_ylim_requires_both_ends():
    assert _padded_ylim(None, 1.0) is None
    assert _padded_ylim(0.0, None) is None
    assert _padded_ylim(None, None) is None


def test_padded_ylim_pads_both_sides():
    lo, hi = _padded_ylim(0.0, 1.0)
    assert lo < 0.0 < 1.0 < hi


def test_padded_ylim_negative_range():
    """Formerly vmax*1.2 shrank the limit for negative vmax."""
    lo, hi = _padded_ylim(-2.0, -1.0)
    assert lo < -2.0 and hi > -1.0


def test_padded_ylim_constant_data():
    lo, hi = _padded_ylim(3.0, 3.0)
    assert lo < 3.0 < hi


def test_x_axis_defaults_to_unit_interval():
    x = _x_axis_1d(5, None)
    assert x[0] == 0.0 and x[-1] == 1.0 and len(x) == 5


def test_frame_title_without_title():
    assert _frame_title(None, 3, 10) == "Step 3 of 9"
    assert _frame_title("Heat", 3, 10) == "Heat | Step 3 of 9"


# ---------------------------------------------------------------------------
# former crash cases
# ---------------------------------------------------------------------------

def test_quick_plot_1d_vmax_only_does_not_crash():
    """Formerly raised TypeError (dereferenced vmin_global when only vmax given)."""
    fig, ax = quick_plot_1d(np.sin(np.linspace(0, np.pi, 20)), vmax_global=1.0)
    assert fig is not None


def test_quick_plot_1d_no_extent():
    """Formerly raised TypeError indexing extent[0] with extent=None."""
    fig, ax = quick_plot_1d(np.ones(10))
    assert ax.lines[0].get_xdata()[-1] == 1.0


def test_quick_plot_1d_sets_ylim_when_range_given():
    fig, ax = quick_plot_1d(np.linspace(-1, 1, 10), vmin_global=-1.0, vmax_global=1.0)
    lo, hi = ax.get_ylim()
    assert lo < -1.0 and hi > 1.0


def test_animated_plot_1d_no_extent_no_title():
    arr = np.sin(np.linspace(0, np.pi, 20))[None, :] * np.linspace(1, 0.5, 5)[:, None]
    anim, fig = animated_plot_1d(arr)  # formerly crashed on extent[0]
    assert anim is not None
    assert fig.axes[0].get_title() == "Step 0 of 4"


def test_animated_plot_2d_title_range():
    arr = np.random.default_rng(0).random((4, 6, 5))
    anim, fig = animated_plot_2d(arr, title="T")
    assert fig.axes[0].get_title() == "T | Step 0 of 3"


def test_animated_plot_requires_two_frames():
    anim, fig = animated_plot_1d(np.ones((1, 10)))
    assert anim is None and fig is None


def test_quick_plot_dispatch_invalid_dimension():
    with pytest.raises(ValueError):
        quick_plot(np.ones(4), dimension="3D")
    with pytest.raises(ValueError):
        animated_plot(np.ones((3, 4)), dimension="3D")
