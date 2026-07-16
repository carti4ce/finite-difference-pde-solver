"""Time integration steppers for the method of lines.

Explicit steppers act on a flat state vector ``u`` and a spatial right-hand
side ``rhs(u)`` (which returns ``du/dt``). Implicit steppers for stiff linear
problems are built directly inside `PDE` from the assembled operator, so they
live there rather than here.
"""
from __future__ import annotations


def explicit_euler(u, rhs, dt):
    """One explicit Euler step: ``u^{n+1} = u^n + dt * rhs(u^n)``."""
    return u + dt * rhs(u)


def rk2(u, rhs, dt):
    """Midpoint (2nd-order Runge-Kutta) step."""
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    return u + dt * k2


def rk4(u, rhs, dt):
    """Classic 4th-order Runge-Kutta step."""
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


EXPLICIT_STEPPERS = {
    "euler": explicit_euler,
    "explicit_euler": explicit_euler,
    "rk2": rk2,
    "rk4": rk4,
}
