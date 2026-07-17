"""End-to-end tests for the /api/solve endpoint via FastAPI's TestClient."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_1d_heat_decay():
    """Dirichlet(0) sine mode should decay towards zero (matches the analytic
    exponential decay rate loosely, but we just check the qualitative trend
    and that the array shapes line up)."""
    body = {
        "grid": {"dimension": "1D", "nx": 64, "lx": 1.0},
        "time_order": 1,
        "terms": [{"type": "diffusion", "coeff": 0.05}],
        "bc": {"type": "dirichlet", "value": 0.0},
        "initial_condition": {"shape": "sine_wave", "intensity": 1.0},
        "integration": {"scheme": "crank_nicolson", "dt": 1e-3, "steps": 200, "record_every": 50},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["dimension"] == "1D"
    assert len(data["times"]) == len(data["history"]) == 5
    assert data["times"][-1] == pytest.approx(0.2)
    first_amp = max(abs(v) for v in data["history"][0])
    last_amp = max(abs(v) for v in data["history"][-1])
    assert last_amp < first_amp


def test_2d_poisson_manufactured_solution():
    body = {
        "grid": {"dimension": "2D", "nx": 32, "ny": 32, "lx": 1.0, "ly": 1.0},
        "time_order": 0,
        "terms": [
            {"type": "diffusion", "coeff": 1.0},
            {"type": "source", "expression": "2*pi**2*sin(pi*x)*sin(pi*y)"},
        ],
        "bc": {"type": "dirichlet", "value": 0.0},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["history"]) == 1  # steady: single frame
    u = np.array(data["history"][0])
    dx, dy = data["dx"], data["dy"]
    x = np.linspace(dx, 1 - dx, data["nx"])
    y = np.linspace(dy, 1 - dy, data["ny"])
    X, Y = np.meshgrid(x, y, indexing="ij")
    exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    assert np.linalg.norm(u - exact) < 0.02


def test_dirichlet_box_edge_values_2d():
    body = {
        "grid": {"dimension": "2D", "nx": 24, "ny": 24, "lx": 1.0, "ly": 1.0},
        "time_order": 0,
        "terms": [{"type": "diffusion", "coeff": 1.0}],
        "bc": {"type": "dirichlet_box", "left": 1.0, "right": 0.0, "bottom": 0.0, "top": 0.0},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 200
    u = np.array(resp.json()["history"][0])
    assert u.min() >= -1e-6 and u.max() <= 1.0 + 1e-6
    assert u[0, :].mean() > u[-1, :].mean()


def test_nonlinear_function_term_fisher_kpp():
    body = {
        "grid": {"dimension": "1D", "nx": 100, "lx": 20.0},
        "time_order": 1,
        "terms": [
            {"type": "diffusion", "coeff": 1.0},
            {"type": "function", "expression": "u * (1 - u)"},
        ],
        "bc": {"type": "dirichlet", "value": 0.0},
        "initial_condition": {"shape": "expression", "expression": "1 if abs(x - 10) < 1 else 0"},
        "integration": {"scheme": "rk4", "dt": 5e-3, "steps": 400, "record_every": 100},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 200
    data = resp.json()
    u_final = np.array(data["history"][-1])
    assert u_final.max() <= 1.001  # bounded by carrying capacity


def test_wave_equation_time_order_2():
    body = {
        "grid": {"dimension": "1D", "nx": 64, "lx": 1.0},
        "time_order": 2,
        "terms": [{"type": "diffusion", "coeff": 1.0}],
        "bc": {"type": "dirichlet", "value": 0.0},
        "initial_condition": {"shape": "sine_wave", "intensity": 1.0},
        "integration": {"scheme": "rk4", "dt": 5e-4, "steps": 200, "record_every": 50},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 200
    assert len(resp.json()["history"]) == 5


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

def test_dangerous_expression_returns_400_not_500():
    body = {
        "grid": {"dimension": "1D", "nx": 20, "lx": 1.0},
        "time_order": 0,
        "terms": [
            {"type": "diffusion", "coeff": 1.0},
            {"type": "source", "expression": "__import__('os').system('x')"},
        ],
        "bc": {"type": "dirichlet", "value": 0.0},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 400
    assert "whitelisted" in resp.json()["detail"] or "unknown" in resp.json()["detail"]


def test_singular_boundary_condition_returns_400_not_500():
    body = {
        "grid": {"dimension": "1D", "nx": 20, "lx": 1.0},
        "time_order": 0,
        "terms": [{"type": "diffusion", "coeff": 1.0}, {"type": "source", "expression": "sin(pi*x)"}],
        "bc": {"type": "neumann", "derivative": 0.0},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 400


def test_missing_integration_for_time_dependent_pde_is_422():
    body = {
        "grid": {"dimension": "1D", "nx": 20, "lx": 1.0},
        "time_order": 1,
        "terms": [{"type": "diffusion", "coeff": 1.0}],
        "bc": {"type": "dirichlet", "value": 0.0},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 422


@pytest.mark.parametrize("field,value", [
    ("nx", 1000),      # exceeds MAX_N
    ("nx", 1),         # below minimum
])
def test_grid_size_limits_enforced(field, value):
    body = {
        "grid": {"dimension": "1D", "lx": 1.0, field: value},
        "time_order": 0,
        "terms": [{"type": "diffusion", "coeff": 1.0}],
        "bc": {"type": "dirichlet", "value": 0.0},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 422


def test_excessive_frame_count_rejected():
    body = {
        "grid": {"dimension": "1D", "nx": 20, "lx": 1.0},
        "time_order": 1,
        "terms": [{"type": "diffusion", "coeff": 1.0}],
        "bc": {"type": "dirichlet", "value": 0.0},
        "integration": {"scheme": "rk4", "dt": 1e-4, "steps": 20000, "record_every": 1},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 422


def test_too_many_terms_rejected():
    body = {
        "grid": {"dimension": "1D", "nx": 20, "lx": 1.0},
        "time_order": 0,
        "terms": [{"type": "diffusion", "coeff": 1.0}] * 9,
        "bc": {"type": "dirichlet", "value": 0.0},
    }
    resp = client.post("/api/solve", json=body)
    assert resp.status_code == 422
