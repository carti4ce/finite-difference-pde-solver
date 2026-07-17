# PDE Studio — web app

React frontend + FastAPI backend over the `finite_differences` package. Build a
custom PDE from composable terms (diffusion, advection, reaction, source,
nonlinear), pick boundary/initial conditions, solve, and watch the solution
animate.

## Run it (two terminals)

Backend (needs the `pdes` conda env, or any env with the repo's requirements plus
`webapp/backend/requirements.txt`):

```bash
cd webapp/backend
python -m uvicorn main:app --port 8000
```

Frontend (Node 18+):

```bash
cd webapp/frontend
npm install
npm run dev        # http://localhost:5173
```

The frontend expects the API at `http://localhost:8000`; override with a
`VITE_API_BASE` env var.

## Layout

| Path | What it is |
|---|---|
| `backend/main.py` | FastAPI app: `POST /api/solve`, `GET /api/health` |
| `backend/schemas.py` | Request/response models + resource limits (grid ≤ 256², steps ≤ 20k, frames ≤ 600) |
| `backend/pde_builder.py` | Translates a validated request into a `finite_differences.PDE` run |
| `backend/expr.py` | Safe AST-walking evaluator for user math expressions (no `eval`) |
| `backend/tests/` | Evaluator security/numerics tests + API round-trip tests |
| `frontend/src/builderState.ts` | Builder model, defaults, presets |
| `frontend/src/components/builder/` | Term/BC/IC/grid/integration editor cards |
| `frontend/src/components/viewer/` | Canvas line chart, heatmap, animation player, colormaps |

## Notes

- User expressions (`source`, `nonlinear`, custom initial conditions) run through
  `expr.py`'s whitelisted-AST evaluator — imports, attribute access, and unknown
  names are rejected before evaluation. Python syntax: `**` for powers, `pi`/`e`,
  `sin/cos/exp/...`, and `a if cond else b` (vectorized to `where`).
- Solve responses return the full recorded history as JSON; the frame budget
  (600) keeps payloads sane. The player animates client-side with scrub/speed
  controls.
- Colors follow the repo's dataviz palette: sequential blue ramp for one-signed
  fields, blue↔gray↔red diverging when the field crosses zero, in both light
  and dark themes.
