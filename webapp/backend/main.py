"""FastAPI service exposing the finite_differences PDE solver to the web UI."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pde_builder import SolveError, run_solve
from schemas import SolveRequest, SolveResponse

app = FastAPI(title="Finite Differences PDE API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    # Local dev only: Vite's default ports.
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    try:
        return run_solve(request)
    except SolveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
