Finite Differences PDE solver

This project provides a modular finite-difference solver for PDEs in 1D and 2D.

Quick start

Install dependencies (recommended into a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run examples in `examples/` or open the notebooks in `notebooks/`.

Running demos

Create a virtualenv and install dependencies (see `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the heat demo (explicit):

```bash
python3 examples/run_demo.py --mode explicit --nx 64 --ny 64 --steps 200
```

Run the Crank-Nicolson demo:

```bash
python3 examples/run_demo.py --mode cn --nx 64 --ny 64 --steps 50
```

Notes:
- Implicit runs require `scipy` (sparse solvers). The explicit demo only requires `numpy` and `matplotlib`.
- Adjust `--dt` and `--alpha` to explore stability and diffusion rates.
