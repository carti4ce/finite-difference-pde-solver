"""Make the backend package (main, schemas, pde_builder, expr) importable
regardless of which directory pytest is invoked from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
