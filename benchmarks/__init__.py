# benchmarks/__init__.py
from .ghz import create_ghz_circuit
from .w_state import create_w_state_circuit

__all__ = ["create_ghz_circuit", "create_w_state_circuit"]

