# noise/reset_error.py
import numpy as np
# Use direct import from qiskit_aer.noise
from qiskit_aer.noise import thermal_relaxation_error, QuantumError

def create_reset_error(p1: float) -> QuantumError:
    """
    Creates a QuantumError for thermal reset to population p1 (Eq. 1 & 2).
    Uses thermal_relaxation_error with T1=T2=inf, time=0.
    """
    if not (0 <= p1 <= 1):
        raise ValueError(f"p1 must be in [0, 1], got {p1}")
    if p1 < 1e-9: # Avoid creating trivial error for p1=0
        return None

    # thermal_relaxation_error returns the QuantumError object directly
    reset_error = thermal_relaxation_error(
        t1=np.inf, t2=np.inf, time=0, excited_state_population=p1
    )
    # ---> Return the object directly, without [0] <---
    return reset_error
