# noise/gate_error.py
import numpy as np
# ---> ADD THIS IMPORT <---
from typing import Optional, List
# -----------------------
# Import QuantumError from noise submodule
from qiskit_aer.noise import QuantumError
# Import the base AerError exception
from qiskit_aer import AerError
# Import Kraus channel from quantum_info
from qiskit.quantum_info import Kraus

# Helper function to check trace-preserving condition
# Add type hint for kraus_ops using imported List
def check_kraus_sum_condition(kraus_ops: List[np.ndarray], num_qubits: int, tol: float = 1e-9) -> bool:
    """Checks if sum K_i^dagger * K_i is close to identity."""
    kraus_sum = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for k_op in kraus_ops:
        if k_op.shape != (2**num_qubits, 2**num_qubits):
             print(f"Warning: Kraus operator has wrong shape {k_op.shape}")
             return False
        kraus_sum += np.dot(k_op.conj().T, k_op)
    identity = np.identity(2**num_qubits, dtype=complex)
    is_identity = np.allclose(kraus_sum, identity, atol=tol)
    # if not is_identity:
    #     print(f"DEBUG: Sum K_i^d K_i check failed. Difference from Identity:\n{kraus_sum - identity}")
    return is_identity

# --- GateError Class ---
class GateError:
    """
    Implements gate errors as dephasing channels based on fidelity,
    following the paper arXiv:2504.08313v1.
    Uses quantum_info.Kraus for intermediate representation.
    """
    def __init__(self, fidelity: float, num_qubits: int):
        if not (0 <= fidelity <= 1):
            raise ValueError("Fidelity must be in [0, 1]")
        if num_qubits not in (1, 2):
            raise ValueError("num_qubits must be 1 or 2")

        self.fidelity = fidelity
        self.num_qubits = num_qubits
        try:
            # Use the type hint here which now works because of the import
            self._error_channel: Optional[QuantumError] = self._create_channel()
            if self._error_channel is None and fidelity < 1.0 - 1e-9:
                 raise ValueError("Failed to create non-trivial error channel.")
        except Exception as e:
             print(f"Error creating QuantumError for fidelity={fidelity}, num_qubits={num_qubits}")
             raise e

    # Add the Optional type hint here
    def _create_channel(self) -> Optional[QuantumError]:
        """Creates the appropriate dephasing QuantumError via quantum_info.Kraus."""
        kraus_matrices = [] # List to hold numpy arrays
        if self.num_qubits == 1:
            delta_1 = 1.5 * (1.0 - self.fidelity)
            delta_1 = min(max(delta_1, 0.0), 0.5)
            if np.isclose(delta_1, 0): return None # No error for fidelity=1

            p0 = 1.0 - delta_1
            p1 = delta_1
            k0 = np.sqrt(p0) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
            k1 = np.sqrt(p1) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
            kraus_matrices = [k0, k1]

        elif self.num_qubits == 2:
            delta_2 = (5.0 / 4.0) * (1.0 - self.fidelity)
            delta_2 = min(max(delta_2, 0.0), 0.75)
            if np.isclose(delta_2, 0): return None # No error for fidelity=1

            p0 = 1.0 - delta_2
            p1 = delta_2 / 3.0
            I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
            Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            ZI = np.kron(Z, I)
            IZ = np.kron(I, Z)
            ZZ = np.kron(Z, Z)
            II = np.kron(I, I)
            k0 = np.sqrt(p0) * II
            k1 = np.sqrt(p1) * ZI
            k2 = np.sqrt(p1) * IZ
            k3 = np.sqrt(p1) * ZZ
            kraus_matrices = [k0, k1, k2, k3]

        if not kraus_matrices:
            return None # Should only happen if fidelity is 1

        # Step 1: Create Kraus channel object
        try:
            # print(f"DEBUG: Creating quantum_info.Kraus channel for {self.num_qubits}-qubit gate, fid={self.fidelity}")
            kraus_channel = Kraus(kraus_matrices)
            if not kraus_channel.is_cptp(atol=1e-8): # Check if it's a valid channel
                 print(f"Warning: Created Kraus channel for fid={self.fidelity} is not CPTP within tolerance.")
                 # Decide whether to raise error or proceed
            # print("DEBUG: quantum_info.Kraus created successfully.")
        except Exception as e:
            print(f"ERROR: Failed to create quantum_info.Kraus object: {e}")
            print("ERROR: Failed Kraus matrices were:")
            for i, k_op in enumerate(kraus_matrices): print(f" K[{i}]:\n{k_op}")
            raise ValueError("Failed to create intermediate Kraus channel object.") from e

        # Step 2: Create QuantumError from the Kraus channel object
        try:
            # print(f"DEBUG: Creating QuantumError from Kraus channel object.")
            quantum_error = QuantumError(kraus_channel)
            # print("DEBUG: QuantumError created successfully from Kraus object.")
            return quantum_error
        except AerError as aer_err:
            print(f"ERROR: qiskit_aer.AerError during QuantumError creation from Kraus object: {aer_err}")
            raise aer_err
        except Exception as e:
             print(f"ERROR: Unexpected error during QuantumError creation from Kraus object: {e}")
             raise e

    # Add the Optional type hint here
    def channel(self) -> Optional[QuantumError]:
        """Returns the pre-computed QuantumError channel, or None if fidelity is 1."""
        return self._error_channel
