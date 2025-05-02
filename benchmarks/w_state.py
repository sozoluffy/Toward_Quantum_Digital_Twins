# benchmarks/w_state.py
from qiskit import QuantumCircuit
import numpy as np

def create_w_state_circuit(num_qubits: int, measure: bool = True) -> QuantumCircuit:
    """
    Creates a W state preparation circuit |10...0> + |01...0> + ... + |00...1>.
    Uses the method described in https://arxiv.org/abs/1606.09290 (Fig. 10).

    Args:
        num_qubits: The number of qubits for the W state (usually >= 3).
        measure: If True, adds measurement operations to the end.

    Returns:
        A QuantumCircuit object for the W state.
    """
    if num_qubits < 2:
        # While technically definable, the standard construction usually starts at 3
        raise ValueError("W state construction typically used for num_qubits >= 2, standard >= 3")

    qc = QuantumCircuit(num_qubits, name=f"w_{num_qubits}")

    # Step 1: Apply Ry rotation to the first qubit
    # Angle calculation: theta_1 = 2 * arccos(sqrt(1/n))
    theta_1 = 2 * np.arccos(np.sqrt(1.0 / num_qubits))
    qc.ry(theta_1, 0)

    # Step 2: Chain of controlled Ry rotations
    for i in range(num_qubits - 1):
        # Angle calculation: theta_{i+2} = 2 * arccos(sqrt(1/(n-i-1)))
        denominator = num_qubits - 1.0 - i
        if denominator <= 0: # Avoid division by zero or sqrt of negative
             # This happens if num_qubits = 2 on the last iteration, angle is 0
             angle = 0
        else:
             angle = 2 * np.arccos(np.sqrt(1.0 / denominator))

        if np.abs(angle) > 1e-9: # Avoid adding identity gates
            qc.cry(angle, i, i + 1)

    # Step 3: Chain of CNOTs
    for i in range(num_qubits - 1, 0, -1):
        qc.cx(i - 1, i)

    # Optional Step 4: Add X gate to first qubit if desired W state is
    # |0...01> + |0...10> + ... + |1...00> (matches Wikipedia/paper definition)
    # Otherwise, the state is |10...0> + |01...0> + ...
    # Let's add it to match the common |W> = (|100> + |010> + |001>)/sqrt(3) form
    qc.x(0)


    if measure:
        # Add measurements if requested
        qc.measure_all(add_bits=True)

    return qc
