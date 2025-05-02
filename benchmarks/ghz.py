# benchmarks/ghz.py
from qiskit import QuantumCircuit

def create_ghz_circuit(num_qubits: int, measure: bool = True) -> QuantumCircuit:
    """
    Creates a standard GHZ state preparation circuit |00...0> + |11...1>.

    Args:
        num_qubits: The number of qubits for the GHZ state.
        measure: If True, adds measurement operations to the end.

    Returns:
        A QuantumCircuit object for the GHZ state.
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")

    qc = QuantumCircuit(num_qubits, name=f"ghz_{num_qubits}")

    # Apply Hadamard gate to the first qubit
    qc.h(0)

    # Apply CNOT gates sequentially
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    if measure:
        # Add measurements if requested
        qc.measure_all(add_bits=True) # Creates classical bits automatically

    return qc
