# tests/test_benchmarks.py
import pytest
from qiskit import QuantumCircuit
# Assumes benchmarks package is importable
from benchmarks import create_ghz_circuit, create_w_state_circuit

@pytest.mark.parametrize("num_qubits", [3, 4, 5])
def test_ghz_creation(num_qubits):
    """Test GHZ circuit creation."""
    qc = create_ghz_circuit(num_qubits, measure=False)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == num_qubits
    # Check if the circuit has operations (depth > 0)
    assert qc.depth() > 0, f"GHZ circuit for {num_qubits} qubits has depth 0"
    # Check for expected gates (H and CX)
    ops = [instr.operation.name for instr in qc.data]
    assert 'h' in ops, f"GHZ circuit for {num_qubits} qubits missing H gate"
    if num_qubits > 1:
        assert 'cx' in ops, f"GHZ circuit for {num_qubits} qubits missing CX gate"


@pytest.mark.parametrize("num_qubits", [3, 4, 5])
def test_w_state_creation(num_qubits):
    """Test W-state circuit creation."""
    qc = create_w_state_circuit(num_qubits, measure=False)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == num_qubits
    # Check if the circuit has operations (depth > 0)
    assert qc.depth() > 0, f"W state circuit for {num_qubits} qubits has depth 0"
    # Check for expected gates (Ry, CRY, CX, X)
    ops = [instr.operation.name for instr in qc.data]
    assert 'ry' in ops, f"W state circuit for {num_qubits} qubits missing Ry gate"
    if num_qubits > 1:
         # Check for controlled Ry (cry) - note: Qiskit might decompose CRY
         # Let's check for cx which is definitely used
         assert 'cx' in ops, f"W state circuit for {num_qubits} qubits missing CX gate"
    assert 'x' in ops, f"W state circuit for {num_qubits} qubits missing X gate"


def test_ghz_measure():
    """Test measurement is added correctly to GHZ."""
    num_qubits = 3
    qc = create_ghz_circuit(num_qubits, measure=True)
    assert qc.num_clbits == num_qubits, f"GHZ circuit measurement failed: expected {num_qubits} clbits, got {qc.num_clbits}"
    assert any(instr.operation.name == 'measure' for instr in qc.data), "GHZ circuit missing measure operation"

def test_w_state_measure():
    """Test measurement is added correctly to W-state."""
    num_qubits = 3
    qc = create_w_state_circuit(num_qubits, measure=True)
    assert qc.num_clbits == num_qubits, f"W state circuit measurement failed: expected {num_qubits} clbits, got {qc.num_clbits}"
    assert any(instr.operation.name == 'measure' for instr in qc.data), "W state circuit missing measure operation"

# Add tests for edge cases if applicable (e.g., num_qubits < minimum required)
def test_ghz_invalid_qubits():
    """Test GHZ creation with invalid number of qubits."""
    with pytest.raises(ValueError, match="GHZ state requires at least 2 qubits"):
        create_ghz_circuit(1)
    with pytest.raises(ValueError, match="GHZ state requires at least 2 qubits"):
        create_ghz_circuit(0)

def test_w_state_invalid_qubits():
    """Test W-state creation with invalid number of qubits."""
    # The W state construction provided starts typically at 3, but the code allows 2.
    # Let's test the boundary based on the code's logic.
    with pytest.raises(ValueError, match="W state construction typically used for num_qubits >= 2, standard >= 3"):
         create_w_state_circuit(1)
    with pytest.raises(ValueError, match="W state construction typically used for num_qubits >= 2, standard >= 3"):
         create_w_state_circuit(0)
    # Test num_qubits = 2, which should work according to the code
    qc_w2 = create_w_state_circuit(2, measure=False)
    assert isinstance(qc_w2, QuantumCircuit)
    assert qc_w2.num_qubits == 2
    assert qc_w2.depth() > 0

