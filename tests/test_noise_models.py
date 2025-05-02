# tests/test_noise_models.py
import pytest
import numpy as np
from noise.reset_error import create_reset_error
from noise.gate_error import GateError
from noise.measurement_error import MeasurementError
from noise.coherent_crosstalk import CoherentCrosstalkGate
# Import specific Qiskit noise objects and exceptions
from qiskit_aer.noise import QuantumError, ReadoutError, thermal_relaxation_error
# Correct import for AerError
from qiskit_aer import AerError

from qiskit.circuit import Gate # For type hinting


# --- Tests for Reset Error ---
def test_reset_error_creation_valid():
    """Test create_reset_error with valid p1."""
    p1_test = 0.05
    error = create_reset_error(p1_test)
    assert isinstance(error, QuantumError)
    # Check if it's a thermal relaxation error with expected population
    # Note: Qiskit Aer might return a composite error or a different internal type
    # Checking the Kraus representation is more robust if possible, or check properties.
    # For now, check if it's a QuantumError and not None.
    assert error is not None, "create_reset_error returned None for valid p1"

def test_reset_error_creation_p1_zero():
    """Test create_reset_error with p1 = 0."""
    error = create_reset_error(0.0)
    assert error is None, "create_reset_error should return None for p1=0"

def test_reset_error_creation_invalid_p1():
    """Test create_reset_error with invalid p1."""
    with pytest.raises(ValueError, match="p1 must be in"):
        create_reset_error(1.1)
    with pytest.raises(ValueError, match="p1 must be in"):
        create_reset_error(-0.1)

# --- Tests for Gate Error ---
def test_gate_error_creation_1q_valid():
    """Test GateError creation for 1-qubit gate with valid fidelity."""
    fidelity = 0.99
    error_obj = GateError(fidelity, 1)
    error_channel = error_obj.channel()
    assert isinstance(error_channel, QuantumError)
    assert error_channel is not None, "GateError channel is None for 1Q valid fidelity"
    # Optional: Check properties of the resulting error channel if possible
    # E.g., check Kraus operators or type if it's a simple depolarizing error

def test_gate_error_creation_2q_valid():
    """Test GateError creation for 2-qubit gate with valid fidelity."""
    fidelity = 0.95
    error_obj = GateError(fidelity, 2)
    error_channel = error_obj.channel()
    assert isinstance(error_channel, QuantumError)
    assert error_channel is not None, "GateError channel is None for 2Q valid fidelity"

def test_gate_error_creation_fidelity_one():
    """Test GateError creation with fidelity = 1."""
    error_obj_1q = GateError(1.0, 1)
    assert error_obj_1q.channel() is None, "GateError channel should be None for fidelity=1 (1Q)"
    error_obj_2q = GateError(1.0, 2)
    assert error_obj_2q.channel() is None, "GateError channel should be None for fidelity=1 (2Q)"

def test_gate_error_creation_invalid_fidelity():
    """Test GateError creation with invalid fidelity."""
    with pytest.raises(ValueError, match="Fidelity must be in"):
        GateError(1.1, 1)
    with pytest.raises(ValueError, match="Fidelity must be in"):
        GateError(-0.1, 2)

def test_gate_error_creation_invalid_num_qubits():
    """Test GateError creation with invalid number of qubits."""
    with pytest.raises(ValueError, match="num_qubits must be 1 or 2"):
        GateError(0.99, 3)
    with pytest.raises(ValueError, match="num_qubits must be 1 or 2"):
        GateError(0.99, 0)

# --- Tests for Measurement Error ---
def test_meas_error_valid():
    """Test MeasurementError creation with valid probabilities."""
    p00_test = 0.95
    p11_test = 0.96
    error_obj = MeasurementError(confusion_matrix_diag=(p00_test, p11_test), qubit_idx=0)
    assert isinstance(error_obj.channel, ReadoutError)
    assert error_obj.index == 0
    # Check the probabilities in the resulting ReadoutError object
    probs = error_obj.channel.probabilities

    # Use a slightly higher tolerance for floating point comparisons
    # Check diagonal elements (P(0|0) and P(1|1))
    assert np.isclose(probs[0][0], p00_test, atol=1e-9)
    assert np.isclose(probs[1][1], p11_test, atol=1e-9)

    # Check off-diagonal elements based on the correct calculation
    # probs[0][1] is P(measure 1 | state 0), which is 1 - P(measure 0 | state 0) = 1 - p00_test
    assert np.isclose(probs[0][1], 1.0 - p00_test, atol=1e-8)
    # probs[1][0] is P(measure 0 | state 1), which is 1 - P(measure 1 | state 1) = 1 - p11_test
    assert np.isclose(probs[1][0], 1.0 - p11_test, atol=1e-8)


def test_meas_error_invalid_prob_range():
    """Test MeasurementError creation with probabilities outside [0, 1]."""
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
         MeasurementError(confusion_matrix_diag=(1.1, 0.90), qubit_idx=0)
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
         MeasurementError(confusion_matrix_diag=(0.95, -0.1), qubit_idx=0)

def test_meas_error_invalid_diag_length():
    """Test MeasurementError creation with incorrect confusion matrix diagonal length."""
    with pytest.raises(ValueError, match="Provide \\[P\\(0\\|0\\), P\\(1\\|1\\)\\]"):
         MeasurementError(confusion_matrix_diag=(0.95,), qubit_idx=0)
    with pytest.raises(ValueError, match="Provide \\[P\\(0\\|0\\), P\\(1\\|1\\)\\]"):
         MeasurementError(confusion_matrix_diag=(0.95, 0.96, 0.97), qubit_idx=0)

# --- Removed test_meas_error_invalid_matrix_qiskit ---
# This test is removed as Qiskit's ReadoutError constructor does not raise
# AerError for matrices with rows not summing to 1 in the user's environment,
# and the MeasurementError class already validates input probabilities.


# --- Tests for Coherent Crosstalk Gate ---
def test_crosstalk_gate_creation_valid():
    """Test CoherentCrosstalkGate creation."""
    beta_hz = 10000.0
    duration = 50e-9
    gate = CoherentCrosstalkGate(beta_hz=beta_hz, duration=duration)
    assert isinstance(gate, Gate)
    assert gate.num_qubits == 2
    assert len(gate.params) == 1
    # Check the calculated theta parameter
    expected_theta = 4.0 * np.pi * duration * beta_hz # Based on RZZGate definition
    assert np.isclose(gate.params[0], expected_theta)
    assert gate.name.startswith("ZZxtalk")
    assert "Hz" in gate.label
    assert "ns" in gate.label

def test_crosstalk_gate_matrix():
    """Test the matrix property of CoherentCrosstalkGate."""
    beta_hz = 5000.0
    duration = 100e-9
    gate = CoherentCrosstalkGate(beta_hz=beta_hz, duration=duration)
    matrix = gate.matrix
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 4) # 2-qubit gate matrix size

    # Optional: Compare with expected RZZGate matrix for the calculated theta
    from qiskit.circuit.library import RZZGate
    expected_matrix = RZZGate(gate.params[0]).to_matrix()
    assert np.allclose(matrix, expected_matrix)

def test_crosstalk_gate_inverse():
    """Test the inverse property of CoherentCrosstalkGate."""
    beta_hz = 10000.0
    duration = 50e-9
    gate = CoherentCrosstalkGate(beta_hz=beta_hz, duration=duration)
    inv_gate = gate.inverse()

    assert isinstance(inv_gate, CoherentCrosstalkGate)
    assert inv_gate.beta_hz == -beta_hz # Beta should be negated
    assert inv_gate.duration == duration
    # Check the theta parameter of the inverse gate
    expected_inv_theta = 4.0 * np.pi * duration * (-beta_hz)
    assert np.isclose(inv_gate.params[0], expected_inv_theta)
    assert inv_gate.name.startswith("ZZxtalk") # Name should be consistent
    assert "_dg" in inv_gate.label # Label should indicate inverse

