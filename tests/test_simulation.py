# tests/test_simulation.py
import pytest
import numpy as np
import os
import yaml
from qiskit import QuantumCircuit
# Use direct import from qiskit_aer
from qiskit_aer import AerSimulator

# Import the function to be tested and necessary dependencies
from simulator import run_simulation_with_layer_noise
from calibration import CalibrationData # Needed to create mock calib file
import tempfile # Use tempfile for creating temporary files

@pytest.fixture
def mock_calibration_file(tmp_path):
    """Fixture to create a minimal mock calibration file."""
    mock_calib_data = {
        'dt': 0.1e-9, # Example dt
        'basis_gates': ['u3', 'rz', 'id', 'measure', 'reset', 'sx', 'x'], # Include common gates
        'qubits': {
            '0': {'T1': 50e-6, 'T2': 20e-6, 'p1': 0.01},
            '1': {'T1': 55e-6, 'T2': 22e-6, 'p1': 0.015},
        },
        'gates': {
            'u3': {'fidelity': 0.999, 'duration': 30e-9},
            'sx': {'fidelity': 0.999, 'duration': 30e-9}, # Add sx/x fidelity/duration
            'x': {'fidelity': 0.999, 'duration': 30e-9},
            'id': {'fidelity': 1.0, 'duration': 30e-9},
            'rz': {'fidelity': 1.0, 'duration': 0.0}, # Virtual
            'measure': {'fidelity': 1.0, 'duration': 1500e-9},
            'reset': {'fidelity': 1.0, 'duration': 50e-9},
            # Add a 2-qubit gate if needed for more complex tests
            # 'cz_0_1': {'fidelity': 0.98, 'duration': 40e-9},
        },
        'readout': {
            '0': {'confusion_matrix_diag': [0.99, 0.98]},
            '1': {'confusion_matrix_diag': [0.985, 0.975]},
        },
        'crosstalk_strength_hz': 0.0 # Start with no crosstalk for simplicity
        # 'coupling_map': [[0, 1]] # Add coupling map if 2Q gates or crosstalk are tested
    }
    calib_path = tmp_path / "mock_sim_calib.yaml"
    with open(calib_path, 'w') as f:
        yaml.dump(mock_calib_data, f)
    return str(calib_path)


def test_run_simulation_basic(mock_calibration_file):
    """
    Test if run_simulation_with_layer_noise runs for a basic circuit
    and returns counts.
    """
    print("\n--- Running test_run_simulation_basic ---") # Debug print

    # Create a simple 1-qubit circuit
    qc = QuantumCircuit(1, 1, name="basic_qc")
    qc.x(0) # Apply X gate
    qc.measure(0, 0) # Measure

    # Define parameters for simulation
    calib_path = mock_calibration_file
    # Need a coupling map even for 1 qubit if the noise_inserter expects it
    # or if the target creation expects it. Let's use an empty or minimal one.
    # Based on create_qiskit_target, coupling_map is optional.
    coupling_map = None # Or [] if needed by noise_inserter logic
    basis_gates = ['u3', 'rz', 'id', 'measure', 'reset', 'sx', 'x'] # Match mock calib
    shots = 100

    try:
        # Run the simulation
        counts = run_simulation_with_layer_noise(
            circuit=qc,
            calib_path=calib_path,
            coupling_map=coupling_map, # Pass coupling map
            basis_gates=basis_gates, # Pass basis gates
            shots=shots,
            scheduling_method='alap', # Use a valid scheduling method
            backend_opts={'method': 'density_matrix'}, # Use density matrix for noise
            seed=123 # For reproducibility
        )

        # Assertions
        assert isinstance(counts, dict)
        assert len(counts) > 0 # Should not be empty counts
        # For an ideal X gate on |0> followed by measure, expect '1'.
        # With noise, expect '0' as well.
        assert '1' in counts or '0' in counts # Should contain keys '0' or '1'
        assert sum(counts.values()) <= shots # Total counts should not exceed shots (or be close)

        print("--- test_run_simulation_basic passed ---") # Debug print

    except Exception as e:
        pytest.fail(f"run_simulation_with_layer_noise failed for basic circuit: {e}")

# Add more tests for different scenarios:
# - Test with a 2-qubit circuit and a coupling map
# - Test with crosstalk enabled
# - Test with different scheduling methods
# - Test with different backend methods (e.g., 'statevector' if applicable, though noise usually needs density matrix)
# - Test with readout errors affecting counts
# - Test with decay errors affecting outcomes (harder to assert specific counts, maybe check for non-zero counts in unexpected states)

