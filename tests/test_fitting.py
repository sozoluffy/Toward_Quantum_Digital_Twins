# tests/test_fitting.py
import pytest
import numpy as np
import os
import yaml
from qiskit import QuantumCircuit
# Assumes fitting package is importable
from fitting import calculate_tvd, fit_parameters
# Need to import CalibrationData and run_simulation_with_layer_noise for the fitting test
from calibration import CalibrationData
from simulator import run_simulation_with_layer_noise
import tempfile # Use tempfile for creating temporary files

# --- Tests for calculate_tvd ---

def test_tvd_identical():
    counts1 = {'00': 500, '11': 500}
    counts2 = {'00': 500, '11': 500}
    shots = 1000
    assert np.isclose(calculate_tvd(counts1, counts2, shots), 0.0)

def test_tvd_disjoint():
    counts1 = {'00': 1000}
    counts2 = {'11': 1000}
    shots = 1000
    assert np.isclose(calculate_tvd(counts1, counts2, shots), 1.0)

def test_tvd_partial_overlap():
    counts1 = {'00': 750, '11': 250}
    counts2 = {'00': 250, '11': 750}
    shots = 1000
    # Expected TVD = 0.5 * (|0.75-0.25| + |0.25-0.75|) = 0.5 * (0.5 + 0.5) = 0.5
    assert np.isclose(calculate_tvd(counts1, counts2, shots), 0.5)

def test_tvd_different_keys():
    counts1 = {'00': 500, '01': 500}
    counts2 = {'10': 500, '11': 500}
    shots = 1000
    # Expected TVD = 0.5 * (|0.5-0| + |0.5-0| + |0-0.5| + |0-0.5|) = 0.5 * (0.5+0.5+0.5+0.5) = 1.0
    assert np.isclose(calculate_tvd(counts1, counts2, shots), 1.0)

def test_tvd_empty():
    counts1 = {}
    counts2 = {}
    shots = 1000
    assert np.isclose(calculate_tvd(counts1, counts2, shots), 0.0)

def test_tvd_zero_shots():
    counts1 = {'00': 10, '11': 0}
    counts2 = {'00': 0, '11': 10}
    shots = 0 # Explicitly test zero shots
    assert np.isclose(calculate_tvd(counts1, counts2, shots), 1.0)

def test_tvd_empty_with_shots():
    counts1 = {}
    counts2 = {}
    shots = 1000
    assert np.isclose(calculate_tvd(counts1, counts2, shots), 0.0)

# --- Test for fit_parameters execution ---

@pytest.fixture
def mock_fitting_data(tmp_path):
    """Fixture to create mock data for fitting test."""
    # Create a minimal mock calibration file
    mock_calib_data = {
        'dt': 0.1e-9,
        'basis_gates': ['u3', 'rz', 'cz', 'id', 'measure', 'reset'],
        'qubits': {
            '0': {'T1': 50e-6, 'T2': 20e-6, 'p1': 0.01},
            '1': {'T1': 55e-6, 'T2': 22e-6, 'p1': 0.015},
        },
        'gates': {
            'u3': {'fidelity': 0.999, 'duration': 30e-9},
            'cz_0_1': {'fidelity': 0.98, 'duration': 40e-9}, # This will be fitted
        },
        'readout': {
            '0': {'confusion_matrix_diag': [0.99, 0.98]},
            '1': {'confusion_matrix_diag': [0.985, 0.975]},
        },
        'crosstalk_strength_hz': 1000.0 # This will be fitted
    }
    calib_path = tmp_path / "mock_calib.yaml"
    with open(calib_path, 'w') as f:
        yaml.dump(mock_calib_data, f)

    # Create a simple mock benchmark circuit (e.g., Bell state)
    qc = QuantumCircuit(2, 2, name="bell_mock")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Create mock target results (ideal Bell state counts)
    mock_target_counts = {'00': 500, '11': 500}
    shots = 1000

    # Define parameters to fit and bounds
    param_names = ['fidelity_cz_0_1', 'crosstalk_strength_hz']
    param_bounds = [(0.9, 1.0), (0, 20000.0)]

    # Define coupling map and basis gates
    coupling_map = [(0, 1)]
    basis_gates = mock_calib_data['basis_gates']

    return {
        'calib_path': str(calib_path),
        'circuits': [qc],
        'target_results': [mock_target_counts],
        'coupling_map': coupling_map,
        'basis_gates': basis_gates,
        'shots': shots,
        'param_names': param_names,
        'param_bounds': param_bounds
    }


def test_fit_parameters_runs(mock_fitting_data):
    """
    Test if the fit_parameters function runs without crashing
    with minimal mock data and optimizer settings.
    """
    print("\n--- Running test_fit_parameters_runs ---") # Debug print
    # Use very minimal optimizer settings for a quick test
    optimizer_opts = {
        'maxiter': 1,
        'popsize': 2,
        'tol': 1.0, # High tolerance for quick exit
        'updating': 'immediate', # Faster for serial
        'workers': 1 # Force serial execution for easier debugging
    }

    try:
        best_params, min_tvd = fit_parameters(
            base_calib_path=mock_fitting_data['calib_path'],
            param_names=mock_fitting_data['param_names'],
            param_bounds=mock_fitting_data['param_bounds'],
            benchmark_circuits=mock_fitting_data['circuits'],
            target_results=mock_fitting_data['target_results'],
            coupling_map=mock_fitting_data['coupling_map'],
            basis_gates=mock_fitting_data['basis_gates'],
            shots=mock_fitting_data['shots'],
            optimizer_opts=optimizer_opts
        )

        # Assert that the function returned values of the expected type
        assert isinstance(best_params, np.ndarray)
        assert len(best_params) == len(mock_fitting_data['param_names'])
        assert isinstance(min_tvd, float)
        assert 0.0 <= min_tvd <= 1.0 # TVD should be between 0 and 1

        print("--- test_fit_parameters_runs passed ---") # Debug print

    except Exception as e:
        pytest.fail(f"fit_parameters failed to run with mock data: {e}")

