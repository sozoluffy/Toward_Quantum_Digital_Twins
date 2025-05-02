# tests/test_calibration.py
import pytest
import os
import yaml
import numpy as np
from calibration import CalibrationData # Assumes calibration is importable
import tempfile # Use tempfile for creating temporary files

# Define path relative to test file location
TEST_DIR = os.path.dirname(__file__)
VALID_CALIB_PATH = os.path.abspath(os.path.join(TEST_DIR, '..', 'calibration', 'device_calib.yaml'))

@pytest.fixture(scope="module")
def valid_calib_data():
    """Fixture to load valid calibration data once per module."""
    # Ensure the base file exists for testing
    if not os.path.exists(VALID_CALIB_PATH):
         pytest.skip(f"Base calibration file not found for testing: {VALID_CALIB_PATH}")
    return CalibrationData(VALID_CALIB_PATH)

@pytest.fixture
def tmp_calib_file(tmp_path):
    """Fixture to provide a path for a temporary calibration file."""
    return tmp_path / "temp_calib.yaml"

def test_load_valid_file(valid_calib_data):
    """Test loading a valid YAML calibration file."""
    # The fixture already loads it, this test just checks the fixture works
    # and that the CalibrationData object is created without error.
    assert isinstance(valid_calib_data, CalibrationData)
    assert valid_calib_data.data is not None
    print(f"Successfully loaded valid calibration file: {VALID_CALIB_PATH}") # Debug print


def test_load_missing_file():
    """Test loading a non-existent file."""
    invalid_path = os.path.abspath(os.path.join(TEST_DIR, '..', 'calibration', 'non_existent_file.yaml'))
    print(f"Testing load of missing file: {invalid_path}") # Debug print
    with pytest.raises(FileNotFoundError, match="Calibration file not found"):
        CalibrationData(invalid_path)

def test_load_invalid_yaml(tmp_calib_file):
    """Test loading a file with invalid YAML syntax."""
    invalid_yaml_content = "qubits: \n  '0': { T1: 40e-6" # Missing closing brace
    tmp_calib_file.write_text(invalid_yaml_content)
    print(f"Testing load of invalid YAML file: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Error reading or parsing"):
         CalibrationData(str(tmp_calib_file))

# --- Tests for Validation Logic ---

def test_validation_invalid_p1(tmp_calib_file):
    """Test validation fails for p1 outside [0, 1]."""
    # Use a minimal valid structure but change p1
    invalid_data = {'qubits': {'0': {'p1': 1.1, 'T1': 1e-5, 'T2': 1e-5}}}
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid p1: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Invalid value for 'p1'"):
         CalibrationData(str(tmp_calib_file))

def test_validation_invalid_t1(tmp_calib_file):
    """Test validation fails for T1 <= 0."""
    invalid_data = {'qubits': {'0': {'p1': 0.1, 'T1': -1e-5, 'T2': 1e-5}}}
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid T1: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Invalid value for 'T1'"):
         CalibrationData(str(tmp_calib_file))

def test_validation_invalid_t2(tmp_calib_file):
    """Test validation fails for T2 <= 0."""
    invalid_data = {'qubits': {'0': {'p1': 0.1, 'T1': 1e-5, 'T2': -1e-5}}}
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid T2: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Invalid value for 'T2'"):
         CalibrationData(str(tmp_calib_file))

def test_validation_missing_qubit_param(tmp_calib_file):
    """Test validation fails for missing qubit parameters."""
    invalid_data = {'qubits': {'0': {'p1': 0.1, 'T1': 1e-5}}} # Missing T2
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with missing qubit param: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="'T2' missing for qubit '0'"):
         CalibrationData(str(tmp_calib_file))

def test_validation_invalid_gate_fidelity(tmp_calib_file):
    """Test validation fails for gate fidelity outside [0, 1]."""
    invalid_data = {'gates': {'u3': {'fidelity': 1.2, 'duration': 32e-9}}}
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid gate fidelity: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Invalid fidelity for gate 'u3'"):
         CalibrationData(str(tmp_calib_file))

def test_validation_invalid_gate_duration(tmp_calib_file):
    """Test validation fails for gate duration < 0."""
    invalid_data = {'gates': {'u3': {'fidelity': 0.99, 'duration': -10e-9}}}
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid gate duration: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Invalid duration for gate 'u3'"):
         CalibrationData(str(tmp_calib_file))

def test_validation_invalid_readout_diag_value(tmp_calib_file):
    """Test validation fails for readout diagonal values outside [0, 1]."""
    invalid_data = {'readout': {'0': {'confusion_matrix_diag': [0.9, 1.1]}}}
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid readout diag value: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Invalid value for confusion_matrix_diag"):
         CalibrationData(str(tmp_calib_file))

def test_validation_invalid_readout_diag_format(tmp_calib_file):
    """Test validation fails for incorrect readout diagonal format."""
    invalid_data = {'readout': {'0': {'confusion_matrix_diag': [0.9, 0.8, 0.7]}}} # Too many values
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid readout diag format: {tmp_calib_file}") # Debug print
    with pytest.raises(ValueError, match="Invalid 'confusion_matrix_diag' format"):
         CalibrationData(str(tmp_calib_file))

def test_validation_invalid_crosstalk_type(tmp_calib_file):
    """Test validation fails for invalid crosstalk type."""
    invalid_data = {'crosstalk_strength_hz': 'abc'} # Not a number
    with open(tmp_calib_file, 'w') as f: yaml.dump(invalid_data, f)
    print(f"Testing validation with invalid crosstalk type: {tmp_calib_file}") # Debug print
    with pytest.raises(TypeError, match="Invalid type for 'crosstalk_strength_hz'"):
         CalibrationData(str(tmp_calib_file))

def test_get_methods(valid_calib_data):
    """Test basic get methods for valid data."""
    assert isinstance(valid_calib_data.get('qubits'), dict)
    assert isinstance(valid_calib_data.get_qubit_params(0)['T1'], (float, np.number))
    assert valid_calib_data.get_qubit_params(0)['T1'] > 0
    assert valid_calib_data.get('non_existent_key') is None
    assert valid_calib_data.get('non_existent_key', default='test') == 'test'
    # Check getting nested keys
    assert isinstance(valid_calib_data.get('qubits', '0'), dict)
    assert isinstance(valid_calib_data.get('qubits', '0', 'T1'), (float, np.number))
    assert valid_calib_data.get('qubits', '99', 'T1', default=123) == 123 # Test default for missing nested key
    # Test getting gate fidelity and duration
    assert isinstance(valid_calib_data.get_gate_fidelity('u3', []), (float, np.number, type(None)))
    assert isinstance(valid_calib_data.get_gate_duration('u3', []), (float, np.number, type(None)))
    assert isinstance(valid_calib_data.get_gate_fidelity('cz', [0, 2]), (float, np.number, type(None)))
    assert isinstance(valid_calib_data.get_gate_duration('cz', [0, 2]), (float, np.number, type(None)))
    # Test getting crosstalk
    assert isinstance(valid_calib_data.get_crosstalk_strength(), (float, np.number))
    # Test getting dt
    assert isinstance(valid_calib_data.get_dt(), (float, np.number, type(None)))


def test_normalization_string_to_number(tmp_calib_file):
    """Test if normalization converts numeric strings."""
    string_data = {
        'qubits': {
            '0': {'p1': '0.05', 'T1': '50e-6', 'T2': '25e-6'}
        },
        'gates': {
            'u3': {'fidelity': '0.998', 'duration': '30e-9'}
        },
        'readout': {
            '0': {'confusion_matrix_diag': ['0.95', '0.94']}
        },
        'crosstalk_strength_hz': '15000.0'
    }
    with open(tmp_calib_file, 'w') as f: yaml.dump(string_data, f)
    calib = CalibrationData(str(tmp_calib_file))

    q0_params = calib.get_qubit_params(0)
    assert isinstance(q0_params['p1'], float)
    assert np.isclose(q0_params['p1'], 0.05)
    assert isinstance(q0_params['T1'], float)
    assert np.isclose(q0_params['T1'], 50e-6)
    assert isinstance(q0_params['T2'], float)
    assert np.isclose(q0_params['T2'], 25e-6)

    u3_fidelity = calib.get_gate_fidelity('u3', [])
    u3_duration = calib.get_gate_duration('u3', [])
    assert isinstance(u3_fidelity, float)
    assert np.isclose(u3_fidelity, 0.998)
    assert isinstance(u3_duration, float)
    assert np.isclose(u3_duration, 30e-9)

    ro0_diag = calib.get_readout_params(0)
    assert isinstance(ro0_diag, list)
    assert len(ro0_diag) == 2
    assert isinstance(ro0_diag[0], float)
    assert np.isclose(ro0_diag[0], 0.95)
    assert isinstance(ro0_diag[1], float)
    assert np.isclose(ro0_diag[1], 0.94)

    crosstalk = calib.get_crosstalk_strength()
    assert isinstance(crosstalk, float)
    assert np.isclose(crosstalk, 15000.0)

