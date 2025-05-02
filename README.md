# Towards a Digital Twin of Noisy Quantum Computers (arXiv:2504.08313v1)

This repository implements a calibration-driven digital twin for simulating noisy quantum computers, based on the model described in the paper "Towards a Digital Twin of Noisy Quantum Computers: Calibration-Driven Emulation of Transmon Qubits" (arXiv:2504.08313v1).

The goal is to emulate the behavior of a specific quantum device by incorporating noise parameters extracted from its calibration data and fitting certain parameters using benchmark circuits.

## Features

* **Calibration Loading:** Loads device parameters (T1, T2, gate fidelities, gate durations, readout errors, coupling map, basis gates) from a YAML file.
* **Layer-Based Noise Model:** Implements noise application based on circuit scheduling and time intervals (layers), applying:
    * Initial state preparation errors (thermal population `p1`).
    * Idle time decoherence (T1 relaxation, T2 dephasing) using `thermal_relaxation_error`.
    * Gate errors modeled as dephasing channels based on fidelity.
    * Readout errors using confusion matrices.
    * Coherent ZZ crosstalk approximation applied during idle periods between coupled qubits.
* **Simulation:** Uses `qiskit-aer`'s `AerSimulator` to run the noisy circuits.
* **Parameter Fitting:** Includes a script using `scipy.optimize.differential_evolution` to fit specified noise parameters (e.g., 2-qubit gate fidelities, crosstalk strength) by minimizing the Total Variation Distance (TVD) between simulation results and target data.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sozoluffy/Toward_Quantum_Digital_Twins.git
    cd Toward_Quantum_Digital_Twins
    ```

2.  **Create a Conda Environment (Recommended):**
    ```bash
    conda create --name qdt_env python=3.9 -y
    conda activate qdt_env
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, for development:
    ```bash
    pip install -e .[test] # Assuming test extras are defined in setup.py if needed
    ```

## Configuration (`calibration/device_calib.yaml`)

The simulation relies on a YAML configuration file specifying the target device's properties. An example based on synthetic data (`device_calib.yaml`) is provided. You should replace this with data for your target device.

Key sections:
* `dt`: Backend time resolution in seconds (required for durations).
* `basis_gates`: List of native gate names supported.
* `coupling_map`: List of connected qubit pairs, e.g., `[[0, 1], [1, 2]]`.
* `qubits`: Dictionary mapping qubit index (as string) to its properties:
    * `T1`: Relaxation time (seconds).
    * `T2`: Dephasing time (seconds).
    * `p1`: Initial thermal population P(|1>) (probability).
* `gates`: Dictionary mapping gate identifiers to properties:
    * Generic gates (e.g., `id`, `sx`, `ecr`): `{fidelity: 0.99, duration: 50e-9}`
    * Specific gates (e.g., `cz_0_2`, `ecr_85_84`): `{fidelity: 0.95, duration: 45e-9}` (Specific overrides generic)
* `readout`: Dictionary mapping qubit index (as string) to:
    * `confusion_matrix_diag`: List `[P(0|0), P(1|1)]`.
* `crosstalk_strength_hz`: Initial guess or fitted value for ZZ crosstalk strength J (in Hz).

**Note:** The current implementation primarily supports `cz` as the 2Q gate in fitting/Target creation unless modified. Adapt the code (mainly `simulator/noise_inserter.py`'s `create_qiskit_target`) if using other native 2Q gates like `ecr`.

## Usage

### Running a Simulation Example

1.  Ensure `calibration/device_calib.yaml` is configured.
2.  Run the example script:
    ```bash
    python examples/run_simulation_example.py
    ```
    This will simulate a 3-qubit GHZ state with noise defined in the calibration file and compare it to an ideal simulation, plotting the results.

### Running Parameter Fitting

1.  **Prepare Target Data:**
    * Create benchmark circuits (e.g., using `benchmarks/`) that exercise the parameters you want to fit.
    * Obtain experimental results (counts dictionaries) for these circuits from the target hardware, or generate synthetic data.
    * Save the target results as a YAML list of dictionaries in `fitting/synthetic_results.yaml`. The order must match the `benchmarks` list in `fitting.py`. (The script can generate initial synthetic data if the file is missing).
2.  **Configure Fitting Script:**
    * Edit `fitting/fitting.py`.
    * Inside the `if __name__ == '__main__':` block:
        * Set `BASE_CALIB_FILE` path.
        * Set `SYNTHETIC_RESULTS_FILE` path.
        * Define `COUPLING_MAP` and ensure `BASIS_GATES` are correct (or read from calibration).
        * **Crucially:** Define `PARAMS_TO_FIT` (list of parameter names like `fidelity_cz_0_2`, `crosstalk_strength_hz`) and `PARAM_BOUNDS` (list of `(min, max)` tuples).
        * Adjust `OPTIMIZER_OPTS_FIT` (`maxiter`, `popsize`, `workers`, etc.).
3.  **Run Fitting:**
    ```bash
    python fitting/fitting.py
    ```
4.  **Update Calibration:** Copy the optimized parameters printed at the end into your `calibration/device_calib.yaml`.

## Testing

Run unit tests using pytest from the project root directory:
```bash
pytest
