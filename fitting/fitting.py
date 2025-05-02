# fitting/fitting.py
# ---> START FIX: Add project root to path <---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---> END FIX <---

import numpy as np
from scipy.optimize import differential_evolution
from qiskit import QuantumCircuit
from simulator import run_simulation_with_layer_noise # This import should now work
from calibration import CalibrationData
from benchmarks import create_ghz_circuit, create_w_state_circuit
from typing import List, Dict, Tuple, Callable, Any # Add Any
import copy, yaml, os, time
import traceback # For printing exceptions during simulation

def calculate_tvd(counts1: Dict[str, int], counts2: Dict[str, int], shots: int) -> float:
    """Calculates the Total Variation Distance (TVD) between two count dictionaries."""
    tvd = 0.0
    all_keys = set(counts1.keys()) | set(counts2.keys())
    if shots <= 0: return 1.0 # Avoid division by zero; max distance
    # Ensure counts are non-negative integers
    c1_total = sum(abs(v) for v in counts1.values() if isinstance(v, int) and v >= 0)
    c2_total = sum(abs(v) for v in counts2.values() if isinstance(v, int) and v >= 0)
    # Use the larger shot count for normalization if they differ, or the target shots
    norm_shots = max(c1_total, c2_total, shots)
    if norm_shots <= 0: return 1.0 # Return max distance if no valid shots

    for key in all_keys:
        p1 = counts1.get(key, 0) / norm_shots
        p2 = counts2.get(key, 0) / norm_shots
        tvd += 0.5 * abs(p1 - p2)
    # Clamp TVD to [0, 1] just in case of floating point issues
    return min(max(tvd, 0.0), 1.0)

# ---> NEW HELPER FUNCTION <---
def convert_numpy_to_native(data: Any) -> Any:
    """Recursively converts NumPy numbers in nested dicts/lists to Python floats/ints."""
    if isinstance(data, dict):
        return {k: convert_numpy_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray): # Convert arrays to lists
        return convert_numpy_to_native(data.tolist())
    else:
        return data
# ---------------------------

# --- objective_function (Modified before yaml.dump, fixed indentation) ---
def objective_function(
    params: np.ndarray, # params from optimizer are numpy arrays/floats
    param_names: List[str],
    base_calib_path: str,
    benchmark_circuits: List[QuantumCircuit],
    target_results: List[Dict[str, int]], # Expecting list of dicts now
    coupling_map: List[Tuple[int, int]],
    basis_gates: List[str],
    shots: int,
    backend_opts: Dict,
    scheduling_method: str,
) -> float:
    """
    Cost function for optimization: calculates mean TVD using the layer-noise simulator.
    Modifies a temporary calibration dict with current params.
    """
    process_id = os.getpid() # Unique ID for parallel execution if needed
    script_dir = os.path.dirname(__file__) # Get directory of this script
    temp_calib_path = os.path.join(script_dir, f"temp_calib_optim_{process_id}.yaml") # Create temp file here
    start_time = time.time()
    param_str = ', '.join([f'{p:.5f}' for p in params])
    # print(f"\n[ObjFunc {process_id}] Evaluating Params: [{param_str}]") # Optional: Verbose logging

    mean_tvd = 1.0 * len(benchmark_circuits) # Default to max penalty

    # Create and save temporary calibration file
    try:
        # Load base calibration data carefully
        # Ensure CalibrationData can load the base file without issues
        base_calib = CalibrationData(base_calib_path)
        base_calib_data = base_calib.data
        temp_calib_dict = copy.deepcopy(base_calib_data) # Work on a copy

        # Update dict with NumPy types from params
        for name, value in zip(param_names, params):
            if name == 'crosstalk_strength_hz':
                temp_calib_dict['crosstalk_strength_hz'] = value # Still numpy type
            elif name.startswith('fidelity_'):
                gate_id = name.replace('fidelity_', '')
                valid_fidelity = min(max(value, 0.0), 1.0) # Clamp to [0, 1]
                # Ensure nested structure exists before assignment
                temp_calib_dict.setdefault('gates', {})[gate_id] = temp_calib_dict['gates'].get(gate_id, {})
                temp_calib_dict['gates'][gate_id]['fidelity'] = valid_fidelity # Still numpy type
            # Add handling for other parameter types if needed

        # ---> CONVERT NUMPY TYPES before dumping <---
        native_calib_dict = convert_numpy_to_native(temp_calib_dict)
        # -------------------------------------------

        with open(temp_calib_path, 'w') as f:
             # Dump the dictionary with native Python types
             yaml.dump(native_calib_dict, f, default_flow_style=False)

    except Exception as e:
        print(f"[ObjFunc {process_id}] ERROR preparing temp calib: {e}")
        traceback.print_exc() # Print traceback for calibration loading error
        if os.path.exists(temp_calib_path):
             try: os.remove(temp_calib_path)
             except OSError: pass
        return mean_tvd # Return max penalty

    # Run simulations and calculate TVD
    total_tvd = 0.0
    num_circuits = len(benchmark_circuits)

    for i, (circuit, target_counts) in enumerate(zip(benchmark_circuits, target_results)):
        circuit_name = circuit.name if hasattr(circuit, 'name') else f'circuit_{i}'
        # print(f"[ObjFunc {process_id}]  Simulating {circuit_name}...") # Optional verbose log
        if not target_counts or not isinstance(target_counts, dict):
             print(f"  Warning: Invalid target counts for {circuit_name}. Skipping.")
             total_tvd += 1.0
             continue

        target_shots = sum(target_counts.values());
        if target_shots <= 0:
             print(f"  Warning: Target counts for {circuit_name} sum to zero. Skipping.")
             total_tvd += 1.0
             continue
        sim_shots = shots if shots > 0 else target_shots

        sim_counts = {} # Initialize
        try:
             sim_start = time.time()
             # This will now load the temp file correctly using CalibrationData's safe_load
             sim_counts = run_simulation_with_layer_noise(
                 circuit=circuit, calib_path=temp_calib_path, coupling_map=coupling_map,
                 basis_gates=basis_gates, shots=sim_shots, backend_opts=backend_opts,
                 scheduling_method=scheduling_method, seed = None
             )
             sim_end = time.time()
             # Print counts and TVD for debugging the TVD=1.0 issue
             sim_counts_total = sum(sim_counts.values()) if isinstance(sim_counts, dict) else 0
             print(f"[ObjFunc {process_id}]    Target Sum={target_shots}, Sim Sum={sim_counts_total}")
             print(f"[ObjFunc {process_id}]    Target Counts ({circuit_name}): {target_counts}")
             print(f"[ObjFunc {process_id}]    Simulated Counts ({circuit_name}): {sim_counts}")

             if not isinstance(sim_counts, dict) or sim_counts_total <=0:
                  print(f"  ERROR: Simulation returned invalid/empty counts. Penalizing.")
                  tvd = 1.0
             else:
                  tvd = calculate_tvd(sim_counts, target_counts, target_shots)
                  print(f"[ObjFunc {process_id}]    TVD for {circuit_name}: {tvd:.6f} (Sim time: {sim_end-sim_start:.2f}s)")
             total_tvd += tvd
        except Exception as e:
             print(f"[ObjFunc {process_id}]  ERROR during simulation run for {circuit_name}: {e}")
             traceback.print_exc() # Print full traceback for simulation error
             total_tvd += 1.0 # Penalize failure

    # Clean up temporary file AFTER the loop finishes
    if os.path.exists(temp_calib_path):
        # ---> CORRECTED INDENTATION <---
        try:
            os.remove(temp_calib_path)
        except OSError as e:
            print(f"Warning: Could not remove temp file {temp_calib_path}: {e}")
        # -----------------------------

    mean_tvd = total_tvd / num_circuits if num_circuits > 0 else 1.0
    end_time = time.time()
    print(f"[ObjFunc {process_id}] Result for Params [{param_str}] -> Mean TVD: {mean_tvd:.6f} (Eval time: {end_time-start_time:.2f}s)")
    return mean_tvd


# --- fit_parameters function ---
def fit_parameters(
    base_calib_path: str,
    param_names: List[str],
    param_bounds: List[Tuple[float, float]],
    benchmark_circuits: List[QuantumCircuit],
    target_results: List[Dict[str, int]], # Expects List[Dict]
    coupling_map: List[Tuple[int, int]],
    basis_gates: List[str],
    shots: int,
    backend_opts: Dict = {'method': 'density_matrix'},
    scheduling_method: str = 'alap',
    optimizer_opts: Dict = {'maxiter': 100, 'popsize': 15, 'tol': 0.01, 'updating': 'deferred', 'workers': -1}
) -> Tuple[np.ndarray, float]:
    """
    Uses Differential Evolution to fit parameters by minimizing mean TVD.
    """
    print(f"--- Starting Optimization ---")
    print(f"Parameters to fit: {param_names}")
    print(f"Bounds: {param_bounds}")
    print(f"# Benchmarks: {len(benchmark_circuits)}")
    print(f"Optimizer settings: {optimizer_opts}")
    print(f"Simulation shots per circuit: {shots}")

    if len(benchmark_circuits) != len(target_results):
        raise ValueError(f"Num circuits ({len(benchmark_circuits)}) != Num targets ({len(target_results)}).")
    if not all(isinstance(item, dict) for item in target_results):
         raise TypeError("Target results must be a list of dictionaries (counts).")

    result = differential_evolution(
        func=objective_function,
        bounds=param_bounds,
        args=(
            param_names, base_calib_path, benchmark_circuits, target_results,
            coupling_map, basis_gates, shots, backend_opts, scheduling_method,
        ),
        **optimizer_opts
    )

    print("\n--- Optimization Finished ---")
    if hasattr(result, 'success') and result.success:
        print(f"Optimization successful after {result.nfev} evaluations!")
        print(f"Best parameters found ({len(param_names)}):")
        for name, val in zip(param_names, result.x): print(f"  {name}: {val:.8f}")
        print(f"Minimum mean TVD achieved: {result.fun:.6f}")
    elif hasattr(result, 'message'):
        print(f"Optimization did not converge or failed: {result.message}")
        if hasattr(result, 'x'): print(f"Last parameters evaluated: {result.x}")
        if hasattr(result, 'fun'): print(f"Objective function value: {result.fun:.6f}")
    else: print(f"Optimization finished, unexpected result object:\n{result}")

    return result.x, result.fun


# --- Example Usage Block (__main__) ---
if __name__ == '__main__':
    # --- Configuration (ADJUST THESE) ---
    NUM_QUBITS_BENCHMARK = 3
    TOTAL_SHOTS = 4096
    script_dir = os.path.dirname(__file__)
    # Use os.path.abspath to ensure the path is absolute
    BASE_CALIB_FILE = os.path.abspath(os.path.join(script_dir, '..', 'calibration', 'device_calib.yaml'))
    SYNTHETIC_RESULTS_FILE = os.path.join(script_dir, 'synthetic_results.yaml')

    COUPLING_MAP = [(0, 2), (1, 2), (2, 3)]
    BASIS_GATES = ['u3', 'rz', 'cz', 'id', 'reset', 'measure']

    PARAMS_TO_FIT = [
        'fidelity_cz_0_2', 'fidelity_cz_1_2', 'fidelity_cz_3_2',
        'crosstalk_strength_hz'
    ]
    PARAM_BOUNDS = [(0.9, 1.0), (0.9, 1.0), (0.9, 1.0), (0, 50000.0)]

    SCHEDULING_METHOD_FIT = 'alap'
    BACKEND_OPTS_FIT = {'method': 'density_matrix'}

    # Use fewer iterations/popsize for faster debugging runs
    OPTIMIZER_OPTS_FIT = {
        'maxiter': 5,     # Low value for quick test run
        'popsize': 4,     # Low value for quick test run
        'tol': 0.01,
        'updating': 'immediate', # Faster for serial run (workers=1)
        'workers': 1     # Set to -1 to use all cores (requires updating='deferred')
        # 'seed': 42      # Optional: for reproducible optimization runs
    }

    # --- Create Benchmark Circuits ---
    print("Creating benchmark circuits...");
    ghz3 = create_ghz_circuit(NUM_QUBITS_BENCHMARK, measure=True)
    w3 = create_w_state_circuit(NUM_QUBITS_BENCHMARK, measure=True)
    benchmarks = [ghz3, w3]
    print(f"Using {len(benchmarks)} benchmark circuits: {[c.name for c in benchmarks]}")

    # --- Generate or Load Synthetic Target Results ---
    print(f"Loading/Generating target results (file: {SYNTHETIC_RESULTS_FILE})...")
    target_data = None
    # ---> Attempt to load first <---
    if os.path.exists(SYNTHETIC_RESULTS_FILE):
        try:
            print(f"Attempting to load existing results from {SYNTHETIC_RESULTS_FILE}")
            with open(SYNTHETIC_RESULTS_FILE, 'r') as f: target_data = yaml.safe_load(f)
            print(f"Loaded {len(target_data)} existing target results.")
            if len(target_data) != len(benchmarks): raise ValueError("Benchmark/target mismatch.")
            if not all(isinstance(item, dict) for item in target_data): raise ValueError("Loaded data not list[dict].")
        except Exception as load_err:
             print(f"Error loading {SYNTHETIC_RESULTS_FILE}: {load_err}. Will regenerate.")
             # Attempt to delete the problematic file before regenerating
             try: os.remove(SYNTHETIC_RESULTS_FILE)
             except OSError: pass
             target_data = None

    # ---> Regenerate ONLY if loading failed or file didn't exist <---
    if target_data is None:
        print("Generating new synthetic results...")
        target_data_regen = []
        true_fidelities = {'cz_0_2': 0.965, 'cz_1_2': 0.955, 'cz_3_2': 0.975}
        true_crosstalk = 18000.0
        temp_gen_calib_path = os.path.join(script_dir, "temp_gen_calib.yaml")
        try:
            # Load base calibration to modify
            base_calib = CalibrationData(BASE_CALIB_FILE)
            gen_calib_dict = copy.deepcopy(base_calib.data)
            # Update with 'true' values
            gen_calib_dict['crosstalk_strength_hz'] = true_crosstalk
            for gate_id, fid in true_fidelities.items():
                gen_calib_dict.setdefault('gates', {})[gate_id] = gen_calib_dict['gates'].get(gate_id, {})
                gen_calib_dict['gates'][gate_id]['fidelity'] = fid
            # Convert numpy types before dumping this 'true' calib file
            native_gen_calib_dict = convert_numpy_to_native(gen_calib_dict)
            with open(temp_gen_calib_path, 'w') as f: yaml.dump(native_gen_calib_dict, f)

            # Simulate results using these 'true' parameters
            for i, circuit in enumerate(benchmarks):
                print(f" Simulating target counts for {circuit.name}...")
                counts = run_simulation_with_layer_noise(
                    circuit=circuit, calib_path=temp_gen_calib_path, coupling_map=COUPLING_MAP,
                    basis_gates=BASIS_GATES, shots=TOTAL_SHOTS, backend_opts=BACKEND_OPTS_FIT,
                    scheduling_method=SCHEDULING_METHOD_FIT, seed=42+i)
                target_data_regen.append(dict(counts)) # Save as dict

            # Save the generated data (list of dicts)
            with open(SYNTHETIC_RESULTS_FILE, 'w') as f:
                yaml.dump(target_data_regen, f, default_flow_style=False)
            print(f"Generated and saved {len(target_data_regen)} results.")
            target_data = target_data_regen # Use the newly generated data
        except Exception as gen_err:
             print(f"ERROR generating synthetic data: {gen_err}")
             traceback.print_exc() # Print full traceback for generation error
             target_data = [] # Prevent fitting from running without data
        finally:
            # Clean up temp generation file with correct indentation
            if os.path.exists(temp_gen_calib_path):
                 try:
                      os.remove(temp_gen_calib_path)
                 except OSError as e:
                      print(f"Warning: Could not remove temp generation calib file: {e}")

    # --- Run Fitting ---
    if target_data and len(target_data) == len(benchmarks) and all(isinstance(item, dict) for item in target_data):
        print("\n--- Starting Parameter Fitting ---")
        best_params, min_tvd = fit_parameters(
            base_calib_path=BASE_CALIB_FILE, param_names=PARAMS_TO_FIT, param_bounds=PARAM_BOUNDS,
            benchmark_circuits=benchmarks, target_results=target_data, coupling_map=COUPLING_MAP,
            basis_gates=BASIS_GATES, shots=TOTAL_SHOTS, backend_opts=BACKEND_OPTS_FIT,
            scheduling_method=SCHEDULING_METHOD_FIT, optimizer_opts=OPTIMIZER_OPTS_FIT
        )
        print("\n--- Fitting Complete ---"); print(f"\nOptimized Parameters:");
        for name, val in zip(PARAMS_TO_FIT, best_params): print(f"  {name}: {val:.8f}")
        print(f"\nAchieved Mean TVD: {min_tvd:.6f}"); print("\nUpdate device_calib.yaml.")
    else:
         print("\n--- Skipping Parameter Fitting: Target data invalid or missing. ---")
