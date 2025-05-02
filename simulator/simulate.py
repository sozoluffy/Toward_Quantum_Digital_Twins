# simulator/simulate.py
from qiskit import QuantumCircuit, transpile
# Use direct import from qiskit_aer
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Local imports remain the same
from .noise_inserter import insert_layer_noise, build_instruction_durations
from calibration import CalibrationData
from typing import List, Tuple, Dict, Any

def run_simulation_with_layer_noise(
    circuit: QuantumCircuit, # This is the *original* ideal circuit
    calib_path: str,
    coupling_map: List[Tuple[int, int]],
    basis_gates: List[str],
    shots: int = 1024,
    scheduling_method: str = 'alap',
    backend_opts: Dict[str, Any] = {'method': 'density_matrix'},
    seed: int = None
) -> Dict[str, int]:
    """
    Runs simulation using the layer-by-layer noise insertion method.
    """
    print("Loading calibration...")
    calib = CalibrationData(calib_path)

    print("Building instruction durations...")
    inst_durations = build_instruction_durations(calib, basis_gates)

    print(f"Inserting noise using '{scheduling_method}' scheduling...")
    noisy_circuit_with_coherent_noise, readout_errors = insert_layer_noise(
        circuit=circuit, # Pass original circuit to noise inserter
        calib=calib,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        inst_durations=inst_durations,
        scheduling_method=scheduling_method
    )
    print("Noise insertion complete.")

    # --- Prepare NoiseModel for Readout Errors ---
    readout_noise_model = NoiseModel(basis_gates=basis_gates)
    if readout_errors:
        print("Adding readout errors to noise model...")
        for ro_error_obj in readout_errors:
            # Ensure ro_error_obj.channel returns the ReadoutError object
            # and ro_error_obj.index gives the qubit index
            readout_noise_model.add_readout_error(ro_error_obj.channel, [ro_error_obj.index])

    # --- Define the circuit to run ---
    # The circuit returned by insert_layer_noise already has coherent noise
    circuit_to_run = noisy_circuit_with_coherent_noise
    print(f"Circuit to run (length {len(circuit_to_run)}): {circuit_to_run.name}")
    # Optional: Draw or print circuit_to_run here for debugging

    # --- Run Simulation ---
    print(f"Running simulation with {shots} shots...")
    backend = AerSimulator(**backend_opts)
    run_options = {'seed_simulator': seed} if seed is not None else {}

    result = backend.run(
        circuit_to_run, # Run the modified circuit
        shots=shots,
        noise_model=readout_noise_model, # Apply only readout errors via model
        **run_options
    ).result()
    print("Simulation finished.")

    # ---> Get counts using the circuit that was ACTUALLY RUN <---
    try:
        # Use the circuit object that was simulated
        counts = result.get_counts(circuit_to_run)
        print(f"Retrieved counts using circuit object key '{circuit_to_run.name}'")
    except Exception as e:
        print(f"Warning: Failed to get counts using executed circuit object key ('{circuit_to_run.name}'): {e}.")
        print("Trying to get counts using index 0 instead.")
        # Fallback for simulators that might not handle circuit object keys well
        try:
            counts = result.get_counts(0) # Get counts for the first (only) experiment
            print("Retrieved counts using index 0.")
        except Exception as e2:
            print(f"Error: Failed to get counts using index 0: {e2}")
            counts = {} # Return empty counts on failure

    return counts
