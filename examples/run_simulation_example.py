# examples/run_simulation_example.py
import sys
import os
# Add project root to path to allow imports if run directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from qiskit (terra) are fine
from qiskit import QuantumCircuit, transpile
# Use direct import from qiskit_aer
from qiskit_aer import AerSimulator
# Local imports remain the same
from benchmarks import create_ghz_circuit
from simulator import run_simulation_with_layer_noise
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

print("Running simulation example...")

# --- Configuration ---
NUM_QUBITS = 3
SHOTS = 4096
# Ensure relative path works when run from root or examples dir
script_dir = os.path.dirname(__file__)
# Use os.path.join for robust path construction
CALIB_FILE = os.path.join(script_dir, '..', 'calibration', 'device_calib.yaml')
# ---> DEFINE BASIS_GATES HERE <---
# Basis gates should match those supported by your Target/Calibration
BASIS_GATES = ['u3', 'rz', 'cz', 'id', 'reset', 'measure']
# Define Coupling Map used for noise simulation (can be None if not needed for ideal)
COUPLING_MAP = [(0, 2), (1, 2)] # Example for 3 qubits (0,1 connected to 2)
SCHEDULING = 'alap'
SIM_METHOD = 'density_matrix' # More accurate for noise

# --- Create Circuit ---
print(f"Creating {NUM_QUBITS}-qubit GHZ circuit...")
# Create circuit *without* measurement for ideal statevector comparison if needed
# But for counts comparison, create with measurement
qc = create_ghz_circuit(NUM_QUBITS, measure=True) # Ensure measure=True
print("Circuit created:")
try:
    print(qc.draw(output='text', fold=-1))
except ImportError:
    print("(Install 'pylatexenc' for text drawing)")
except Exception as e:
     print(f"(Drawing failed: {e})")


# --- Run Simulation with Layer Noise ---
print(f"\nRunning noisy simulation ({SCHEDULING}, {SIM_METHOD})...")
noisy_counts = run_simulation_with_layer_noise(
    circuit=qc, # Pass original ideal circuit
    calib_path=CALIB_FILE,
    coupling_map=COUPLING_MAP, # Pass coupling map
    basis_gates=BASIS_GATES,  # Pass basis gates
    shots=SHOTS,
    scheduling_method=SCHEDULING,
    backend_opts={'method': SIM_METHOD},
    seed = 123 # Optional seed for reproducibility
)
print("Noisy simulation counts:")
print(noisy_counts)

# --- Run Ideal Simulation (for comparison) ---
print("\nRunning ideal simulation...")
# Use the imported AerSimulator directly
sim_ideal = AerSimulator()
# Transpile ideally for the target basis gates (important for comparison)
# Ensure BASIS_GATES is defined before this line
print(f"Transpiling ideal circuit to basis gates: {BASIS_GATES}")
ideal_transpiled = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
print("Ideal circuit transpiled.")
# Run simulation
result_ideal = sim_ideal.run(ideal_transpiled, shots=SHOTS, seed_simulator=456).result()
print("Ideal simulation finished.")
# ---> Get counts using the circuit that was ACTUALLY run for ideal sim <---
try:
    # Use the transpiled circuit object as the key
    ideal_counts = result_ideal.get_counts(ideal_transpiled)
    print(f"Retrieved ideal counts using circuit object key '{ideal_transpiled.name}'")
except Exception as e:
    print(f"Warning: Failed to get ideal counts using circuit object key: {e}.")
    print("Trying to get ideal counts using index 0 instead.")
    try:
        ideal_counts = result_ideal.get_counts(0)
        print("Retrieved ideal counts using index 0.")
    except Exception as e2:
        print(f"Error: Failed to get ideal counts using index 0: {e2}")
        ideal_counts = {} # Return empty counts on failure
print("Ideal simulation counts:")
print(ideal_counts)

# --- Plot Results ---
print("\nPlotting results...")
legend = ['Ideal', 'Noisy Simulation (Layered)']
# Ensure both counts dictionaries are valid before plotting
if ideal_counts or noisy_counts: # Proceed if at least one has data
    try:
        hist = plot_histogram([ideal_counts, noisy_counts], legend=legend, figsize=(10, 6),
                              title=f"{NUM_QUBITS}-Qubit GHZ State Simulation ({SCHEDULING} Scheduling)")
        plt.show()
        print("Plot displayed. Close plot window to exit.")
    except Exception as e:
        print(f"Plotting failed. Ensure matplotlib is installed ({e})")
else:
     print("Skipping plot: No data in ideal or noisy counts.")

print("\nExample finished.")
