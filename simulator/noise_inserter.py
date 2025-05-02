# simulator/noise_inserter.py
import numpy as np
# ---> CORE QISKIT IMPORTS AT THE VERY TOP <---
from qiskit import QuantumCircuit, transpile
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager, Layout, InstructionDurations, Target, CouplingMap
from qiskit.transpiler import InstructionProperties
from qiskit.transpiler.passes import SetLayout
from qiskit.circuit import Measure, Reset, Parameter, Qubit, Clbit
from qiskit.circuit.library import U3Gate, RZGate, CZGate, CXGate, IGate, SXGate, XGate
# ---------------------------------------------

# Use direct imports from qiskit_aer.noise
from qiskit_aer.noise import thermal_relaxation_error, QuantumError

# Local project imports
from calibration import CalibrationData
from noise.reset_error import create_reset_error
from noise.gate_error import GateError
from noise.coherent_crosstalk import CoherentCrosstalkGate
from noise.measurement_error import MeasurementError

# Import typing hints AFTER standard modules
from typing import List, Tuple, Dict, Optional, Type
from qiskit.circuit import Gate # For type hinting

# --- build_instruction_durations function (remains the same) ---
def build_instruction_durations(calib: CalibrationData, basis_gates: List[str]) -> InstructionDurations:
    # (Code from previous responses)
    durations = []
    unique_gates_with_duration = set()
    for gate_id, gate_data in calib.get('gates', default={}).items():
        duration = gate_data.get('duration')
        if duration is not None:
            parts = gate_id.split('_')
            name = parts[0]
            qubits = tuple(int(q) for q in parts[1:]) if len(parts) > 1 else None
            if name in basis_gates:
                 durations.append((name, qubits, duration))
                 unique_gates_with_duration.add((name, qubits))
    for gate in basis_gates:
        if (gate, None) not in unique_gates_with_duration:
            duration = calib.get_gate_duration(gate, [])
            if duration is not None:
                durations.append((gate, None, duration))
                unique_gates_with_duration.add((gate, None))
    if 'u3' in basis_gates:
         duration_u3 = calib.get_gate_duration('u3', [])
         if duration_u3 is not None:
              derived_gates_needed = []
              for dg in ['u', 'p', 'u1', 'u2', 'rx', 'ry', 'sx', 'x', 'y', 's', 'sdg', 't', 'tdg', 'h']:
                    if dg not in basis_gates: derived_gates_needed.append(dg)
              for derived_gate in derived_gates_needed:
                   if (derived_gate, None) not in unique_gates_with_duration and \
                      not any(g[0] == derived_gate for g in unique_gates_with_duration if g[1] is not None):
                        durations.append((derived_gate, None, duration_u3))
    dt = calib.get_dt()
    print(f"Building InstructionDurations with dt={dt}")
    return InstructionDurations(durations, dt=dt)


# --- Function to create a basic Target (remains the same) ---
def create_qiskit_target(
    num_qubits: int,
    basis_gates: List[str],
    inst_durations: InstructionDurations,
    coupling_map: Optional[List[Tuple[int, int]]] = None,
    dt: Optional[float] = None
) -> Target:
    # (Code from previous responses - logic is ok now)
    print(f"Initializing Target with num_qubits={num_qubits}, dt={dt}")
    target = Target(num_qubits=num_qubits, dt=dt)
    gate_map: Dict[str, Tuple[Type[Gate], int, int]] = {}
    gate_class_imports = {
        "u3": ("qiskit.circuit.library", "U3Gate"), "rz": ("qiskit.circuit.library", "RZGate"),
        "cz": ("qiskit.circuit.library", "CZGate"), "id": ("qiskit.circuit.library", "IGate"),
        "reset": ("qiskit.circuit", "Reset"), "measure": ("qiskit.circuit", "Measure"),
        "cx": ("qiskit.circuit.library", "CXGate"), "sx": ("qiskit.circuit.library", "SXGate"),
        "x": ("qiskit.circuit.library", "XGate"), "h": ("qiskit.circuit.library", "HGate"),
        "p": ("qiskit.circuit.library", "PhaseGate"), "rx": ("qiskit.circuit.library", "RXGate"),
        "ry": ("qiskit.circuit.library", "RYGate"), "u": ("qiskit.circuit.library", "UGate"),
        "u1": ("qiskit.circuit.library", "U1Gate"), "u2": ("qiskit.circuit.library", "U2Gate"),
    }
    gate_param_counts = {"u3": 3, "rz": 1, "cz": 0, "id": 0, "reset": 0, "measure": 0, "cx": 0, "sx": 0, "x": 0, "h": 0, "p": 1, "rx": 1, "ry": 1, "u": 3, "u1": 1, "u2": 2}
    gate_qubit_counts = {"u3": 1, "rz": 1, "cz": 2, "id": 1, "reset": 1, "measure": 1, "cx": 2, "sx": 1, "x": 1, "h": 1, "p": 1, "rx": 1, "ry": 1, "u": 1, "u1": 1, "u2": 1}
    imported_classes = {"Measure": Measure, "Reset": Reset, "Parameter": Parameter}
    print("Loading gate classes for basis gates:", basis_gates)
    for gate_name in basis_gates:
        if gate_name in gate_class_imports:
            module_name, class_name = gate_class_imports[gate_name]
            if class_name not in imported_classes:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    gate_class = getattr(module, class_name)
                    imported_classes[class_name] = gate_class
                except Exception: continue
            else: gate_class = imported_classes[class_name]
            num_q = gate_qubit_counts.get(gate_name); num_p = gate_param_counts.get(gate_name)
            if num_q is not None and num_p is not None: gate_map[gate_name] = (gate_class, num_q, num_p)
        elif gate_name not in ['delay']: print(f"Warning: Basis gate '{gate_name}' not in predefined import map.")
    print("Adding instructions to Target:")
    for gate_name, (gate_class, num_gate_qubits, num_params) in gate_map.items():
         if gate_name not in basis_gates: continue
         params = [Parameter(f"p_{i}") for i in range(num_params)] if num_params > 0 else []
         try: inst = gate_class(*params) if params else gate_class()
         except Exception: continue
         properties_map = {}
         if num_gate_qubits == 1: qubit_tuples = [(q,) for q in range(num_qubits)]
         elif num_gate_qubits == 2 and coupling_map:
             qubit_tuples = list(coupling_map)
             if gate_name in ["cz"]: qubit_tuples.extend([(q1, q0) for q0, q1 in coupling_map])
         elif num_gate_qubits == 2 and not coupling_map:
             qubit_tuples = [(q0, q1) for q0 in range(num_qubits) for q1 in range(num_qubits) if q0 != q1]
         else: qubit_tuples = [()]
         for q_tup in set(qubit_tuples):
              if any(q_idx >= num_qubits for q_idx in q_tup): continue
              duration = None
              try: duration = inst_durations.get(gate_name, q_tup)
              except Exception:
                   try: duration = inst_durations.get(gate_name, None)
                   except Exception: pass
              if duration is not None: properties_map[q_tup] = InstructionProperties(duration=duration, error=None)
         if properties_map or gate_name in ['reset', 'measure']: target.add_instruction(inst, properties_map if properties_map else None)
         else: target.add_instruction(inst, None)
    print("Target object creation finished.")
    return target


# --- insert_layer_noise function (Simplified transpile calls again) ---
def insert_layer_noise(
    circuit: QuantumCircuit,
    calib: CalibrationData,
    coupling_map: List[Tuple[int, int]],
    basis_gates: List[str],
    inst_durations: InstructionDurations,
    scheduling_method: str = 'alap',
) -> Tuple[QuantumCircuit, List[MeasurementError]]:
    """
    Inserts noise operations layer-by-layer into a circuit DAG based on scheduling
    performed by the main transpile function using a Target object.
    """
    num_qubits = circuit.num_qubits
    circuit_qubits = list(circuit.qubits)
    circuit_clbits = list(circuit.clbits)

    qiskit_target = create_qiskit_target(
        num_qubits=num_qubits, basis_gates=basis_gates,
        inst_durations=inst_durations, coupling_map=coupling_map, dt=calib.get_dt()
    )

    # ---> Attempt scheduled transpilation (simplified call) <---
    print(f"Attempting scheduled transpilation (scheduling_method='{scheduling_method}')...")
    scheduled_dag = None
    try:
        # Rely only on Target and scheduling_method. Qiskit should use durations from Target.
        transpiled_scheduled_circuit = transpile(
            circuit,
            target=qiskit_target,
            scheduling_method=scheduling_method.lower(),
            optimization_level=0 # Keep structure
        )
        scheduled_dag = circuit_to_dag(transpiled_scheduled_circuit)
        print("Scheduled transpilation successful.")
    except Exception as e:
        print(f"Warning: Scheduled transpilation failed: {e}")
        print("Falling back to basic transpilation using Target.")
        try:
             # Fallback: Just transpile to target basis gates, no explicit scheduling call
             transpiled_circuit = transpile(
                 circuit,
                 target=qiskit_target,
                 optimization_level=0
             )
             scheduled_dag = circuit_to_dag(transpiled_circuit)
             print("Fallback transpilation successful.")
        except Exception as fallback_e:
             print(f"Fallback transpilation also failed: {fallback_e}")
             # Last resort: use original circuit's DAG (likely inaccurate)
             scheduled_dag = circuit_to_dag(circuit)
             print("Warning: Using original circuit DAG for noise insertion.")


    # --- Build the noisy QuantumCircuit ---
    print("Building noisy circuit from DAG...")
    noisy_qc = QuantumCircuit(circuit.qubits, circuit.clbits, name=circuit.name + "_noisy")
    qubit_stop_times = {q: 0.0 for q in circuit_qubits}
    readout_errors = []

    # Apply initial Reset error
    for q in circuit_qubits:
        q_idx = circuit_qubits.index(q)
        q_params = calib.get_qubit_params(q_idx)
        reset_err = create_reset_error(q_params['p1'])
        if reset_err:
            try: noisy_qc.append(reset_err, [noisy_qc.qubits[q_idx]])
            except Exception as append_e: print(f"Warning: Could not append reset error for qubit {q_idx}: {append_e}")

    # Iterate through scheduled/transpiled DAG nodes
    for node in scheduled_dag.topological_op_nodes():
        op = node.op
        qargs_dag = node.qargs
        cargs_dag = node.cargs
        qargs_indices = [circuit_qubits.index(q) for q in qargs_dag] if qargs_dag else []
        qargs_noisyqc = [noisy_qc.qubits[i] for i in qargs_indices]
        cargs_noisyqc = [noisy_qc.clbits[circuit_clbits.index(c)] for c in cargs_dag] if cargs_dag else []

        # Timing calculation relies on inst_durations, as schedule times aren't reliable from DAG props
        start_time = max((qubit_stop_times[circuit_qubits[i]] for i in qargs_indices), default=0.0)
        duration = 0.0
        try: duration = inst_durations.get(op.name, qargs_indices)
        except Exception:
            try: duration = inst_durations.get(op.name, None)
            except Exception: duration = 0.0
        stop_time = start_time + duration

        # Apply Idle Decay BEFORE
        for i, q_idx in enumerate(qargs_indices):
            q_orig = circuit_qubits[q_idx]; q_noisy = qargs_noisyqc[i]
            idle_start_time = qubit_stop_times[q_orig]
            idle_duration = start_time - idle_start_time
            if idle_duration > 1e-15:
                q_params = calib.get_qubit_params(q_idx)
                T1, T2 = q_params['T1'], q_params['T2']
                if T1 != np.inf or T2 != np.inf:
                    decay_err_tuple = thermal_relaxation_error(T1, T2, idle_duration)
                    if decay_err_tuple:
                        try: noisy_qc.append(decay_err_tuple, [q_noisy])
                        except Exception as append_e: print(f"Warning: Could not append decay error for qubit {q_idx}: {append_e}")

        # Apply Operation
        try: noisy_qc.append(op, qargs_noisyqc, cargs_noisyqc)
        except Exception as append_e: print(f"Warning: Could not append operation {op.name} on {qargs_indices}: {append_e}")

        # Apply Gate Error AFTER
        if op.name not in ['measure', 'reset', 'delay', 'barrier', 'snapshot', 'id']:
            fidelity = calib.get_gate_fidelity(op.name, qargs_indices)
            if fidelity is not None and fidelity < 1.0 - 1e-9:
                gate_err = GateError(fidelity, len(qargs_noisyqc)).channel()
                if gate_err is not None:
                    try: noisy_qc.append(gate_err, qargs_noisyqc)
                    except Exception as append_e: print(f"Warning: Could not append gate error for {op.name} on {qargs_indices}: {append_e}")

        # Handle Measurement
        if op.name == 'measure':
            q_idx = qargs_indices[0]
            ro_diag = calib.get_readout_params(q_idx)
            if ro_diag: readout_errors.append(MeasurementError(tuple(ro_diag), q_idx))

        # Update stop times
        for q_idx in qargs_indices: qubit_stop_times[circuit_qubits[q_idx]] = stop_time

    # Apply Crosstalk (Placeholder)
    crosstalk_strength_hz = calib.get_crosstalk_strength()
    if crosstalk_strength_hz > 1e-9:
            print("Warning: Coherent crosstalk insertion based on idle time is complex and not implemented in this version.")

    print("Finished building noisy circuit.")
    return noisy_qc, readout_errors
