# noise/coherent_crosstalk.py
import numpy as np
from typing import List, Tuple, Optional
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library import RZZGate

class CoherentCrosstalkGate(Gate):
    # --- Paste the CoherentCrosstalkGate class code here ---
    # (From previous response, using RZZGate)
    """
    Implements the coherent ZZ crosstalk interaction exp(-i*beta*duration*Z⊗Z)
    as a Qiskit Gate using RZZGate. beta is coupling strength (Eq. 10).
    """
    def __init__(self, beta_hz: float, duration: float, label: Optional[str] = None):
        """
        Args:
            beta_hz: The ZZ coupling strength parameter β in Hz (from Eq. 10).
            duration: The duration 'd_uv' over which the interaction occurs (layer duration).
            label: An optional label for the gate.
        """
        # Calculate RZZ angle: θ = 2 * pi * duration * beta_hz
        theta = 2.0 * np.pi * duration * beta_hz
        # Use Qiskit's RZZGate's definition
        super().__init__(name=f"ZZxtalk({theta:.2e})", num_qubits=2, params=[theta], label=label or f"ZZxtalk({beta_hz:.1f}Hz,{duration*1e9:.0f}ns)")
        self.beta_hz = beta_hz
        self.duration = duration

    def _define(self):
        """Define the gate in terms of RZZ."""
        qc = QuantumCircuit(2, name=self.name)
        theta = self.params[0]
        qc.rzz(theta, 0, 1)
        self.definition = qc

    def inverse(self):
        # RZZ inverse is RZZ(-theta)
        return CoherentCrosstalkGate(beta_hz=-self.beta_hz, duration=self.duration, label=f"{self.label}_dg" if self.label else None)

    @property
    def matrix(self):
        """Return the matrix representation."""
        theta = self.params[0]
        # Matrix for exp(-i * theta/2 * Z⊗Z)
        # cos = np.cos(theta / 2.0)
        # sin = np.sin(theta / 2.0)
        # return np.array([
        #     [cos - 1j * sin, 0, 0, 0],
        #     [0, cos + 1j * sin, 0, 0],
        #     [0, 0, cos + 1j * sin, 0],
        #     [0, 0, 0, cos - 1j * sin]
        # ], dtype=complex)
        # Or simply use Qiskit's gate
        return RZZGate(theta).to_matrix()
