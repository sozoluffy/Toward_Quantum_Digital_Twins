# noise/coherent_crosstalk.py
import numpy as np
from typing import List, Tuple, Optional
# Imports from qiskit (terra) are fine
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library import RZZGate

class CoherentCrosstalkGate(Gate):
    """
    Implements the coherent ZZ crosstalk interaction exp(-i*beta*duration*Z⊗Z)
    as a Qiskit Gate using RZZGate. beta is coupling strength (Eq. 10).
    """
    def __init__(self, beta_hz: float, duration: float, label: Optional[str] = None):
        """
        Args:
            beta_hz: The ZZ coupling strength parameter β in Hz (from Eq. 10).
                     Assumes this is the effective coupling J fitted or calculated.
            duration: The duration 'd_uv' over which the interaction occurs (e.g., layer duration).
            label: An optional label for the gate.
        """
        # Calculate RZZ angle: θ = 2 * pi * duration * beta_hz
        # RZZGate(theta) implements exp(-i * theta/2 * Z⊗Z)
        # We want exp(-i * duration * beta_rad * Z⊗Z) = exp(-i * duration * (2*pi*beta_hz) * Z⊗Z)
        # So, theta/2 = duration * 2*pi*beta_hz  => theta = 4 * pi * duration * beta_hz
        theta = 4.0 * np.pi * duration * beta_hz # Adjusted calculation for RZZGate definition
        # Use Qiskit's RZZGate's definition
        super().__init__(name=f"ZZxtalk({theta:.2e})", num_qubits=2, params=[theta], label=label or f"ZZxtalk({beta_hz:.1f}Hz,{duration*1e9:.0f}ns)")
        self.beta_hz = beta_hz
        self.duration = duration
        self._theta = theta # Store theta for matrix property

    def _define(self):
        """Define the gate in terms of RZZ."""
        # Ensure the definition uses the stored theta
        qc = QuantumCircuit(2, name=self.name)
        qc.rzz(self._theta, 0, 1)
        self.definition = qc

    def inverse(self):
        # RZZ inverse is RZZ(-theta)
        # Need to recalculate beta for the inverse or just pass negative theta
        # Passing negative beta works if theta calculation is consistent
        return CoherentCrosstalkGate(beta_hz=-self.beta_hz, duration=self.duration, label=f"{self.label}_dg" if self.label else None)

    @property
    def matrix(self):
        """Return the matrix representation using Qiskit's RZZGate."""
        # Use the stored theta
        return RZZGate(self._theta).to_matrix()