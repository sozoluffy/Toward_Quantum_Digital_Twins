# noise/measurement_error.py
from typing import Tuple, List
# Use direct import from qiskit_aer.noise
from qiskit_aer.noise import ReadoutError
import numpy as np # Import numpy for checks

class MeasurementError:
    """
    Readout confusion matrix wrapper. C=[[P(0|0), P(0|1)],[P(1|0), P(1|1)]] (Eq.5)
    Stores the qubit index it applies to.
    """
    def __init__(self, confusion_matrix_diag: Tuple[float, float], qubit_idx: int):
        """
        Args:
            confusion_matrix_diag: A tuple [P(0|0), P(1|1)].
            qubit_idx: The index of the qubit this error applies to.
        """
        if len(confusion_matrix_diag) != 2:
            raise ValueError("Provide [P(0|0), P(1|1)]")
        # Validate input probabilities are within [0, 1]
        if not (0 <= confusion_matrix_diag[0] <= 1 and 0 <= confusion_matrix_diag[1] <= 1):
             raise ValueError("Probabilities must be between 0 and 1")

        self.p00, self.p11 = confusion_matrix_diag
        self.qubit_idx = qubit_idx
        self._channel = self._create_channel()

    def _create_channel(self) -> ReadoutError:
        """Creates the Qiskit ReadoutError object."""
        # The input is [P(0|0), P(1|1)]
        p00 = self.p00 # P(measure 0 | state 0)
        p11 = self.p11 # P(measure 1 | state 1)

        # Calculate the error probabilities based on row sums = 1
        p10 = 1.0 - p00 # P(measure 1 | state 0)
        p01 = 1.0 - p11 # P(measure 0 | state 1)

        # Construct the probability matrix for ReadoutError(probabilities)
        # probabilities[i][j] is the probability of measuring outcome `j` given the true state was `i`.
        # Row 0 (true state 0): [P(measure 0 | state 0), P(measure 1 | state 0)] = [p00, p10]
        # Row 1 (true state 1): [P(measure 0 | state 1), P(measure 1 | state 1)] = [p01, p11]
        probs = [[p00, p10], [p01, p11]]

        # Check row sums for debugging (should be close to 1)
        # print(f"DEBUG: Readout matrix row 0 sum: {probs[0][0] + probs[0][1]}") # Should be ~1
        # print(f"DEBUG: Readout matrix row 1 sum: {probs[1][0] + probs[1][1]}") # Should be ~1

        return ReadoutError(probs)

    @property
    def channel(self) -> ReadoutError:
        """Return the Qiskit ReadoutError object."""
        return self._channel

    @property
    def index(self) -> int:
        """Return the qubit index this error applies to."""
        return self.qubit_idx

