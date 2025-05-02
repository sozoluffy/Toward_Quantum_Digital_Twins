# calibration/calibration.py
import json
import yaml
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

class CalibrationData:
    """
    Loads, normalizes, validates, and provides access to device calibration parameters.
    Includes a normalization step to attempt conversion of numeric strings.
    """
    def __init__(self, path: str):
        """Loads, normalizes, and validates the calibration data."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found: {path}")
        print(f"Loading calibration data from: {os.path.abspath(path)}")
        self.data: Dict[str, Any] = self._load_file(path)
        # ---> ADDED NORMALIZATION STEP <---
        self._normalize(self.data)
        # ---------------------------------
        self._validate(self.data)
        print("Calibration data loaded and validated successfully.")

    def _load_file(self, path: str) -> Dict[str, Any]:
        """Loads data from JSON or YAML file."""
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, 'r') as f: # Explicitly open as UTF-8 might help
            # with open(path, 'r', encoding='utf-8') as f: # Try this if encoding issues suspected
                if ext == '.json':
                    print("Parsing as JSON...")
                    return json.load(f)
                if ext in ('.yaml', '.yml'):
                    print("Parsing as YAML...")
                    return yaml.safe_load(f)
                raise ValueError(f"Unsupported format: {ext}")
        except Exception as e:
            raise ValueError(f"Error reading or parsing {path}: {e}") from e

    def _normalize(self, raw: Dict[str, Any]):
        """Attempts to convert string representations of numbers to floats/ints."""
        print("Normalizing loaded data (attempting string->number conversion)...")
        if isinstance(raw.get('qubits'), dict):
            for q, params in raw['qubits'].items():
                if isinstance(params, dict):
                    for key in ('p1', 'T1', 'T2'):
                        val = params.get(key)
                        if isinstance(val, str):
                            try:
                                # Try converting string to float
                                params[key] = float(val)
                                print(f"  Normalized qubits.{q}.{key}: string '{val}' -> float {params[key]}")
                            except ValueError:
                                print(f"  Warning: Could not convert qubits.{q}.{key} string '{val}' to float.")
                                # Keep original string if conversion fails, validation will catch it
                                pass

        if isinstance(raw.get('gates'), dict):
            for g, gd in raw['gates'].items():
                if isinstance(gd, dict):
                    for key in ('fidelity', 'duration'):
                         val = gd.get(key)
                         if isinstance(val, str):
                             try:
                                 gd[key] = float(val)
                                 print(f"  Normalized gates.{g}.{key}: string '{val}' -> float {gd[key]}")
                             except ValueError:
                                 print(f"  Warning: Could not convert gates.{g}.{key} string '{val}' to float.")
                                 pass

        if isinstance(raw.get('readout'), dict):
             for q, rd in raw['readout'].items():
                  if isinstance(rd, dict):
                       diag = rd.get('confusion_matrix_diag')
                       if isinstance(diag, list):
                            new_diag = []
                            changed = False
                            for i, x in enumerate(diag):
                                if isinstance(x, str):
                                     try:
                                          new_diag.append(float(x))
                                          print(f"  Normalized readout.{q}.diag[{i}]: string '{x}' -> float {new_diag[-1]}")
                                          changed = True
                                     except ValueError:
                                          print(f"  Warning: Could not convert readout.{q}.diag[{i}] string '{x}' to float.")
                                          new_diag.append(x) # Keep original
                                else:
                                     new_diag.append(x)
                            if changed:
                                 rd['confusion_matrix_diag'] = new_diag

        cs_key = 'crosstalk_strength_hz'
        cs_val = raw.get(cs_key)
        if isinstance(cs_val, str):
             try:
                  raw[cs_key] = float(cs_val)
                  print(f"  Normalized {cs_key}: string '{cs_val}' -> float {raw[cs_key]}")
             except ValueError:
                  print(f"  Warning: Could not convert {cs_key} string '{cs_val}' to float.")
                  pass
        print("Normalization finished.")


    def _validate(self, raw: Dict[str, Any]):
        """Validates the structure and values of the (normalized) calibration data."""
        print("Starting validation...")
        # Qubits validation
        if 'qubits' not in raw or not isinstance(raw.get('qubits'), dict):
            print("Warning: 'qubits' section missing or not a dictionary.")
        else:
            for q, params in raw.get('qubits', {}).items():
                q_label = f"qubit '{q}'"
                if not isinstance(params, dict): raise ValueError(f"Invalid format for {q_label}: Expected dict.")

                # Validate p1
                p1 = params.get('p1')
                # REMOVED p1 DEBUG PRINT - can be added back if needed
                if p1 is None: raise ValueError(f"'p1' missing for {q_label}")
                # Use stricter type check AFTER normalization
                if not isinstance(p1,(float,int)): # Check if it's now a standard number
                    # If still not float/int, raise TypeError including np.number for clarity
                    if not isinstance(p1, np.number):
                         raise TypeError(f"Invalid type for 'p1' in {q_label}: Expected number, got {type(p1)}.")
                if not (0 <= p1 <= 1): raise ValueError(f"Invalid value for 'p1' in {q_label}: {p1}. Must be in [0, 1].")

                # Validate T1, T2
                for t in ('T1', 'T2'):
                    v = params.get(t)
                    if v is None: raise ValueError(f"'{t}' missing for {q_label}")
                    # Check type AFTER normalization
                    if not isinstance(v,(float,int)):
                         if not isinstance(v, np.number):
                              raise TypeError(f"Invalid type for '{t}' in {q_label}: Expected number, got {type(v)}.")
                    if v <= 0: raise ValueError(f"Invalid value for '{t}' in {q_label}: {v}. Must be > 0.")
            print("Qubit validation passed.")

        # Gates validation
        if 'gates' not in raw or not isinstance(raw.get('gates'), dict):
            print("Warning: 'gates' section missing or not a dictionary.")
        else:
            for g, gd in raw.get('gates', {}).items():
                g_label = f"gate '{g}'"
                if not isinstance(gd, dict): raise ValueError(f"Invalid format for {g_label}")
                fid = gd.get('fidelity')
                dur = gd.get('duration')
                if fid is not None:
                     if not isinstance(fid,(float,int, np.number)): raise TypeError(f"Invalid type for fidelity in {g_label}: Expected number, got {type(fid)}")
                     if not (0<=fid<=1): raise ValueError(f"Invalid fidelity for {g_label}: {fid}")
                if dur is not None:
                     if not isinstance(dur, (float, int, np.number)): raise TypeError(f"Invalid type for duration in {g_label}: Expected number, got {type(dur)}")
                     if dur < 0: raise ValueError(f"Invalid duration for {g_label}: {dur}")
            print("Gate validation passed.")

        # Readout validation
        if 'readout' not in raw or not isinstance(raw.get('readout'), dict):
            print("Warning: 'readout' section missing or not a dictionary.")
        else:
            for q, rd in raw.get('readout', {}).items():
                r_label = f"readout for qubit '{q}'"
                if not isinstance(rd, dict): raise ValueError(f"Invalid format for {r_label}")
                diag = rd.get('confusion_matrix_diag')
                if diag is None: raise ValueError(f"'confusion_matrix_diag' missing for {r_label}")
                if not isinstance(diag,list) or len(diag)!=2: raise ValueError(f"Invalid 'confusion_matrix_diag' format for {r_label}")
                for i, x in enumerate(diag):
                     if not isinstance(x,(float,int, np.number)): raise TypeError(f"Invalid type for confusion_matrix_diag[{i}] in {r_label}: Expected number, got {type(x)}")
                     if not (0<=x<=1): raise ValueError(f"Invalid value for confusion_matrix_diag[{i}] in {r_label}: {x}")
            print("Readout validation passed.")

        # Crosstalk validation
        cs_key = 'crosstalk_strength_hz'
        cs_val = raw.get(cs_key)
        if cs_val is not None and not isinstance(cs_val,(float,int, np.number)): raise TypeError(f"Invalid type for '{cs_key}': Expected number, got {type(cs_val)}")
        print("Crosstalk validation passed.")
        print("All validation checks complete.")


    # --- get, get_qubit_params, get_readout_params, etc. methods remain the same ---
    # --- Paste the rest of the CalibrationData methods here ---
    def get(self, *keys: str, default: Any=None) -> Any:
        node=self.data
        for k in keys:
            if isinstance(node, dict):
                node_get = node.get(k)
                if node_get is None and isinstance(k, int): node_get = node.get(str(k))
                if node_get is None and isinstance(k, str) and k.isdigit(): node_get = node.get(int(k))
                if node_get is None: return default
                node = node_get
            elif isinstance(node, list) and isinstance(k, int) and 0 <= k < len(node): node = node[k]
            else: return default
        return node

    def get_qubit_params(self, qubit_idx: int) -> Dict[str, float]:
        q_params = self.get('qubits', str(qubit_idx), default={})
        return {'T1': q_params.get('T1', np.inf), 'T2': q_params.get('T2', np.inf), 'p1': q_params.get('p1', 0.0)}

    def get_readout_params(self, qubit_idx: int) -> Optional[List[float]]:
        return self.get('readout', str(qubit_idx), 'confusion_matrix_diag', default=None)

    def get_gate_fidelity(self, gate_name: str, qubits: List[int]) -> Optional[float]:
        fidelity = None
        if qubits:
            specific_name = f"{gate_name}_{'_'.join(map(str, qubits))}"
            fidelity = self.get('gates', specific_name, 'fidelity')
        if fidelity is None: fidelity = self.get('gates', gate_name, 'fidelity')
        return fidelity

    def get_gate_duration(self, gate_name: str, qubits: Optional[List[int]]) -> Optional[float]:
        duration = None
        qubit_indices = list(qubits) if qubits else []
        if qubit_indices:
            specific_name = f"{gate_name}_{'_'.join(map(str, qubit_indices))}"
            duration = self.get('gates', specific_name, 'duration')
        if duration is None: duration = self.get('gates', gate_name, 'duration')
        return duration

    def get_crosstalk_strength(self) -> float:
        return self.get('crosstalk_strength_hz', default=0.0)

    def get_dt(self) -> Optional[float]:
         return self.get('dt', default=None)
