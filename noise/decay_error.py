try:
    from qiskit_aer.noise import thermal_relaxation_error, QuantumError
except ImportError:
    from qiskit.providers.aer.noise import thermal_relaxation_error, QuantumError
class DecayError:
    """Idle relaxation/dephasing (Eqs.3–4)"""
    def __init__(self,T1:float,T2:float,dt:float=1e-9):
        if T1<=0 or T2<=0 or dt<=0: raise ValueError("T1,T2,dt>0")
        self.T1,self.T2,self.dt=T1,T2,dt
    def channel(self,qubit_idx:int)->QuantumError:
        return thermal_relaxation_error(self.T1,self.T2,self.dt)
