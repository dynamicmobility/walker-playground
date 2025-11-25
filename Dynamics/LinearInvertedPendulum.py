from SingleDomainSystem import SingleDomainSystem
import numpy as np
from numpy.typing import ArrayLike

g = 9.81
class HybridLinearInvertedPendulum(SingleDomainSystem):
    '''
    Model of a linear inverted pendulum with hybrid behavior
    Args
    ---
        z0 (float): Nominal Height of pendulum
        step_nom (float): 
    '''
    def __init__(self, z0: float, step_nom: float):
        super().__init__()
        self.z0 = z0
        self.step_nom = step_nom
    
    def f(self, t: float, x: np.ndarray):
        p, v = x.ravel()
        return np.array([v, p*g/self.z0])

    def reset(self, t:float, x: np.ndarray):
        p, v = x.ravel()
        return np.array([-p, v])

    def guard(self, t: float, x: np.ndarray):
        p, v = x.ravel()
        return p - self.step_nom/2
