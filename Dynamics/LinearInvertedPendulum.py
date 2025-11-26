from .SingleDomainSystem import SingleDomainSystem
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.collections import LineCollection

# Dynamics taken from Xiong, Ames - 3-D Underactuated Bipedal Walking via H-LIP Based Gait Synthesis and Stepping Stabilization
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
    
    def f(self, t: float, x: np.ndarray, *args):
        p, v = x.ravel()
        return np.array([v, p*g/self.z0])

    def reset(self, t:float, x: np.ndarray):
        p, v = x.ravel()
        return np.array([-p, v])

    def guard(self, t: float, x: np.ndarray):
        p, v = x.ravel()
        return p - self.step_nom/2
    
    def draw_system(self, t, x, *args, **kwargs):
        p, v = x.ravel()
        line0 = [(0., 0.), (p, self.z0)]
        return LineCollection([line0])
