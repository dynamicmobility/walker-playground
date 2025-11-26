from .SingleDomainSystem import SingleDomainSystem
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.collections import LineCollection

# Dynamics taken from Russ Tedrake - Underactuated Robotics https://underactuated.mit.edu/simple_legs.html 
g = 9.81
class SpokedWheel(SingleDomainSystem):
    '''
    Model of a nonlinear spoked wheel (inverted pendulum) with hybrid behavior
    Args
    ---
        alpha (float): angle between spokes
        gamma (float): angle of ground
    '''
    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def f(self, t: float, x: np.ndarray):
        th, thd = x.ravel()
        return np.array([thd, np.sin(th)])

    def reset(self, t:float, x: np.ndarray):
        th, thd = x.ravel()
        return np.array([2*self.gamma - th, np.cos(2*self.alpha)*thd])

    def guard(self, t: float, x: np.ndarray):
        th, thd = x.ravel()
        return th - self.alpha - self.gamma

    # TODO: Finish 
    def draw_system(self, t, x, *args, **kwargs):
        th, thd = x.ravel()
        line0 = [(0., 0.), (np.sin(th), np.cos(th))]
        return LineCollection([line0], kwargs)
