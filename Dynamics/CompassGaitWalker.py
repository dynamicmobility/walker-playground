from .SingleDomainSystem import SingleDomainSystem
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.collections import LineCollection

# Dynamics taken from Hybrid Limit Cycles of Walking Robots 
# Underactuated Robotics (https://underactuated.mit.edu/simple_legs.html) Also uses these dynamics but the coordinates are flipped
g = 9.81
class CompassGaitWalker(SingleDomainSystem):
    '''
    Model of a compass gait walker 
    Args
    ---
        mh (float): mass at hip
        m (float): mass at each leg
        a (float): distance from foot to COM of leg
        b (float): distance from COM of leg to hip
        gamma (float): angle of ground
    '''
    def __init__(self, mh: float, m: float, a: float, b: float, gamma: float):
        super().__init__()
        self.mh = mh
        self.m = m
        self.a = a
        self.b = b
        self.l = a+b
        self.gamma = gamma
    
    def f(self, t: float, x: np.ndarray, ctrl_func):
        tsw, tst, tswd, tstd = x.ravel()
        u = ctrl_func(t, x)

        M = np.array([[self.m*self.b**2, -self.m*self.l*self.b*np.cos(tst - tsw)],
                      [-self.m*self.l*self.b*np.cos(tst - tsw), (self.mh + self.m)*self.l**2 + self.m*self.a**2]])

        C = np.array([[0, self.m*self.l*self.b*np.sin(tst - tsw)*tstd],
                      [-self.m*self.l*self.b*np.sin(tst - tsw)*tswd, 0]])

        G = np.array([[self.m*self.b*g*np.sin(tsw)],
                      [-(self.mh*self.l + self.m*self.a + self.m*self.l)*g*np.sin(tst)]])

        B = np.array([[1],
                      [-1]])
        qd = np.array([[tswd], [tstd]])
        qdd = np.linalg.inv(M)@(B*u - C@qd - G)

        return np.hstack((qd.flatten(), qdd.flatten()))

    def reset(self, t:float, x: np.ndarray):
        tsw, tst, tswd, tstd = x.ravel()
        alpha = (tsw-tst)*0.5
        Qm = np.array([[-self.m*self.a*self.b, -self.m*self.a*self.b + (self.mh*self.l**2 + 2*self.m*self.a*self.l)*np.cos(2*alpha)],
                       [0, -self.m*self.a*self.b]])
        Qp = np.array([[self.m*self.b*(self.b-self.l*np.cos(2*alpha)), self.m*self.l*(self.l-self.b*np.cos(2*alpha)) + self.m*self.a**2 + self.mh*self.l**2],
                       [self.m*self.b**2, -self.m*self.b*self.l*np.cos(2*alpha)]])
        qdm = np.array([[tswd], [tstd]])
        qdp = np.linalg.inv(Qp)@Qm@qdm # Relabel velocities
        qp = np.array([tst + 0.001, tsw])
        return np.hstack((qp, qdp.flatten()))

    def guard(self, t: float, x: np.ndarray):
        tsw, tst, tswd, tstd = x.ravel()
        return (tsw + tst - 2*self.gamma)
    
    def draw_system(self, t, x, offset=np.zeros((2,)), *args, **kwargs):
        tsw, tst, tswd, tstd = x.ravel()
        ph = np.array([self.l*np.sin(tst), self.l*np.cos(tst)]) + offset
        pf = np.array([self.l*np.sin(tst) - self.l*(np.sin(tsw)), self.l*np.cos(tst)- self.l*np.cos(tsw)]) + offset
        line0 = [offset, ph] # Stance Leg 
        line1 = [ph, pf] # Swing Leg
        line2 = [offset - np.array([np.cos(-self.gamma), np.sin(-self.gamma)]), offset + np.array([np.cos(-self.gamma), np.sin(-self.gamma)])] # Floor
        return LineCollection([line0, line1, line2], colors=['r', 'g', 'k'])
    
    def sw_foot_pos(self, x):
        tsw, tst, tswd, tstd = x.ravel()
        return np.array([self.l*np.sin(tst) - self.l*(np.sin(tsw)), self.l*np.cos(tst)- self.l*np.cos(tsw)]) # Needed for finding the global offset that advances the robot every step


