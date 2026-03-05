import numpy as np
from sklearn.utils import Bunch

def solve_ivp(f, tspan, x0, events, args, dense_output=True, max_step=0.001):
    # local variables
    t = tspan[0]
    idx = 0
    x = x0
    step = max_step
    ttape = np.zeros((int(np.ceil((tspan[1]-tspan[0])/step)), ))
    xtape = np.zeros((int(np.ceil((tspan[1]-tspan[0])/step)), len(x0)))
    event_sign = events(t, x) > 0

    end_cond = lambda t, x: events.terminal and (not event_sign and events(t, x) > 0)

    while(idx < len(ttape) and not end_cond(t, x)):
        
        ttape[idx] = t
        xtape[idx, :] = x
        k1 = f(t, x, *args)
        k2 = f(t + step/2, x + step/2*k1, *args)
        k3 = f(t + step/2, x + step/2*k2, *args)
        k4 = f(t + step, x + step*k3, *args)
        x = x + step/6*(k1 + 2*k2 + 2*k3 + k4)
        t += step
        idx += 1

    return Bunch(t=ttape[0:idx], y=xtape[0:idx, :].T)