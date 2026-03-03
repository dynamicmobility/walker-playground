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
        x = x + step*f(t, x, *args)
        t += step
        idx += 1

    return Bunch(t=ttape[0:idx], y=xtape[0:idx, :].T)