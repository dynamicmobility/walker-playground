import numpy as np
from Dynamics import HybridLinearInvertedPendulum, CompassGaitWalker, SpokedWheel
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

type = "Compass"
tspan = [0, 1]
nsteps = 1

if type == "Compass":
    sys = CompassGaitWalker(5., 10., 0.5, 0.5, -0.05625)
    x0 = np.array([0, 0, 0.4, -2.0])
    ctrl_func = lambda t, x: 0
    args = (ctrl_func, )

# TODO: Test
elif type == "SpokedWheel":
    sys = SpokedWheel(np.pi/6)
    x0 = np.array([0, 0.1])
    args = None

elif type == "HLIP":
    sys = HybridLinearInvertedPendulum(1, 0.4)
    x0 = np.array([0, 0.1])
    args = None

event = lambda t, x, *args: sys.guard(t, x)
event.terminal = True
event.direction = 1

traj = solve_ivp(sys.f, tspan, x0, events=event, args=args)

ts = traj.t
ys = traj.y

fig, ax = plt.subplots()
for i in range(len(ts)):
    ax.add_collection(sys.draw_system(ts[i], ys[:, i]))


plt.show()