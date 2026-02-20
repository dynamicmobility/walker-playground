import numpy as np
from Dynamics import HybridLinearInvertedPendulum, CompassGaitWalker, SpokedWheel
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, Euler, ConstantStepSize, Event
import optimistix as optx

type = "Compass"
tspan = [0, 1]
nsteps = 1

sys = CompassGaitWalker(m=5., mh=10., a=0.5, b=0.5, gamma=0.0525)
x0 = np.array([0, 0, -2.0, 0.4])
# x0 = np.array([0.32596612, -0.22096612, 0.38193607, 1.0887043])
# x0 = np.array([ 0.3259661 , -0.22096609,  0.3819361 ,  1.0887045 ])
x0 = np.array([0.3238649,  -0.21885487,  0.37617958,  1.0930421])
ctrl_func = lambda t, x: 0.0
args = (ctrl_func, )



# Simulate Compass Gait for one step
term = ODETerm(sys.f)
solver=Tsit5()

cond_fn = sys.guard
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event = Event(cond_fn, root_finder, direction=True)

traj = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.01, y0=x0, args=ctrl_func, event=event)

ts = traj.ts
ys = traj.ys

print("tf: ", ts)
print("xf: ", repr(ys))
print("x0: ", repr(sys.reset(ts, ys)))
print("x0_nom: ", repr(x0))

# fig1, ax1 = plt.subplots()
# for i in range(len(ts)):
#     ax1.add_collection(sys.draw_system(ts[i], ys[:, i]))

# print(sys.reset(ts[-1], ys[:, -1]))
# fig2, ax2 = plt.subplots()

# ax2.plot(ys[0, :], ys[2, :])


# plt.show()