import numpy as np
from Dynamics import HybridLinearInvertedPendulum, CompassGaitWalker, SpokedWheel
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

type = "Compass"
tspan = [0, 1]
nsteps = 1

if type == "Compass":
    sys = CompassGaitWalker(m=5., mh=10., a=0.5, b=0.5, gamma=0.0525)
    # x0 = np.array([0, 0, -2.0, 0.4])
    x0 = np.array([ 0.3259661 , -0.22096609,  0.3819361 ,  1.0887045 ])
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

traj = solve_ivp(sys.f, tspan, x0, events=event, args=args, dense_output=True, max_step=0.01)

ts = traj.t
ys = traj.y

fig1, ax1 = plt.subplots()
for i in range(len(ts)):
    ax1.add_collection(sys.draw_system(ts[i], ys[:, i]))

print("tf: ", ts[-1])
print("xf: ", ys[:, -1])
print("x0: ", sys.reset(ts[-1], ys[:, -1]))
print("x0_nom: ", x0)
fig2, ax2 = plt.subplots()

ax2.plot(ys[0, :], ys[2, :])
ax2.plot(ys[1, :], ys[3, :])
ax2.set_xlabel("theta")
ax2.set_ylabel("thetadot")
ax2.legend(["Stance Leg", "Swing Leg"])
ax2.set_title("Phase plot")


fig3, ax3 = plt.subplots()
ax3.plot(ts, [sys.guard(ts[i], ys[:, i]) for i in range(len(ts))])

fig4, ax4 = plt.subplots()
ax4.plot(ts, ys[0, :])
ax4.plot(ts, ys[2, :])
ax4.legend(["Stance Leg", "Swing Leg"])

plt.show()