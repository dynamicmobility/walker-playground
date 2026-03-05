import numpy as np
from Dynamics import HybridLinearInvertedPendulum, CompassGaitWalker, SpokedWheel
# from scipy.integrate import solve_ivp
from Integrator.euler_integrate import solve_ivp
import matplotlib.pyplot as plt

plot = True

type = "Compass"
tspan = [0, 1]
nsteps = 30

if type == "Compass":
    sys = CompassGaitWalker(m=5., mh=10., a=0.5, b=0.5, gamma=0.0525)
    # x0 = np.array([0, 0, -2.0, 0.4])
    x0 = np.array([ 0.3159661 , -0.22096609,  0.3819361 ,  1.0887045 ])
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
global_offset = np.array([0.0, 0.0])
guard_offset_vec = np.array([1e-5, 0, 0, 0])

if(plot):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    if(type == "Compass"):
        fig3, ax3 = plt.subplots()


for i in range(nsteps):
    traj = solve_ivp(sys.f, tspan, x0, events=event, args=args, dense_output=True, max_step=0.001)
    ts = traj.t
    ys = traj.y
    xf = ys[:, -1]
    tspan = [ts[-1], ts[-1]+10]
    print("Step dur: ", ts[-1] - ts[0])
    print("Final state: ", repr(xf))
    global_offset += sys.sw_foot_pos(xf)
    x0 = sys.reset(ts[-1], xf) + guard_offset_vec
    print("New initial state: ", repr(x0))
    if(plot):
        for j in range(0, len(ts), 20):
            ax1.add_collection(sys.draw_system(ts[j], ys[:, j], offset=global_offset))
        ax2.plot(ts, [sys.guard(ts[i], ys[:, i]) for i in range(len(ts))])
        if(type == "Compass"):
            color_idx = float(i) / nsteps
            colors = plt.cm.viridis(color_idx)
            ax3.plot(ys[0, :], ys[2, :], color=colors, label="Stance Leg" if i == 0 else "")
            ax3.plot(ys[1, :], ys[3, :], color=colors, label="Swing Leg" if i == 0 else "")
            if i == 0:
                ax3.legend()
    # print(global_offset)

if(plot):
    ax1.set_xlim(left=-1, right=global_offset[0]+0.5)
    ax1.set_ylim(bottom=global_offset[1]-0.5, top=1.25)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(ts, ys[0, :])

    plt.show()