import numpy as np
from scipy.interpolate import make_interp_spline
import casadi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

show_animation = True
NX = 6  # x, y, phi, vx, vy, omega
NU = 2  # alpha, moment

# vehicle parameters
M = 1725                # mass
J = 2780.1358078125004  # moment of inertia
Lf = 1.3                # distance from the center of mass to the front axle
Lr = 1.5                # distance from the center of mass to the rear axle
D = .752                # half distance between left and right wheels
Rw = .331               # wheel radius
# Jw = 4                # wheel moment of inertia
# Cx = 10**2            # longitudinal tire stiffness factor
Cy = 10 ** 4            # lateral tire stiffness factor
Ca = 2                  # air (and other types) resistance coefficient

# MPC parameters
T = 10                            # horizon length
Q = np.diag([1, 1, 0, 0, 0, 0])   # state cost matrix
Qf = np.diag([1, 1, 0, 0, 0, 0])  # state final matrix
R = np.diag([0, 0])               # input cost matrix

Dt = .05                          # integration step
V_ref = 2                         # reference velocity [m/s]
Ds = V_ref * Dt                   # distance between points

# Restrictions on control parameters
Alpha_lb = -np.pi / 3
Alpha_ub = np.pi / 3
M_lb = -10**5
M_ub = 4 * 10**3

GOAL_DIS = 1                      # goal distance
MAX_TIME = 60                     # max simulation time
N_IND_SEARCH = 10                 # search index number

def alpha_l(alpha):
    return np.arctan((Lr + Lf) * np.tan(alpha) / (Lr + Lf - D * np.tan(alpha)))

def alpha_r(alpha):
    return np.arctan((Lr + Lf) * np.tan(alpha) / (Lr + Lf + D * np.tan(alpha)))

def calc_spline_course(x_points, y_points, spline_degree=3, ds=Ds):
    x = np.arange(0, max(x_points), ds)
    spline = make_interp_spline(x_points, y_points, k=spline_degree)
    y = spline(x)

    # make the points along the trajectory equidistant
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.hypot(xd, yd)   # distances from point yo point
    u = np.cumsum(dist)       # path length (sum of all distances between points)
    u = np.hstack([0, u])
    s = np.arange(0, u[-1], ds)   # parameter along the path

    cx = np.interp(s, u, x)
    cy = np.interp(s, u, y)

    cphi = np.arctan2(np.diff(cy), np.diff(cx))
    cphi = np.append(cphi, cphi[-1])
    return cx, cy, cphi


def plot_arrow(x, y, phi, length=Lf + Lr, width=2 * D, fc="r", ec="k"):
    plt.arrow(x, y, length * np.cos(phi), length * np.sin(phi), fc=fc, ec=ec,
              length_includes_head=True, head_width=width, head_length=length)


def calc_nearest_index(state, cx, cy, pind):
    dx = [state[0] - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state[1] - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d)) + pind
    return ind


def calc_ref_trajectory(state, cx, cy, pind):
    xy_ref = np.zeros((6, T + 1))
    ncourse = len(cx)
    ind = calc_nearest_index(state, cx, cy, pind)
    if pind >= ind:
        ind = pind

    for i in range(T + 1):
        if (ind + i) < ncourse:
            xy_ref[0, i] = cx[ind + i]
            xy_ref[1, i] = cy[ind + i]
        else:
            xy_ref[0, i] = cx[ncourse - 1]
            xy_ref[1, i] = cy[ncourse - 1]
    return xy_ref, ind


def check_goal(state, goal):
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    d = np.hypot(dx, dy)
    if d <= GOAL_DIS:
        return True
    else:
        return False


def update_state(state, alpha, m):
    x, y, phi, vx, vy, omega = state[0], state[1], state[2], state[3], state[4], state[5]

    alphal = alpha_l(alpha)
    alphar = alpha_r(alpha)

    v_xfl = vx - omega * D
    v_xfr = vx + omega * D
    v_yfl = vy + omega * Lf
    v_yfr = v_yfl

    v_yrl = vy - omega * Lr
    v_yrr = v_yrl

    v_wyfl = -v_xfl * np.sin(alphal) + v_yfl * np.cos(alphal)
    v_wyfr = -v_xfr * np.sin(alphar) + v_yfr * np.cos(alphar)

    f_wxfl = m / Rw
    f_wxfr = m / Rw

    f_wyfl = -Cy * v_wyfl
    f_wyfr = -Cy * v_wyfr
    f_yrl = -Cy * v_yrl
    f_yrr = -Cy * v_yrr

    state[0] = x + (vx * np.cos(phi) - vy * np.sin(phi)) * Dt
    state[1] = y + (vx * np.sin(phi) + vy * np.cos(phi)) * Dt
    state[2] = phi + omega * Dt
    state[3] = vx + (vy * omega + (f_wxfl * np.cos(alphal) - f_wyfl * np.sin(alphal) +
                                   f_wxfr * np.cos(alphar) - f_wyfr * np.sin(alphar) -
                                   Ca * vx ** 2) / M) * Dt
    state[4] = vy + (-vx * omega + (f_wxfl * np.sin(alphal) + f_wyfl * np.cos(alphal) +
                                    f_wxfr * np.sin(alphar) + f_wyfr * np.cos(alphar) +
                                    f_yrl + f_yrr) / M) * Dt
    state[5] = omega + ((Lf * (f_wxfl * np.sin(alphal) + f_wyfl * np.cos(alphal) +
                               f_wxfr * np.sin(alphar) + f_wyfr * np.cos(alphar)) +
                         D * (f_wxfr * np.cos(alphar) - f_wyfr * np.sin(alphar) -
                              f_wxfl * np.cos(alphal) + f_wyfl * np.sin(alphal)) -
                         Lr * (f_yrl + f_yrr)) / J) * Dt

    return state


def mpc_control(xy_ref, state, alphahor, mhor):
    opti = casadi.Opti()
    x = opti.variable(NX, T + 1)
    u = opti.variable(NU, T)

    #cost = casadi.bilin(Qf, x[:, T] - xy_ref[:, T], x[:, T] - xy_ref[:, T])
    cost = (x[0, T]-xy_ref[0, T])**2 + (x[1, T]-xy_ref[1, T])**2
    opti.subject_to(x[:, 0] == state)

    for k in range(T):
        if k != 0:
            cost += (x[0, k]-xy_ref[0, k])**2 + (x[1, k]-xy_ref[1, k])**2
        #cost += casadi.bilin(R, u[:, k], u[:, k])
        opti.subject_to(x[:, k + 1] == update_state(x[:, k], u[0, k], u[1, k]))

    opti.subject_to(x[:, 0] == state)
    opti.subject_to(opti.bounded(Alpha_lb, u[0, :], Alpha_ub))
    opti.subject_to(opti.bounded(M_lb, u[1, :], M_ub))

    opti.minimize(cost)
    p_opts = {'print_time': False}
    s_opts = {'max_iter': 1500, 'print_level': 0}
    opti.solver('ipopt', p_opts, s_opts)
    opti.set_initial(u[0, :], np.append(alphahor[1:], alphahor[-1]))
    opti.set_initial(u[1, :], np.append(mhor[1:], mhor[-1]))

    sol = opti.solve()
    alphahor, mhor = sol.value(u)

    return alphahor, mhor


if __name__ == '__main__':

    x_points = [0, 10, 20, 30, 40]
    y_points = [0, 10, 20, 10, 0]

    cx, cy, cphi = calc_spline_course(x_points, y_points, spline_degree=3, ds=Ds)  # course x, y, phi
    goal = [cx[-1], cy[-1]]  # goal point
    # results arrays
    x, y, phi, vx, vy, omega = [cx[0]], [cy[0]], [cphi[0]], [0], [0], [0]
    alpha, m, t, time = [0], [0], [0], 0
    alphahor, mhor = [0] * T, [0] * T
    # current state
    state = [cx[0], cy[0], cphi[0], 0, 0, 0]

    target_ind = calc_nearest_index(state, cx, cy, 0)
    t_ind = [target_ind]

    while MAX_TIME >= time:
        xy_ref, target_ind = calc_ref_trajectory(state, cx, cy, target_ind)
        alphahor, mhor = mpc_control(xy_ref, state, alphahor, mhor)

        state = update_state(state, alphahor[0], mhor[0])
        time = time + Dt

        x.append(state[0])
        y.append(state[1])
        phi.append(state[2])
        vx.append(state[3])
        vy.append(state[4])
        omega.append(state[5])
        alpha.append(alphahor[0])
        m.append(mhor[0])
        t_ind.append(target_ind)
        t.append(time)

        print('state:', state)
        print('time:', time)

        if check_goal(state, goal):
            print("Goal")
            break

    t_ind.append(target_ind)
    figure, ax = plt.subplots()


    def animation_function(i):
        plt.cla()
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(x[:i + 1], y[:i + 1], "-b", label="trajectory")
        for k in range(T):
            plt.plot(cx[t_ind[i + 1] + k], cy[t_ind[i + 1] + k], "og", label="target")
        plot_arrow(x[i], y[i], phi[i])
        plt.axis("equal")
        plt.grid(True)
        plt.title("speed: " + str(round(np.hypot(vx[i], vy[i]), 2)) + "; time: " + str(t[i]))


    animation = FuncAnimation(figure, func=animation_function, frames=len(t), interval=100)
    # animation.save('nmpc.gif', writer='imagemagick', fps=30)
    plt.show()

    plt.subplots()
    plt.plot(cx, cy, "-r", label="reference")
    plt.plot(x, y, "-b", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax1.plot(t, vx, "-r", label="speed")
    ax1.set(ylabel=r'$v_x$', ylim=[min(vx), max(vx)])
    ax1.grid(True)
    ax2.plot(t, alpha, label=r'$\alpha$')
    ax2.set(ylabel=r'$\alpha$', ylim=[Alpha_lb, Alpha_ub])
    ax2.grid(True)
    ax3.plot(t, m, label='m')
    ax3.set(ylabel=r'$M$', xlabel='time', xlim=[t[0], t[-1]], ylim=[min(m), max(m)])
    ax3.grid(True)
    fig.align_ylabels((ax1, ax2, ax3))
    plt.show()
