import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint


def make_chain(functions):
    # example
    # input [f=[f_1, f_2], g=[g_1, g_2]]
    # where f_i, g_i are returned values of funcs
    # this yields f_1, f_2, f_3, f_4
    for func in functions:
        for item in func:
            yield item


def join_functions(*functions):
    # joins outputs of functions together
    # used in animation.Funcanimation for the 'func'
    # and 'init_func' parameters to allow the drawing of
    # multiple pendulums
    return lambda *args: tuple(make_chain(f(*args) for f in functions))


class DoublePendulum:
    def __init__(self, l1, l2, m1, m2, g, dt, t_max, th1, th2, w1, w2, ax, p_col, t_col):
        self.l1 = l1                # length of first rod
        self.m1 = m1                # mass 1
        self.l2 = l2                # length of second rod
        self.m2 = m2                # mass 2
        self.g = g                  # acceleration due to gravity
        self.dt = dt                # time-step
        self.t_max = t_max          # time of animation
        self.th1 = th1              # angle of first rod (with vertical)
        self.th2 = th2              # angle of second rod (with vertical)
        self.w1 = w1                # angular velocity of mass 1
        self.w2 = w2                # angular velocity of mass 2
        self.ax = ax                # axis to plot on
        self.p_col = p_col          # colour of pendulum
        self.t_col = t_col          # colour of trace
        self.l = self.l1 + self.l2  # total length of pendulum

        # solve the system of ODEs
        self.state_init = np.array([th1, w1, th2, w2])
        self.time = np.arange(0, t_max, dt)
        self.sol = odeint(self.system, self.state_init, self.time)

        # determine (x, y) co-ordinates for each mass
        self.x1 = self.l1 * np.sin(self.sol[:, 0])
        self.y1 = -self.l1 * np.cos(self.sol[:, 0])
        self.x2 = self.x1 + self.l2 * np.sin(self.sol[:, 2])
        self.y2 = self.y1 - self.l2 * np.cos(self.sol[:, 2])

        # set up axes for plotting on
        self.ax.set_xlim(-self.l - 1, self.l + 1)
        self.ax.set_ylim(-self.l - 1, self.l + 1)
        self.line, = self.ax.plot([], [], self.p_col, lw=2)
        self.trace, = self.ax.plot([], [], self.t_col, lw=1)

    def system(self, y, t):
        # solves the system of ODEs describing double pendulum
        # uses odeint from SciPy.integrate
        delta_th = y[0] - y[2]
        const_1 = (self.l1 / self.l2) * np.cos(delta_th)  # g1,g2,f1,f2
        const_2 = (self.m2 / (self.m1 + self.m2)) * (self.l2 / self.l1) * np.cos(delta_th)
        denom = (const_1 * const_2) - 1
        const_3 = (self.l1 / self.l2) * np.sin(delta_th) * y[1] * y[1] - (self.g / self.l2) * np.sin(y[2])
        const_4 = -(self.m2 / (self.m1 + self.m2)) * (self.l2 / self.l1) * np.sin(delta_th) * y[3] * y[3] - (self.g / self.l1) * np.sin(y[0])

        f0 = y[1]
        f1 = ((const_3 * const_2) - const_4) / denom
        f2 = y[3]
        f3 = ((const_1 * const_4) - const_3) / denom

        return np.array([f0, f1, f2, f3])

    def animate(self, i):
        # func that is called by animation.Funcanimation at each frame
        self.line.set_data([0, self.x1[i], self.x2[i]], [0, self.y1[i], self.y2[i]])
        self.trace.set_data(self.x2[:i+1], self.y2[:i+1])
        return self.line, self.trace

    def init_func(self):
        # initialises empty axis and pendulum for each new animation
        self.line.set_data([], [])
        self.trace.set_data([], [])
        return self.line, self.trace


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for ax in (ax1, ax2):
        ax.grid()

    pend1 = DoublePendulum(l1=1, l2=1, m1=1, m2=1, g=9.81, dt=0.02, t_max=10, th1=np.radians(45), th2=np.radians(60), w1=0, w2=0, ax=ax1, p_col='bo-', t_col='r-')
    pend2 = DoublePendulum(l1=1, l2=1, m1=1, m2=1, g=9.81, dt=0.02, t_max=10, th1=np.radians(45), th2=np.radians(90), w1=0, w2=0, ax=ax2, p_col='bo-', t_col='r-')
    pend3 = DoublePendulum(l1=1, l2=1, m1=1, m2=1, g=9.81, dt=0.02, t_max=10, th1=np.radians(180), th2=np.radians(90), w1=0, w2=0, ax=ax1, p_col='co-', t_col='m-')
    pend4 = DoublePendulum(l1=1, l2=1, m1=1, m2=1, g=9.81, dt=0.02, t_max=10, th1=np.radians(45), th2=np.radians(65), w1=0, w2=0, ax=ax2, p_col='co-', t_col='m-')

    # trying to make as smooth as possible by choice of interval parameter
    # haven't found the best choice yet
    # dt^2 works best on my screen but different value may be needed for you
    ani = animation.FuncAnimation(fig,
                                  join_functions(pend1.animate, pend2.animate, pend3.animate, pend4.animate),
                                  len(pend1.sol),
                                  interval=pend1.dt*pend1.dt,
                                  blit=True,
                                  init_func=join_functions(pend1.init_func, pend2.init_func, pend3.init_func, pend4.init_func))

    plt.tight_layout()
    plt.show()
