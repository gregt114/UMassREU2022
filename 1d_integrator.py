import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.optimize import root

from numba import jit

import time


# Parameters
n = 400
OMEGA = 0.2
MU = 1
TOL = 1e-8
bc = 0 # Bondary Condition


# Grid to solve over
x0 = -24
xf = 24
dx = (xf - x0) / n
xs = np.linspace(x0, xf, n)



@jit(nopython=True)
def f(v):
    res = np.zeros_like(v, dtype=np.float32)
    for i in range(0, len(v)):
        before = v[i-1] if i != 0 else bc
        now = v[i]
        after = v[i+1] if i != len(v)-1 else bc
        x = x0 + i*dx

        val = -0.5*(after - 2*now + before)/dx**2 + (np.abs(now)**2)*now + (0.5*OMEGA**2)*(x**2)*now - MU*now
        res[i] = val

    return res


# Want to solve differential eqation dfdt = 0
@jit(nopython=True)
def dfdt(v):
    res = np.zeros_like(v, dtype=np.complex128)
    for i in range(0, len(v)):
        before = v[i-1] if i != 0 else bc
        now = v[i]
        after = v[i+1] if i != len(v)-1 else bc
        x = x0 + i*dx

        val = -0.5*(after - 2*now + before)/dx**2 - (np.abs(now)**2)*now - (0.5*OMEGA**2)*(x**2)*now
        res[i] = val

    return -1j*res


# Return next value in integration using RK4
@jit(nopython=True)
def step_rk4(v, dt):
    k1 = dfdt(v)
    k2 = dfdt(v + dt*k1/2)
    k3 = dfdt(v + dt*k2/2)
    k4 = dfdt(v + dt*k3)
    return v + (k1 + 2*k2 + 2*k3 + k4)*dt/6



def main():
    
    # Guess to use
    guess0 = np.exp(-OMEGA*xs**2 / 2)
    guess = guess0

    # Time domain to integrate over
    t0 = 0
    tf = 20
    nt = 4000 # number of time points to use (= number of frames)
    dt = (tf - t0) / nt
    ts = np.linspace(t0, tf, nt)

    skip = 1 # How many frames to skip when animating (skip=1 means dont skip any)



    start = time.time()

    # Find initial stationary solution
    soln = root(f, guess, method="df-sane", tol=TOL).x

    # Integrate
    final_soln = soln.copy()
    plotting = [] # list of arrays, use for animation
    for t in ts:
        plotting.append(final_soln)
        final_soln = step_rk4(final_soln, dt)
    print(f'Time: {time.time() - start}')


    fig = plt.figure(figsize=[10,6])


    # For plotting norm
    ax = plt.axes(xlim=(x0, xf), ylim=(0,4))
    line, = ax.plot([], [], lw=2)
    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        line.set_data(xs, np.abs(plotting[skip*i])**2)
        ax.set_title(f't = {skip * i / (nt / (tf - t0)):.1f}')
        return line,
    plt.ylabel(r'$|\psi|^2$')
    
    # For plotting real and imag parts
    # ax = plt.axes(xlim=(x0, xf), ylim=(-4,4))
    # line1, = ax.plot([], [], lw=2, label="real")
    # line2, = ax.plot([], [], lw=2, label='imag')
    # def init():
    #     line1.set_data([], [])
    #     line2.set_data([], [])
    #     return [line1, line2]
    # def animate(i):
    #     line1.set_data(xs, plotting[skip*i].real)
    #     line2.set_data(xs, plotting[skip*i].imag)
    #     ax.set_title(f't = {skip * i / (nt / (tf - t0)):.1f}')
    #     return [line1, line2]
    # plt.legend()


    anim = FuncAnimation(fig, animate, init_func=init, frames=nt//skip, interval=20, blit=True)

    plt.xlabel('x')
    plt.title('Time Evolution of 1D NLS')

    #anim.save('1d_nls.gif') # uncomment if you want to save animation to .gif file
    plt.show()


if __name__ == "__main__":
    main()

