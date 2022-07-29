import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import time

from methods.newton import newton



# Domain to solve over
x0 = -24
xf = 24
n = 450
xs = np.linspace(x0, xf, n)
dx = (xf - x0) / n

# Parameters
OMEGA = 0.2
MU = 1
TOL = 1e-7
bc = 0 # Bondary Conditions


@jit(nopython=True)
def f(v):
    res = np.zeros_like(v)
    for i in range(0, len(v)):
        before = v[i-1] if i != 0 else bc
        now = v[i]
        after = v[i+1] if i != len(v)-1 else bc
        x = x0 + i*dx

        val = (after - 2*now + before)/dx**2 - (2*np.abs(now)**2)*now - (OMEGA**2)*(x**2)*now + 2*MU*now
        res[i] = val

    return res


@jit(nopython=True)
def jac(v):
    res = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if(j == i-1 or j == i+1):
                res[i][j] = 1/dx**2
            elif(j == i):
                x = x0 + j*dx
                res[i][j] = -2/dx**2 - 6*v[j]**2 - (OMEGA**2)*x**2 + 2*MU
            else:
                res[i][j] == 0
    return res


def main():
    guess0 = np.exp(-OMEGA*xs**2 / 2)
    guess1 = 0.5*xs * np.exp(-OMEGA*xs**2 / 2)
    guess2 = 0.18*(xs**2 - 0.333**2) * np.exp(-OMEGA * xs**2 / 2)  # need n >= 400
    guess3 = (0.1*xs**3 - 0.75*xs) * np.exp(-OMEGA * xs**2 / 2)    # need n >= 400
    guess4 = (0.02*xs**4 - 0.35*xs**2 + 0.5) * np.exp(-OMEGA * xs**2 / 2) # need n >= 200


    # Setup plotting
    plt.figure(figsize=[10,6])
    plotnum = 1

    start = time.time()
    for guess in [guess0, guess1, guess2, guess3, guess4]:
        plt.subplot(2,3,plotnum)
        soln = newton(f, guess, TOL, 'numpy', jac=jac, verbose=True)
        plt.plot(xs, soln)
        plt.title(f'State {plotnum - 1}')
        plt.xlabel('x')
        plt.ylabel(r'$\phi(x)$')
        plotnum += 1
    end = time.time()
    print(f'Time: {end - start}')


    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()

