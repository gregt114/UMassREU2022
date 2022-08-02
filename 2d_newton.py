
import numpy as np
from numpy.linalg import norm

from numba import jit

import matplotlib.pyplot as plt

from scipy.optimize import root
from scipy.optimize import newton_krylov
from scipy.sparse import diags


# Parameters
n = 160
OMEGA = 0.2
MU = 1
TOL = 8e-5
bc = 0 # Bondary Condition


# Grid to solve over
x0 = -14
xf = 14
y0 = -14
yf = 14
dx = (xf - x0) / n
dy = (yf - y0) / n
xs = np.linspace(x0, xf, n)
ys = np.linspace(y0, yf, n)
X, Y = np.meshgrid(xs, ys)




@jit(nopython=True)
def f(v):
    ret = np.zeros_like(v, dtype=np.float64)

    for i in range(n**2):
        beforeX = v[i-1] if i > 0 else bc
        afterX = v[i+1] if i+1 < len(v) else bc
        beforeY = v[i-n] if i-n >= 0 else bc
        afterY = v[i+n] if i+n < len(v) else bc
        here = v[i]

        x = x0 + (i % n)*dx
        y = y0 + (i // n)*dy

        val = -0.5*((afterX - 2*here + beforeX)/dx**2 + (afterY - 2*here + beforeY)/dy**2)
        val += (np.abs(here)**2) * here + (OMEGA**2 / 2)*(x**2 + y**2)*here - MU*here
 
        ret[i] = val

    return ret





import time
def main():

    # Guesses    
    gs0 = [np.exp(-OMEGA * (X**2 + Y**2) / 2)]
    gs1 = [
        X * np.exp(-OMEGA * (X**2 + Y**2) / 2),
        Y * np.exp(-OMEGA * (X**2 + Y**2) / 2)
    ]
    gs2 = [
        0.5*X**2 * np.exp(-OMEGA * (X**2 + Y**2) / 2),
        0.5*Y**2 * np.exp(-OMEGA * (X**2 + Y**2) / 2),
        X * Y * np.exp(-OMEGA * (X**2 + Y**2) / 2)
        ]
    gs3 = [
        0.1*(X**3) * np.exp(-OMEGA * (X**2 + Y**2) / 2),
        0.5*(X**2 * Y) * np.exp(-OMEGA * (X**2 + Y**2) / 2)
    ]

    # Reshape guess
    gs = gs3
    gs = [g.reshape(1,-1)[0] for g in gs]


    # Solve
    solutions = [] # list to hold solutions
    start = time.time()
    for g in gs:
        soln = newton_krylov(f, g, f_tol=TOL, method='lgmres', verbose=True)
        solutions.append(soln)
        print(f"Max Residual: {np.abs(f(soln)).max()}")
        print('----------------')
    end = time.time()
    print(f"Time: {end - start}")
  


    # Reshape and take prob. density for plotting
    solutions = [np.abs(sol.reshape(n,n))**2 for sol in solutions]

    # Setup plotting
    plt.figure(figsize=[12,6])
    num_solns = len(solutions)
    plotnum = 1
    cmap='inferno'

    # Plot each solution
    for i in range(len(solutions)):
        sol = solutions[i]

        plt.subplot(1, num_solns, plotnum)
        im = plt.imshow(sol, origin="lower", extent=(x0,xf,y0,yf), cmap=cmap)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plotnum += 1



    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()

