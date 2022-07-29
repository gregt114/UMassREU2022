
import numpy as np
from numpy.linalg import norm

from numba import jit

import matplotlib.pyplot as plt

from scipy.optimize import root
from scipy.optimize import newton_krylov
from scipy.sparse import diags


# Parameters
n = 100
OMEGA = 0.2
MU = 0.65
TOL = 1e-8
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
    guess0 = np.exp(-OMEGA * (X**2 + Y**2) / 2)
    guess1 = X * np.exp(-OMEGA * (X**2 + Y**2) / 2)
    guess2 = X * Y * np.exp(-OMEGA * (X**2 + Y**2) / 2)
    

    # Reshape guess
    guess = guess2
    guess = guess.reshape(1, -1)[0]


    # Solve
    start = time.time()

    soln = newton_krylov(f, guess, f_rtol=TOL, method='lgmres', verbose=True)
    
    end = time.time()
    print(f"Time: {end - start}")
    print(f"Max Residual: {np.abs(f(soln)).max()}")


    # Prepare for plotting
    guess = guess.reshape(n,n)
    guess = np.abs(guess)**2

    soln = soln.reshape(n,n)
    soln = np.abs(soln)**2


    plt.figure(figsize=[10,6])
    plt.subplot(1,2,1)
    plt.imshow(guess, origin="lower", extent=(x0,xf,y0,yf))
    plt.colorbar()
    plt.title("Guess")

    plt.subplot(1,2,2)
    plt.title("Solution")
    plt.imshow(soln, origin="lower", extent=(x0,xf,y0,yf))
    plt.colorbar()


    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()

