from hashlib import new
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton_krylov
from scipy.integrate import simps

from numba import jit

from methods.newton import newton
from methods.utils import norm2D as norm

import time

x0 = -12
xf = 12

y0 = -12
yf = 12

n = 160

dx = (xf - x0) / n
dy = (yf - y0) / n

# Grid to solve over
xs = np.linspace(x0, xf, n)
ys = np.linspace(y0, yf, n)
X, Y = np.meshgrid(xs, ys)


# Bondary Conditions
bc = 0

OMEGA = 0.2



@jit(nopython=True)
def f(v, mu):
    ret = np.zeros_like(v, dtype=np.complex128)

    for i in range(n**2):
        beforeX = v[i-1] if i > 0 else bc
        afterX = v[i+1] if i+1 < len(v) else bc
        beforeY = v[i-n] if i-n >= 0 else bc
        afterY = v[i+n] if i+n < len(v) else bc
        here = v[i]

        x = x0 + (i % n)*dx
        y = y0 + (i // n)*dy

        val = -0.5*((afterX - 2*here + beforeX)/dx**2 + (afterY - 2*here + beforeY)/dy**2)
        val += (np.abs(here)**2) * here + (OMEGA**2 / 2)*(x**2 + y**2)*here - mu*here
 
        ret[i] = val

    return ret



def main():

    # Constants
    n_mu = 40
    mu_0 = 1
    mu_f = 0
    mus = np.linspace(mu_0, mu_f, n_mu)
    
    TOL = 1e-4 # was 1e-6

    

    # Ground state
    guess0 = np.exp(-OMEGA * (X**2 + Y**2) / 50)

    # First Excited
    guess1 = X * np.exp(-OMEGA * (X**2 + Y**2) / 2)
    vortex1 = (X + 1j*Y) * np.exp(-OMEGA * (X**2 + Y**2) / 2) # single vortex

    # 2nd Excited
    blobs2 = X**2 * np.exp(-OMEGA * (X**2 + Y**2) / 2) # 2nd excited, 2 blobs + stripe in middle
    radial2 = X * Y * np.exp(-OMEGA * (X**2 + Y**2) / 2) # 2nd excited, need n >= 150 with domain (-12,12) (radially symmetric)

    
    

    # Reshape guess   
    guesses = [guess0, guess1, vortex1, blobs2, radial2]
    guesses = [ guess.reshape(1, -1)[0] for guess in guesses]


    # Solve
    start = time.time()
    i = 0
    for guess in guesses:
        ns = [] # list to hold norms
        isComplex = False
        if guess.dtype == np.complex128:
            isComplex = True

        for mu in mus:
            if isComplex:
                soln = newton(lambda x: f(x,mu), guess, TOL, method='complex')
            else:
                soln = newton_krylov(lambda x : f(x,mu), guess, f_rtol=TOL)
            guess = soln
            normalized = norm(soln, xs, ys)
            ns.append(normalized)

        plt.plot(mus, ns, label=f"{i}")
        i += 1
    print(f"Time: {time.time() - start}")


    plt.title('Norm of solutions to 2D NLS')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$N = \int \int |\psi^2|dxdy$')

    plt.xlim(0, 1.1)


    plt.grid()
    plt.legend()

    plt.show()




if __name__ == "__main__":
    main()