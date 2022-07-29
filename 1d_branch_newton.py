import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from methods.newton import newton
from methods.utils import norm1D as norm


from scipy.optimize import newton_krylov, root

import time


# Domain to solve over
x0 = -24
xf = 24
n = 450
xs = np.linspace(x0, xf, n)
dx = (xf - x0) / n

# Parameters
OMEGA = 0.2
TOL = 1e-5
bc = 0 # Bondary Conditions


@jit(nopython=True)
def f(v, mu):
    res = np.zeros(len(v))
    for i in range(0, len(v)):
        before = v[i-1] if i != 0 else bc
        now = v[i]
        after = v[i+1] if i != len(v)-1 else bc
        x = x0 + i*dx

        val = (after - 2*now + before)/dx**2 - (2*np.abs(now)**2)*now - (OMEGA**2)*(x**2)*now + 2*mu*now
        res[i] = val

    return res


@jit(nopython=True)
def jac(v, mu):
    res = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if(j == i-1 or j == i+1):
                res[i][j] = 1/dx**2
            elif(j == i):
                x = x0 + j*dx
                res[i][j] = -2/dx**2 - 6*v[j]**2 - (OMEGA**2)*x**2 + 2*mu
            else:
                res[i][j] == 0
    return res



# Values of mu to solve over
mu0 = 1
muf = 0
n_mu = 200
mus = np.linspace(mu0, muf, n_mu)


# Initial guesses for each branch
guess0 = np.exp(-OMEGA*xs**2 / 2)
guess1 = 0.5*xs * np.exp(-OMEGA*xs**2 / 2)
guess2 = 0.18*(xs**2 - 0.333**2) * np.exp(-OMEGA * xs**2 / 2)  # need n >= 400
guess3 = (0.1*xs**3 - 0.75*xs) * np.exp(-OMEGA * xs**2 / 2)  # need n >= 400
guess4 = (0.02*xs**4 - 0.35*xs**2 + 0.5) * np.exp(-OMEGA * xs**2 / 2)


plt.figure(figsize=[10,6])

start = time.time()
for guess in [guess0, guess1, guess2, guess3, guess4]:
    ns = []

    for mu in mus:
        func = lambda x: f(x, mu)
        soln = newton(lambda x: f(x, mu), guess, TOL, 'numpy', jac=lambda x: jac(x, mu), verbose=False)

        normalized = norm(soln, dx)
        ns.append(normalized)

        guess = soln

    plt.plot(mus, ns)
    print(f"----")
end = time.time()
print(f"Time: {end - start}")

plt.title(r'Norm of solutions to 1D NLS')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$N = \int |\phi|^2dx$')

plt.xlim(0, 1.1)

plt.grid()

plt.show()


