import numpy as np
from rtrunc import td_optimizer as rtrunc
from scipy.special import zeta

n = int(input("n: "))
k = int(input("k: "))
gamma = 0.501

xs = np.arange(n)+1
v = xs**(-gamma)

# normalize
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))


# instantiate trace distance optimization
tdo = rtrunc.TDOptimizer(k, v)
fid = tdo.fid
eps = np.sqrt(1-fid**2)
print("Generated length-{} random vector: {}.".format(n, v))
print(" Optimal pure TD approx: {}".format(eps))
print("Solving TD optimization problem for k={}...".format(k))
m,td = tdo.getOptimalTDMeas()
input("Solved. Optimal randomized TD approx: {:4f}. Press enter to see plot.".format(td))
m=list(m)


# form guess for optimal measurement
a = 1/(np.sqrt(4*(1-eps)))
b = np.sqrt(1-a**2*(1-eps**2))
s = int(k**(2*gamma))
print(s)
term1 = np.concatenate((v[:k], np.zeros(n-k)))
term2 = np.concatenate((np.zeros(k), np.ones(s-k), np.zeros(n-s)))
m_guess = a*term1 + (b/np.sqrt(s))*term2

import matplotlib.pyplot as plt
plt.step([*range(len(m))], m, label="m")
plt.step([*range(len(m))], v, label="v")
plt.step([*range(len(m))], m_guess, label="Guess for m")
plt.legend()
plt.title("k={}. Pure TD approx: {:4f}. Random TD approx: {:4f}".format(k, eps,td))
plt.show()