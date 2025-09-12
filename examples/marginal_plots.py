import numpy as np
from rtrunc import td_optimizer as rtrunc
import matplotlib.pyplot as plt


n = 500
k = 100
gamma = 2.2

# initialize state vector
v = np.arange(n)+1
v=v**(-gamma)

# normalize
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))

# instantiate trace distance optimization
tdo = rtrunc.TDOptimizer(k, v)
fid = tdo.fid
print("Generated length-{} random vector: {}.".format(n, v))
print(" Optimal pure TD approx: {}".format(np.sqrt(1-fid**2)))
print("Solving TD optimization problem for k={}...".format(k))
m,td = tdo.getOptimalTDMeas()
r = tdo.r
l = tdo.l

# compute rtrunc marginal vector, including initial ones
marginals = np.zeros(n)
marginals[:k-r-1] = np.ones(k-r-1)
marginals[k-r-1:l-1] = tdo.getMarginals()

xs = np.arange(1, n+1, 1)
plt.plot(xs, marginals, 'o', label='$\\gamma={:.2f}$'.format(gamma))
plt.legend()
plt.show()
