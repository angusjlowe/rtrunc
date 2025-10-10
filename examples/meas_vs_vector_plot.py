import numpy as np
from rtrunc import td_optimizer as rtrunc
from rtrunc import rob_optimizer as rob_rtrunc
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('font', size=14)

# matrix visualization
from matplotlib import colors

n = 200
k = 100

xs = np.arange(1, n+1, 1)
# power law
gamma = 0.8
v = (n-xs)**2

# normalize
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))


# instantiate trace distance optimization
tdo = rtrunc.TDOptimizer(k, v)
ro = rob_rtrunc.RobustnessExperiment(k, v)
fid = tdo.fid
print("Generated length-{} random vector: {}.".format(n, v))
print(" Optimal pure TD approx: {}".format(np.sqrt(1-fid**2)))
print("Solving TD optimization problem for k={}...".format(k))
m,td = tdo.getOptimalTDMeas()

input("Solved. Optimal randomized TD approx: {:4f}. Press enter to see plot.".format(td))
m=list(m)
theta = tdo.theta
k = tdo.k
r = tdo.r
l = tdo.l


import matplotlib.pyplot as plt
xs = np.arange(1, n+1, 1)
xspartial = np.arange(k-r,l,1)
plt.plot(xs, v, '--', label="$v$", color='blue')
plt.plot(xs, m, '-', label="$m$", color='red')
plt.axvline(x=k-r, color='grey', linestyle=':')
plt.axvline(x=l-1, color='grey', linestyle=':')
plt.legend()
plt.xlim((-10,210))
plt.show()
#plt.savefig('meas_vs_vector_plot_1.pdf')

marginal = np.concatenate((np.ones(k-r-1), v[k-r-1:l-1]/theta(r,l,td) - td, np.zeros(n-l+1)))
plt.plot(xs, marginal, '-', label='$q$', color='black')
plt.axvline(x=k-r, color='grey', linestyle=':')
plt.axvline(x=l-1, color='grey', linestyle=':')
plt.legend()
plt.xlim((-10,210))
#plt.savefig('meas_vs_vector_plot_2.pdf')
plt.show()
print("k={}, r={}, l={}".format(k,r,l))
