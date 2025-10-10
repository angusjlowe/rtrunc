import numpy as np
from rtrunc import td_optimizer as rtrunc
from rtrunc import rob_optimizer as rob_rtrunc
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('font', size=16)

# matrix visualization
from matplotlib import colors

np.random.seed(41)


n = int(input("n: "))
k = int(input("k: "))

# normal, random
v = np.random.normal(0,1,n)

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

sigma = tdo.getOptimalTDState()

idxs = np.arange(n)
sigmanew = np.array([[sigma[idxs[i],idxs[j]] for i in range(n)] for j in range(n)])
sigma = sigmanew
vtrunc = np.concatenate((v[:k],np.zeros(n-k)))
vtrunc = vtrunc/np.linalg.norm(vtrunc)
vtruncnew = np.array([vtrunc[idxs[j]] for j in range(n)])
vtrunc = vtruncnew
vnew = np.array([v[idxs[j]] for j in range(n)])
v = vnew
vmin = 0
vmax = v[0]**2
cmap = plt.cm.binary
norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(4,4))
ax.matshow(np.outer(v,v), cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("$v v^\\top$", pad=10)
plt.savefig('density_matrix_plot_1.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(4,4))
ax.matshow(np.outer(vtrunc,vtrunc), cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("${\\widetilde{v}}_{1:k} {(\\widetilde{v}_{1:k})}^\\top$", pad=10)
plt.savefig('density_matrix_plot_2.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(4,4))
ax.matshow(sigma, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("$\\sigma^\\star$", pad=10)
plt.savefig('density_matrix_plot_3.pdf')
plt.close()

# colorbar beside ax3 only
#cbar = fig.colorbar(rmat, ax=ax3, orientation="vertical",
#                    fraction=0.046, pad=0.04)
#cbar.set_label("Value")
#plt.show()
