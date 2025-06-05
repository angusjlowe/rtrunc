import matplotlib.pyplot as plt
import numpy as np
import new_td_optimization as rtrunc

n = int(input("n: "))
k = int(input("k: "))

# normal, random
v = np.random.normal(0,1,n)

# power law
#gamma = 0.05
#xs = np.arange(n)+1
#v = xs**(-gamma)

# normalize
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))

newTd = rtrunc.NewTDExperiment(k, v)
fid = newTd.fid
print("Generated length-{} random vector: {}.".format(n, v))
print(" Optimal pure TD approx: {}".format(np.sqrt(1-fid**2)))
print("Solving TD optimization problem for k={}...".format(k))
m,td = newTd.getOptimalTDMeas()
input("Solved. Optimal randomized TD approx: {:4f}. Press enter to see plot.".format(td))
m=list(m)
plt.step([*range(len(m))], m, label="m")
plt.step([*range(len(m))], v, label="v")
plt.legend()
plt.title("k={}. Pure TD approx: {:4f}. Random TD approx: {:4f}".format(k, np.sqrt(1-fid**2),td))
plt.show()

print("Beginning sampling procedure...")


sigma = newTd.getOptimalTDState()
td = rtrunc.traceDistance(np.outer(v,v), sigma)
print("Getting optimal state leads to a trace distance: {}.".format(td))
input("Press enter to see visualization.")


# matrix visualization
from matplotlib import colors

idxs = np.arange(n)
np.random.shuffle(idxs)
sigmanew = np.array([[sigma[idxs[i],idxs[j]] for i in range(n)] for j in range(n)])
sigma = sigmanew
vmin = 0
vmax = v[0]**2
cmap = plt.cm.plasma
norm = colors.Normalize(vmin=vmin, vmax=vmax)
fig, (ax1,ax2,ax3) = plt.subplots(1,3)
axes = (ax1,ax2,ax3)
vtrunc = np.concatenate((v[:k],np.zeros(n-k)))
vtrunc = vtrunc/np.linalg.norm(vtrunc)
vtruncnew = np.array([vtrunc[idxs[j]] for j in range(n)])
vtrunc = vtruncnew
vnew = np.array([v[idxs[j]] for j in range(n)])
v = vnew
im1 = ax1.matshow(np.outer(v,v), cmap=cmap, norm=norm)
ax1.set_title("$|v\\rangle\\langle v|$")
ax2.matshow(np.outer(vtrunc,vtrunc), cmap=cmap, norm=norm)
ax2.set_title("$|v_{1:k}\\rangle \\langle v_{1:k}|$")
ax3.matshow(sigma, cmap=cmap, norm=norm)
ax3.set_title("$\\sigma^\\star$")
plt.tight_layout()
plt.show()
    
