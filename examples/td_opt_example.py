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

import matplotlib.pyplot as plt
plt.step([*range(len(m))], m, label="m")
plt.step([*range(len(m))], v, label="v")
plt.legend()
plt.title("k={}. Pure TD approx: {:4f}. Random TD approx: {:4f}".format(k, np.sqrt(1-fid**2),td))
plt.show()

print("Getting worst-case meas. for deterministic approx.")
# store deterministic truncation
vtrunc = np.concatenate((v[:k],np.zeros(n-k)))
vtrunc = vtrunc/np.linalg.norm(vtrunc)
# compute worst-case measurement
(evals, evecs) = np.linalg.eig(np.outer(v,v)-np.outer(vtrunc, vtrunc))
max_index = np.argmax(evals)
m_det = evecs[:,max_index]
m_det = m_det/np.linalg.norm(m_det)

true_expec = np.abs(np.dot(m_det, v))**2
det_trunc_expec = np.abs(np.dot(m_det, vtrunc))**2

print("Begin sampling procedure...")

n_samples = 1000
expec_samples = []
rob_expec_samples = []
for j in range(n_samples):
    if (j+1) % 20 == 0:
        print("Sample {}".format(j+1))
    phi = tdo.sampleOptimalTDState()
    expec = np.abs(np.dot(phi, m_det))**2
    expec_samples.append(expec)
    phi = ro.sampleOptimalRobState()
    expec = np.abs(np.dot(phi, m_det))**2
    rob_expec_samples.append(expec)


means = np.array(list(map(lambda x: np.mean(expec_samples[:x]), [*range(1,n_samples+1)])))
stds = np.array(list(map(lambda x: np.std(expec_samples[:x], ddof=1)/np.sqrt(x), [*range(2,n_samples+1)])))
stds = np.concatenate(([0], stds))
xs = np.arange(n_samples)+1
ys = np.abs(means - true_expec)
plt.plot(xs, ys, '-', label='rtrunc (trace distance)', color='blue')
plt.fill_between(xs, ys-stds, ys+stds, color='blue', alpha=0.2)

rob_means = np.array(list(map(lambda x: np.mean(rob_expec_samples[:x]), [*range(1,n_samples+1)])))
rob_stds = np.array(list(map(lambda x: np.std(expec_samples[:x], ddof=1)/np.sqrt(x), [*range(2,n_samples+1)])))
rob_stds = np.concatenate(([0], stds))
ys = np.abs(rob_means - true_expec)
plt.plot(xs, ys, '-', label='rtrunc (robustness)', color='orange')
plt.fill_between(xs, ys-stds, ys+stds, color='orange', alpha=0.2)
#plt.plot(xs, np.ones(n_samples)*np.abs(true_expec-, '--', color='black', label='true expectation')
#plt.plot(xs, np.ones(n_samples)*true_expec, '--', color='green', label='true expec.')
plt.plot(xs, np.ones(n_samples)*np.abs(det_trunc_expec - true_expec), '--', color='red', label='dtrunc')
#print("Computing optimal density matrix...")
#sigma_expec = np.linalg.trace(np.dot(sigma, np.outer(m_det, m_det)))
#plt.plot(xs, np.ones(n_samples)*np.abs(sigma_expec - true_expec), '--', color='red', label='closed-form rtrunc expec. diff.')
plt.xlabel('no. of samples')
plt.legend()
plt.ylabel('error')
#title1 = "Estimating worst-case observable for $|v_{1:k}\\rangle$."
#title2 = " n={}, k={}.".format(n, k)
#plt.title(title1 + title2)
plt.show()

#td = rtrunc.traceDistance(np.outer(v,v), sigma)
#print("Getting optimal state leads to a trace distance: {}.".format(td))
input("Press enter to see visualization.")

sigma = tdo.getOptimalTDState()

idxs = np.arange(n)
np.random.shuffle(idxs)
sigmanew = np.array([[sigma[idxs[i],idxs[j]] for i in range(n)] for j in range(n)])
sigma = sigmanew
vmin = 0
vmax = v[0]**2
cmap = plt.cm.seismic
norm = colors.Normalize(vmin=vmin, vmax=vmax)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(10,4))
axes = (ax1,ax2,ax3)
vtruncnew = np.array([vtrunc[idxs[j]] for j in range(n)])
vtrunc = vtruncnew
vnew = np.array([v[idxs[j]] for j in range(n)])
v = vnew
im1 = ax1.matshow(np.outer(v,v), cmap=cmap, norm=norm, aspect="auto")
ax1.set_xticks([])
ax1.set_yticks([])
ax2.matshow(np.outer(vtrunc,vtrunc), cmap=cmap, norm=norm, aspect="auto")
ax2.set_xticks([])
ax2.set_yticks([])
rmat = ax3.matshow(sigma, cmap=cmap, norm=norm, aspect="auto")
ax3.set_xticks([])
ax3.set_yticks([])

ax1.set_title("$v v^T$", pad=10)
ax2.set_title("$\\widetilde{v}_{1:k} \\left(\\widetilde{v}_{1:k}\\right)^T$", pad=10)
ax3.set_title("$\\sigma^\\star$", pad=10)



# colorbar beside ax3 only
cbar = fig.colorbar(rmat, ax=ax3, orientation="vertical",
                    fraction=0.046, pad=0.04)
#cbar.set_label("Value")
plt.show()


    
