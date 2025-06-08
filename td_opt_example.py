import matplotlib.pyplot as plt
import numpy as np
import new_td_optimization as rtrunc

n = int(input("n: "))
k = int(input("k: "))

# normal, random
v = np.random.normal(0,1,n)

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

print("Getting worst-case meas. for deterministic approx.")
vtrunc = np.concatenate((v[:k],np.zeros(n-k)))
vtrunc = vtrunc/np.linalg.norm(vtrunc)
(evals, evecs) = np.linalg.eig(np.outer(v,v)-np.outer(vtrunc, vtrunc))
max_index = np.argmax(evals)
print("top eval is {}".format(evals[max_index]))
m_det = evecs[:,max_index]
m_det = m_det/np.linalg.norm(m_det)

true_expec = np.abs(np.dot(m_det, v))**2
det_trunc_expec = np.abs(np.dot(m_det, vtrunc))**2


print("I'm calculating {} for trace distance now".format(np.abs(true_expec-det_trunc_expec)))

print("Begin sampling procedure...")

n_samples = 200
expec_samples = []
for j in range(n_samples):
    if (j+1) % 10 == 0:
        print("Sample {}".format(j+1))
    phi = newTd.sampleOptimalTDState()
    expec = np.abs(np.dot(phi, m_det))**2
    expec_samples.append(expec)

means = np.array(list(map(lambda x: np.mean(expec_samples[:x]), [*range(1,n_samples+1)])))
stds = np.array(list(map(lambda x: np.std(expec_samples[:x], ddof=1)/np.sqrt(x), [*range(2,n_samples+1)])))
stds = np.concatenate(([0], stds))
xs = np.arange(n_samples)+1
plt.plot(xs, means, '-', label='rtrunc estimate', color='blue')
plt.fill_between(xs, means-stds, means+stds, color='blue', alpha=0.2)
plt.plot(xs, np.ones(n_samples)*true_expec, '--', color='green', label='true expec.')
plt.plot(xs, np.ones(n_samples)*det_trunc_expec, '--', color='orange', label='det. trunc. expec. diff.')
plt.xlabel('no. of samples')
plt.legend()
plt.show()



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
cmap = plt.cm.viridis
norm = colors.Normalize(vmin=vmin, vmax=vmax)
fig, (ax1,ax2,ax3) = plt.subplots(1,3)
axes = (ax1,ax2,ax3)
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


    
