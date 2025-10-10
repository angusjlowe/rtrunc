import numpy as np
from rtrunc import td_optimizer as rtrunc
from rtrunc import rob_optimizer as rob_rtrunc
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('font', size=16)

np.random.seed(41)

# get dimension and sparsity parameters
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
print("Optimal pure TD approx: {}".format(np.sqrt(1-fid**2)))
print("Solving TD optimization problem for k={}...".format(k))
m,td = tdo.getOptimalTDMeas()

# display optimal randomized TD
input("Solved. Optimal randomized TD approx: {:4f}. Press enter to see plot.".format(td))
m=list(m)

# store deterministic truncation
print("Getting worst-case meas. for deterministic approx.")
vtrunc = np.concatenate((v[:k],np.zeros(n-k)))
vtrunc = vtrunc/np.linalg.norm(vtrunc)

# compute worst-case measurement
(evals, evecs) = np.linalg.eig(np.outer(v,v)-np.outer(vtrunc, vtrunc))
max_index = np.argmax(evals)
m_det = evecs[:,max_index]
m_det = m_det/np.linalg.norm(m_det)

# get true expectation value and dtrunc
true_expec = np.abs(np.dot(m_det, v))**2
det_trunc_expec = np.abs(np.dot(m_det, vtrunc))**2

# sample using different randomized truncation schemes
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

# compute means and stds for TD rtrunc
means = np.array(list(map(lambda x: np.mean(expec_samples[:x]), [*range(1,n_samples+1)])))
stds = np.array(list(map(lambda x: np.std(expec_samples[:x], ddof=1)/np.sqrt(x), [*range(2,n_samples+1)])))
stds = np.concatenate(([0], stds))
xs = np.arange(n_samples)+1
ys = np.abs(means - true_expec)
plt.plot(xs, ys, '-', label='rtrunc (trace distance)', color='blue')
plt.fill_between(xs, ys-stds, ys+stds, color='blue', alpha=0.2)

# compute means and stds for robustness rtrunc
rob_means = np.array(list(map(lambda x: np.mean(rob_expec_samples[:x]), [*range(1,n_samples+1)])))
rob_stds = np.array(list(map(lambda x: np.std(rob_expec_samples[:x], ddof=1)/np.sqrt(x), [*range(2,n_samples+1)])))
rob_stds = np.concatenate(([0], rob_stds))
ys = np.abs(rob_means - true_expec)
plt.plot(xs, ys, '-', label='rtrunc (robustness)', color='orange')
plt.fill_between(xs, ys-rob_stds, ys+rob_stds, color='orange', alpha=0.2)

# plot deterministic estimate (horizontal line)
plt.plot(xs, np.ones(n_samples)*np.abs(det_trunc_expec - true_expec), '--', color='red', label='dtrunc')
plt.xlabel('no. of samples')
plt.legend()
plt.ylabel('error')
plt.show()