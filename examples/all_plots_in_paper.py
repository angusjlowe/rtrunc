# This code will produce all the plots in the paper arXiv:2510.08518

import numpy as np
from rtrunc import td_optimizer as rtrunc
from rtrunc import rob_optimizer as rob_rtrunc
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('font', size=16)

# ----------------------------------------------------------------------
# Figure 1
# ----------------------------------------------------------------------

# matrix visualization
from matplotlib import colors

np.random.seed(41)


n = 35
k = 15

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


# ----------------------------------------------------------------------
# Figure 3
# ----------------------------------------------------------------------
plt.rc('font', size=14)

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


xs = np.arange(1, n+1, 1)
xspartial = np.arange(k-r,l,1)
plt.plot(xs, v, '--', label="$v$", color='blue')
plt.plot(xs, m, '-', label="$m$", color='red')
plt.axvline(x=k-r, color='grey', linestyle=':')
plt.axvline(x=l-1, color='grey', linestyle=':')
plt.legend()
plt.xlim((-10,210))
plt.savefig('meas_vs_vector_plot_1.pdf')
plt.close()

marginal = np.concatenate((np.ones(k-r-1), v[k-r-1:l-1]/theta(r,l,td) - td, np.zeros(n-l+1)))
plt.plot(xs, marginal, '-', label='$q$', color='black')
plt.axvline(x=k-r, color='grey', linestyle=':')
plt.axvline(x=l-1, color='grey', linestyle=':')
plt.legend()
plt.xlim((-10,210))
plt.savefig('meas_vs_vector_plot_2.pdf')
print("k={}, r={}, l={}".format(k,r,l))
plt.close()



# ----------------------------------------------------------------------
# Figure 5 Data Generation
# ----------------------------------------------------------------------
from mps_helpers import *

np.random.seed(52)
n=10
gammas = [0.1, 0.2, 0.4, 0.8]

def haar_random_state(d):
    # Draw complex Gaussian entries
    x = np.random.randn(d) + 1j * np.random.randn(d)
    # Normalize
    psi = x / np.linalg.norm(x)
    return psi

for numb in range(4):
    # parameter setup
    d = 2
    bond_dim = 2**(n//2)
    gamma = gammas[numb-1]
    n_samples = 100
    ks = np.arange(2,bond_dim+1,1)
    Z = np.array([[1,0],[0,-1]], dtype=float)
    X = np.array([[0,1],[1,0]], dtype=float)
    I = np.eye(d)

    # observable setup
    Zs = [Z] * n
    Xs = [X] * n
    Is = [I] * n
    obs = Is
    obs[4] = Z
    obs[5] = Z
    #obs[4] = Z
    #obs[8] = X

    # random mps setup
    psi = haar_random_state(d**n)
    psi_tensors_original = tensor_to_fixedbond_mps(psi, d, n, bond_dim)
    psi_tensors_original = power_law_schmidt_coeffs(psi_tensors_original, gamma)

    psi_tensors_original = normalize(psi_tensors_original)

    # original expecs
    orig_expec = np.real(mps_expec(psi_tensors_original, psi_tensors_original, obs))
    print("true expec is {:.5f}".format(orig_expec))

    rtrunc_means = []
    rtrunc_stds = []
    dtrunc_expecs = []
    for k in ks:
        print("k={}".format(k))
        print("Computing dtrunc expec")
        # dtrunc state computation
        psi_tensors = copy.deepcopy(psi_tensors_original)
        for l in range(1,n):
            psi_tensors = mps_dtrunc(psi_tensors, k, l)
            # dtrunc expecs
        dtrunc_expec = np.real(mps_expec(psi_tensors, psi_tensors, obs))
        print("dtrunc estimate is {:.5f}".format(dtrunc_expec))
        dtrunc_expecs.append(dtrunc_expec)

        # rtrunc states and expecs
        print("Computing rtrunc states")
        rtrunc_expecs = []
        for j in range(n_samples):
            if (j+1) % 10 == 0:
                print("Samples collected: {}".format(j+1))
            psi_tensors = copy.deepcopy(psi_tensors_original)
            for l in range(1,n):
                psi_tensors = mps_rtrunc(psi_tensors, k, l)
            trunc_expec = np.real(mps_expec(psi_tensors, psi_tensors, obs))
            rtrunc_expecs.append(trunc_expec)
        rtrunc_mean = np.mean(rtrunc_expecs)
        print("rtrunc estimate is {:.5f}".format(rtrunc_mean))
        rtrunc_std = np.std(rtrunc_expecs, ddof=1)/np.sqrt(n_samples)
        rtrunc_means.append(rtrunc_mean)
        rtrunc_stds.append(rtrunc_std)

    np.savez("mps_random_plot_{}.npz".format(numb),
            orig_expec=orig_expec,
            ks=ks,
            rtrunc_means=np.array(rtrunc_means),
            rtrunc_stds=np.array(rtrunc_stds),
            dtrunc_expecs=np.array(dtrunc_expecs))

    print("Saved results to mps_random_plot_{}.npz".format(numb))

# ----------------------------------------------------------------------
# Figure 5 Plots
# ----------------------------------------------------------------------

# parameter setup
d = 2
bond_dim = 2**(n//2)
datasets = []
for j in range(4):
    dataset = []
    data = np.load("mps_random_plot_{}.npz".format(j+1))
    dataset.append(data["orig_expec"])
    dataset.append(data["ks"])
    dataset.append(data["rtrunc_means"])
    dataset.append(data["rtrunc_stds"])
    dataset.append(data["dtrunc_expecs"])
    dataset.append(gammas[j])
    datasets.append(dataset)


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rc('font', size=18)

# --- Create a 2x2 grid ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)

# Enforce identical left margins for ALL subplots →
# this prevents size changes from long/short y-tick labels
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.10, top=0.93,
                    wspace=0.25, hspace=0.30)

# Flatten axes for easy iteration
axs = axs.ravel()

for ax, (orig_expec, ks_data, rmean, rstd, dexpec, gamma_val) in zip(axs, datasets):

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(2, bond_dim+1, 4))
    ax.set_xlabel("bond dim. cutoff")
    ax.set_ylabel(r"$\langle Z_{5}Z_{6}\rangle$ rel. error")

    # rtrunc with error bars
    ax.errorbar(
        ks_data,
        (rmean - orig_expec) / abs(orig_expec),
        yerr=rstd / abs(orig_expec),
        fmt='o',
        capsize=4,
        color='blue',
        label="rtrunc (TD)"
    )

    # dtrunc
    ax.plot(
        ks_data,
        (dexpec - orig_expec) / abs(orig_expec),
        's',
        color='red',
        label='dtrunc'
    )

    # zero line
    ax.plot(ks_data, np.zeros_like(ks_data), '--', color='black')

    ax.set_title(r"$\gamma={}$".format(gamma_val))
    ax.legend()


fig.tight_layout()
plt.savefig("plots_2x2_grid.pdf")
