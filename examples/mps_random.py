from mps_helpers import *
import numpy as np
from rtrunc import td_optimizer as tdo

def rtrunc(tensors, k, l):
    """"
    Get random k-truncation at lth site. Expects l
    between 1 and n-1. Assumes schmidt values already sorted.
    """
    new_tensors = mixed_canonical_form(tensors, l)
    schmidts = np.diag(new_tensors[l])
    if k < schmidts.size:
        newTDOpt = tdo.TDOptimizer(k, schmidts)
        m,td = newTDOpt.getOptimalTDMeas()
        new_schmidts = newTDOpt.sampleOptimalTDState()
    else:
        new_schmidts = schmidts
    new_tensors[l] = np.diag(new_schmidts)
    new_tensors = get_mps_tensors_from_canonical(new_tensors)
    return new_tensors

# parameter setup
d=2
bond_dim = 40
k = 10
gamma = 1.2
n_samples = 50

Z = np.array([[1,0],[0,-1]], dtype=float)
X = np.array([[0,1],[1,0]], dtype=float)
I = np.array([[1,0],[0,1]], dtype=float)

# what happens with the optimal k-incoherent density matrix?
ns = [*range(9,13)]
rtrunc_means = []
rtrunc_stds = []
dtrunc_expecs = []
orig_expecs = []
for n in ns:
    print("\n n = {} sites".format(n))

    # observable setup
    Zs = [Z] * n
    Xs = [X] * n
    Is = [I] * n
    obs = Is
    obs[0] = X
    obs[-1] = X
    
    # random mps setup
    psi_tensors_original = get_random_mps(n,d,bond_dim)
    print("Setting power law Schmidt coefficients...")
    psi_tensors_original = power_law_schmidt_coeffs(psi_tensors_original, gamma)

    # original expecs
    orig_expec = np.real(mps_expec(psi_tensors_original, psi_tensors_original, obs))
    orig_expecs.append(orig_expec)

    # dtrunc state computation
    psi_tensors = copy.deepcopy(psi_tensors_original)
    print("Computing dtrunc state")
    for l in range(1,n):
        psi_tensors = dtrunc(psi_tensors, k, l)

    # dtrunc expecs
    dtrunc_expec = np.real(mps_expec(psi_tensors, psi_tensors, obs))
    dtrunc_expecs.append(dtrunc_expec)

    # rtrunc states and expecs
    print("Computing rtrunc states")
    rtrunc_expecs = []
    for j in range(n_samples):
        if (j+1) % 10 == 0:
            print("Samples collected: {}".format(j+1))
        psi_tensors = copy.deepcopy(psi_tensors_original)
        for l in range(1,n-1):
            psi_tensors = rtrunc(psi_tensors, k, l)
        trunc_expec = np.real(mps_expec(psi_tensors, psi_tensors, obs))
        rtrunc_expecs.append(trunc_expec)
    rtrunc_mean = np.mean(rtrunc_expecs)
    rtrunc_std = np.std(rtrunc_expecs, ddof=1)/np.sqrt(n_samples)
    rtrunc_means.append(rtrunc_mean)
    rtrunc_stds.append(rtrunc_std)


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title("Dtrunc vs. rtrunc on random MPSs with {} samples".format(n_samples))
plt.xlabel("# of sites n")
plt.ylabel("$\\langle X_1X_n\\rangle$")
print("Plot things, yerr size = {}, means size = {}".format(len(rtrunc_stds), len(rtrunc_means)))
plt.errorbar(ns, rtrunc_means, yerr=rtrunc_stds, label="rtrunc", linestyle='none', capsize=4)
plt.plot(ns, dtrunc_expecs, 's', label='dtrunc')
plt.plot(ns, orig_expecs, 'o', label="true expec.")
print("Plotted")
plt.legend()
plt.show()