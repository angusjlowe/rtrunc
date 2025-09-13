from mps_helpers import *
import numpy as np
from rtrunc import td_optimizer as tdo
np.random.seed(42)

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
n = 11
d = 2
bond_dim = 30
gamma = 0.30
n_samples = 100
ks = np.arange(5,6,1)
Z = np.array([[1,0],[0,-1]], dtype=float)
X = np.array([[0,1],[1,0]], dtype=float)
I = np.array([[1,0],[0,1]], dtype=float)

# observable setup
Zs = [Z] * n
Xs = [X] * n
Is = [I] * n
obs = Is
obs[5] = Z

# random mps setup
psi_tensors_original = get_random_mps(n,d,bond_dim)
psi_tensors_original = power_law_schmidt_coeffs(psi_tensors_original, gamma)

# original expecs
orig_expec = np.real(mps_expec(psi_tensors_original, psi_tensors_original, obs))


rtrunc_means = []
rtrunc_stds = []
dtrunc_expecs = []
for k in ks:
    print("k={}".format(k))
    print("Computing dtrunc expec")
    # dtrunc state computation
    psi_tensors = copy.deepcopy(psi_tensors_original)
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
plt.rcParams['text.usetex'] = True
plt.rc('font', size=16)

rtrunc_means = np.array(rtrunc_means)
rtrunc_stds = np.array(rtrunc_stds)
dtrunc_expecs = np.array(dtrunc_expecs)
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#plt.title("Dtrunc vs. rtrunc on random MPSs. bd = {}, k={}".format(bond_dim, k))
plt.xlabel("bond dim. cutoff")
plt.ylabel("estimate /$\\ \\langle Z_{6}\\rangle$")
#print("Plot things, yerr size = {}, means size = {}".format(len(rtrunc_stds), len(rtrunc_means)))
plt.errorbar(ks, rtrunc_means/orig_expec, yerr=rtrunc_stds/np.abs(orig_expec),
             fmt='o', label="rtrunc mps", capsize=4, color='blue')
plt.plot(ks, dtrunc_expecs/orig_expec, 's', label='dtrunc mps', color='red')
#plt.plot(ks, np.ones(ks.size)*orig_expec, '--', label='true expec.')
xs = np.arange(ks[0]-1,ks[-1]+1,0.1)
plt.plot(ks, np.ones(ks.size), '--', color='black')
plt.legend()
plt.show()