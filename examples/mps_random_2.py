from mps_helpers import *
import numpy as np
from rtrunc import td_optimizer as tdo

def rtrunc(tensors, k, l, n_samples=1):
    """"
    Get random k-truncation at lth site. Expects l
    between 1 and n-1. Assumes schmidt values already sorted.
    """
    new_tensors = mixed_canonical_form(tensors, l)
    schmidts = np.diag(new_tensors[l])
    new_schmidtss = []
    if k < schmidts.size:
        newTDOpt = tdo.TDOptimizer(k, schmidts)
        m,td = newTDOpt.getOptimalTDMeas()
        for samp in range(n_samples):
            new_schmidts = newTDOpt.sampleOptimalTDState()
            new_schmidtss.append(new_schmidts)
            if samp % 10 == 0:
                print("Collected {} samples".format(samp))
    else:
        new_schmidtss = np.array([list(schmidts)] * n_samples)
    new_tensorss = []
    for j in range(n_samples):
        new_tensorsj = copy.deepcopy(new_tensors)
        new_tensorsj[l] = np.diag(new_schmidtss[j])
        new_tensorsj = get_mps_tensors_from_canonical(new_tensorsj)
        new_tensorss.append(new_tensorsj)
    return new_tensorss

# parameter setup
d=2
bond_dim = 50
k = 25
gamma = 0.2
n_samples = 100

Z = np.array([[1,0],[0,-1]], dtype=float)
X = np.array([[0,1],[1,0]], dtype=float)
I = np.array([[1,0],[0,1]], dtype=float)

# what happens with the optimal k-incoherent density matrix?
ns = [*range(10,11)]
n_random_tensors = 15
rtrunc_errss = []
dtrunc_errss = []
for n in ns:
    print("\n n = {} sites".format(n))
    rtrunc_means = []
    rtrunc_stds = []
    dtrunc_expecs = []
    orig_expecs = []
    for i in range(n_random_tensors):
        print("Observable no. {}".format(i+1))
        # observable setup
        Zs = [Z] * n
        Xs = [X] * n
        Is = [I] * n
        obs = Is
        obs[0] = X
        obs[-1] = X
        #obs[int(n/2)] = X
        print("Computing dtrunc state")

        for _ in range(100):
            # random mps setup
            psi_tensors_original = get_random_mps(n,d,bond_dim)
            psi_tensors_original = power_law_schmidt_coeffs(psi_tensors_original, gamma)

            # original expecs
            orig_expec = np.real(mps_expec(psi_tensors_original, psi_tensors_original, obs))

            # dtrunc state computation
            psi_tensors = copy.deepcopy(psi_tensors_original)
            for l in range(1,n):
                psi_tensors = dtrunc(psi_tensors, k, l)
            # dtrunc expecs
            dtrunc_expec = np.real(mps_expec(psi_tensors, psi_tensors, obs))
            # look for a case where the relative error is reasonable
            rel_error = np.abs(dtrunc_expec - orig_expec)/np.abs(orig_expec)
            if rel_error > 0.1 or n < 10:
                print("Instance found. Additive error: {:.5f}".format(np.abs(dtrunc_expec-orig_expec)))
                break

        orig_expecs.append(orig_expec)
        dtrunc_expecs.append(dtrunc_expec)


        # rtrunc states and expecs
        print("Computing rtrunc states")
        rtrunc_expecs = []
        psi_tensors = copy.deepcopy(psi_tensors_original)
        for l in range(1,int(n/2)):
            psi_tensors = dtrunc(psi_tensors, k, l)
        rtrunc_tensorss = rtrunc(psi_tensors, k, int(n/2), n_samples=n_samples)
        for j in range(len(rtrunc_tensorss)):
            rtrunc_tensors = rtrunc_tensorss[j]
            if (j+1)%10==0:
                print("Computing expectation with sample: {}".format(j+1))
            psi_tensorsj = copy.deepcopy(rtrunc_tensors)
            for l in range(int(n/2)+1, n):
                psi_tensorsj = dtrunc(psi_tensorsj, k, l)
            trunc_expec = np.real(mps_expec(psi_tensorsj, psi_tensorsj, obs))
            rtrunc_expecs.append(trunc_expec)
        rtrunc_mean = np.mean(rtrunc_expecs)
        rtrunc_std = np.std(rtrunc_expecs, ddof=1)/np.sqrt(n_samples)
        rtrunc_means.append(rtrunc_mean)
        rtrunc_stds.append(rtrunc_std)
    rtrunc_means = np.array(rtrunc_means)
    dtrunc_expecs = np.array(dtrunc_expecs)
    orig_expecs = np.array(orig_expecs)
    rtrunc_errs = np.abs(rtrunc_means-orig_expecs)/np.abs(orig_expecs)
    dtrunc_errs = np.abs(dtrunc_expecs-orig_expecs)/np.abs(orig_expecs)
    rtrunc_errss.append(rtrunc_errs)
    dtrunc_errss.append(dtrunc_errs)

#dtrunc_data = np.array(dtrunc_errss).T
#rtrunc_data = np.array(rtrunc_errss).T
#rtrunc_err_means = np.array([np.mean(x) for x in rtrunc_errss])
#dtrunc_err_means = np.array([np.mean(x) for x in dtrunc_errss])
#rtrunc_err_stds = np.array([np.std(x)/np.sqrt(len(x)) for x in rtrunc_errss])
#dtrunc_err_stds = np.array([np.std(x)/np.sqrt(len(x)) for x in dtrunc_errss])

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title("Dtrunc vs. rtrunc on random MPSs. bd = {}, k={}".format(bond_dim, k))
plt.xlabel("1=dtrunc, 2=rtrunc")
plt.ylabel("rel. error for $\\langle X_{\\lfloor n/2 \\rfloor}\\rangle$")
print("Plot things, yerr size = {}, means size = {}".format(len(rtrunc_stds), len(rtrunc_means)))
for j in range(len(ns)):
    n = ns[j]
    rtrunc_errs = rtrunc_errss[j]
    drtrunc_errs = dtrunc_errss[j]
    err_data = np.array([dtrunc_errs,rtrunc_errs]).T
    plt.violinplot(err_data)
print("Plotted")
#plt.legend()
plt.show()