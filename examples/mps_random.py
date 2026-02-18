from mps_helpers import *
import numpy as np
from rtrunc import td_optimizer as tdo

np.random.seed(52)

n = int(input("number of sites: "))
numb = int(input("gamma setting number: "))
gammas = [0.1, 0.2, 0.4, 0.8]

def haar_random_state(d):
    # Draw complex Gaussian entries
    x = np.random.randn(d) + 1j * np.random.randn(d)
    # Normalize
    psi = x / np.linalg.norm(x)
    return psi

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