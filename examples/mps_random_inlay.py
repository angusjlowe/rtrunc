from mps_helpers import *
import numpy as np
from rtrunc import td_optimizer as tdo

np.random.seed(44)

def rtrunc(tensors, k, l):
    """"
    Get random k-truncation at lth site. Expects l
    between 1 and n-1.
    """
    new_tensors = mixed_canonical_form(tensors, l)
    schmidts = np.diag(new_tensors[l]).copy()
    cut = 1e-12  # MPS tolerance to handle near-product case
    schmidts[schmidts < cut] = 0.0
    if k < schmidts.size:
        newTDOpt = tdo.TDOptimizer(k, schmidts)
        m,td = newTDOpt.getOptimalTDMeas()
        if td < 1e-12:
            new_schmidts = np.concatenate((schmidts[:k], np.zeros(schmidts.size-k)))
        else:
            new_schmidts = newTDOpt.sampleOptimalTDState()
    else:
        new_schmidts = schmidts
    new_schmidts[new_schmidts < cut] = 0.0
    new_tensors[l] = np.diag(new_schmidts)
    new_tensors = get_mps_tensors_from_canonical(new_tensors)
    return new_tensors


def dtrunc(tensors, k, l):
    """"
    Get deterministic k-truncation at lth site. Expects l
    between 1 and n-1.
    """
    new_tensors = mixed_canonical_form(tensors, l)
    #print("Just to be safe: (l+1)th tensor has shape: {}".format(new_tensors[l].shape))
    schmidts = np.diag(new_tensors[l]).copy()
    cut = 1e-12  # MPS tolerance to handle near-product case
    schmidts[schmidts < cut] = 0.0
    if k < schmidts.size:
        new_schmidts = np.concatenate((schmidts[:k], np.zeros(schmidts.size-k)))
    else:
        new_schmidts = schmidts
    new_schmidts = new_schmidts/np.linalg.norm(new_schmidts + 1e-16)
    new_schmidts[new_schmidts < cut] = 0.0
    #print(schmidts, new_schmidts)
    #print(np.linalg.norm(schmidts), np.linalg.norm(new_schmidts))
    new_tensors[l] = np.diag(new_schmidts)
    new_tensors = get_mps_tensors_from_canonical(new_tensors)
    return new_tensors


def haar_random_state(d):
    # Draw complex Gaussian entries
    x = np.random.randn(d) + 1j * np.random.randn(d)
    # Normalize
    psi = x / np.linalg.norm(x)
    return psi

# parameter setup
n = 9
d = 2
bond_dim = 2**(n//2)
gamma = 0.8
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
obs[int(n/2)] = Z
#obs[4] = Z
#obs[8] = X

# random mps setup
psi = haar_random_state(d**n)
psi_tensors_original = tensor_to_fixedbond_mps(psi, d, n, bond_dim)
psi_tensors_original = power_law_schmidt_coeffs(psi_tensors_original, gamma)

psi_tensors_original = normalize(psi_tensors_original)
psi_can = mixed_canonical_form(psi_tensors_original, n//2)

schmidts = np.diag(psi_can[n//2])

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('font', size=14)
edges = list(np.arange(bond_dim+1)+0.5)
plt.xticks(list(np.arange(2,bond_dim+1,2)))
plt.xlim(0.6,2**(n//2)+0.4)
#plt.yticks([])
plt.stairs(schmidts**2, color='black', linewidth=2, fill=False, edges=edges)
plt.savefig("inlay_4.pdf")
