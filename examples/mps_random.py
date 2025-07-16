import numpy as np
from rtrunc import td_optimizer as tdo
from scipy.stats import unitary_group
import copy

def mps_expec(tensors1, tensors2, obs_tensors):
    n = len(tensors1)
    if n != len(tensors2) or n!=len(obs_tensors):
        raise ValueError("Mismatch in number of sites.")

    # Start contraction with left boundary
    A = tensors1[0]           # shape (d, D)
    B = obs_tensors[0]        # shape (d, d)  
    C = np.conj(tensors2[0])  # shape (d, D)
    contraction = np.einsum('ia,ij,jb->ab', A, B, C)

    # Contract bulk tensors
    for j in range(1, n-1):
        A = tensors1[j]           # shape (D, d, D)
        B = obs_tensors[j]        # shape (d, d)
        C = np.conj(tensors2[j])  # shape (D, d, D)

        # Contract left edge
        contraction = np.einsum('ab,aic,ij,bjd->cd', contraction, A, B, C)
    # Final tensors
    A = tensors1[-1]          # shape (D, d)
    B = obs_tensors[-1]       # shape (d, d)
    C = np.conj(tensors2[-1]) # shape (D, d)

    # Contract with last site
    contraction = np.einsum('ab,ai,ij,bj->', contraction, A, B, C)

    return contraction
    
def mps_inner_prod(tensors1, tensors2):
    n = len(tensors1)
    if n != len(tensors2):
        raise ValueError("Mismatch in number of sites.")
    d = tensors1[0].shape[0]
    I = np.eye(d)
    obs_tensors = [I] * n
    return mps_expec(tensors1, tensors2, obs_tensors)

def right_canonical_form(tensors):
    n = len(tensors)
    d = tensors[0].shape[0]
    new_tensors = copy.deepcopy(tensors)

    # last tensor
    matrix = new_tensors[n-1]
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    S = np.diag(s)
    new_tensors[n-1] = Vh.reshape(-1,d)
    new_tensors[n-2] = np.einsum('aib,bc,cd->aid', new_tensors[n-2], U, S)

    # right sweep
    for j in range(n-2, 1, -1):
        dims = new_tensors[j].shape
        tensor = new_tensors[j]
        matrix = tensor.reshape(dims[0],-1)
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
        S = np.diag(s)
        new_tensors[j] = Vh.reshape(-1, d, dims[2])
        new_tensors[j-1] = np.einsum('aib,bc,cd->aid', new_tensors[j-1], U, S)
    
    # last tensor in right sweep
    dims = new_tensors[1].shape
    tensor = new_tensors[1]
    matrix = tensor.reshape(dims[0],-1)
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    S = np.diag(s)
    new_tensors[1] = Vh.reshape(-1, d, dims[2])
    new_tensors[0] = np.einsum('ia,ab,bc->ic', new_tensors[0], U, S)
    return new_tensors


def mixed_canonical_form(tensors, l):
    """
    Brings an MPS into mixed canonical form ending at the lth site.
    Equivalent to Schmidt decomposition being at lth cut.

    Assume l is between 1 and n-1
    """
    n = len(tensors)
    d = tensors[0].shape[0]
    new_tensors = right_canonical_form(tensors)
    # first tensor
    if l > 1:
        matrix = new_tensors[0]
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
        S = np.diag(s)
        new_tensors[0] = U
        new_tensors[1] = np.einsum('ab,bc,cid->aid', S, Vh, new_tensors[1])

        # second to (l-1)th tensor
        for j in range(1, l-1):
            dims = new_tensors[j].shape
            tensor = new_tensors[j]
            matrix = tensor.reshape(-1, dims[-1])
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
            S = np.diag(s)
            new_tensors[j] = U.reshape(dims[0], d, -1)
            new_tensors[j+1] = np.einsum('ab,bc,cid->aid', S, Vh, new_tensors[j+1])

    # lth tensor
    dims = new_tensors[l-1].shape
    tensor = new_tensors[l-1]
    matrix = tensor.reshape(-1, dims[-1])
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    #print(U.conj().T @ U)
    #print(s)
    Sl = np.diag(s)
    if l!=1:
        new_tensors[l-1] = U.reshape(dims[0], d, -1)
    else:
        new_tensors[l-1] = U
    if l!=n-1:
        new_tensors[l] = np.einsum('ab,bic->aic', Vh, new_tensors[l])
    else:
        new_tensors[l] = np.einsum('ab,bi->ai', Vh, new_tensors[l])


    # output has n+1 tensors, since there is Schmidt diagonal matrix in middle
    output = new_tensors[:l] + [Sl] + new_tensors[l:]
    #print("Shapes are {}".format([x.shape for x in output]))
    return output

def get_mps_tensors_from_canonical(tensors):
    n = len(tensors)
    tensors_copy = copy.deepcopy(tensors)
    new_tensors = [tensors_copy[0]]
    for j in range(1, n-2):
        if tensors_copy[j].ndim == 2:
            tensors_copy[j+1] = np.einsum('ab,bic->aic', tensors_copy[j], tensors_copy[j+1])
        else:
            new_tensors.append(tensors_copy[j])
    if tensors_copy[n-2].ndim == 2:
        tensor = np.einsum('ab,bi->ai', tensors_copy[n-2], tensors_copy[n-1])
        new_tensors.append(tensor)
    if tensors_copy[n-2].ndim == 3:
        new_tensors.append(tensors_copy[n-2])
        new_tensors.append(tensors_copy[n-1])
    return new_tensors


def get_vector_from_mps(tensors):
    """
    Contracts an MPS (possibly with a diagonal matrix in the middle)
    into a full state vector.
    
    Supports both standard MPS and mixed-canonical form with
    a 2D matrix (e.g., Schmidt coefficients) somewhere in the list.
    """
    n = len(tensors)
    tensor = tensors[0]
    for j in range(1, n):
        next_tensor = tensors[j]
        if next_tensor.ndim == 2:
            # No physical index: contract virtual legs directly
            tensor = np.einsum('ia,ab->ib', tensor, next_tensor)
        elif next_tensor.ndim == 3:
            # Standard MPS tensor: contract physical and virtual legs
            tensor = np.einsum('ia,ajb->ijb', tensor, next_tensor)
            tensor = tensor.reshape(-1, tensor.shape[-1])
        else:
            raise ValueError(f"Unexpected tensor rank {next_tensor.ndim} at position {j}")
    
    # Final tensor should now be shape (k, 1)
    return tensor.reshape(-1)


def entanglement_spectrum(psi, j, n, tol=1e-12):
    assert 0 < j < n, "j must be between 1 and n-1"
    dimL = 2**j
    dimR = 2**(n - j)

    # Reshape into bipartite matrix
    psi_matrix = psi.reshape((dimL, dimR))

    # Singular values of the bipartition
    s = np.linalg.svd(psi_matrix, compute_uv=False)

    # Squared singular values = eigenvalues of reduced density matrix
    spectrum = s**2

    # Discard tiny values (numerical noise)
    spectrum = spectrum[spectrum > tol]

    return spectrum

def get_random_mps(n,d,bond_dim):
    # create random mps
    tensors = []
    for j in range(n):
        if j == 0:
            Areal = np.random.rand(d, bond_dim)-1
            Aimag = np.random.rand(d, bond_dim)-1
        if j == n-1:
            Areal = np.random.rand(bond_dim, d)-1
            Aimag = np.random.rand(bond_dim, d)-1
        if j != 0 and j != n-1:
            Areal = np.random.rand(bond_dim, d, bond_dim)-1
            Aimag = np.random.rand(bond_dim, d, bond_dim)-1
        tensors.append(Areal + 1j * Aimag)

    # normalize
    norm = np.sqrt(mps_inner_prod(tensors, tensors))
    for j in range(n):
        tensors[j] = tensors[j] / norm**(1/n)
    return tensors

def power_law_schmidt_coeffs(tensors, gamma):
    n = len(tensors)
    psi_tensors = copy.deepcopy(tensors)
    for l in range(1,n-1):
        psi_can = mixed_canonical_form(psi_tensors, l)
        r = psi_can[l].shape[0]
        xs = np.arange(1,r+1)
        new_schmidts = xs**(-gamma)
        new_schmidts = new_schmidts/np.linalg.norm(new_schmidts)
        psi_can[l] = np.diag(new_schmidts)
        psi_tensors = get_mps_tensors_from_canonical(psi_can)
    return psi_tensors

def dtrunc(tensors, k, l):
    """"
    Get deterministic k-truncation at lth site. Expects l
    between 1 and n-1. Assumes schmidt values already sorted.
    """
    new_tensors = mixed_canonical_form(tensors, l)
    #print("Just to be safe: (l+1)th tensor has shape: {}".format(new_tensors[l].shape))
    schmidts = np.diag(new_tensors[l])
    if k < schmidts.size:
        new_schmidts = np.concatenate((schmidts[:k], np.zeros(schmidts.size-k)))
    else:
        new_schmidts = schmidts
    new_schmidts = new_schmidts/np.linalg.norm(new_schmidts)
    #print(schmidts, new_schmidts)
    #print(np.linalg.norm(schmidts), np.linalg.norm(new_schmidts))
    new_tensors[l] = np.diag(new_schmidts)
    new_tensors = get_mps_tensors_from_canonical(new_tensors)
    return new_tensors



    

n=10
d=2
l=5
bond_dim = 60
tensors = get_random_mps(n,d,bond_dim)
tensors = power_law_schmidt_coeffs(tensors,0.2)
# check if resulting vector has unit norm
psi = get_vector_from_mps(tensors)
print("norm is {:.3f}".format(np.linalg.norm(psi)))

# check right canonical form
print("\nChecking right canonical form...\n")
psi_original = get_vector_from_mps(tensors)
right = right_canonical_form(tensors)
psi_right = get_vector_from_mps(right)
print("Norm preserved?", np.allclose(np.linalg.norm(psi_original),np.linalg.norm(psi_right),1))
right_mps = get_mps_tensors_from_canonical(right)
print("equal?", np.allclose(1., mps_inner_prod(tensors, right_mps)))

# check if mixed_canonical_form preserves norm
print("\nChecking mixed canonical form...\n")
psi_original = get_vector_from_mps(tensors)
mixed = mixed_canonical_form(tensors, l)
psi_mixed = get_vector_from_mps(mixed)
print("Norm preserved?", np.allclose(np.linalg.norm(psi_original), np.linalg.norm(psi_mixed), 1))
new_mps = get_mps_tensors_from_canonical(mixed)
print("equal?", np.allclose(1., mps_inner_prod(tensors, new_mps)))


# check if mixed canonical form has normalized schmidts
print("\nChecking normalization of Schmidt coefficients...\n")
Sl = mixed[l]
print("Dimensions of (l+1)th tensor is {}".format(Sl.shape))
print("Norm is: {:.3f}".format(np.linalg.norm(Sl)))


# Check multiple truncations
print("\nChecking multiple truncations...\n")
psi_tensors_original = get_random_mps(n,d,bond_dim)
print("Setting power law Schmidt coefficients...")
psi_tensors_original = power_law_schmidt_coeffs(psi_tensors_original, 1.2)
psi_tensors = copy.deepcopy(psi_tensors_original)
k = 10
for l in range(1,n):
    psi_tensors = dtrunc(psi_tensors, k, l)
    overlap_sq = np.abs(mps_inner_prod(psi_tensors, psi_tensors_original))**2
    td = np.sqrt(max(1-overlap_sq,0))
    print("td is {:.5f}".format(td))

print("\nFinal TD is...\n")
overlap_sq = np.abs(mps_inner_prod(psi_tensors, psi_tensors_original))**2
td = np.sqrt(1-overlap_sq)
print("TD with k={} is {:.5f}".format(k, td))

# observable expectation
print("\nObservable expectations...\n")
Z = np.array([[1,0],[0,-1]], dtype=float)
X = np.array([[0,1],[1,0]], dtype=float)
Zs = [Z] * n
Xs = [X] * n
orig_expec = np.real(mps_expec(psi_tensors_original, psi_tensors_original, Xs))
trunc_expec = np.real(mps_expec(psi_tensors, psi_tensors, Xs))
print("Original expec. = {:.5f}".format(orig_expec))
print("Truncated MPS expec. = {:.5f}".format(trunc_expec))


# what happens with the optimal k-incoherent density matrix?

    
