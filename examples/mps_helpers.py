import numpy as np
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
    norm = np.linalg.norm(s)
    s_normalized = s / (norm + 1e-16)
    S = np.diag(s_normalized)
    new_tensors[n-1] = Vh.reshape(-1,d)
    new_tensors[n-2] = norm*np.einsum('aib,bc,cd->aid', new_tensors[n-2], U, S)

    # right sweep
    for j in range(n-2, 1, -1):
        dims = new_tensors[j].shape
        tensor = new_tensors[j]
        matrix = tensor.reshape(dims[0],-1)
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
        norm = np.linalg.norm(s)
        s_normalized = s / (norm + 1e-16)
        S = np.diag(s_normalized)
        new_tensors[j] = Vh.reshape(-1, d, dims[2])
        US_matrix = np.einsum('bc,cd->bd', U, S)
        new_tensors[j-1] = norm*np.einsum('aib,bd->aid', new_tensors[j-1], US_matrix)
    
    # last tensor in right sweep
    dims = new_tensors[1].shape
    tensor = new_tensors[1]
    matrix = tensor.reshape(dims[0],-1)
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    norm = np.linalg.norm(s)
    s_normalized = s / (norm + 1e-16)
    S = np.diag(s)
    new_tensors[1] = Vh.reshape(-1, d, dims[2])
    new_tensors[0] = norm*np.einsum('ia,ab,bc->ic', new_tensors[0], U, S)
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
        norm = np.linalg.norm(s)
        s_normalized = s / (norm + 1e-16)
        S = np.diag(s_normalized)
        new_tensors[0] = U
        new_tensors[1] = norm*np.einsum('ab,bc,cid->aid', S, Vh, new_tensors[1])

        # second to (l-1)th tensor
        for j in range(1, l-1):
            dims = new_tensors[j].shape
            tensor = new_tensors[j]
            matrix = tensor.reshape(-1, dims[-1])
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
            norm = np.linalg.norm(s)
            s_normalized = s / (norm + 1e-16)
            S = np.diag(s_normalized)
            new_tensors[j] = U.reshape(dims[0], d, -1)
            new_tensors[j+1] = norm*np.einsum('ab,bc,cid->aid', S, Vh, new_tensors[j+1])

    # lth tensor
    dims = new_tensors[l-1].shape
    tensor = new_tensors[l-1]
    matrix = tensor.reshape(-1, dims[-1])
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    norm = np.linalg.norm(s)
    s_normalized = s / (norm + 1e-16)
    Sl = np.diag(s_normalized)
    if l!=1:
        new_tensors[l-1] = U.reshape(dims[0], d, -1)
    else:
        new_tensors[l-1] = U
    if l!=n-1:
        new_tensors[l] = norm*np.einsum('ab,bic->aic', Vh, new_tensors[l])
    else:
        new_tensors[l] = norm*np.einsum('ab,bi->ai', Vh, new_tensors[l])


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

def normalize(tensors):
    new_tensors = []
    norm = np.sqrt(mps_inner_prod(tensors, tensors))
    n = len(tensors)
    for j in range(n):
        new_tensors.append(tensors[j] / norm**(1/n))
    return new_tensors

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
    for l in range(1,n):
        psi_can = mixed_canonical_form(psi_tensors, l)
        r = psi_can[l].shape[0]
        xs = np.arange(1,r+1)
        new_schmidts = xs**(-gamma)
        new_schmidts = new_schmidts/np.linalg.norm(new_schmidts)
        psi_can[l] = np.diag(new_schmidts)
        psi_tensors = get_mps_tensors_from_canonical(psi_can)
    return psi_tensors


def tensor_to_fixedbond_mps(psi, d, n, chi=None):
    """
    Convert a quantum state into an open-boundary MPS with uniform bond dimension.

    Parameters
    ----------
    psi : np.ndarray
        State vector of length d**n (or shape (d,)*n).
    d : int
        Physical dimension at each site.
    n : int
        Number of sites.
    chi : int or None
        Desired bond dimension. If None, use chi = d**(n//2).

    Returns
    -------
    mps : list of np.ndarray
        List of tensors [A0, A1, ..., A_{n-1}] with shapes:
          A0: (d, chi)
          Ai: (chi, d, chi) for 1 <= i <= n-2
          A_{n-1}: (chi, d)
    """
    psi = np.asarray(psi)
    if psi.ndim == 1:
        psi = psi.reshape((d,) * n)
    elif psi.shape != (d,) * n:
        raise ValueError(f"psi must have shape {(d,)*n} or length d**n = {d**n}")

    if chi is None:
        chi = d ** (n // 2)
    chi = int(chi)

    mps = []
    tensor = psi
    left_dim = 1  # leftmost virtual leg initially 1-dim

    for site in range(n - 1):
        # reshape tensor to matrix: (left_dim * d, remaining)
        tensor = tensor.reshape(left_dim * d, -1)
        U, S, Vh = np.linalg.svd(tensor, full_matrices=False)

        # truncate/pad to fixed bond dimension chi
        U = U[:, :min(chi, U.shape[1])]
        S = S[:min(chi, len(S))]
        Vh = Vh[:min(chi, Vh.shape[0])]

        # pad up to chi if needed
        if U.shape[1] < chi:
            pad = chi - U.shape[1]
            U = np.pad(U, ((0, 0), (0, pad)), mode='constant')
            S = np.pad(S, (0, pad), mode='constant')
            Vh = np.pad(Vh, ((0, pad), (0, 0)), mode='constant')

        # reshape U to (left_dim, d, chi)
        A = U.reshape(left_dim, d, chi)
        # remove singleton leftmost index at first site
        if site == 0:
            A = A[0, :, :]  # shape (d, chi)
        mps.append(A)

        # propagate remainder
        tensor = np.diag(S) @ Vh
        left_dim = chi

    # last site tensor: shape (chi, d)
    A_last = tensor.reshape(chi, d)
    mps.append(A_last)

    return mps