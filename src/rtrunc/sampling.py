import numpy as np
from scipy.special import logsumexp

MAX_LOG_EXP = 500

def log_partialWeight(k, C, ws):
    """partialWeight computed in log-space using DP."""
    if k < 0 or k > len(C):
        return -np.inf
    if k == 0:
        return 0.0

    log_ws = np.log(np.clip([ws[i] for i in C], 1e-300, None))

    log_dp = np.full(k + 1, -np.inf)
    log_dp[0] = 0.0  # log(1)

    for log_w in log_ws:
        # Use slicing and vectorized update instead of a loop
        log_dp[1:k+1] = logsumexp(
            np.stack([
                log_dp[1:k+1],        # without current w
                log_dp[0:k] + log_w   # with current w
            ]),
            axis=0
        )

    return log_dp[k]

def getWeightsFromCoverage(ps, k, max_iter=5000, tol=1e-6):
    """
    Computes the vector w for the maximum entropy distribution over subsets of size k,
    given specified marginals p, using computeSingleMarginalFromWeights() to compute marginals.

    Parameters:
    - ps: array-like, shape (N,)
        Desired marginal probabilities (must sum to k).
    - k: int
        Target subset size.
    - max_iter: int
        Maximum number of iterations.
    - tol: float
        Convergence tolerance (L1 norm of marginal error).

    Returns:
    - ws: ndarray, shape (N,)
        The exponential of the dual parameters defining the max-entropy distribution.
    """
    if abs(np.sum(ps) - k) > tol:
        print("Not a valid set of coverage probabilities. sum = {} when k = {}".format(np.sum(ps), k))
        return None
    ps = np.array(ps)
    for j in range(np.size(ps)):
        if ps[j] < tol:
            ps[j] = tol
        if ps[j] > 1-tol:
            ps[j] = 1-tol
    log_ws = np.log(ps.copy())

    for it in range(max_iter):
        if it > max_iter/2 and (it + 1) % 100==0:
            print("On it: {}".format(it+1))
        # Compute expected marginals under current w
        log_ws = np.clip(log_ws, -MAX_LOG_EXP, MAX_LOG_EXP)
        expected_ps = computeSingleMarginalFromWeights(k, np.exp(log_ws))

        # Check convergence
        error = np.linalg.norm(expected_ps - ps, 1)
        if error < tol:
            break

        diag_hessian = (expected_ps*(1-expected_ps))
        for j in range(np.size(ps)):
            if diag_hessian[j] < tol:
                diag_hessian[j] = tol
        
        # Update step (gradient descent on dual)
        log_ws += 1/(diag_hessian + 1e-7) * (ps - expected_ps)

    return np.exp(log_ws)


def computeSingleMarginalFromWeights(k, ws):
    ws = np.array(ws, dtype=float)
    n = ws.size
    S = list(range(n))

    # Compute log Z(S)
    log_ek = log_partialWeight(k, S, ws)

    # Pre-allocate marginals
    p1 = np.zeros(n)

    # Compute single marginals using formula from lemma
    for i in S:
        subset = S.copy()
        subset.remove(i)
        log_ek_minus_1 = log_partialWeight(k - 1, subset, ws)
        log_p = np.log(ws[i]) + log_ek_minus_1 - log_ek
        p1[i] = np.exp(log_p)
    return p1


# (assumes distinct i,j, zero indexed) get (i,j)th marginal from weights
def computePairMarginalFromWeights(k, ws):
    ws = np.array(ws, dtype=float)
    n = ws.size
    S = list(range(n))
    # Compute log e_k(S)
    log_ek = log_partialWeight(k, S, ws)

    # pre-allocate marginals
    p2 = np.zeros((n, n))

    # Compute double marginals
    for i in range(n):
        for j in range(i + 1, n):
            subset = S.copy()
            subset.remove(i)
            subset.remove(j)
            log_ek_minus_2 = log_partialWeight(k - 2, subset, ws)
            log_p = np.log(ws[i]) + np.log(ws[j]) + log_ek_minus_2 - log_ek
            p2[i, j] = p2[j, i] = np.exp(log_p)
    p1 = computeSingleMarginalFromWeights(k, ws)
    for i in range(n):
        p2[i,i] = p1[i]
    return p2