import numpy as np
from scipy.special import logsumexp
import time



def log_partialWeight(k, C, ws):
    """Fast log-space dynamic programming for partialWeight."""
    if k < 0 or k > len(C):
        return -np.inf
    if k == 0:
        return 0.0

    log_ws = np.log(np.clip([ws[i] for i in C], 1e-300, None))
    #m = len(log_ws)

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

def getWeightsFromCoverage(ps, n, k, max_iter=1000, tol=1e-6):
    """
    Computes the vector w for the maximum entropy distribution over subsets of size k,
    given specified marginals p, using function to compute marginals.

    Parameters:
    - p: array-like, shape (N,)
        Desired marginal probabilities (must sum to k).
    - k: int
        Target subset size.
    - computeSingleMarginalFromWeights: function
        Function taking (k, w) and returning a vector of estimated marginals.
    - max_iter: int
        Maximum number of iterations.
    - tol: float
        Convergence tolerance (L1 norm of marginal error).

    Returns:
    - ws: ndarray, shape (N,)
        The exponential of the dual parameters defining the MaxEnt distribution.
    """
    if abs(np.sum(ps) - k) > tol:
        print("Not a valid set of coverage probabilities.")
        return None
    ps = np.array(ps)
    log_ws = np.log(ps.copy())

    for it in range(max_iter):
        if it > max_iter/2:
            print("On it: {}".format(it+1))
        # Compute expected marginals under current w
        expected_ps = computeSingleMarginalFromWeights(k, np.exp(log_ws))

        # Check convergence
        error = np.linalg.norm(expected_ps - ps, 1)
        if error < tol:
            break

        # Update step (gradient ascent on dual)
        log_ws += 1/(expected_ps*(1-expected_ps)) * (ps - expected_ps)

    return np.exp(log_ws)


def getWeightsFromCoverage2(ps, n, k, accuracy='3', maxIter=1000):
    MAX_LOG_EXP = 500  # safe limit for exp to avoid inf
    acc = int(accuracy)
    tol = 10**(-acc)
    if abs(np.sum(ps) - k) > tol:
        print("Not a valid set of coverage probabilities.")
        return None

    ws = np.clip(np.array(ps, dtype=float), 1e-8, None)
    #ws = np.ones(ps.size)
    S = list(range(n))

    for it in range(maxIter):
        try:
            log_common_num = log_partialWeight(k - 1, S[1:], ws)
        except:
            print(f"Iteration {it}: Failed to compute log_common_num")
            return None
        log_ws_rest = []
        for j in range(1, n):
            Sel = S[:j] + S[j+1:]
            log_denom = log_partialWeight(k - 1, Sel, ws)

            if not np.isfinite(log_denom) or ps[j] < 0:
                log_ws_j = -np.inf
            else:
                log_ws_j = np.log(ps[j]) + log_common_num - log_denom

            log_ws_rest.append(log_ws_j)

        # Clamp log values and exponentiate safely
        log_ws_rest = np.array(log_ws_rest)
        log_ws_rest = np.clip(log_ws_rest, -MAX_LOG_EXP, MAX_LOG_EXP)
        ws_rest = np.exp(log_ws_rest)
        
        
        # full weights vector
        ws_new = np.concatenate(([ps[0]], ws_rest))

        # normalize and clip
        ws_new /= np.sum(ws_new)
        ws_new = np.clip(ws_new, 1e-8, None)
        ws_new *= k  # optional: match total marginal mass

        diff = np.max(np.abs(ws_new - ws))
        if diff < tol:
            return ws_new
        
        if it > maxIter/2:
            print("iter: {}, diff: {:.3f}".format(it, diff))
        
        ws = ws_new

    print("Warning: maxIter reached without convergence")
    return ws


def computeSingleMarginalFromWeights(k, ws):
    ws = np.array(ws, dtype=float)
    n = ws.size
    S = list(range(n))
    # Compute log e_k(S)
    log_ek = log_partialWeight(k, S, ws)

    # Pre-allocate marginals
    p1 = np.zeros(n)

    # Compute single marginals
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