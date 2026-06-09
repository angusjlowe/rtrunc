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
        # Use slicinxg and vectorized update instead of a loop
        log_dp[1:k+1] = logsumexp(
            np.stack([
                log_dp[1:k+1],        # without current w
                log_dp[0:k] + log_w   # with current w
            ]),
            axis=0
        )

    return log_dp[k]

def getWeightsFromCoverage(ps, k, max_iter=5000, tol=1e-5):
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
    log_ws = np.log(ps)

    for it in range(max_iter):
        if it > max_iter/2 and (it + 1) % 100==0:
            print("On it: {:<3} error: {:.7f}".format(it+1, error))
        # Compute expected marginals under current w
        log_ws = np.clip(log_ws, -MAX_LOG_EXP, MAX_LOG_EXP)
        expected_ps = computeSingleMarginalFromWeights_fast(k, np.exp(log_ws))

        # Check convergence
        error = np.linalg.norm(expected_ps - ps, 1)
        if error < tol:
            break

        diag_hessian = (expected_ps*(1-expected_ps)) + tol
        
        # Update step (gradient descent on dual)
        log_ws += 1/(diag_hessian + 1e-10) * (ps - expected_ps)

    return np.exp(log_ws)


def computeSingleMarginalFromWeights_fast(k, ws):
    ws = np.asarray(ws, dtype=float)
    n = ws.size

    if k < 0 or k > n:
        return np.zeros(n)
    if k == 0:
        return np.zeros(n)

    log_ws = np.log(np.clip(ws, 1e-300, None))

    # F[i, r] = log elementary symmetric sum of degree r
    # using ws[0], ..., ws[i-1]
    F = np.full((n + 1, k + 1), -np.inf)
    F[0, 0] = 0.0

    for i in range(n):
        F[i + 1, 0] = 0.0
        m = min(k, i + 1)
        for r in range(1, m + 1):
            F[i + 1, r] = np.logaddexp(F[i, r], F[i, r - 1] + log_ws[i])

    # B[i, r] = log elementary symmetric sum of degree r
    # using ws[i], ..., ws[n-1]
    B = np.full((n + 1, k + 1), -np.inf)
    B[n, 0] = 0.0

    for i in range(n - 1, -1, -1):
        B[i, 0] = 0.0
        m = min(k, n - i)
        for r in range(1, m + 1):
            B[i, r] = np.logaddexp(B[i + 1, r], B[i + 1, r - 1] + log_ws[i])

    log_ek = F[n, k]

    p1 = np.zeros(n)
    for i in range(n):
        terms = F[i, :k] + B[i + 1, k - 1::-1]
        log_ek_minus_1_excluding_i = logsumexp(terms)
        p1[i] = np.exp(log_ws[i] + log_ek_minus_1_excluding_i - log_ek)

    return p1


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


def elementary_symmetric(weights, max_degree):
    """
    e[m] = sum_{|B|=m} prod_{i in B} weights[i].
    """
    e = np.zeros(max_degree + 1, dtype=float)
    e[0] = 1.0
    for w in weights:
        # update backwards
        for m in range(max_degree, 0, -1):
            e[m] += w * e[m - 1]
    return e

def R_without_j(ws, C, j, degree):
    """
    Compute R(degree, C \\ {j}).
    """
    C_without_j = [i for i in C if i != j]
    if degree < 0:
        return 0.0
    if degree == 0:
        return 1.0
    if degree > len(C_without_j):
        return 0.0
    e = elementary_symmetric(ws[C_without_j], degree)
    return e[degree]

def sampleSubset(ws, ps, n, k, tol=1e-4):
        if list(ws) == []:
            ws = getWeightsFromCoverage(ps, k)
        ws = ws
        #ws = np.clip(np.array(ws, dtype=float), tol, None)
        S = list(range(n))
        A = []

        q_prev = ps / k
        q_prev = np.clip(q_prev, 0, None)
        q_prev /= np.sum(q_prev)
        i1 = np.random.choice(S, size=1, p=q_prev)[0]
        A.append(i1)

        for el in range(1, k):
            Sel = [j for j in S if j not in A]
            i_prev = A[-1]
            q_el = np.zeros(n)
            denom_eps = tol

            for j in Sel:
                num = ws[i_prev] * q_prev[j] - ws[j] * q_prev[i_prev]
                denom = (k - el) * (ws[i_prev] - ws[j]) * q_prev[i_prev]

                if abs(denom) < denom_eps or q_prev[i_prev] == 0:
                    q_el[j] = 0.0
                else:
                    q_el[j] = max(num / denom, 0.0)

            total = np.sum(q_el)

            if total < tol or np.isnan(total):
                remaining = [j for j in Sel if j not in A]
                if not remaining:
                    break
                ik = np.random.choice(remaining)
                q_prev = ps / k
            else:
                q_el /= total
                ik = np.random.choice(S, size=1, p=q_el)[0]
                q_prev = q_el

            A.append(int(ik))

        return A