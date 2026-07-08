import logging
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

def getWeightsFromCoverage(ps, k, max_iter=5000, tol=1e-5, init_mode=0,
                           fixed_iters=False, return_error=False,
                           return_history=False):
    """
    Computes weights w for the max-entropy k-DPP with specified marginals ps.

    init_mode=0: initialise log_ws = log(ps)       (default, reliable)
    init_mode=1: initialise log_ws = logit(ps)      (can oscillate when ps near 0/1)
    """
    if return_history and return_error:
        raise ValueError("return_history and return_error are mutually exclusive")
    if abs(np.sum(ps) - k) > tol:
        raise ValueError(
            f"Coverage probabilities must sum to k={k}, got sum={np.sum(ps):.6g}"
        )

    ps = np.array(ps, dtype=float)
    ps = np.clip(ps, tol, 1 - tol)

    if init_mode == 0:
        log_ws = np.log(ps)
    else:
        log_ws = np.log(ps / (1 - ps))
    log_ws -= np.mean(log_ws)

    error = np.inf
    history = [] if return_history else None
    for it in range(max_iter):
        if it > max_iter / 2 and (it + 1) % 100 == 0:
            logging.warning("getWeightsFromCoverage: slow convergence at iteration %d, error=%.7f", it + 1, error)
        log_ws = np.clip(log_ws, -MAX_LOG_EXP, MAX_LOG_EXP)
        expected_ps = computeSingleMarginalFromWeights_fast(k, np.exp(log_ws))

        error = np.linalg.norm(expected_ps - ps, 1)
        if return_history:
            history.append(error)
        if not fixed_iters and error < tol:
            break

        diag_hessian = np.maximum(expected_ps * (1 - expected_ps), 1e-2)
        log_ws += (ps - expected_ps) / diag_hessian

    if return_history:
        return np.exp(log_ws), history
    if return_error:
        return np.exp(log_ws), error
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
        F[i + 1, 1:m + 1] = np.logaddexp(F[i, 1:m + 1], F[i, 0:m] + log_ws[i])

    # B[i, r] = log elementary symmetric sum of degree r
    # using ws[i], ..., ws[n-1]
    B = np.full((n + 1, k + 1), -np.inf)
    B[n, 0] = 0.0

    for i in range(n - 1, -1, -1):
        B[i, 0] = 0.0
        m = min(k, n - i)
        B[i, 1:m + 1] = np.logaddexp(B[i + 1, 1:m + 1], B[i + 1, 0:m] + log_ws[i])

    log_ek = F[n, k]

    # For each i, we need logsumexp(F[i, :k] + B[i+1, k-1::-1]).
    # Stack all n rows into one (n, k) matrix and call logsumexp once.
    combined = F[:n, :k] + B[1:, k - 1::-1]   # shape (n, k)
    p1 = np.exp(log_ws + logsumexp(combined, axis=1) - log_ek)

    return p1


# (assumes distinct i,j, zero indexed) get (i,j)th marginal from weights
def computePairMarginalFromWeights(k, ws):
    ws = np.array(ws, dtype=float)
    n = ws.size
    S = list(range(n))
    log_ek = log_partialWeight(k, S, ws)

    p2 = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            subset = S.copy()
            subset.remove(i)
            subset.remove(j)
            log_ek_minus_2 = log_partialWeight(k - 2, subset, ws)
            log_p = np.log(ws[i]) + np.log(ws[j]) + log_ek_minus_2 - log_ek
            p2[i, j] = p2[j, i] = np.exp(log_p)

    p1 = computeSingleMarginalFromWeights_fast(k, ws)
    for i in range(n):
        p2[i, i] = p1[i]
    return p2


def sampleSubset(ws, ps, n, k, tol=1e-4):
        if len(ws) == 0:
            ws = getWeightsFromCoverage(ps, k)
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