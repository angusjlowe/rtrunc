import numpy as np


def t_helper(i, C, ws):
    return np.sum(ws[C]**i)

# compute partial weight, dynamic programming
# for fast recursive computation
def partialWeight(k, C, ws):
    if k <= 0:
        return 1
    if k > len(C):
        return 0
    ws_np = np.array(ws, dtype=float)

    # dp_table[i] will store partialWeight(i, C, ws)
    # Initialize with zeros, dp_table[0] = 1.0 for k=0 case
    dp_table = np.zeros(k + 1, dtype=float)
    dp_table[0] = 1.0
    for current_k in range(1, k+1):
        # f = lambda x: (-1)**(x+1) * t_helper(x, C, ws) * partialWeight(k-x, C, ws)
        sum_f_terms = 0.0
        for j in range(1, current_k + 1):
            sum_f_terms += (-1)**(j + 1) * t_helper(j, C, ws_np) * dp_table[current_k - j]
        dp_table[current_k] = (1.0 / current_k) * sum_f_terms
    return dp_table[k]


# get weights in max entropy model from marginal probabilities
def getWeightsFromCoverage(ps, n, k, accuracy='5', maxIter=1000):
    acc = int(accuracy)
    tol = 10**(-acc-1)
    if np.abs(np.sum(ps) - k) > tol:
        print("Not a valid set of coverage probabilities.")
        return None
    ws = ps
    S = [*range(n)]
    for _ in range(maxIter):
        ws_rest_list = []
        common_num_term = partialWeight(k-1, S[1:], ws)
        for j in range(1, n):
            indices_for_denom = S[0:j] + S[j+1:n]
            denom_term = partialWeight(k-1, indices_for_denom, ws)
            if denom_term == 0:
                val = 0
            else:
                val = (ps[j] * common_num_term) / denom_term
            ws_rest_list.append(val)
        wsNew = np.concatenate(([ps[0]], ws_rest_list))
        if np.max(np.abs(np.array(wsNew) - ws)) < tol:
            break
        else:
            ws = wsNew
    return ws

# (assumes zero indexed i) get ith marginal from weights
def computeSingleMarginalFromWeights(i, k, ws):
    n = np.size(ws)
    S = [*range(n)]
    return ws[i]*partialWeight(k-1, S[0:i] + S[i+1:n], ws)/partialWeight(k, S, ws)

# (assumes distinct i,j, zero indexed) get (i,j)th marginal from weights
def computePairMarginalFromWeights(i, j, k, ws):
    if i==j:
        return computeSingleMarginalFromWeights(i,k,ws)
    if i!=j and k<=1:
        return 0 
    else:
        n = np.size(ws)
        C = [*range(n)]
        C.remove(i)
        C.remove(j)
        return ws[i]*ws[j]*partialWeight(k-2, C, ws)/partialWeight(k, [*range(n)], ws)

