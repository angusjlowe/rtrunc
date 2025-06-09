import numpy as np

def traceDistance(rho, sigma):
    return np.linalg.norm(rho-sigma, ord='nuc')/2.

# take a numpy array as input and compute the top-k norm
def topKNorm(k, v):
    sortedV = -np.sort(-np.abs(v))
    return np.sqrt(np.sum(sortedV[:k]**2))

# compute k-supp norm
def kSuppNorm(k, vs):
    r = rCompute(vs, k)
    return np.sqrt(np.sum(vs[:k-r-1]**2) + pSum(k-r-1,vs)**2/(r+1))

# get partial sum from index j to end of vector (zero-indexed)
def pSum(j, vs):
    n = np.size(vs)
    res = 0
    if j in [*range(n)]:
        res = np.sum(vs[j:])
    return res

# (for the next 4 definitions: assumes sorted, nonnegative vs!)
# upper bound boolean for determining r value
def upperTest(vs, k, r):
    val = False
    if r == k-1:
        val = True
    else:
        val = vs[k-r-2] > pSum(k-r-1, vs)/(r+1)
    return val

# lower bound boolean for determining r value
def lowerTest(vs, k, r):
    val =  pSum(k-r-1, vs)/(r+1) >= vs[k-r-1]
    return val

# compute the unique r value in k-supp norm computation
def rCompute(vs, k):
    rs = np.array([*range(0, k)])
    filter = list(map(lambda r: upperTest(vs, k ,r) and lowerTest(vs, k, r), rs))
    res = rs[filter]
    if len(res)!=1:
        print("Note: something went wrong computing the unique integer r.")
    return res[0]
