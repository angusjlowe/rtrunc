import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import itertools

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
    psum = pSum(k-r-1, vs)/(r+1)
    vRhs = vs[k-r-1]
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

# compute k-supp norm
def kSuppNorm(k, vs):
    r = rCompute(vs, k)
    return np.sqrt(np.sum(vs[:k-r-1]**2) + pSum(k-r-1,vs)**2/(r+1))

# the partial weight function R(k, C) from the 1994 paper by Chen et al.
def partialWeight(k, C, ws):
    subsets = [*itertools.combinations(C, k)]
    res = 0
    S = [*range(np.size(ws))]
    for B in subsets:
        filter = list(map(lambda j: j in B, S))
        res += np.prod(ws[filter])
    return res

# get weights in max entropy model from marginal probabilities
def getWeightsFromCoverage(ps, n, k, accuracy='3', maxIter=300):
    acc = int(accuracy)
    if np.abs(np.sum(ps) - k) > 0.01:
        print("Not a valid set of coverage probabilities.")
        return None
    ws = ps
    S = [*range(n)]
    for _ in range(maxIter):
        wsRest = [(ps[j] * partialWeight(k-1, S[1:], ws))
                  /partialWeight(k-1,S[0:j] + S[j+1:n], ws) for j in range(1,n)]
        wsNew = np.concatenate(([ps[0]], wsRest))
        if np.max(np.abs(wsNew - ws)) < 10**(-acc-1):
            break
        else:
            ws = wsNew
    return ws

# get pmf over subsets from weights
def pmfFromWeights(ws, B):
    S = [*range(np.size(ws))]
    filter = list(map(lambda j: j in B, S))
    return np.prod(ws[filter])

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


class RobustnessExperiment():
    def __init__(self, k, v, verbose=0):
        self.k = k
        self.v = -np.sort(-np.abs(v))
        self.n = np.size(v)
        self.r = rCompute(self.v, k)
        self.ps = (self.r+1)*self.v[k-self.r-1:]/pSum(k-self.r-1, self.v) # psum is zero indexed and want s_{k-r}
        if np.abs(np.sum(self.ps) - (self.r+1)) > 0.001:
            print("Something went wrong: sum of ps is not r+1.")
            print("sum is: {}".format(np.sum(self.ps)))

    def getLastBlock(self):
        ws = getWeightsFromCoverage(self.ps, self.n-self.k+self.r+1, self.r+1)
        p = lambda i: computeSingleMarginalFromWeights(i, self.r+1, ws)
        q = lambda i, j: computePairMarginalFromWeights(i, j, self.r+1, ws)
        t = self.k - self.r - 1 # shift in starting index
        M = [[self.v[t+i]*self.v[t+j]*q(i,j)/(p(i)*p(j)) for i in [*range(0, self.n-t)]]
         for j in [*range(0, self.n-t)]]
        return np.array(M)

    def getTauTilde(self):
        vtrunc = np.concatenate((self.v[:self.k-self.r-1],np.zeros(self.n-self.k+self.r+1)))
        w = np.concatenate((np.zeros(self.k-self.r-1),self.v[self.k-self.r-1:]))
        outerv = np.outer(vtrunc, vtrunc)
        term2 = np.outer(vtrunc, w) + np.outer(w, vtrunc)
        final = np.zeros((self.n, self.n))
        final[self.k-self.r-1:, self.k-self.r-1:] = self.getLastBlock()
        return outerv + term2 + final
    
    def getOptimalRobustness(self):
        tauTilde = self.getTauTilde()
        rob = np.linalg.trace(tauTilde) - np.linalg.norm(self.v)**2
        sigma = (tauTilde - np.outer(self.v, self.v))/rob
        return (rob, sigma)
    
def main():
    n = int(input("n: "))
    k = int(input("k: "))
    v = np.random.rand(n)
    v = v/np.linalg.norm(v)
    v = -np.sort(-np.abs(v))
    print("Generated length-{} random, sorted vector: {}.".format(n, v))
    analyticRob = kSuppNorm(k, v)**2 - 1
    print("Formula for robustness from k-support norm leads to: {}.".format(analyticRob))
    newRob = RobustnessExperiment(k, v)
    rob, sigma = newRob.getOptimalRobustness()
    print("Closed-form solution for optimal sigma is: {}...".format(sigma))
    print("This leads to a robustness value of: {}.".format(rob))
    print("k, r = {}, {}".format(k, newRob.r))
    
if __name__ == '__main__':
    main()

    