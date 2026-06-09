import numpy as np
from .helpers import *
from .sampling import *

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

        self.ws = []

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


    # Samples a random pure state according to the right (max-entropy)
    # distribution using Procedure 1 in Sec. 3 of Chen et al.
    def sampleOptimalRobState(self):
        if self.r < 0:
            raise ValueError("r not yet computed. Run optimization first.")
        r = self.r
        k = self.k
        n = self.n
        s = pSum(k-r-1, self.v)
        if list(self.ps) == []:
            self.ps = self.v[k-r-1:]*(r+1)/s
        ps = self.ps
        if list(self.ws) == []:
            self.ws = getWeightsFromCoverage(ps, self.r+1)
        S = sampleSubset(self.ws, ps, n-k+r+1, r+1)
        phi1 = self.v[:k-r-1]
        phi2 = np.zeros(n-k+r+1,dtype=float)
        for idx in S:
            phi2[idx] = s/(r+1)
        phi = np.concatenate((phi1, phi2))
        phi = phi/np.linalg.norm(phi)
        return phi
    
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

    