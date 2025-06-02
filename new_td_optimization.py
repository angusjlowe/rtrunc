import numpy as np
from numpy.polynomial import Polynomial
import itertools

def traceDistance(rho, sigma):
    eigs = np.linalg.eigvals(rho-sigma)
    return np.sum(np.abs(eigs))/2.

# take a numpy array as input and compute the top-k norm
def topKNorm(k, v):
    sortedV = -np.sort(-np.abs(v))
    return np.sqrt(np.sum(sortedV[:k]**2))

# the partial weight function R(k, C) from the 1994 paper by Chen et al.
# TODO: refactor using recursive expression in Chen et al. paper.
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


class NewTDExperiment():
    def __init__(self, k, v, verbose=0):
        self.k = k
        self.t = 0
        # sort in nonincreasing order of abs value
        self.v = -np.sort(-np.abs(v))
        self.r=-1
        self.l=-1
        self.n = np.size(v)
        self.verbose = verbose
        self.fid = topKNorm(k, v)

        # expensive sums
        self.a = np.sum(v[:k-1]**2)
        self.b0 = np.sum(v[k-1:])
        self.b1 = np.sum(v[k-1:])
        self.c = np.sum(v[k-1:]**2)

        # store optimal quantities
        self.ps = []
        self.m = []
        self.t = -1

    def getCubicSols(self,r,l):
        # with a,b,c,d,e defined as below, the cubic eqn to be satisfied becomes
        # a/(1+x) + b/(d+ex) + c/x - 1 = 0, which is equivalent to the polynomial
        # -ex^3+(b-d-e+ae+ce)x^2+(b-d+ad+c(d+e))x + cd = 0.
        k = self.k
        a = self.a
        b = (self.b0 - self.b1)**2
        c = self.c
        d, e = r+1, l-k+r
        c0, c1, c2, c3 = c*d, (b-d+a*d+c*(d+e)), (b-d-e+a*e+c*e), -e
        cubic = Polynomial([c0, c1, c2, c3])
        return list(filter(lambda x: np.isreal(x) and 0<=x<=1, cubic.roots()))

    def theta(self,r,l,t):
        k=self.k
        Trl = self.b0 - self.b1
        return Trl/(r+1+(l-k+r)*t)
    
    # get marginals. Should only be called after 
    # optimizing
    def getMarginals(self):
        if self.r < 0:
            raise ValueError("Integer r not yet set. Run optimization first.")
        r=self.r
        l=self.l
        v=self.v
        t=self.t
        theta=self.theta(r,l,t)
        k=self.k
        return v[k-r-1:l-1]/theta - t
    
    def rTest(self,r,l,t,tol):
        v=self.v
        k=self.k
        theta=self.theta
        if r==k-1:
            return v[k-r-1]/(1+t)-tol <= theta(r,l,t)
        else:
            return v[k-r-1]/(1+t)-tol <= theta(r,l,t) < v[k-r-2]/(1+t)+tol
        
    def lTest(self,r,l,t,tol):
        v=self.v
        n=np.size(v)
        theta=self.theta
        if l==n+1:
            return t*theta(r,l,t)>0
        else:
            return v[l-1]/t-tol < theta(r,l,t) <= v[l-2]/t+tol
    
    def rlTest(self,r,l,t,tol=10**(-4)):
        return self.rTest(r,l,t,tol) and self.lTest(r,l,t,tol)
    
    def formMeas(self,r,l,t):
        k=self.k
        theta=self.theta
        v=self.v
        n=np.size(v)
        a=np.array([])
        b=np.repeat(theta(r,l,t),l-k+r)
        c=np.array([])
        if r!=k-1:
            a=v[:k-r-1]/(1+t)
        if l!=n+1:
            c=v[l-1:]/t
        mtilde=np.concatenate((a,b,c))
        m=mtilde/np.linalg.norm(mtilde)
        return m
    
    # return optimal meas and td value as tuple (m,td)
    def getOptimalTDMeas(self):
        v = self.v
        k = self.k
        n = self.n
        b1init = self.b1
        cinit = self.c
        for r in range(0,k):
            if r > 0:
                self.a -= v[k-r-1]**2
                self.b0 += v[k-r-1]
            self.b1 = b1init
            self.c = cinit
            for l in range(k,n+2):
                if l > k:
                    self.b1 -= v[l-2]
                    self.c -= v[l-2]**2
                if l == n+1:
                    self.b1 = 0
                    self.c = 0
                ts = self.getCubicSols(r, l)
                for t in ts:
                    if self.rlTest(r,l,t):
                        self.m = self.formMeas(r,l,t)
                        self.t = t
                        self.r=r
                        self.l=l
                        return self.m,t
        raise RuntimeError("Valid r and ell not found.")
    
    # should probably just be a function that samples
    # a random pure state according to the right
    # distribution. Use Procedure 1 in Sec. 3 of Chen et al.
    def sampleOptimalTDState(self):
        return None
    
    def getLastBlock(self, theta):
        ps = self.ps
        k = self.k
        r = self.r
        l = self.l
        ws = getWeightsFromCoverage(ps, l-k+r, r+1)
        q = lambda i, j: computePairMarginalFromWeights(i, j, r+1, ws)
        M = [[q(i,j)*theta**2 for i in [*range(0, l-k+r)]]
         for j in [*range(0, l-k+r)]]
        return np.array(M)
    
    # output the optimal density matrix.
    # (Not useful in practice. And not 100% sure it works.)
    def getOptimalTDState(self):
        if len(self.m) < self.n:
            raise ValueError("m not yet computed. Run optimization first.")
        m = self.m
        t = self.t
        k = self.k
        r = self.r
        l = self.l
        n = self.n
        self.ps = self.getMarginals()
        ps = self.ps
        theta = self.theta(r,l,t)
        mtrunc = np.concatenate((m[:k-r-1],np.zeros(n-k+r+1)))
        w = np.concatenate((np.zeros(k-r-1),theta*ps,np.zeros(n-l+1)))
        term2 = np.outer(mtrunc, w) + np.outer(w, mtrunc)
        outerm = np.outer(mtrunc, mtrunc)
        final = np.zeros((n, n))
        final[k-r-1:l-1,k-r-1:l-1] = self.getLastBlock(theta)
        sigma = outerm + term2 + final
        return sigma/np.linalg.trace(sigma)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = int(input("n: "))
    k = int(input("k: "))
    
    # normal, random
    v = np.random.normal(0,1,n)

    # power law
    #gamma = 0.05
    #xs = np.arange(n)+1
    #v = xs**(-gamma)

    # normalize
    v = v/np.linalg.norm(v)
    v = -np.sort(-np.abs(v))

    newTd = NewTDExperiment(k, v)
    fid = newTd.fid
    print("Generated length-{} random vector: {}.".format(n, v))
    print(" Optimal pure TD approx: {}".format(np.sqrt(1-fid**2)))
    print("Solving TD optimization problem for k={}...".format(k))
    m,td = newTd.getOptimalTDMeas()
    input("Solved. Optimal randomized TD approx: {:4f}. Press enter to see plot.".format(td))
    m=list(m)
    plt.step([*range(len(m))], m, label="m")
    plt.step([*range(len(m))], v, label="v")
    plt.legend()
    plt.title("k={}. Pure TD approx: {:4f}. Random TD approx: {:4f}".format(k, np.sqrt(1-fid**2),td))
    plt.show()
    
    #sigma = newTd.getOptimalTDState()
    #td = traceDistance(np.outer(v,v), sigma)
    #print("Getting optimal state leads to a trace distance: {}.".format(td))
    #input("Press enter to see visualization.")
    

    # matrix visualization
    
    #from matplotlib import colors
    #vmin = 0
    #vmax = v[0]**2
    #cmap = plt.cm.viridis
    #norm = colors.Normalize(vmin=vmin, vmax=vmax)

    #fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    #axes = (ax1,ax2,ax3)
    #vtrunc = np.concatenate((v[:k],np.zeros(n-k)))
    #vtrunc = vtrunc/np.linalg.norm(vtrunc)
    #im1 = ax1.matshow(np.outer(v,v), cmap=cmap, norm=norm)
    #ax1.set_title("$|v\\rangle\\langle v|$")
    #ax2.matshow(np.outer(vtrunc,vtrunc), cmap=cmap, norm=norm)
    #ax2.set_title("$|v_{1:k}\\rangle \\langle v_{1:k}|$")
    #ax3.matshow(sigma, cmap=cmap, norm=norm)
    #ax3.set_title("$\\sigma^\\star$")
    #plt.tight_layout()
    #plt.show()
    
