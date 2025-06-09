import numpy as np
from numpy.polynomial import Polynomial
from .sampling import *
from .helpers import *

# Solve for the optimal randomized truncation in trace distance.
# Assumes v is sorted in nonincreasing order
class TDOptimizer():
    def __init__(self, k, v, verbose=0):
        self.k = k

        # sort in nonincreasing order of abs value, redundant
        self.v = -np.sort(-np.abs(v))

        # initiate intermediate values
        self.r=-1
        self.l=-1
        self.n = np.size(v)
        self.verbose = verbose # not used currently
        self.fid = topKNorm(k, v)

        # expensive sums
        self.a = np.sum(v[:k-1]**2)
        self.b0 = np.sum(v[k-1:])
        self.b1 = np.sum(v[k-1:])
        self.c = np.sum(v[k-1:]**2)

        # store optimal quantities
        self.ps = []
        self.m = []
        self.t = -1 # negative value indicates the optimizer has not run
        self.m_top_k_norm = -1

    def getCubicSols(self,r,l):
        # with a,b,c,d,e defined as below, the cubic norm eqn to be satisfied becomes
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
    
    # helper function for rl test, this and below
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
            return t*theta(r,l,t) <= v[n-1]+tol and t!=0 # bit hacky
        else:
            return v[l-1]-tol < t*theta(r,l,t) <= v[l-2]+tol
    
    # test if r and ell are valid
    def rlTest(self,r,l,t,tol=10**(-5)):
        return self.rTest(r,l,t,tol) and self.lTest(r,l,t,tol)
    
    # given r and ell, form the optimal measurement
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
        b1init = self.b1 # b1 is s_{ell} (with current guess for ell)
        cinit = self.c # c is 2-norm squared of v_{ell:d} (-"-)
        for r in range(0,k):
            if r > 0:
                self.a -= v[k-r-1]**2 # a is 2-norm squared of v_{1:k-r-1} (-"-)
                self.b0 += v[k-r-1] # b0 is s_{k-r} (-"-)
            self.b1 = b1init # reinitialize when ell is reinitialized
            self.c = cinit # (-"-)
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
                        print("r = {}, l={}, t={}".format(r, l, t))
                        self.m = self.formMeas(r,l,t)
                        self.t = t
                        self.r=r
                        self.l=l
                        return self.m,t
        raise RuntimeError("Valid r and ell not found.")
    
    # draw-by-draw sampling procedure from Chen et al.
    # intermediate marginals with set S given by q(j, S)
    def sampleSubset(self, ps, n, k):
        ws = getWeightsFromCoverage(ps, n, k)
        S = [*range(n)]
        A = []
        q_init = ps/k
        i1 = np.random.choice(S, size=1, p=q_init)[0]
        A.append(int(i1))
        q_prev = q_init
        if k > 1:
            for el in range(1, k):
                Sel = S.copy()
                for i in A:
                    Sel.remove(i)
                ielminus1 = A[-1]
                qel = lambda j: ((ws[ielminus1]*q_prev[j]-ws[j]*q_prev[ielminus1]) 
                                 / ((k-el)*(ws[ielminus1]-ws[j])*q_prev[ielminus1])) if (j in Sel) else 0
                qelnew = lambda j: qel(j) if qel(j) > 0 else 0
                qels = [qelnew(j) for j in S]
                qels = qels/np.sum(qels)
                ik = np.random.choice(S, size=1, p=qels)[0]
                A.append(int(ik))
                q_prev = qels
        return A


    # Samples a random pure state according to the right (max-entropy)
    # distribution using Procedure 1 in Sec. 3 of Chen et al.
    def sampleOptimalTDState(self):
        if self.r < 0:
            raise ValueError("r not yet computed. Run optimization first.")
        if self.m_top_k_norm == -1:
            self.m_top_k_norm = topKNorm(self.k, self.m)
        if self.ps == []:
            ps = self.getMarginals()
        S = self.sampleSubset(ps, self.l-self.k+self.r, self.r+1)
        phi1 = self.m[:self.k-self.r-1]
        phi2 = np.zeros(self.l-self.k+self.r,dtype=float)
        theta = self.theta(self.r, self.l, self.t)
        for idx in S:
            phi2[idx] = theta
        phi3 = np.zeros(self.n - self.l + 1)
        phi = np.concatenate((phi1, phi2, phi3))
        phi = phi/np.linalg.norm(phi)
        return phi
    
    def getLastBlock(self, theta, ps):
        k = self.k
        r = self.r
        l = self.l
        ws = getWeightsFromCoverage(ps, l-k+r, r+1)
        q = lambda i, j: computePairMarginalFromWeights(i, j, r+1, ws)
        M = [[q(i,j)*theta**2 for i in [*range(0, l-k+r)]]
         for j in [*range(0, l-k+r)]]
        return np.array(M)
    
    # output the optimal density matrix. Probably not useful in pracrtice.
    def getOptimalTDState(self):
        if len(self.m) < self.n:
            raise ValueError("m not yet computed. Run optimization first.")
        m = self.m
        t = self.t
        k = self.k
        r = self.r
        l = self.l
        n = self.n
        if self.ps == []:
            self.ps = self.getMarginals()
        ps = self.ps
        theta = self.theta(r,l,t)
        mtrunc = np.concatenate((m[:k-r-1],np.zeros(n-k+r+1)))
        w = np.concatenate((np.zeros(k-r-1),theta*ps,np.zeros(n-l+1)))
        term2 = np.outer(mtrunc, w) + np.outer(w, mtrunc)
        outerm = np.outer(mtrunc, mtrunc)
        final = np.zeros((n, n))
        final[k-r-1:l-1,k-r-1:l-1] = self.getLastBlock(theta, ps)
        sigma = outerm + term2 + final
        return sigma/np.linalg.trace(sigma)

