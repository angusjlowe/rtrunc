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

def t_helper(i, C, ws):
    return np.sum(ws[C]**i)

# compute partial weight, dynamically
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
        self.m_top_k_norm = -1

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
            return t*theta(r,l,t) <= v[n-1]+tol
        else:
            return v[l-1]-tol < t*theta(r,l,t) <= v[l-2]+tol
    
    def rlTest(self,r,l,t,tol=10**(-5)):
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


    # should probably just be a function that samples
    # a random pure state according to the right
    # distribution. Use Procedure 1 in Sec. 3 of Chen et al.
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
        ps = self.getMarginals()
        theta = self.theta(r,l,t)
        mtrunc = np.concatenate((m[:k-r-1],np.zeros(n-k+r+1)))
        w = np.concatenate((np.zeros(k-r-1),theta*ps,np.zeros(n-l+1)))
        term2 = np.outer(mtrunc, w) + np.outer(w, mtrunc)
        outerm = np.outer(mtrunc, mtrunc)
        final = np.zeros((n, n))
        final[k-r-1:l-1,k-r-1:l-1] = self.getLastBlock(theta, ps)
        sigma = outerm + term2 + final
        return sigma/np.linalg.trace(sigma)

