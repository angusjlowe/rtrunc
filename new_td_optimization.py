import numpy as np
from numpy.polynomial import Polynomial

# take a numpy array as input and compute the top-k norm
def topKNorm(k, v):
    sortedV = -np.sort(-np.abs(v))
    return np.sqrt(np.sum(sortedV[:k]**2))

class NewTDExperiment():
    def __init__(self, k, v, verbose=0):
        self.k = k
        self.t = 0
        # sort in nonincreasing order of abs value
        self.v = -np.sort(-np.abs(v))
        self.r=0
        self.l=k
        self.n = np.size(v)
        self.verbose = verbose
        self.fid = topKNorm(k, v)

    def getCubicSols(self,r,l):
        # with a,b,c,d,e defined as below, the cubic eqn to be satisfied becomes
        # a/(1+x) + b/(d+ex) + c/x - 1 = 0, which is equivalent to the polynomial
        # -ex^3+(b-d-e+ae+ce)x^2+(b-d+ad+c(d+e))x + cd = 0.
        v=self.v
        k=self.k
        a, b, c, d, e = np.sum(v[:k-r-1]**2), np.sum(v[k-r-1:l-1])**2, np.sum(v[l-1:]**2), r+1, l-k+r
        c0, c1, c2, c3 = c*d, (b-d+a*d+c*(d+e)), (b-d-e+a*e+c*e), -e
        cubic = Polynomial([c0, c1, c2, c3])
        return list(filter(lambda x: np.isreal(x) and 0<=x<=1, cubic.roots()))

    def theta(self,r,l,t):
        v=self.v
        k=self.k
        return np.sum(v[k-r-1:l-1])/(r+1+(l-k+r)*t)
    
    def getMartginals(self):
        r=self.r
        l=self.l
        v=self.v
        n=self.n
        t=self.t
        theta=self.theta
        k=self.k
        return np.concatenate((np.ones(k-r-1), v[k-r-1:l-1]/theta(r,l,t) - t, np.zeros(n-l+1)))
    
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
    # TODO: the above applies to edge cases for r and ell!
    # Related to other loose end
    def getOptimalTDMeas(self):
        k=self.k
        n=self.n
        for r in range(0,k):
            for l in range(k,n+2):
                ts = self.getCubicSols(r, l)
                for t in ts:
                    #print(t)
                    #print(self.rlTest(r,l,t))
                    if self.rlTest(r,l,t):
                        m = self.formMeas(r,l,t)
                        self.r=r
                        self.l=l
                        self.t = t
                        return m,t
        raise RuntimeError("Valid r and ell not found.")
    
    # should probably just be a function that samples
    # a random pure state according to the right
    # distribution
    def getOptimalTDState(self):
        return None

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = int(input("n: "))
    k = int(input("k: "))
    # normal, random
    #v = np.random.normal(0,1,n)

    # power law
    gamma=1.5
    v = np.arange(n)+1
    v=v**(-gamma)

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
