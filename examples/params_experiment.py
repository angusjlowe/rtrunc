import numpy as np
import matplotlib.pyplot as plt
from rtrunc import td_optimizer as rtrunc
from rtrunc.helpers import *

def compute_a(r, m, v):
    if r==k-1:
        return 0
    else:
        return m[0]/v[0]

def compute_b(r, l, m):
    if l-k+r==0:
        return 0
    else:
        return m[k-r-1]

def compute_c(l, m, v):
    n==np.size(m)
    if l==n+1:
        return 0
    else:
        return m[l-1]/v[l-1]

n = 500
gamma = 0.75
start = 10
ks = np.arange(start, n-10, int(n/50))

# power law
v = np.arange(n)+1
v = v**(-gamma)

# normal
#v = np.random.normal(0, 1, n)

# normalize
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))

# we want r,l,a,b,c
rs = []
ls = []
aas = []
bs = []
cs = []
fids = []
for k in ks:
    if k % 1 == 0:
        print("k = {}".format(k))
    tdo = rtrunc.TDOptimizer(k, v)
    fid = tdo.fid
    m,td = tdo.getOptimalTDMeas()
    r = tdo.r
    l = tdo.l
    a = compute_a(r, m, v)
    b = compute_b(r, l, m)
    c = compute_c(l, m, v)
    rs.append(r)
    ls.append(l)
    aas.append(a)
    bs.append(b)
    fids.append(fid)
    cs.append(c)

fids = np.array(fids)
epss=np.sqrt(1-fids**2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.plot(epss, rs, '--o', label='r')
ax1.plot(epss, ls, '--o', label='l')
ax1.set_xlabel("$\\epsilon$")
ax1.legend()
ax1.set_title("Optimal r and ell. n = {}, gamma = {:.2f}".format(n, gamma))
ax2.plot(epss, aas, '--o', label='a')
ax2.plot(epss, bs, '--o', label='b')
ax2.plot(epss, cs, '--o', label='c')
ax2.set_xlabel("$\\epsilon$")
ax2.legend()
ax2.set_title("Values of a, b, c. n = {}, gamma = {:.2f}".format(n, gamma))
plt.tight_layout()
plt.show()

