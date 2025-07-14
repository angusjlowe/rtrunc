import numpy as np
import matplotlib.pyplot as plt
from rtrunc import td_optimizer as rtrunc
from rtrunc.helpers import *

ns = [100,200,400,800,1600]

gamma = 0.75
colors = ['orange', 'blue', 'green', 'red', 'black']
startingepss = []
start = 10
for j in range(len(ns)):
    n = ns[j]
    # power law
    v = np.arange(n)+1
    v=v**(-gamma)

    # normalize
    v = v/np.linalg.norm(v)
    v = -np.sort(-np.abs(v))

    ks = np.arange(start,n - 10,int(n/10))

    fids = []
    ubs = []
    tds = []
    for k in ks:
        if k % 1 == 0:
            print("on iteration {}".format(k))
        tdo = rtrunc.TDOptimizer(k, v)
        fid = tdo.fid
        fids.append(fid)
        m,td = tdo.getOptimalTDMeas()
        tds.append(td)
        kSupp = kSuppNorm(k, v)
        robUB = 1-kSupp**(-2)
        ubs.append(robUB)

    print("finished n = {}".format(n))
    ubs=np.array(ubs)
    tds=np.array(tds)
    fids=np.array(fids)
    epss=np.sqrt(1-fids**2)
    startingepss.append(epss[0])
    plt.plot(epss, ubs/epss**(2*gamma), '--', color=colors[j], label='Rob. UB, n={}'.format(n))
    plt.plot(epss, tds/epss**(2*gamma), '-', color=colors[j], label='Opt. TD, n={}'.format(n))

maxeps = max(startingepss)
epsrange = np.arange(0.,maxeps+0.01,0.01)
plt.plot(epsrange, epsrange**2/epsrange**(2*gamma), '--', label='best possible, $\\epsilon^2$')
plt.plot(epsrange, epsrange/epsrange**(2*gamma), '--', color='gray', label='best pure approx.' )
plt.title("Mixed approx. to power law pure state. $\\gamma=${}.".format(gamma))

plt.legend()
plt.ylabel('val.$/\\epsilon^{2\\gamma}$')
plt.xlabel('$\\epsilon$')
plt.show()