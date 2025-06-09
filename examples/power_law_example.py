import numpy as np
import matplotlib.pyplot as plt
from rtrunc import td_optimizer as rtrunc
from rtrunc.helpers import *

n = 500

gammas = [0.25, 0.5, 0.75, 1., 1.25]
colors = ['orange', 'blue', 'green', 'red', 'black']
startingepss = []
start = 10
ks = np.arange(start,n - 10,int(n/50))

for j in range(len(gammas)):
    gamma = gammas[j]
    # power law
    v = np.arange(n)+1
    v=v**(-gamma)

    # normalize
    v = v/np.linalg.norm(v)
    v = -np.sort(-np.abs(v))

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

    print("finished gamma = {}".format(gamma))
    ubs=np.array(ubs)
    tds=np.array(tds)
    fids=np.array(fids)
    epss=np.sqrt(1-fids**2)
    startingepss.append(epss[0])
    plt.plot(epss, ubs/epss**2, '--', color=colors[j], label='Rob. UB, $\\gamma$={}'.format(gamma))
    plt.plot(epss, tds/epss**2, '-', color=colors[j], label='Opt. TD, $\\gamma$={}'.format(gamma))

maxeps = max(startingepss)
epsrange = np.arange(0.,maxeps+0.01,0.01)
plt.plot(epsrange, np.ones(np.size(epsrange)), '--', label='best possible, $\\epsilon^2$')
plt.plot(epsrange, 1/epsrange, '--', color='gray', label='best pure approx.' )
plt.title("Mixed approx. to power law pure state. n={}.".format(n))

plt.legend()
plt.ylabel('val.$/\\epsilon^2$')
plt.xlabel('$\\epsilon$')
plt.show()