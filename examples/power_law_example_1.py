import numpy as np
import matplotlib.pyplot as plt
from rtrunc import td_optimizer as rtrunc
from rtrunc.helpers import *
plt.rcParams['text.usetex'] = True
plt.rc('font', size=16) 

n = 100

gammas = [0.25, 0.75, 1., 1.25]
colors = ['orange', 'blue', 'green', 'red']
startingepss = []
start = 1
ks = np.arange(start,n,1)

epsrange = np.arange(0.,1.01,0.01)
plt.plot(epsrange, epsrange, '-', label='$\\gamma$=0',color='gray')

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
    plt.plot(epss, ubs/epss, '--', color=colors[j])
    plt.plot(epss, tds/epss, '-', color=colors[j], label='$\\gamma$={}'.format(gamma))
    #plt.plot(epss, epss**(2*gamma-1), '.', color=colors[j], label='$\\epsilon^{2\\gamma-1}$')

#plt.plot(epsrange, np.ones(np.size(epsrange)), '-', color='gray', label='best pure' )
#plt.title("Mixed approx. to power law pure state. n={}.".format(n))
plt.legend()
plt.ylabel('$T/\\varepsilon$')
plt.xlabel('$\\varepsilon$')
plt.show()