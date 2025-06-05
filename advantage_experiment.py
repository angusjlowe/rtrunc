import numpy as np
import matplotlib.pyplot as plt
import new_td_optimization as ntd
import robustness_optimization as ro

ns = [500]

gamma = 0.75
startingepss = []
start = 2

for n in ns:
    # power law
    v = np.arange(n)+1
    v=v**(-gamma)

    # normalize
    v = v/np.linalg.norm(v)
    v = -np.sort(-np.abs(v))
        

    ks1 = np.arange(start, n-50, int(n/100))
    ks2 = np.arange(n-50, n-10, 10)
    ks = np.concatenate((ks1,ks2))

    fids = []
    ubs = []
    tds = []
    for k in ks:
        if k % 1 == 0:
            print("on iteration {}".format(k))
        newTd = ntd.NewTDExperiment(k, v)
        fid = newTd.fid
        fids.append(fid)
        m,td = newTd.getOptimalTDMeas()
        tds.append(td)
        kSupp = ro.kSuppNorm(k, v)
        robUB = 1-kSupp**(-2)
        ubs.append(robUB)


    ubs=np.array(ubs)
    tds=np.array(tds)
    fids=np.array(fids)
    epss=np.sqrt(1-fids**2)
    startingepss.append(epss[0])
    plt.plot(epss, ubs/epss**2, '-', label='Rob. UB')
    plt.plot(epss, tds/epss**2, '-', label='opt. td')

maxeps = max(startingepss)
epsrange = np.arange(0.,maxeps+0.01,0.01)
plt.plot(epsrange, np.ones(np.size(epsrange)), '--', label='best possible, $\\epsilon^2$')
plt.plot(epsrange, 1/epsrange, '--', label='best pure approx.' )
plt.title("Mixed approx. to power law pure state. gamma={:.2f}, n={}.".format(gamma,n))

plt.legend()
#plt.yscale('log')
plt.ylabel('advantage $\\alpha$, $T_k=\\epsilon^{1+\\alpha}$')
plt.xlabel('$\\epsilon$')
#plt.xscale('log')
plt.show()

m=list(m)
plt.step([*range(len(m))], m, label="m")
plt.step([*range(len(m))], v, label="v")
plt.show()