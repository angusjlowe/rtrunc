import numpy as np
import matplotlib.pyplot as plt
import new_td_optimization as ntd
import robustness_optimization as ro

ns = [1000]

gamma = 0.51
startingepss = []
start = 10

for n in ns:
    # power law
    v = np.arange(n)+1
    v=v**(-gamma)

    # normalize
    v = v/np.linalg.norm(v)
    v = -np.sort(-np.abs(v))
        

    ks = np.arange(start,n - 10,int(n/50))

    fids = []
    ubs = []
    tds = []
    rs = []
    ls = []
    for k in ks:
        if k % 1 == 0:
            print("on iteration {}".format(k))
        newTd = ntd.NewTDExperiment(k, v)
        fid = newTd.fid
        fids.append(fid)
        m,td = newTd.getOptimalTDMeas()
        tds.append(td)
        rs.append(newTd.r)
        ls.append(newTd.l)
        #kSupp = ro.kSuppNorm(k, v)
        #robUB = 1-kSupp**(-2)
        #ubs.append(robUB)


    ubs=np.array(ubs)
    tds=np.array(tds)
    fids=np.array(fids)
    epss=np.sqrt(1-fids**2)
    startingepss.append(epss[0])
    plt.plot(ks, rs, '-', label='r')
    plt.plot(ks, ls, '-', label='ell')
    #plt.plot(epss, ubs/epss**2, '-', label='Rob. UB')
    #plt.plot(epss, tds/epss**2, '-', label='opt. td')

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

plt.plot(ks, rs, '-o', label='r')
plt.plot(ks, ls, '-o', label='ell')
plt.legend()
plt.show()

m=list(m)
plt.step([*range(len(m))], m, label="m")
#plt.step([*range(len(m))], v, label="v")
plt.legend()
plt.title('n = {}, k= {}'.format(n, ks[-1]))
plt.show()