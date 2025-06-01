import numpy as np
import matplotlib.pyplot as plt
import new_td_optimization as ntd

n = 1000

gamma=1.

# power law
v = np.arange(n)+1
v=v**(-gamma)

# normalize
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))


mode = input("Mode: press c for comparison, m for optimal measurement, a for advantage.\n")

if mode=='c':

    ks = np.arange(20, 501, 20)

    fids = []
    tds = []
    for k in ks:
        if k % 20 ==0:
            print("on iteration {}".format(k))
        newTd = ntd.NewTDExperiment(k, v)
        fid = newTd.fid
        fids.append(fid)
        m,td = newTd.getOptimalTDMeas()
        tds.append(td)


    plt.plot(ks, np.sqrt(1-np.array(fids)**2), '-', label="best pure")
    plt.plot(ks, tds, '-', label="best mixed")
    #plt.plot(ks, 1-np.sqrt(np.array(fids)), '-', label="FvDG LB")
    #plt.plot(ks, 1-np.array(fids)**2, '-', label='square of best pure')

    plt.title("Approximation to power law pure state. dim={}, gamma={}".format(n, gamma))

    plt.legend()
    plt.yscale('log')
    plt.ylabel('trace distance')
    plt.xlabel('k')
    plt.show()

if mode=='m':
    k=700
    newTd=ntd.NewTDExperiment(k,v)
    fid=newTd.fid
    m,td=newTd.getOptimalTDMeas()
    plt.plot(np.arange(n)+1,m,'-',label='optimal meas. m')
    plt.plot(np.arange(n)+1,v,'-',label='target v')
    plt.title("best pure = {:4f}. best mixed = {:4f}".format(np.sqrt(1-fid**2),td))
    plt.yscale('log')
    plt.legend()
    plt.show()

if mode=='a':
    ks = np.arange(20, n, 5)

    fids = []
    tds = []
    for k in ks:
        if k % 20 ==0:
            print("on iteration {}".format(k))
        newTd = ntd.NewTDExperiment(k, v)
        fid = newTd.fid
        fids.append(fid)
        m,td = newTd.getOptimalTDMeas()
        tds.append(td)


    #plt.plot(ks, np.sqrt(1-np.array(fids)**2), '-', label="best pure")
    plt.plot(np.sqrt(1-np.array(fids)**2), tds, 'o-', label="best mixed")
    #plt.plot(ks, 1-np.sqrt(np.array(fids)), '-', label="FvDG LB")
    #plt.plot(ks, 1-np.array(fids)**2, '-', label='square of best pure')

    plt.title("Approximation to power law pure state. dim={}, gamma={:4f}".format(n, gamma))

    plt.legend()
    #plt.yscale('log')
    plt.ylabel('trace distance')
    plt.xlabel('eps')
    #plt.xscale('log')
    plt.show()