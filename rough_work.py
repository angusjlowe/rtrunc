import numpy as np
import new_td_optimization as rtrunc
n = int(input("n: "))
k = int(input("k: "))

v = np.random.normal(0,1,n)
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))

newTd = rtrunc.NewTDExperiment(k, v)
m, td = newTd.getOptimalTDMeas()
ps = newTd.getMarginals()
A = newTd.sampleSubset(ps, n, k)
print(A)