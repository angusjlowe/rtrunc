import numpy as np
import new_td_optimization as rtrunc
n = int(input("n: "))
k = int(input("k: "))

v = np.random.normal(0,1,n)
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))

newTd = rtrunc.NewTDExperiment(k, v)
print("beginning optimization")
m, td = newTd.getOptimalTDMeas()
phi = newTd.sampleOptimalTDState()
print(phi)