import numpy as np
from rtrunc import td_optimizer as rtrunc
from rtrunc import helpers

k=3
n=6
v = np.array([9.99169762e-01, 4.07192495e-02, 1.29582183e-03, 2.23335185e-04, 1.06616974e-05, 1.00387176e-06])
tdo=rtrunc.TDOptimizer(k,v)
fid=tdo.fid
m, t = tdo.getOptimalTDMeas()
n_samples = 10000
rho = 0
for _ in range(n_samples):
    phi = tdo.sampleOptimalTDState()
    outer = np.outer(phi, phi)
    rho += outer/n_samples

empirical_td = helpers.traceDistance(rho, np.outer(v,v))
det_trunc = np.concatenate((v[:k],np.zeros(n-k)))
det_trunc = det_trunc/np.linalg.norm(det_trunc)
det_outer = np.outer(det_trunc, det_trunc)
det_td = helpers.traceDistance(det_outer, np.outer(v,v))
print("True rtrunc TD: {},Empirical TD: {}, Deterministic TD: {}".format(t, empirical_td, det_td))