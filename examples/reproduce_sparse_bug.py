import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rtrunc import td_optimizer as rtrunc

k = 4
v = np.array([9.99800087e-01, 1.99946681e-02, 1.19877401e-05, 2.39738812e-07,
              4.66538159e-10, 9.33014086e-12, 5.59385650e-15, 1.11869668e-16])

print("v =", v)
print("k =", k)
print("n =", len(v))
print()

tdo = rtrunc.TDOptimizer(k, v)
m, td = tdo.getOptimalTDMeas()

print("Optimizer found:")
print("  r =", tdo.r)
print("  l =", tdo.l)
print("  t =", tdo.t)
print("  middle block length = l - k + r =", tdo.l - k + tdo.r)
print()

ps = tdo.getMarginals()
print("getMarginals() returns:", ps)
print("sum(ps) =", np.sum(ps), "  (should equal r+1 =", tdo.r + 1, ")")
print()

print("Calling sampleOptimalTDState() -- expect error below:")
phi = tdo.sampleOptimalTDState()
print("phi =", phi)
