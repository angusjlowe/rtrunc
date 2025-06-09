import unittest
from rtrunc import td_optimizer as tdo
from rtrunc.helpers import *
import numpy as np

np.random.seed(21)

class TestTDSetup(unittest.TestCase):
    def testTopKNorm(self):
        self.assertAlmostEqual(np.sqrt(2/3.),
                               topKNorm(2, np.array([1./np.sqrt(3)]*3)),5)
    

class TestTDComputation(unittest.TestCase):
    def testFid(self):
        eps = np.random.rand(1)[0]
        v = np.array([np.sqrt(eps),np.sqrt(1-eps)])
        newTd = tdo.TDOptimizer(1, v)
        self.assertAlmostEqual(newTd.fid,np.sqrt(np.max((eps,1-eps))),5)

    def testSimple(self):
        eps = np.random.rand(1)[0]
        v = np.array([np.sqrt(eps),np.sqrt(1-eps)])
        newTd = tdo.TDOptimizer(1, v)
        _, t = newTd.getOptimalTDMeas()
        self.assertAlmostEqual(np.sqrt(eps*(1-eps)),t,4)

    def testUniform(self):
        n=13
        k=7
        eps=1-k/float(n)
        v=np.array([1./np.sqrt(n)]*n)
        newTd = tdo.TDOptimizer(k, v)
        self.assertAlmostEqual(np.sqrt(k/float(n)),newTd.fid,4)
        _, t = newTd.getOptimalTDMeas()
        robustness = 1/(1-eps) - 1
        self.assertGreaterEqual(robustness,t,2)
        self.assertAlmostEqual(t,0.461539,2) # answer obtained in Mathematica

        

if __name__ == '__main__':
    unittest.main()