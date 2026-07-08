import unittest
import numpy as np
from rtrunc import td_optimizer as tdo
from rtrunc.helpers import topKNorm
from rtrunc.sampling import getWeightsFromCoverage

np.random.seed(21)


class TestHelpers(unittest.TestCase):

    def test_top_k_norm(self):
        self.assertAlmostEqual(
            topKNorm(2, np.array([1. / np.sqrt(3)] * 3)),
            np.sqrt(2 / 3.), places=5)


class TestTDOptimizer(unittest.TestCase):

    def test_fidelity(self):
        eps = np.random.rand(1)[0]
        v = np.array([np.sqrt(eps), np.sqrt(1 - eps)])
        optimizer = tdo.TDOptimizer(1, v)
        self.assertAlmostEqual(optimizer.fid, np.sqrt(np.max((eps, 1 - eps))), places=5)

    def test_binary_td_value(self):
        eps = np.random.rand(1)[0]
        v = np.array([np.sqrt(eps), np.sqrt(1 - eps)])
        optimizer = tdo.TDOptimizer(1, v)
        _, td = optimizer.getOptimalTDMeas()
        self.assertAlmostEqual(td, np.sqrt(eps * (1 - eps)), places=4)

    def test_uniform_fidelity_and_td(self):
        n, k = 13, 7
        v = np.array([1. / np.sqrt(n)] * n)
        optimizer = tdo.TDOptimizer(k, v)
        self.assertAlmostEqual(optimizer.fid, np.sqrt(k / float(n)), places=4)
        _, td = optimizer.getOptimalTDMeas()
        robustness = 1 / (1 - (1 - k / float(n))) - 1
        self.assertGreaterEqual(robustness, td)
        self.assertAlmostEqual(td, 0.461539, places=2)  # verified in Mathematica


class TestErrorHandling(unittest.TestCase):

    def test_weights_from_coverage_bad_sum_raises(self):
        # sum([0.5, 0.5]) = 1 != k=3
        with self.assertRaises(ValueError):
            getWeightsFromCoverage([0.5, 0.5], 3)

    def test_weights_from_coverage_empty_raises(self):
        # sum([]) = 0 != k=1
        with self.assertRaises(ValueError):
            getWeightsFromCoverage([], 1)

    def test_get_marginals_before_optimization_raises(self):
        optimizer = tdo.TDOptimizer(1, np.array([0.8, 0.6]))
        with self.assertRaises(ValueError):
            optimizer.getMarginals()

    def test_sample_before_optimization_raises(self):
        optimizer = tdo.TDOptimizer(1, np.array([0.8, 0.6]))
        with self.assertRaises(ValueError):
            optimizer.sampleOptimalTDState()


class TestUniformSampling(unittest.TestCase):
    """
    For v = uniform on n entries with k=1, the optimal TD is (n-1)/n,
    marginals are uniform, and samples cover all indices equally.
    """

    def setUp(self):
        self.n = 10
        self.v = np.ones(self.n) / np.sqrt(self.n)
        self.optimizer = tdo.TDOptimizer(1, self.v)
        _, self.td = self.optimizer.getOptimalTDMeas()

    def test_td_value(self):
        self.assertAlmostEqual(self.td, (self.n - 1) / self.n, places=5)

    def test_marginals_are_uniform(self):
        ps = self.optimizer.getMarginals()
        np.testing.assert_allclose(ps, np.ones(self.n) / self.n, atol=1e-5)

    def test_sample_has_unit_norm(self):
        phi = self.optimizer.sampleOptimalTDState()
        self.assertAlmostEqual(np.linalg.norm(phi), 1.0, places=5)

    def test_sample_indices_are_uniform(self):
        """Each index should appear with frequency 1/n (within 5 standard deviations)."""
        n_samples = 2000
        counts = np.zeros(self.n, dtype=int)
        for _ in range(n_samples):
            phi = self.optimizer.sampleOptimalTDState()
            counts[np.argmax(np.abs(phi))] += 1
        expected = n_samples / self.n
        sigma = np.sqrt(n_samples * (1 / self.n) * (1 - 1 / self.n))
        for c in counts:
            self.assertAlmostEqual(c, expected, delta=5 * sigma)


class TestNearSparseVector(unittest.TestCase):
    """
    Regression: near-sparse vectors trigger a degenerate solution (r=0, l=k)
    where the middle block is empty.  sampleOptimalTDState previously crashed
    by passing an empty ps to getWeightsFromCoverage.
    """

    def setUp(self):
        self.v = np.array([
            9.99800087e-01, 1.99946681e-02, 1.19877401e-05,
            2.39738812e-07, 4.66538159e-10, 9.33014086e-12,
            5.59385650e-15, 1.11869668e-16,
        ])
        self.optimizer = tdo.TDOptimizer(4, self.v)
        self.optimizer.getOptimalTDMeas()

    def test_middle_block_is_empty(self):
        opt = self.optimizer
        self.assertEqual(opt.l - opt.k + opt.r, 0)

    def test_sample_does_not_raise(self):
        phi = self.optimizer.sampleOptimalTDState()
        self.assertIsNotNone(phi)

    def test_sample_has_unit_norm(self):
        phi = self.optimizer.sampleOptimalTDState()
        self.assertAlmostEqual(np.linalg.norm(phi), 1.0, places=5)

    def test_td_is_small(self):
        _, td = self.optimizer.getOptimalTDMeas()
        self.assertLess(td, 1e-4)


if __name__ == '__main__':
    unittest.main()
