"""
Convergence curves for the k=2 DPP regime that causes logit-init oscillation.

With k=2 and power-law weights ws[i]=(i+1)^{-gamma}, increasing gamma
concentrates mass on the first element: ps[0] -> 1 while ps[i>0] stay small.
This is the regime encountered in mps_random.py where the logit initialisation
causes large oscillating Newton steps.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rtrunc.sampling import (getWeightsFromCoverage, getWeightsFromCoverage_v2,
                             getWeightsFromCoverage_v3,
                             computeSingleMarginalFromWeights_fast)

N         = 10
K         = 2
GAMMAS    = [0.5, 2.0, 5.0]
MAX_ITERS = 500


def make_marginals(n, k, gamma):
    """Valid k-DPP marginals from power-law weights ws[i] = (i+1)^{-gamma}."""
    ws = np.arange(1, n + 1, dtype=float) ** (-gamma)
    return computeSingleMarginalFromWeights_fast(k, ws)


def run_benchmark():
    results = {}
    for gamma in GAMMAS:
        ps = make_marginals(N, K, gamma)
        print(f"gamma={gamma:4.1f}  ps[0]={ps[0]:.6f}  ps[-1]={ps[-1]:.2e}")

        _, hist0 = getWeightsFromCoverage(
            ps, K, max_iter=MAX_ITERS, init_mode=0,
            fixed_iters=True, return_history=True)
        _, hist1 = getWeightsFromCoverage(
            ps, K, max_iter=MAX_ITERS, init_mode=1,
            fixed_iters=True, return_history=True)
        _, histv2 = getWeightsFromCoverage_v2(
            ps, K, max_iter=MAX_ITERS, init_mode=1,
            fixed_iters=True, return_history=True)
        _, histv3 = getWeightsFromCoverage_v3(
            ps, K, max_iter=MAX_ITERS,
            fixed_iters=True, return_history=True)

        results[gamma] = (ps, hist0, hist1, histv2, histv3)
    return results


def plot_results(results):
    iters = np.arange(1, MAX_ITERS + 1)
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(5 * len(GAMMAS), 4),
                             sharey=False)
    fig.suptitle(f"L1 error vs iteration  (n={N}, k={K})", fontsize=13)

    for i, gamma in enumerate(GAMMAS):
        ax = axes[i]
        ps, hist0, hist1, histv2, histv3 = results[gamma]
        ax.semilogy(iters, hist0,  color='blue',   lw=1.2,
                    label='Newton, log(p) init')
        ax.semilogy(iters, hist1,  color='orange', lw=1.2,
                    label='Newton, logit(p) init')
        ax.semilogy(iters, histv2, color='green',  lw=1.2,
                    label='Grad desc (α=4), logit(p) init')
        ax.semilogy(iters, histv3, color='red',    lw=1.2,
                    label='Newton, θ=0 init')
        ax.set_title(f'γ = {gamma}  [ps[0] = {ps[0]:.4f}]')
        ax.set_xlabel('iteration')
        if i == 0:
            ax.set_ylabel('L1 error')
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_init_modes_2.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    results = run_benchmark()
    plot_results(results)
