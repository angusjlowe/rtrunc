"""
Profiles the runtime of TDOptimizer methods, with the expectation that most
time is spent inside the sampling_v2 submodule.

Usage:
    python profile_td_optimizer.py            # prints summary, writes profile.prof
    snakeviz profile.prof                     # opens interactive flamegraph in browser
"""

import cProfile
import pstats
import io
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rtrunc import td_optimizer as tdo

# --- Test vector ---
n, k = 200, 100
gamma = 0.2  # power law exponent: larger gamma = faster decay
xs = np.arange(1, n + 1, dtype=float)
v = xs ** (-gamma)
v /= np.linalg.norm(v)

N_SAMPLES = 1  # number of sampleOptimalTDState calls to profile

def workload():
    optimizer = tdo.TDOptimizer(k, v)
    optimizer.getOptimalTDMeas()
    for _ in range(N_SAMPLES):
        optimizer.sampleOptimalTDState()

# --- Run profiler ---
profiler = cProfile.Profile()
profiler.enable()
workload()
profiler.disable()

# --- Save .prof file for snakeviz ---
prof_path = os.path.join(os.path.dirname(__file__), "profile.prof")
profiler.dump_stats(prof_path)
print(f"Profile data written to {prof_path}")
print("Visualize with:  snakeviz profiling/profile.prof\n")

# --- Print text summary ---
stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream)
stats.strip_dirs()

print("=== Top 20 functions by cumulative time (time spent in fn + callees) ===")
stats.sort_stats("cumulative")
stats.print_stats(20)
print(stream.getvalue())

stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream)
stats.strip_dirs()

print("=== Top 20 functions by total time (time spent in fn body only) ===")
stats.sort_stats("tottime")
stats.print_stats(20)
print(stream.getvalue())
