import numpy as np
import time
import math
from simulation import _find_interactions_spatial

def reference_interactions(agents, threshold):
    """Original O(N^2) implementation for reference."""
    n = agents.shape[0]
    diff = agents[:, None, :2] - agents[None, :, :2]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, 2.0 * threshold)
    interactions = distances < threshold
    rows, cols = np.where(interactions)
    return rows, cols

def test_correctness():
    print("Testing correctness...")
    hive_bounds = np.array([-100, 100, -100, 100])
    threshold = 5.0
    
    # Random agents
    rng = np.random.default_rng(42)
    for n in [10, 100, 500]:
        agents = rng.uniform(-100, 100, (n, 3))
        
        ref_rows, ref_cols = reference_interactions(agents, threshold)
        new_rows, new_cols = _find_interactions_spatial(agents[:, :2], threshold, hive_bounds)
        
        # Sort results for comparison
        ref_pairs = sorted(zip(ref_rows, ref_cols))
        new_pairs = sorted(zip(new_rows, new_cols))
        
        assert ref_pairs == new_pairs, f"Mismatch for N={n}"
        print(f"  N={n}: OK")

def test_performance():
    print("Testing performance...")
    hive_bounds = np.array([-200, 200, -100, 100])
    threshold = 2.0
    rng = np.random.default_rng(42)
    
    for n in [1000, 2000, 5000]:
        agents = rng.uniform(-100, 100, (n, 3))
        
        # Reference
        start = time.perf_counter()
        ref_rows, ref_cols = reference_interactions(agents, threshold)
        ref_time = time.perf_counter() - start
        
        # New
        start = time.perf_counter()
        new_rows, new_cols = _find_interactions_spatial(agents[:, :2], threshold, hive_bounds)
        new_time = time.perf_counter() - start
        
        print(f"  N={n}:")
        print(f"    Ref time: {ref_time:.4f}s")
        print(f"    New time: {new_time:.4f}s")
        print(f"    Speedup:  {ref_time / new_time:.2f}x")
        
        assert len(ref_rows) == len(new_rows), f"Mismatch in count for N={n}"

if __name__ == "__main__":
    test_correctness()
    test_performance()
