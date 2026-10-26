import numpy as np
from simulation import run_full_factorial_ablation, SimulationConfig, default_simulation_config

def test_factorial_ablation():
    print("Testing full factorial ablation...")
    config = default_simulation_config()
    config.sim.sim_duration = 50 # Short duration for testing
    
    factors = ["disable_walls", "disable_homing"]
    results = run_full_factorial_ablation(
        density_factor=0.1,
        factors=factors,
        config=config,
        rhythmicity_permutations=0 # Skip stats for speed
    )
    
    expected_labels = ["baseline", "no_walls", "no_homing", "no_walls+homing"]
    print(f"Generated labels: {list(results.keys())}")
    
    for label in expected_labels:
        assert label in results, f"Missing expected label: {label}"
    
    print("Full factorial ablation test: OK")

if __name__ == "__main__":
    test_factorial_ablation()
