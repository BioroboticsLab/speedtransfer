# Simulation Rebuttal Tasks

Extracted from `Reviewer_comments_and_rebuttal_plan.md`.

## Model Logic & Distributions
- [x] **Gamma Distribution for Speed:** Change speed sampling to use a Gamma distribution instead of clipped Gaussian. (Reviewer 1)
  - *Status:* Implemented in `simulation.py` (`_sample_gamma_from_stats`).
- [x] **Von Mises Distribution for Turning:** Change orientation changes to use a Von Mises distribution. (Reviewer 1)
  - *Status:* Implemented in `simulation.py` (`_sample_vonmises_from_stats`).
- [x] **Equations & Parameters:** Provide mathematical equations and parameter values used in the model for the methods section. (Reviewer 2, Minor 1)
  - *Status:* Complete. `model_equations.tex` created with full specification and parameter table. Removed redundant `speed_transfer_scale` parameter (was always 1.0).

## Ablation Studies (Promised)
The rebuttal explicitly promises a "list of model variants derived through ablation" and specifically "location preference".

- [x] **Systematic Component Ablation (The "List of Variants"):**
    - **No Walls:** Test effect of confinement.
    - **No Homing (Location Preference):** Test effect of spatial fidelity (explicitly promised).
    - **No Speed Transfer:** Test effect of the mechanism itself.
    - **No Turning Noise:** Test effect of stochasticity.
    - *Status:* Full factorial ablation run (`run_comprehensive_ablation.py`) covers all combinations of these.

## Parameter Sensitivity
- [x] **Density × Decay Sweep:** Sweep density factors and speed transfer decay values to show sensitivity.
  - *Status:* Implemented in `run_density_decay_sweep.py` with caching and multi-seed aggregation.
- [x] **Homing Asymmetry Investigation:** Test whether the asymmetric homing pull for Group 1 affects results across group sizes.
  - *Status:* Done — robust testing (5 seeds × 4 densities, production settings) showed no meaningful effect. Asymmetric homing term removed from model.

## Lower Priority
- [ ] **Speed Transfer Variants:** Try different transfer mechanisms (e.g., sum vs max). (Lower priority)
- [ ] **Coupled Oscillators Discussion:** Discuss observations in context of coupled oscillator theory. (Reviewer 2)
  - *Status:* Requires literature review; interpretation of simulation results.

## Documentation
- [ ] **Physical Analogies:** Revise "momentum transfer" analogy if misleading. (Reviewer 1)

## New Tasks
- [x] **Reproduce Figure 3b:** Reproduce Figure 3b with the new simulation (using the baseline). 
  - Code: [`aggregate_phase_per_loaction.py`](https://github.com/BioroboticsLab/speedtransfer/blob/wero/simulation/aggregate_phase_per_loaction.py)
  - *Status:* Done. Script `reproduce_figure_3b.py` runs baseline simulation and produces `visualizations/figure_3b_simulated.pdf`.
