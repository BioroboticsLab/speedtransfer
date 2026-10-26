# Handoff: Manuscript Revision with Tracked Changes

## Objective

Revise the speed transfer paper (`main.tex`) to address reviewer comments from the rebuttal plan. Create `main_revised.tex` with visual change tracking (blue = added, red strikethrough = deleted) togglable via a LaTeX boolean.

## Repository Layout

```
/Users/tim/Code/speedtransfer/
├── simulation/                    # Simulation code
│   ├── simulation.py              # Core simulation (1140 lines)
│   ├── tasks_for_tim.md           # Rebuttal tasks checklist (48 items)
│   ├── model_equations.tex        # Mathematical model (already written)
│   ├── run_sensitivity_sweeps.py  # Parameter sensitivity script
│   ├── run_comprehensive_ablation.py
│   ├── visualizations/
│   │   ├── ablation_study.tex     # Ablation + sensitivity PDF (5 pages, compiled)
│   │   ├── ablation_study.pdf
│   │   ├── ablation_table.tex     # LaTeX table for ablation results
│   │   ├── ablation_single_factor.pdf
│   │   ├── ablation_factorial_heatmap.pdf
│   │   ├── sensitivity_density.pdf
│   │   ├── sensitivity_day_duration.pdf
│   │   └── sensitivity_decay.pdf
│   └── tests/                     # Test suite
└── speed_transfer_paper/          # Paper repo (moved here by user)
    ├── main.tex                   # Original manuscript (505 lines)
    ├── references.bib
    └── figure_*.png               # All figures
```

## What Has Been Done (Simulation Side)

1. **Ablation study** — Full 2⁴ factorial over walls, homing, speed transfer, turning noise. Speed transfer is the only essential mechanism. Results in `ablation_study.pdf`.
2. **Parameter sensitivity sweeps** — Single-parameter sweeps over density, day duration, decay rate. Three two-panel figures + CSV data. Supplementary section written in `ablation_study.tex`.
3. **Model equations** — Full mathematical formulation in `model_equations.tex`.
4. **Homing asymmetry** — Investigated and removed (Group 1 had an extra homing pull term that was not documented; ablation showed negligible effect).
5. **Tests** — Phase-shift integration test, sampling distribution tests, homing asymmetry test all pass.

## What Needs to Be Done (Manuscript Side)

### Step 1: File Setup

Duplicate `main.tex` → `main_revised.tex` and add this preamble block after `\usepackage{xcolor}` (line 18):

```latex
\usepackage{soul}          % for \st (strikethrough)

\newif\iftrackchanges
\trackchangestrue          % toggle: true = show markup, false = clean

\iftrackchanges
  \newcommand{\added}[1]{\textcolor{blue}{#1}}
  \newcommand{\deleted}[1]{\textcolor{red}{\st{#1}}}
  \newcommand{\replaced}[2]{\deleted{#1}\added{#2}}
\else
  \newcommand{\added}[1]{#1}
  \newcommand{\deleted}[1]{}
  \newcommand{\replaced}[2]{#2}
\fi
```

### Step 2: Text Changes (in priority order)

#### 2a. Methods §ABM (line 159–160)
- **Expand** agent-based model description with key equations from `model_equations.tex`
- **Mention** Gamma distribution for speed, Von Mises for turning angles
- **Add** parameter values table

#### 2b. Discussion (lines 95–115)
- **Tone down** adaptive claims (line 102–103): "may hold adaptive benefits" → express as speculative hypothesis
- **Add** coupled oscillator paragraph: discuss observations in coupled oscillator framework (weaker coupling for young bees = longer phase lag)
- **Add** movement-as-proxy limitation paragraph
- **Clarify** model vs real data distinction (reviewer comment on line 686)
- **Address** "model excludes mechanisms" (reviewer comment on line 684) — note that model intentionally tests minimal mechanism

#### 2c. Results §2.2 (lines 79–86)
- **Revise** "momentum transfer" analogy (line 82): speed transfer is NOT conserved like momentum; replace with more accurate analogy
- **Clarify** "fast bees lose speed" (Fig 2a): distinguish regression to mean from actual transfer effect

#### 2d. Results §2.3 (lines 88–93)
- **Add** reference to ablation study in SI
- **Expand** model description paragraph with more detail

#### 2e. Abstract + Introduction (lines 56, 61–68)
- **Abstract**: Add sentence about novelty (whole-colony tracking vs individual studies)
- **Introduction**: Clarify "circadian rhythm" = overt behavioral rhythm (movement speed), not molecular clock
- **Introduction**: Strengthen novelty statement about tracking thousands

#### 2f. Supplementary (after line 501)
- **Add** ablation study section: include `ablation_table.tex`, `ablation_single_factor.pdf`, `ablation_factorial_heatmap.pdf`
- **Add** parameter sensitivity section: include `sensitivity_density.pdf`, `sensitivity_day_duration.pdf`, `sensitivity_decay.pdf` with summary text from `ablation_study.tex`

## Key Reviewer Points (Quick Reference)

| Reviewer | Key Concern | Where |
|----------|-------------|-------|
| R1 | Physical analogies misleading ("momentum") | Results §2.2, line 82 |
| R1 | Adaptive claims too strong | Discussion, lines 102-103 |
| R1 | Movement as proxy limitations | Discussion (new paragraph) |
| R1 | Ablation studies needed | SI (done, needs adding) |
| R1 | Sensitivity analysis | SI (done, needs adding) |
| R2 | Novelty unclear | Abstract + Introduction |
| R2 | Coupled oscillator discussion | Discussion (new paragraph) |
| R2 | Equations in methods | Methods §ABM |
| R2 | Colony differences | Discussion (needs Tim's input) |

## User's Comment on Review Report

> "sensitivity to interaction dimensions, time windows, and thresholds — this relates to the analysis of empirical data, not that from the simulation"

This is noted: the reviewer's concern about interaction dimensions/thresholds relates to the **empirical data analysis** (rectangle size, 30-sec window), not the simulation parameter sweeps. The simulation sensitivity analysis (density, day duration, decay) is a separate contribution.

## Files to Copy into Paper Repo

When adding SI sections, these files need to be copied from `simulation/visualizations/` into `speed_transfer_paper/`:
- `ablation_single_factor.pdf`
- `ablation_factorial_heatmap.pdf`
- `ablation_table.tex`
- `sensitivity_density.pdf`
- `sensitivity_day_duration.pdf`
- `sensitivity_decay.pdf`
