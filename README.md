# 🐝 Collective Flow of Circadian Clock Information in Honeybee Colonies

<div align="center">

[![Preprint](https://img.shields.io/badge/bioRxiv-2024.07.29.605620-red?style=flat-square)](https://www.biorxiv.org/content/10.1101/2024.07.29.605620v1)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square&logo=python)](https://python.org)
[![Conda](https://img.shields.io/badge/Conda-Environment-brightgreen?style=flat-square&logo=anaconda)](environment.yml)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)](Dockerfile)
[![SLURM](https://img.shields.io/badge/SLURM-Compatible-orange?style=flat-square)](https://slurm.schedmd.com/)
[![Zenodo](https://img.shields.io/badge/Data-Zenodo-blue?style=flat-square&logo=zenodo)](https://zenodo.org/records/10869728)

*Analyzing social behaviour and rhythmicity in honey bee colonies through advanced computational methods*

</div>

## 📑 Table of Contents

- [🎯 Goal](#-goal)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📊 Analysis Pipeline](#-analysis-pipeline)
- [📊 Visualization & Figures](#-visualization--figures)
- [🔬 Simulation Framework](#-simulation-framework)
- [💾 Data Access](#-data-access)
- [🏆 Citation](#-citation)

## 🎯 Goal
This project provides a comprehensive implementation of methods for studying and analyzing behavioral rhythmicity in honey bees, as detailed in our paper: **[Collective flow of circadian clock information in honeybee colonies](https://www.biorxiv.org/content/biorxiv/early/2024/07/30/2024.07.29.605620.full.pdf)**.

### 🔬 Key Objectives

| Objective | Description |
|-----------|-------------|
| 📊 **Data Acquisition** | Implement methods for collecting trajectory data of marked bees using the [BeesBook](https://github.com/BioroboticsLab/bb_main/wiki) system |
| 🕐 **Cosinor Modeling** | Develop algorithms for fitting cosine curves to individual bee movement speeds to analyze rhythmic expression |
| 🤝 **Interaction Analysis** | Implement techniques for detecting and analyzing bee interactions based on trajectory data |
| 📈 **Statistical Validation** | Conduct statistical tests to validate observed behavior patterns against null models |
| 📊 **Visualization** | Provide comprehensive visualization tools for interpreting and presenting analysis results |

> 📅 **Analysis Period**: August 1-25, 2016 and August 20 - September 14, 2019

<div align="center">

![Overview Figure](figures%2Fimgs%2Foverview_figure.png)
*Figure: Overview of the collective flow analysis framework*

</div>

## 📁 Project Structure

<details>
<summary>📂 <strong>Click to expand directory structure</strong></summary>

```
speedtransfer/
├── 📊 analysis/                    # Core analysis scripts
│   ├── cosinor_fit_per_bee.py     # Individual rhythm analysis
│   ├── velocity_change_per_interaction.py  # Interaction detection
│   └── ...
├── 📈 figures/                     # Visualization notebooks
│   ├── figure_panel_1.ipynb       # Movement speed analysis
│   ├── figure_panel_2.ipynb       # Speed transfer analysis
│   └── imgs/                       # Generated figures
├── 📋 aggregated_results/          # Processed data outputs
│   ├── 2016/                       # 2016 analysis results
│   └── 2019/                       # 2019 analysis results
├── 💾 data/                        # Raw data (download required)
│   ├── 2016/
│   └── 2019/
├── 🔧 simulation/                  # Simulation scripts
├── ⚙️ path_settings.py            # Path configuration
├── 📦 environment.yml             # Conda environment
└── 🐳 Dockerfile                  # Container setup
```

</details>

### 📂 Directory Details

| Directory | Purpose | Key Features |
|-----------|---------|--------------|
| 🔬 **`analysis/`** | Core analysis scripts | SLURM-compatible, modular design |
| 📊 **`figures/`** | Jupyter visualization notebooks | Interactive plots, publication-ready figures |
| 📈 **`aggregated_results/`** | Processed data for visualization | Pre-computed results for faster plotting |
| 💾 **`data/`** | Raw trajectory data | [Download from Zenodo](https://zenodo.org/records/10869728) |
| 🔧 **`simulation/`** | MATLAB simulation scripts | Agent-based modeling |

## 🚀 Getting Started

### 📦 Installation

<details>
<summary><strong>Option 1: Conda Environment</strong></summary>

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate speedtransfer
```

> ⚠️ **Note**: This environment only supports Linux.

</details>

<details>
<summary><strong>Option 2: VS Code Dev Containers (Recommended)</strong></summary>

1. Install the [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) extension in VS Code
2. Click `Reopen in Container` when prompted, or:
   - Press `Ctrl+Shift+P` (Linux/Windows) or `Cmd+Shift+P` (Mac)
   - Type `Dev Containers: Reopen in Container`

</details>

<details>
<summary><strong>Option 3: Docker</strong></summary>

```bash
# Build the container
docker build -t speedtransfer .

# Run the container
docker run -it speedtransfer
```

</details>

### 📊 Analysis Pipeline

Our analysis framework provides comprehensive tools for studying bee rhythmicity and social interactions:

#### 🔬 Core Analysis Scripts

| Script | Function | Key Features |
|--------|----------|--------------|
| 🕐 [`cosinor_fit_per_bee.py`](analysis/cosinor_fit_per_bee.py) | **Individual Rhythm Analysis** | Fits cosine curves to movement speeds, detects 24h rhythms |
| 📊 [`mean_velocity_per_age_group.py`](analysis/mean_velocity_per_age_group.py) | **Age-based Velocity Analysis** | Generates mean velocity data per age group (10-min intervals) |
| 🌐 [`social_network_interaction_tree.py`](analysis/social_network_interaction_tree.py) | **Social Network Analysis** | Maps sequential interaction cascades, constructs graph-theoretic trees |
| 🤝 [`velocity_change_per_interaction.py`](analysis/velocity_change_per_interaction.py) | **Interaction Detection** | Identifies bee interactions, measures speed changes |
| 🎲 [`velocity_change_per_interaction_null_model.py`](analysis/velocity_change_per_interaction_null_model.py) | **Null Model Generation** | Creates randomized interaction models for statistical validation |

> 💡 **Note**: These scripts are designed for [slurmhelper](https://github.com/walachey/slurmhelper) compatibility for high-performance computing environments.

#### ⚙️ SLURM Configuration

For scripts with SLURM support, adapt the job configuration:
```python
# create job
job = SLURMJob("foo_job_name", "foo_job_directory") # Choose job name and a directory where the job is stored
job.map(run_job_2016, generate_jobs_2016()) # either _2016 or _2019

# set job parameters for slurm settings
job.qos = "standard"
job.partition = "main,scavenger"
job.max_memory = "{}GB".format(2)
job.n_cpus = 1
job.max_job_array_size = 5000
job.time_limit = datetime.timedelta(minutes=60)
job.concurrent_job_limit = 100
job.custom_preamble = "#SBATCH --exclude=g[013-015],b[001-004],c[003-004],g[009-015]"
job.exports = "OMP_NUM_THREADS=2,MKL_NUM_THREADS=2"
job.set_postprocess_fun(concat_jobs_2016) # either _2016 or _2019
```

```bash
# Create SLURM job array
python analysis_script.py --create

# Run jobs automatically
python analysis_script.py --autorun

# Or run manually
python analysis_script.py --run

# Post-process/concatenate results
python analysis_script.py --postprocess
```

> 📚 **More Information**: See [slurmhelper documentation](https://github.com/walachey/slurmhelper) or run `python script.py --help`

The remaining scripts can either be executed locally or on a HPC cluster using the provided bash scripts:
- `run_individual_job.sh` - Single job execution
- `run_parallel_jobs.sh` - Parallel job execution

## 📊 Visualization & Figures

Explore our comprehensive Jupyter notebooks for data visualization and figure generation:

### 📈 Available Notebooks

| Notebook | Content | Key Visualizations |
|----------|---------|-------------------|
| 📊 [`figure_panel_1.ipynb`](figures/figure_panel_1.ipynb) | **Movement & Rhythm Analysis** | Group/individual speeds, cosinor fits, age-based rhythmicity |
| 🤝 [`figure_panel_2.ipynb`](figures/figure_panel_2.ipynb) | **Interaction Analysis** | Speed transfer patterns, body-location mapping |
| 🗺️ [`figure_panel_3.ipynb`](figures/figure_panel_3.ipynb) | **Spatial Analysis** | Activity phase distribution, interaction cascades |
| 📋 [`figure_supplementary.ipynb`](figures/figure_supplementary.ipynb) | **Supplementary Data** | Observation statistics, additional metrics |
| 🎬 [`animations.ipynb`](figures/animations.ipynb) | **Dynamic Visualizations** | Animated activity flow patterns |

### 🎯 Quick Start with Figures

```bash
# Navigate to figures directory
cd figures/

# Launch Jupyter Lab
jupyter lab

# Open any notebook and run all cells
```

## 🔬 Simulation Framework

Our MATLAB-based simulation environment provides agent-based modeling capabilities for validating theoretical predictions:

### 🔧 Simulation Components

| Script | Purpose |
|--------|---------|
| `run_simulation.m` | Main simulation runner |
| `sketch_agents_2D.m` / `sketch_agents_3D.m` | Agent visualization |
| `draw_positions_from_gaussian.m` | Spatial position generation |
| `draw_velocities.m` | Velocity distribution modeling |
| `fit_sine.m` | Sine wave fitting for rhythm analysis |

## 💾 Data Access

<div align="center">

[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10869728-blue?style=flat-square&logo=zenodo)](https://zenodo.org/records/10869728)

</div>

### 📥 Download Instructions

```bash
# Download data from Zenodo (adjust link for different files)
wget https://zenodo.org/records/10869728/files/interactions_side0_2019.zip

# Extract to data directory
unzip data.zip

# Verify structure
ls data/
```

> ⚠️ **Important**: Data must be placed in the `data/` directory for proper path imports in analysis scripts. Alternatively adjust the `path_settings.py` file.

## 🏆 Citation

If you use this code or methodology in your research, please cite our work:

```bibtex
@article{mellert2024collective,
  title={Collective flow of circadian clock information in honeybee colonies},
  author={Mellert, Julia and K{\l}os, Weronika and Dormagen, David M and Wild, Benjamin and Zachariae, Adrian and Smith, Michael L and Galizia, C Giovanni and Landgraf, Tim},
  journal={bioRxiv},
  pages={2024--07},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

<div align="center">

**🐝 Made with 💛 for bee research**

[📖 Paper](https://www.biorxiv.org/content/10.1101/2024.07.29.605620v1) • [💾 Data](https://zenodo.org/records/10869728) • [🐛 Issues](../../issues) • [🤝 Contributing](../../pulls)

</div>
