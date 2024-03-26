# Speed transfer contributes to circadian activity

Analyzing social behaviour and rhythmicity in honey bee colony.

Preprint:

## Goal
The goal of this project is to provide a comprehensive implementation of methods for studying and analyzing behavioral 
rhythmicity in bees which is handled in the paper [Speed transfer contributes to circadian activity](). The repository 
is offering a toolkit for data acquisition, preprocessing, and analysis, focusing on the following key objectives:
1) Data Acquisition and Preprocessing: Implement methods for collecting trajectory data of marked bees using the 
[BeesBook](10.3389/frobt.2018.00035) system. 
2) Cosinor Model of Individuals’ Activity: Develop algorithms for fitting cosine curves to individual bee movement 
speeds to analyze rhythmic expression and detect 24-hour activity patterns.
3) Interaction Modeling: Implement techniques for detecting and analyzing interactions between bees based on trajectory 
data, including timing, duration, and impact on movement speed.
4) Statistical Analysis and Validation: Conduct statistical tests to validate the significance of observed behavior 
patterns and interactions compared to null models and simulated data.
5) Visualization and Interpretation: Provide visualization tools to interpret and present the results of the analysis.

The analysis was done for the period 01.08.-25.08.2016 and 20.08-14.09.2019, but can be easily adapted for different
time periods and data sets.

![overview_figure.png](figures%2Fimgs%2Foverview_figure.png)

## Structure
The code contains several folders containing python scripts, jupyter notebooks or data.

This folder contains several python scripts for creating data structures, data models, but also for analysing the data. 
Most of them are designed to be run on a SLURM system. Further use of the scripts can be found in [Usage](#usage).
```
analysis\
```

This folder contains several jupyter notebooks for visualizing the data. Each panel figure of the is visualized by one 
of the notebooks. The panel figures and sub figures are stored in the imgs sub folder as .png and .svg files.
```
figures\
    \imgs
```

This folder contains aggregates of the data which are used for faster visualizing in the figure notebooks.
```
\aggregated_results
    \2016
    \2019
```

This file contains the output dataframes of the analysis scripts. The data is used for further analysis in the figure 
notebooks. The data folder is currently empty but can be downloaded into the data directory [here](zenododata).
```
\data
    \2016
    \2019
```

This file contains all the paths used in the repository and needs to be modified in case different data and data paths 
are used.
```
path_settings.py
```

## Usage
### Analysis
Here scripts are provided for bee rhythm behavior analysis, including data acquisition, preprocessing, and statistical 
validation. It focuses on trajectory data collection, cosine curve fitting for activity patterns, interaction analysis, 
and statistical validation, with most scripts designed for use with a Slurm manager.

The following scripts are designed with [slurmhelper](htttps://ww.github.com/walachey/slurmhelper) and provide the following functionalities:
* [cosinor_fit_per_bee.py](analysis%2Fcosinor_fit_per_bee.py):  It implements a method to analyze the rhythm expression of individual bees by fitting 
cosine curves to their movement speeds using a cosinor model. It divides for a given time period the speeds into 
three-day intervals, calculates fits for each day, and determines the presence of a 24-hour rhythm based on statistical 
significance tests, providing outputs such as phase mapping and fit strength indicators.
* [mean_velocity_per_age_group.py](analysis%2Fmean_velocity_per_age_group.py): 
This script generates a pandas dataframe containing the mean velocity data per 10 minutes for each age.
* [social_network_interaction_tree.py](analysis%2Fsocial_network_interaction_tree.py): This script provides a method to capture the cumulative effect of sequential 
interactions among bees, beyond just one-on-one interactions. It focuses on interactions occurring between 10 am and 3 pm, 
selecting a random subset of significantly rhythmic bees younger than 5 days old, with peak activity after 12 pm. By recursively 
tracing back interactions that positively impacted the focal bee's speed, it constructs a graph-theoretic tree structure, 
with a maximum time window of 30 minutes between interactions and a maximum cascade duration of 2 hours.
* [velocity_change_per_interaction.py](analysis%2Fvelocity_change_per_interaction.py): The script implements an interaction detection algorithm adn how the speed
of a bee is affected by such an interaction. It identifies interactions between bees by simultaneous detection, requiring 
a confidence threshold of 0.25 and thorax marking distances within 14 mm. If detections occur within a 1-second interval, 
interactions are consolidated, accounting for overlapping bodies and counting both bees as focal once. For each interaction, 
the script records information including timing, duration, partner age, rhythmicity features (phase and R²), speed change 
after interaction, and overlap image indicating contact position and angle. Speed change is computed as the difference in 
average speed before and after the interaction.
* [velocity_change_per_interaction_null_model.py](analysis%2Fvelocity_change_per_interaction_null_model.py): This script implements the creation of an interaction null model.
The model is created by taking the distribution of the start and end times of the interactions and selecting two random 
bees at those times that the bees were detected in the hive at that time. These pairs of bees are considered as 
"interacting" and their speed change is calculated.

Each file contains at the end a slurmhelper job definition which needs to be adapted:
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

To create a slurmarray with these jobs:
```
python foo.py --create
```
To run it:
```
python foo.py --autorun
```
or
```
python foo.py --run
```
and to concatenate the output of the slurm arrays to a single dataframe. It is often the case that the output files are
too large to be processed without slurm. If that is the case this line should be copied into the [run_individual_job.sh](analysis%2Frun_individual_job.sh)
script and then run using it accordingly.
```
python foo.py --postprocess
```
More flags and further information can be found [here](htttps://ww.github.com/walachey/slurmhelper) or 
with ```python foo.py --help```.


### Figures
Dive into the jupyter notebooks which contain visualizations of the data. 
``figure_panel_1.ipynb``:
* movements speeds of the bees as a group or as an individual
* cosinor fit
* age distribution and rhythmicity among different age groups
* cosinor estimates in relation to distance to hive exit

``figure_panel_2.ipyng``:
* post-interaction movement speed change and speed transfer
* body-locations involved in speed transfer

``figure_panel_3.ipyng``:
* spatial distribution of activity phase
* spatial and temporal distribution of post-interaction movement speed change
* spatial distribution of phase of simulated agents
* spatial distribution of activating interaction cascades

``figure_supplementary.ipyng``:
* Number of bees observed in the two time periods

## Simulation

## Data
The data for the project can be found and downloaded [here](zenododdata). The data should be moved to the data folder to
have valid path imports in the scripts.

## Citation
