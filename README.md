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
The scripts are 

To create a slurmarray with the jobs:
```
python cosinor_fit_per_bee.py --create
```

To run it:
```
python cosinor_fit_per_bee.py --autorun
```
or
```
python cosinor_fit_per_bee.py --run
```
and to concatenate the output of the slurm arrays to a single dataframe.
```
python cosinor_fit_per_bee.py --postprocess
```

More flags and further information can be found [here](htttps://ww.github.com/walachey/slurmhelper) or 
with ``python foo.py --help``.

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
