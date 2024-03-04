from slurmhelper import SLURMJob
import pandas as pd

"""
In this script the resuslt of a slurmhelper job array are collected. The resulting 10min mean velocity dataframes per 
bee are grouped by age and the mean velocity per 60min was calculated for each age.
"""

# create mean job from existing results
job = SLURMJob(
    "BB2016_bayesian_velocity_mean_10min_all_new_bees", "/home/juliam98/slurm_tryouts"
)

# for each 10min mean velocity per bee combine to one
result_df = pd.DataFrame(columns=["time", "velocity", "age"])
for kwarg, result in job.items():
    result_df = pd.concat([result_df, result], ignore_index=True)

# get mean velocity df per age
result_df = result_df.groupby(["time", "age"])["velocity"].mean().reset_index()

# save df
result_df.to_pickle(
    "~/diurnal_rhythm_paper/data/dataframes/velocity_mean_10min_all_bb2016_bayesian_bees.pkl"
)
