from slurmhelper import SLURMJob
import pandas as pd

"""
In this script the resuslt of a slurmhelper job array are collected. The resulting 10min mean velocity dataframes per 
bee are grouped by age and the mean velocity per 60min was calculated for each age.
"""

MEAN_VELOCITY_DF_PATH_2016 = "../data/2016/mean_velocity.pkl"
MEAN_VELOCITY_DF_PATH_2019 = "../data/2019/mean_velocity.pkl"

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
    MEAN_VELOCITY_DF_PATH_2016
)
