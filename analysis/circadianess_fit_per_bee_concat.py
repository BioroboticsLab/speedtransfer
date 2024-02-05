from slurmhelper import SLURMJob
import pandas as pd

"""
In this script the results of a slurmhelper job array are collected. The resulting cosinor fit parameter dataframes per 
bee are concated to one df.
"""

# create job from existing results
job = SLURMJob("2016_circadian_velocities_cosinor_std", "/scratch/juliam98/")

# collect all dfs per bee and combine to one df
result_list = []
for kwarg, result in job.items(ignore_open_jobs=True):
    # case of death or non-existent tracking data
    if type(result) is dict:
        continue
    else:
        result_list.append(result)
df = pd.concat(result_list)

# save df
df.to_pickle("~/diurnal_rhythm_paper/data/dataframes/circadianess_2016/cosinor_std.pkl")
