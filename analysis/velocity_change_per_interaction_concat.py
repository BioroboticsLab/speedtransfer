import pandas as pd
from bb_rhythm import interactions
from slurmhelper import SLURMJob

job = SLURMJob("velocity_change_per_interaction_null_2019_side1", "/scratch/juliam98/")

result_list = []
for kwarg, result_df in job.items(ignore_open_jobs=True):
    try:
        result_list.append(pd.DataFrame(result_df))
    except AttributeError:
        continue
df = pd.concat(result_list)
circadian_df = pd.read_pickle("~/../../scratch/juliam98/zenodo/2019/cosinor.pkl")
circadian_df_subset = circadian_df[
    ["bee_id", "amplitude", "r_squared", "date", "p_value", "phase", "age"]
]
del circadian_df
df = interactions.add_circadianess_to_interaction_df(
    df.reset_index(), circadian_df_subset
)
df.to_pickle("/scratch/juliam98/data/2019/interactions_side1_null_model.pkl")
