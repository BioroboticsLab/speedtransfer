import pandas as pd
from slurmhelper import SLURMJob

import bb_rhythm.interactions

"""
In this script the results of a slurmhelper job array are collected. The resulting interaction dataframes per 
bee are concatenated to one df and the estimates of the cosinor dataframe is added.
"""

COSINOR_DF_PATH_2016 = "../data/2016/cosinor.pkl"
COSINOR_DF_PATH_2019 = "../data/2019/cosinor.pkl"

INTERACTION_SIDE_1_DF_PATH_2016 = "../data/2016/interactions_side1.pkl"
INTERACTION_SIDE_2_DF_PATH_2016 = "../data/2016/interactions_side2.pkl"
INTERACTION_SIDE_1_DF_PATH_2019 = "../data/2019/interactions_side1.pkl"
INTERACTION_SIDE_2_DF_PATH_2019 = "../data/2019/interactions_side2.pkl"

INTERACTION_SIDE_1_DF_PATH_2016_NULL_MODEL = (
    "../data/2016/interactions_side1_null_model.pkl"
)
INTERACTION_SIDE_1_DF_PATH_2019_NULL_MODEL = (
    "../data/2019/interactions_side1_null_model.pkl"
)

# define job
job = SLURMJob("velocity_change_per_interaction_null_2019_side1", "/scratch/juliam98/")

# iterate through job results and concat to dataframe
result_list = []
for kwarg, result_df in job.items(ignore_open_jobs=True):
    try:
        result_list.append(pd.DataFrame(result_df))
    except AttributeError:
        continue
interaction_df = pd.concat(result_list)

# read cosinor dataframe
cosinor_df = pd.read_pickle(COSINOR_DF_PATH_2019)
cosinor_df_subset = cosinor_df[
    ["bee_id", "amplitude", "r_squared", "date", "p_value", "phase", "age"]
]
del cosinor_df

# merge interaction dataframe and cosinor dataframe so cosinor fit paramters are added to interaction dataframe
interaction_df = bb_rhythm.interactions.add_circadianess_to_interaction_df(
    interaction_df.reset_index(), cosinor_df_subset
)

# save interaction dataframe
interaction_df.to_pickle(INTERACTION_SIDE_1_DF_PATH_2019_NULL_MODEL)
