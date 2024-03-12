from slurmhelper import SLURMJob

import bb_rhythm.network

"""
In this script the results of a slurmhelper job array are collected. The resulting interaction trees are collected and 
each node of all paths in the interaction trees are concatenated to one dataframe.
"""


INTERACTION_TREE_DF_PATH_2016 = "../data/2016/sampled_interaction_tree_paths.pkl"
INTERACTION_TREE_DF_PATH_2019 = "../data/2019/sampled_interaction_tree_paths.pkl"

job = SLURMJob("interaction_model_tree_2019_long", "/scratch/juliam98/")
path_df = bb_rhythm.network.tree_to_path_df(job)
path_df.reset_index(inplace=True, drop=True)
path_df.to_pickle(INTERACTION_TREE_DF_PATH_2019)
