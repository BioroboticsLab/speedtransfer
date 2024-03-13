import pandas as pd
import numpy as np
import math

import bb_rhythm.interactions

from .. import path_settings


# load interaction df
interaction_df = pd.read_pickle(path_settings.INTERACTION_SIDE_2_DF_PATH_2016)

# filter overlap
interaction_df = bb_rhythm.interactions.filter_overlap(interaction_df)

# combine df so all bees are considered as focal
interaction_df = bb_rhythm.interactions.combine_bees_from_interaction_df_to_be_all_focal(
    interaction_df
)

# map phase to 24h
interaction_df["phase_focal"] = (
    (-24 * interaction_df["phase_focal"] / (2 * np.pi)) + 12
) % 24
interaction_df["phase_non_focal"] = (
    (-24 * interaction_df["phase_non_focal"] / (2 * np.pi)) + 12
) % 24

# aggregate by position
interaction_df["x_pos_start_focal"] = interaction_df["x_pos_start_focal"].round()
interaction_df["y_pos_start_focal"] = interaction_df["y_pos_start_focal"].round()

# calculate dist exit
exit_pos_2016 = (0, 250)
exit_pos_2019 = (5, 264)
interaction_df["entrance_dist_focal"] = [
    math.dist(exit_pos_2016, (row["x_pos_start_focal"], row["y_pos_start_focal"]))
    for index, row in interaction_df.iterrows()
]

# aggregate per hive position and for each parameter calculate median, std and count
dist_agg_df = interaction_df.groupby(["x_pos_start_focal", "y_pos_start_focal"]).agg(
    theta_start_focal_median=("theta_start_focal", "median"),
    theta_start_focal_count=("theta_start_focal", "count"),
    theta_start_focal_std=("theta_start_focal", "std"),
    age_focal_median=("age_focal", "median"),
    age_focal_count=("age_focal", "count"),
    age_focal_std=("age_focal", "std"),
    r_squared_focal_median=("r_squared_focal", "median"),
    r_squared_focal_count=("r_squared_focal", "count"),
    r_squared_focal_std=("r_squared_focal", "std"),
    phase_focal_median=("phase_focal", "median"),
    phase_focal_count=("phase_focal", "count"),
    phase_focal_std=("phase_focal", "std"),
    entrance_dist_focal_median=("entrance_dist_focal", "median"),
    entrance_dist_focal_count=("entrance_dist_focal", "count"),
    entrance_dist_focal_std=("entrance_dist_focal", "std"),
)

# save pickle
dist_agg_df.to_pickle("../data/2016/dist_exit_df_2016.pkl")
