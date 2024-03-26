import os
"""
In this file the paths for the project are provided.
"""

# Path to data
DATA_PATH = os.path.join(".." , "data")

# Path to aggregated data
AGG_DATA_PATH_2016 = os.path.join(".." , "aggregated_results", "2016")
AGG_DATA_PATH_2019 = os.path.join(".." , "aggregated_results", "2019")

# mean velocities
MEAN_VELOCITY_DF_PATH_2016 = os.path.join(DATA_PATH, "mean_velocity_2016.csv")
MEAN_VELOCITY_DF_PATH_2019 = os.path.join(DATA_PATH, "mean_velocity_2019.csv")

# cherry-picked velocities 2019
VELOCITY_2088_DF_PATH_2019 = os.path.join(DATA_PATH, "velocity_2088_2019.csv")
VELOCITY_5101_DF_PATH_2019 = os.path.join(DATA_PATH, "velocity_5101_2019.csv")

# velocities
VELOCITY_DF_PATH_2016 = os.path.join(DATA_PATH, "velocities_1_8-25_8_2016")
VELOCITY_DF_PATH_2019 = os.path.join(DATA_PATH, "velocities_20_8-14_9_2019")

# cosinor fit
COSINOR_DF_PATH_2016 = os.path.join(DATA_PATH, "cosinor_2016.csv")
COSINOR_DF_PATH_2019 = os.path.join(DATA_PATH, "cosinor_2019.csv")

# cosinor estimates aggregated in hive position
DIST_EXIT_SIDE_0_DF_PATH_2016 = os.path.join(AGG_DATA_PATH_2016, "dist_exit_df_side0_2016.csv")
DIST_EXIT_SIDE_1_DF_PATH_2016 = os.path.join(AGG_DATA_PATH_2016, "dist_exit_df_side1_2016.csv")
DIST_EXIT_SIDE_0_DF_PATH_2019 = os.path.join(AGG_DATA_PATH_2019, "dist_exit_df_side0_2019.csv")
DIST_EXIT_SIDE_1_DF_PATH_2019 = os.path.join(AGG_DATA_PATH_2019, "dist_exit_df_side1_2019.csv")

# Interactions
INTERACTION_SIDE_0_DF_PATH_2016 = os.path.join(DATA_PATH, "interactions_side0_2016.csv")
INTERACTION_SIDE_1_DF_PATH_2016 = os.path.join(DATA_PATH, "interactions_side1_2016.csv")
INTERACTION_SIDE_0_DF_PATH_2019 = os.path.join(DATA_PATH, "interactions_side0_2019.csv")
INTERACTION_SIDE_1_DF_PATH_2019 = os.path.join(DATA_PATH, "interactions_side1_2019.csv")

# Interaction null model
INTERACTION_SIDE_0_DF_PATH_2016_NULL_MODEL = os.path.join(DATA_PATH, "interactions_side0_null_model_2016.csv")
INTERACTION_SIDE_0_DF_PATH_2019_NULL_MODEL = os.path.join(DATA_PATH, "interactions_side0_null_model_2019.csv")

# Interaction trees
INTERACTION_TREE_DF_PATH_2016 = os.path.join(DATA_PATH, "interaction_tree_paths_2016.csv")
INTERACTION_TREE_DF_PATH_2019 = os.path.join(DATA_PATH, "interaction_tree_paths_2019.csv")


def set_parameters(year: int, side: int) -> tuple[str, tuple[int, int], str]:
    """
    Set parameters for given year and side of hive.
    """
    if year == 2016:
        exit_pos = (0, 250)
        if side == 0:
            interaction_df_path = INTERACTION_SIDE_0_DF_PATH_2016
            save_to = os.path.join("..", "aggregated_results", "2016", "dist_exit_df_side0_2016.csv")
        elif side == 1:
            interaction_df_path = INTERACTION_SIDE_1_DF_PATH_2016
            save_to = os.path.join("..", "aggregated_results", "2016", "dist_exit_df_side1_2016.csv")
        else:
            assert ValueError(f"No data for the side '{side}' available. Possible options are '0' and '1'.")
    elif year == 2019:
        exit_pos = (5, 264)
        if side == 0:
            interaction_df_path = INTERACTION_SIDE_0_DF_PATH_2019
            save_to = os.path.join("..", "aggregated_results", "2019", "dist_exit_df_side0_2019.csv")
        elif side == 1:
            interaction_df_path = INTERACTION_SIDE_1_DF_PATH_2019
            save_to = os.path.join("..", "aggregated_results", "2019", "dist_exit_df_side1_2019.csv")
        else:
            assert ValueError(f"No data for the side '{side}' available. Possible options are '0' and '1'.")
    else:
        assert ValueError(f"No data for the year '{year}' available. Possible options are '2016' and '2019'.")
    return interaction_df_path, exit_pos, save_to