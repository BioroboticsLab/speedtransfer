import os
"""
In this file the paths for the project are provided.
"""

# Path to data
DATA_PATH_2016 = os.path.join(".." , "data", "2016")
DATA_PATH_2019 = os.path.join(".." , "data", "2019")

# Path to aggregated data
AGG_DATA_PATH_2016 = os.path.join(".." , "aggregated_results", "2016")
AGG_DATA_PATH_2019 = os.path.join(".." , "aggregated_results", "2019")

# mean velocities
MEAN_VELOCITY_DF_PATH_2016 = os.path.join(DATA_PATH_2016, "mean_velocity.pkl")
MEAN_VELOCITY_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "mean_velocity.pkl")

# cherry-picked velocities 2019
VELOCITY_2088_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "velocity_2088.pkl")
VELOCITY_5101_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "velocity_5101.pkl")

# velocities
VELOCITY_DF_PATH_2016 = os.path.join(DATA_PATH_2016, "velocities_1_8-25_8_2016")
VELOCITY_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "velocities_20_8-14_9_2019")

# cosinor fit
COSINOR_DF_PATH_2016 = os.path.join(DATA_PATH_2016, "cosinor.pkl")
COSINOR_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "cosinor.pkl")

# cosinor estimates aggregated in hive position
DIST_EXIT_SIDE_0_DF_PATH_2016 = os.path.join(AGG_DATA_PATH_2016, "dist_exit_df_side0.csv")
DIST_EXIT_SIDE_1_DF_PATH_2016 = os.path.join(AGG_DATA_PATH_2016, "dist_exit_df_side1.csv")
DIST_EXIT_SIDE_0_DF_PATH_2019 = os.path.join(AGG_DATA_PATH_2019, "dist_exit_df_side0.csv")
DIST_EXIT_SIDE_1_DF_PATH_2019 = os.path.join(AGG_DATA_PATH_2019, "dist_exit_df_side1.csv")

# Interactions
INTERACTION_SIDE_0_DF_PATH_2016 = os.path.join(DATA_PATH_2016, "interactions_side0.pkl")
INTERACTION_SIDE_1_DF_PATH_2016 = os.path.join(DATA_PATH_2016, "interactions_side1.pkl")
INTERACTION_SIDE_0_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "interactions_side0.pkl")
INTERACTION_SIDE_1_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "interactions_side1.pkl")

# Interaction null model
INTERACTION_SIDE_0_DF_PATH_2016_NULL_MODEL = os.path.join(DATA_PATH_2016, "interactions_side0_null_model.pkl")
INTERACTION_SIDE_0_DF_PATH_2019_NULL_MODEL = os.path.join(DATA_PATH_2019, "interactions_side0_null_model.pkl")

# Interaction trees
INTERACTION_TREE_DF_PATH_2016 = os.path.join(DATA_PATH_2016, "interaction_tree_paths.pkl")
INTERACTION_TREE_DF_PATH_2019 = os.path.join(DATA_PATH_2019, "interaction_tree_paths.pkl")


def set_parameters(year: int, side: int) -> tuple[str, tuple[int, int], str]:
    """
    Set parameters for given year and side of hive.
    """
    if year == 2016:
        exit_pos = (0, 250)
        if side == 0:
            interaction_df_path = INTERACTION_SIDE_0_DF_PATH_2016
            save_to = os.path.join("..", "aggregated_results", "2016", "dist_exit_df_2016_side0.csv")
        elif side == 1:
            interaction_df_path = INTERACTION_SIDE_1_DF_PATH_2016
            save_to = os.path.join("..", "aggregated_results", "2016", "dist_exit_df_2016_side1.csv")
        else:
            assert ValueError(f"No data for the side '{side}' available. Possible options are '0' and '1'.")
    elif year == 2019:
        exit_pos = (5, 264)
        if side == 0:
            interaction_df_path = INTERACTION_SIDE_0_DF_PATH_2019
            save_to = os.path.join("..", "aggregated_results", "2019", "dist_exit_df_2019_side0.csv")
        elif side == 1:
            interaction_df_path = INTERACTION_SIDE_1_DF_PATH_2019
            save_to = os.path.join("..", "aggregated_results", "2019", "dist_exit_df_2016_side1.csv")
        else:
            assert ValueError(f"No data for the side '{side}' available. Possible options are '0' and '1'.")
    else:
        assert ValueError(f"No data for the year '{year}' available. Possible options are '2016' and '2019'.")
    return interaction_df_path, exit_pos, save_to