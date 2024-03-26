import pandas as pd
import numpy as np
import math
import sys
import os
import argparse
from pathlib import Path

import bb_rhythm.interactions

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
)
parser.add_argument(
    "--side", type=int, help="Which side of the hive to analyze the data for. (0 or 1)"
)

args = parser.parse_args()


def set_parameters(year: int, side: int) -> tuple[str, tuple[int, int], str]:
    """
    Set parameters for given year and side of hive.
    """
    if year == 2016:
        exit_pos = (0, 250)
        if side == 0:
            interaction_df_path = path_settings.INTERACTION_SIDE_0_DF_PATH_2016
            save_to = path_settings.DIST_EXIT_SIDE_0_DF_PATH_2016
        elif side == 1:
            interaction_df_path = path_settings.INTERACTION_SIDE_1_DF_PATH_2016
            save_to = path_settings.DIST_EXIT_SIDE_1_DF_PATH_2016
        else:
            assert ValueError(f"No data for the side '{side}' available. Possible options are '0' and '1'.")
    elif year == 2019:
        exit_pos = (5, 264)
        if side == 0:
            interaction_df_path = path_settings.INTERACTION_SIDE_0_DF_PATH_2019
            save_to = path_settings.DIST_EXIT_SIDE_0_DF_PATH_2019
        elif side == 1:
            interaction_df_path = path_settings.INTERACTION_SIDE_1_DF_PATH_2019
            save_to = path_settings.DIST_EXIT_SIDE_1_DF_PATH_2019
        else:
            assert ValueError(f"No data for the side '{side}' available. Possible options are '0' and '1'.")
    else:
        assert ValueError(f"No data for the year '{year}' available. Possible options are '2016' and '2019'.")
    return interaction_df_path, exit_pos, save_to


def map_phase_to_24_hours(interaction_df: pd.DataFrame, columns: list):
    """
    Maps acrophase in radians to hours.
    """
    for column in columns:
        interaction_df[column] = ((-24 * interaction_df[column] / (2 * np.pi)) + 12) % 24


if __name__ == "__main__":
    # set sys path and import path settings
    sys.path.append(str(Path("aggregate_cosinor_estimates_per_hive_position.py").resolve().parents[1]))
    import path_settings

    interaction_df_path, exit_pos, save_to = set_parameters(args.year, args.side)

    # load interaction df
    interaction_df = pd.read_pickle(interaction_df_path)

    # filter overlap
    interaction_df = bb_rhythm.interactions.filter_overlap(interaction_df)

    # combine df so all bees are considered as focal
    interaction_df = bb_rhythm.interactions.combine_bees_from_interaction_df_to_be_all_focal(
        interaction_df
    )

    # map phase to 24h
    map_phase_to_24_hours(interaction_df, ["phase_focal", "phase_non_focal"])

    # aggregate by position
    interaction_df["x_pos_start_focal"] = interaction_df["x_pos_start_focal"].round()
    interaction_df["y_pos_start_focal"] = interaction_df["y_pos_start_focal"].round()

    # calculate dist exit
    interaction_df["entrance_dist_focal"] = [
        math.dist(exit_pos, (row["x_pos_start_focal"], row["y_pos_start_focal"]))
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
    dist_agg_df.to_csv(save_to, index=False)
