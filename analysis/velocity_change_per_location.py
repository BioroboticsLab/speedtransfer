import numpy as np
import pandas as pd
import os
import pickle
import argparse
from bb_rhythm import rhythm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_settings

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
)
parser.add_argument(
    "--side", type=int, help="Which side of the hive to analyze the data for. (0 or 1)"
)
args = parser.parse_args()

_, path, _, _, save_to, exit_pos = path_settings.set_parameters(args.year, args.side)

def read_data(path):
    # Read interaction data.
    df = pd.read_pickle(path)

    # Remove unnecessary columns.
    df = df[
        [
            "x_pos_start_bee0",
            "y_pos_start_bee0",
            "vel_change_bee0",
            "rel_change_bee0",
            "x_pos_start_bee1",
            "y_pos_start_bee1",
            "vel_change_bee1",
            "rel_change_bee1",
            "interaction_start",
        ]
    ]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df


def replace_time_with_hour(df):
    df["hour"] = df["interaction_start"].dt.hour
    df.drop(columns=["interaction_start"], inplace=True)
    return df


def swap_focal_bee(df):
    """Sets the bee with the higher increase in speed to be the focal one.

    Args:
        df (pd.DataFrame): DataFrame with speed changes and positions of both focal bees.

    Returns:
        pd.DataFrame: Data for speed change of focal bee.
    """
    new_row_ls = []

    df = df.to_dict(orient="records")

    spatial_bin_size = 1

    for row in df:
        if row["vel_change_bee0"] > row["vel_change_bee1"]:
            new_row_ls.append(
                [
                    int(round(row["x_pos_start_bee0"] / spatial_bin_size)),
                    int(round(row["y_pos_start_bee0"] / spatial_bin_size)),
                    row["vel_change_bee0"],
                    row["hour"],
                ]
            )
        elif row["vel_change_bee1"] > row["vel_change_bee0"]:
            new_row_ls.append(
                [
                    int(round(row["x_pos_start_bee1"] / spatial_bin_size)),
                    int(round(row["y_pos_start_bee1"] / spatial_bin_size)),
                    row["vel_change_bee1"],
                    row["hour"],
                ]
            )

    res = pd.DataFrame(new_row_ls, columns=["x_grid", "y_grid", "vel_change", "hour"])
    return res


def concat_grids_over_time(df, var="vel_change", aggfunc="median", scale=False):
    """Creates a 3d numpy array with velocity changes for each hour and x,y-position.

    Args:
        df (pd.DataFrame): Data containeing x_grid, y_grid, hour and vel_change columns.
        var (str): Variable to aggregate.
        aggfunc (str): Which function to use for aggregating (mean or median).
        scale (bool, optional): Whether to scale the timeseries at each location to a rangebetween 0 and 1. Defaults to False.

    Returns:
        np.array: Accumulator of shape 24 x height x width.
    """

    y_vals = sorted(np.unique(df.y_grid))
    x_vals = sorted(np.unique(df.x_grid))

    # Create accumulator.
    h, w = len(y_vals), len(x_vals)
    accumulator = np.zeros((24, h, w))

    # Create grid for each hour and add to accumulator.
    for hour in range(24):
        subset = df.loc[df.hour == hour]
        subset = subset.drop(columns=["hour"])
        grid = pd.pivot_table(
            data=subset, index="y_grid", columns="x_grid", values=var, aggfunc=aggfunc
        )
        grid = grid.reindex(index=y_vals, columns=x_vals)
        grid = grid.to_numpy()
        accumulator[hour] = grid

    if scale:
        for i in range(h):
            for j in range(w):
                accumulator[:, i, j] = rhythm.min_max_scaling(accumulator[:, i, j])

    return accumulator


def convert_grid_to_df(grid_3d):
    hours, h, w = grid_3d.shape

    # Convert to df.
    row_ls = []
    for hour in range(hours):
        for y in range(h):
            for x in range(w):
                val = grid_3d[hour, y, x]
                row_ls.append([x, y, hour, val])

    return pd.DataFrame(row_ls, columns=["x_grid", "y_grid", "hour", "vel_change"])


def get_vel_ch_vs_dist(df, aggfunc="mean"):

    # Calculate distance to exit.
    exit_x = exit_pos[0]
    exit_y = exit_pos[1]

    df["dist"] = np.sqrt((df.x_grid - exit_x) ** 2 + (df.y_grid - exit_y) ** 2)
    df.drop(columns=["x_grid", "y_grid"], inplace=True)

    # Bin distance into quantiles and use lower boundary for label.
    df["dist"] = pd.cut(df["dist"], 18, precision=0).map(lambda x: int(x.right))

    # Create pivot for heatmap.
    pivot = pd.pivot_table(
        df, index="dist", columns="hour", values="vel_change", aggfunc=aggfunc,
    )

    return pivot


if __name__ == "__main__":

    var = "vel_change"
    aggfunc = "mean"
    scale = False

    df = read_data(path)
    df = replace_time_with_hour(df)
    df = swap_focal_bee(df)
    
    if scale:
        grid_3d = concat_grids_over_time(df, var, aggfunc, scale=scale)
        df = convert_grid_to_df(grid_3d)

    vel_ch_vs_dist_and_hour = get_vel_ch_vs_dist(df, aggfunc)
    save_to = os.path.join(
        os.pardir, "aggregated_results", str(args.year), "vel_change_per_dist_and_hour_grid.pkl"
    )
    vel_ch_vs_dist_and_hour.to_pickle(save_to)
