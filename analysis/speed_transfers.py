"""
Compute change in speed depending on start velocity and phase of interacting bees.
The results can be compared to null model data.
"""

import argparse
import os

import numpy as np
import pandas as pd


def read_necessary_data(year, null):
    # Select correct file from data directory.
    suffix = "null_model" if null else "side1"
    path = os.path.join(os.pardir, "data", year, f"interactions_{suffix}.pkl")
    df = pd.read_pickle(path)

    # Select necessary columns.
    col_subset = [
        "vel_change_bee0",
        "vel_change_bee1",
        "rel_change_bee0",
        "rel_change_bee1",
        "phase_bee0",
        "phase_bee1",
        "r_squared_bee0",
        "r_squared_bee1",
        "age_bee0",
        "age_bee1",
    ]
    if not null:
        col_subset.append("overlapping")
    df = df[col_subset]

    # Compute start velocity.
    df["start_vel_bee0"] = df["vel_change_bee0"] * 100 / df["rel_change_bee0"]
    df["start_vel_bee1"] = df["vel_change_bee1"] * 100 / df["rel_change_bee1"]
    df.drop(columns=["rel_change_bee0", "rel_change_bee1"], inplace=True)

    # Filter for overlapping only.
    if not null:
        df = df[df["overlapping"].values]
        df.drop(columns=["overlapping"], inplace=True)

    # Convert phase to hours.
    df["phase_bee0"] = ((-24 * df["phase_bee0"] / (2 * np.pi)) + 12) % 24
    df["phase_bee1"] = ((-24 * df["phase_bee1"] / (2 * np.pi)) + 12) % 24

    return df


def make_both_bees_focal(
    interactions_df,
    var_list=["vel_change", "start_vel", "phase", "r_squared", "age"],
    extra_cols=[],
):
    """Combine data such that both bees are considered focal once for each interaction."""
    df = pd.DataFrame()

    for var in var_list:
        df["%s_focal" % var] = pd.concat(
            [interactions_df["%s_bee0" % var], interactions_df["%s_bee1" % var]]
        )
        df["%s_non_focal" % var] = pd.concat(
            [interactions_df["%s_bee1" % var], interactions_df["%s_bee0" % var]]
        )

    for col in extra_cols:
        df[col] = pd.concat([interactions_df[col], interactions_df[col]])

    return df


def bin_quantiles(df, var):
    """Bin the 'var'-columns for the focal and non-focal bees into 6 quantiles."""
    df["%s_focal" % var] = pd.qcut(
        np.array(df["%s_focal" % var]), 6, labels=[1, 2, 3, 4, 5, 6], duplicates="drop"
    )
    df["%s_non_focal" % var] = pd.qcut(
        np.array(df["%s_non_focal" % var]),
        6,
        labels=[1, 2, 3, 4, 5, 6],
        duplicates="drop",
    )
    return df


def create_combination_matrix(df, var):
    """Get median speed change for each combination of given variable values
    for the focal and non-focal bee.
    """
    return pd.pivot_table(
        data=df,
        values="vel_change_focal",
        index="%s_non_focal" % var,
        columns="%s_focal" % var,
        aggfunc="median",
    ).to_numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--null",
        required=False,
        action="store_true",
        help="Whether to perform analysis for null model or real data.",
    )
    parser.add_argument(
        "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
    )
    parser.add_argument(
        "--side",
        type=int,
        help="Which side of the hive to analyze the data for. (0 or 1)",
    )

    args = parser.parse_args()
    null = args.null
    year = args.year

    # Prepare data.
    interactions_df = read_necessary_data(year, null)
    combined_df = make_both_bees_focal(interactions_df)

    # Save aggregated results.
    for var in ["phase", "start_vel", "r_squared", "age"]:
        combined_df = bin_quantiles(combined_df, var)
        combination_matrix = create_combination_matrix(combined_df, var)
        suffix = [var, str(args.side)]
        if null:
            suffix.append("null")
        save_to = os.path.join(
            os.pardir,
            "aggregated_results",
            str(year),
            f"speed_trans_vs_{'_'.join(suffix)}.npy",
        )
        np.save(save_to, combination_matrix)
