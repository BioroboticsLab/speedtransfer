import pandas as pd
import numpy as np
import sys
import os
import argparse
from pathlib import Path

import bb_rhythm.interactions
import bb_rhythm.utils
import bb_rhythm.statistics
import bb_rhythm.plotting

"""
This script groups the interaction data frame and its null model in 6-equally sized bins
and compares them statistically. For that a Welch-test is performed and its precondition
(normality and unequal variance of samples) is tested. Different parameters can be chosen
for binning the data frame e.g. start velocity, age, phase and r_squared.

Example workflow for the analysis of 2019 data of side 0:

    # First compute area of overlap for small batches of interactions in parallel.
    comparison_interaction_null_model_test.py --year 2019 --side 0 --binning_var 'start_vel'
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
)
parser.add_argument(
    "--side", type=int, help="Which side of the hive to analyze the data for. (0 or 1)"
)
parser.add_argument(
    "--binning_var",
    type=str,
    help="Which parameter the interactions are binned with. ('start_vel', 'age', 'phase', 'r_squared')",
)

args = parser.parse_args()


def set_binning_var(binning_var_flag: str) -> dict:
    if binning_var_flag == "start_vel":
        return {
            "velocity_start_focal": "Start velocity of focal bee [mm/s]",
            "velocity_start_non_focal": "Start velocity of non-focal bee [mm/s]",
        }
    if binning_var_flag == "age":
        return {
            "age_focal": "Age of focal bee [d]",
            "age_non_focal": "Age of non-focal bee [d]",
        }
    if binning_var_flag == "phase":
        return {
            "phase_focal": "Phase of focal bee [h]",
            "phase_non_focal": "Phase of non-focal bee [h]",
        }
    if binning_var_flag == "r_squared":
        return {
            "r_squared_focal": "R² of focal bee",
            "r_squared_non_focal": "R² of non-focal bee",
        }
    if binning_var_flag not in ["start_vel", "age", "phase", "r_squared"]:
        assert ValueError(
            f"data can't be binned by '{binning_var_flag}'. Possible options are 'start_vel', 'age', 'phase' or 'r_squared'."
        )


def prepare_df_for_testing(df: pd.DataFrame, binning_dict: dict, overlap: bool):
    if overlap:
        # filter overlap
        df = bb_rhythm.interactions.filter_overlap(df)

    # combine df so all bees are considered as focal
    df = bb_rhythm.interactions.combine_bees_from_interaction_df_to_be_all_focal(df)

    # calculate start velocity
    bb_rhythm.interactions.get_start_velocity(df)

    # subset
    df = df.loc[
        :,
        list(binning_dict.keys())
        + ["vel_change_bee_focal", "vel_change_bee_non_focal"],
    ]

    # filter and replace nan
    df.replace({-np.inf: np.nan, np.inf: np.nan}, inplace=True)
    df.dropna(inplace=True)

    # add bins
    binning = bb_rhythm.utils.Binning(
        bin_name="bins_focal", bin_parameter=list(binning_dict.keys())[0]
    )
    df = binning.add_bins_to_df(
        df, n_bins=6, step_size=None, bin_max_n=None, bin_labels=None
    )
    binning = bb_rhythm.utils.Binning(
        bin_name="bins_non_focal", bin_parameter=list(binning_dict.keys())[1]
    )
    df = binning.add_bins_to_df(
        df, n_bins=6, step_size=None, bin_max_n=None, bin_labels=None
    )
    return df


def sample_sizes(df: pd.DataFrame) -> dict:
    sample_size_dict = (
        df[["bins_focal", "bins_non_focal", "vel_change_bee_focal"]]
        .groupby(["bins_focal", "bins_non_focal"])
        .count()
        .to_dict()
    )["vel_change_bee_focal"]
    return sample_size_dict


def get_in_case_reverse_tuple(key_tuple, sample_sizes):
    try:
        s_size = sample_sizes[key_tuple]
    except KeyError:
        s_size = sample_sizes[(key_tuple[1], key_tuple[0])]
    return s_size


def extract_test_stats(
    test_results: dict, sample_sizes: dict, test_name: str
) -> pd.DataFrame:
    p_values = []
    test_statistics = []
    sample_size = []
    bin_pair = []
    for key, value in test_results.items():
        bin_pair.append(key)
        p_values.append(value.pvalue)
        test_statistics.append(value.statistic)
        if type(key[0]) == str:
            sample_size.append(get_in_case_reverse_tuple(key, sample_sizes))
        if not type(key[0]) == str:
            sample_size.append((get_in_case_reverse_tuple(key[0], sample_sizes), get_in_case_reverse_tuple(key[1], sample_sizes)))
    test_stats_df = pd.DataFrame(
        {
            "test_name": len(p_values) * [test_name],
            "test_statistic": test_statistics,
            "p_value": p_values,
            "sample_size": sample_size,
            "bin_pair": bin_pair,
        }
    )
    return test_stats_df


if __name__ == "__main__":
    # set sys path and import path settings
    sys.path.append(
        str(Path("comparison_interaction_null_model_test.py").resolve().parents[1])
    )
    import path_settings

    cosinor_df_path, interaction_df_path, interaction_df_null_path, interaction_tree_df_path, agg_data_path, exit_pos = path_settings.set_parameters(
         args.year, args.side
    )

    # get binning labels and set variables
    binning_dict = set_binning_var(args.binning_var)

    # load null model frame
    df_null = pd.read_csv(interaction_df_null_path)

    # clean df and add bins
    df_null = prepare_df_for_testing(df_null.copy(), binning_dict, overlap=False)

    # load interaction frame
    interaction_df = pd.read_csv(interaction_df_path)

    # clean df and add bins
    interaction_df = prepare_df_for_testing(
        interaction_df.copy(), binning_dict, overlap=True
    )

    # get sample size
    sample_sizes_dict = sample_sizes(interaction_df)
    sample_sizes_dict.update(sample_sizes(df_null))

    # test if normally distributed
    normally_distributed_bins_test_null = extract_test_stats(
        bb_rhythm.statistics.test_normally_distributed_bins(df_null, printing=False),
        sample_sizes_dict,
        "Kalgomorov Smirnoff test",
    )
    normally_distributed_bins_test_interaction = extract_test_stats(
        bb_rhythm.statistics.test_normally_distributed_bins(interaction_df, printing=False),
        sample_sizes_dict,
        "Kalgomorov Smirnoff test",
    )

    # test if equal variance
    equal_variance_comparison_bins_test = extract_test_stats(
        bb_rhythm.statistics.test_for_comparison_bins(
            df_null,
            interaction_df,
            bb_rhythm.statistics.test_bins_have_equal_variance,
            args=(True, "mean"),
            printing=False,
        ),
        sample_sizes_dict,
        "Levene test",
    )

    # test if significantly different mean
    unequal_mean_comparison_bins_test = extract_test_stats(
        bb_rhythm.statistics.test_for_comparison_bins(
            df_null,
            interaction_df,
            bb_rhythm.statistics.test_bins_have_unequal_mean,
            args=(True, False),
            printing=False,
        ),
        sample_sizes_dict,
        "Welch test",
    )

    # save test results
    save_to = os.path.join(
        agg_data_path, f"test_vel_change_side{args.side}_{args.year}_{args.binning_var}"
    )
    test_stats_df = pd.concat(
        [
            normally_distributed_bins_test_null,
            normally_distributed_bins_test_interaction,
            equal_variance_comparison_bins_test,
            unequal_mean_comparison_bins_test,
        ],
        ignore_index=True,
    )
    test_stats_df.to_csv(f"{save_to}.csv")

    # get summary of each test
    test_stats_df.groupby("test_name").apply(lambda x: print(x.test_name.iloc[0], "\n", x.describe()))
