import argparse
import os
import sys
from pathlib import Path

import bb_rhythm.statistics
import bb_rhythm.utils
import numpy as np
import pandas as pd
from bb_rhythm.interactions import combine_bees_from_interaction_df_to_be_all_focal, filter_overlap, get_start_velocity

"""
This script groups the interaction data frame and its null model in bins/quantiles
and compares them statistically. For that a Welch-test is performed and its precondition
(normality and unequal variance of samples) is tested. Different parameters can be chosen
for binning the data frame e.g. start velocity, age, phase and r_squared.

Example workflow for the analysis of 2019 data of side 0:

    comparison_interaction_null_model_test.py --year 2019 --side 0 --binning_var 'start_vel' --n_bins 6
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year",
    type=int,
    required=True,
    help="Which year to analyze the data for. (2016 or 2019)",
)
parser.add_argument(
    "--side",
    type=int,
    required=True,
    help="Which side of the hive to analyze the data for. (0 or 1)",
)
parser.add_argument(
    "--binning_var",
    type=str,
    required=True,
    help="Which parameter the interactions are binned with. ('start_vel', 'age', 'phase', 'r_squared')",
)
parser.add_argument(
    "--n_bins",
    type=int,
    default=6,
    help="Number of bins/quantiles to divide the data into (default: 6)",
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
    raise ValueError(
        f"data can't be binned by '{binning_var_flag}'. "
        "Possible options are 'start_vel', 'age', 'phase' or 'r_squared'."
    )


def get_required_columns(binning_dict: dict, overlap: bool = False) -> list:
    """Return a minimal list of columns needed from the raw CSV to reduce RAM requirements.

    Args:
        binning_dict (dict): Dictionary with binning variables.
        overlap (bool): Whether to include the 'overlapping' column (should be False for the null model data).

    Returns:
        list: List of required columns.
    """
    cols = list(binning_dict.keys())
    cols = [col.replace("non_focal", "bee1").replace("focal", "bee0") for col in cols]
    cols.extend(["vel_change_bee0", "vel_change_bee1"])

    if "velocity_start_bee0" in cols:
        cols = [col.replace("velocity_start", "rel_change") for col in cols]

    if overlap:
        cols.append("overlapping")

    # remove duplicates if any
    return list(set(cols))


def prepare_df_for_testing(
    df: pd.DataFrame,
    binning_dict: dict,
    overlap: bool,
    n_bins: int,
) -> pd.DataFrame:
    """
    Prepare the DataFrame for statistical testing by filtering for actual interactions, combining bees,
    and adding bins.

    Args:
        df (pd.DataFrame): The DataFrame containing interaction data.
        binning_dict (dict): Dictionary with binning variables.
        overlap (bool): Whether to filter for overlapping interactions.
        n_bins (int): Number of bins to create.

    Returns:
        pd.DataFrame: The prepared DataFrame with bins added and NaN values handled.
    """
    if overlap:
        # filter overlap
        df = filter_overlap(df)

    # combine df so all bees are considered as focal
    df = combine_bees_from_interaction_df_to_be_all_focal(df)

    # calculate start velocity (if not already present)
    if "velocity_start_focal" in binning_dict.keys() and "velocity_start_focal" not in df.columns:
        get_start_velocity(df)

    # filter and replace nan/infs
    df.replace({-np.inf: np.nan, np.inf: np.nan}, inplace=True)
    df.dropna(inplace=True)

    # add bins for focal
    bin_focal = bb_rhythm.utils.Binning(
        bin_name="bins_focal",
        bin_parameter=list(binning_dict.keys())[0],
    )
    df = bin_focal.add_bins_to_df(df, n_bins=n_bins, step_size=None, bin_max_n=None, bin_labels=None)

    # add bins for non-focal
    bin_non = bb_rhythm.utils.Binning(
        bin_name="bins_non_focal",
        bin_parameter=list(binning_dict.keys())[1],
    )
    df = bin_non.add_bins_to_df(df, n_bins=n_bins, step_size=None, bin_max_n=None, bin_labels=None)

    return df


def sample_sizes(df: pd.DataFrame) -> dict:
    """
    Calculate the sample sizes for each combination of focal and non-focal bins (i.e. each entry in the
    aggregation matrix).

    Args:
        df (pd.DataFrame): The DataFrame containing the interaction data with bins.

    Returns:
        dict: A dictionary with tuples of (focal_bin, non_focal_bin) as keys and sample sizes as values.
    """
    sample_size_dict = (
        df[["bins_focal", "bins_non_focal", "vel_change_bee_focal"]]
        .groupby(["bins_focal", "bins_non_focal"])
        .count()
        .to_dict()
    )["vel_change_bee_focal"]
    return sample_size_dict


def aggregate_for_plotting(
    df: pd.DataFrame,
    binning_dict: dict,
    null: bool = False,
    year: int = 2019,
    side: int = 0,
    n_bins: int = 6,
    aggfunc: str = "median",
) -> None:
    """
    Aggregate the DataFrame by the binning variables and apply the specified aggregation function
    for each combination of binning values.

    Args:
        df (pd.DataFrame): The DataFrame containing the interaction data with bins.
        binning_dict (dict): Dictionary with binning variables.
        null (bool): Whether this is the null model data (default: False).
        year (int): The year of the data (default: 2019).
        side (int): The side of the hive (0 or 1) (default: 0).
        n_bins (int): Number of bins used for aggregation (default: 6).
        aggfunc (str): Aggregation function to use ('mean', 'median', etc.) (default: 'median').

    Returns:
        None: Saves the aggregated results to a .npy file to be used for plotting.
    """
    var = list(binning_dict.keys())[0].replace("focal", "").replace("non_focal", "").strip("_")

    aggregation_matrix = pd.pivot_table(
        data=df,
        values="vel_change_bee_focal",
        index="bins_non_focal",
        columns="bins_focal",
        aggfunc=aggfunc,
    ).to_numpy()

    suffix = [var, str(side), f"{n_bins}bins"]

    if null:
        suffix.append("null")

    save_to = Path(os.pardir) / "aggregated_results" / str(year) / f"speed_trans_vs_{'_'.join(suffix)}.npy"
    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_to = str(save_to)
    np.save(save_to, aggregation_matrix)


def get_in_case_reverse_tuple(key_tuple, sample_sizes):
    """
    Get the sample size for a given key tuple, considering that the keys may be symmetric pairs.
    If the key tuple is (a, b), it will return the sample size for (a, b) or (b, a).
    If the key is a string, it will return the sample size for that string.

    Args:
        key_tuple (tuple or str): The key tuple or string to look up in the sample sizes dictionary.
        sample_sizes (dict): Dictionary containing sample sizes for each key tuple.

    Returns:
        int: The sample size for the given key tuple, considering symmetry.
    """
    try:
        return sample_sizes[key_tuple]
    except KeyError:
        return sample_sizes[(key_tuple[1], key_tuple[0])]


def extract_test_stats(test_results: dict, sample_sizes: dict, test_name: str) -> pd.DataFrame:
    """
    Extract test statistics from the results of statistical tests and format them into a DataFrame.

    Args:
        test_results (dict): Dictionary containing the results of statistical tests.
        sample_sizes (dict): Dictionary containing sample sizes for each key tuple.
        test_name (str): Name of the statistical test performed.

    Returns:
        pd.DataFrame: A DataFrame containing the test name, test statistics, p-values, sample sizes, and bin pairs.
    """
    p_values = []
    test_statistics = []
    sample_size = []
    bin_pair = []
    for key, value in test_results.items():
        bin_pair.append(key)
        p_values.append(value.pvalue)
        test_statistics.append(value.statistic)
        # sample sizes may be symmetric pairs
        if isinstance(key[0], str):
            sample_size.append(get_in_case_reverse_tuple(key, sample_sizes))
        else:
            sample_size.append(
                (
                    get_in_case_reverse_tuple(key[0], sample_sizes),
                    get_in_case_reverse_tuple(key[1], sample_sizes),
                )
            )
    return pd.DataFrame(
        {
            "test_name": [test_name] * len(p_values),
            "test_statistic": test_statistics,
            "p_value": p_values,
            "sample_size": sample_size,
            "bin_pair": bin_pair,
        }
    )


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    import path_settings

    # determine file paths
    (
        cosinor_df_path,
        interaction_df_path,
        interaction_df_null_path,
        interaction_tree_df_path,
        agg_data_path,
        exit_pos,
    ) = path_settings.set_parameters(args.year, args.side)

    # get binning labels and variables
    binning_dict = set_binning_var(args.binning_var)

    # read only required columns for null model
    usecols_null = get_required_columns(binning_dict, overlap=False)
    df_null = pd.read_csv(interaction_df_null_path, usecols=usecols_null)

    # clean null df and add bins
    df_null = prepare_df_for_testing(df_null.copy(), binning_dict, overlap=False, n_bins=args.n_bins)

    # save aggregated results for plotting
    aggregate_for_plotting(
        df_null,
        binning_dict,
        null=True,
        year=args.year,
        side=args.side,
        n_bins=args.n_bins,
    )

    # read only required columns for interaction data (needs overlap)
    usecols_inter = get_required_columns(binning_dict, overlap=True)
    interaction_df = pd.read_csv(interaction_df_path, usecols=usecols_inter)

    # clean interaction df and add bins
    interaction_df = prepare_df_for_testing(interaction_df.copy(), binning_dict, overlap=True, n_bins=args.n_bins)

    # save aggregated results for plotting
    aggregate_for_plotting(
        interaction_df,
        binning_dict,
        null=False,
        year=args.year,
        side=args.side,
        n_bins=args.n_bins,
    )

    # get combined sample sizes
    sample_sizes_dict = sample_sizes(interaction_df)
    sample_sizes_dict.update(sample_sizes(df_null))

    # test normality
    normally_null = extract_test_stats(
        bb_rhythm.statistics.test_normally_distributed_bins(df_null, printing=False),
        sample_sizes_dict,
        "Kolmogorov-Smirnov test",
    )
    normally_inter = extract_test_stats(
        bb_rhythm.statistics.test_normally_distributed_bins(interaction_df, printing=False),
        sample_sizes_dict,
        "Kolmogorov-Smirnov test",
    )

    # test variance equality
    equal_var = extract_test_stats(
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

    # test mean differences
    unequal_mean = extract_test_stats(
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

    # combine and save results
    test_stats_df = pd.concat(
        [normally_null, normally_inter, equal_var, unequal_mean],
        ignore_index=True,
    )
    save_to = os.path.join(
        agg_data_path,
        f"test_vel_change_side{args.side}_{args.year}_{args.binning_var}_bins{args.n_bins}",
    )
    test_stats_df.to_csv(f"{save_to}.csv", index=False)

    # print summary per test
    for test_name, group in test_stats_df.groupby("test_name"):
        print(test_name, "\n", group.describe(), "\n")
