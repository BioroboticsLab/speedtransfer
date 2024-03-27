import pandas as pd
import os
import numpy as np
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import bb_rhythm.interactions
import bb_rhythm.utils
import bb_rhythm.statistics
import bb_rhythm.plotting

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
)
parser.add_argument(
    "--side", type=int, help="Which side of the hive to analyze the data for. (0 or 1)"
)

args = parser.parse_args()

if __name__ == "__main__":
    # set sys path and import path settings
    sys.path.append(
        str(Path("comparison_interaction_null_model_test.py").resolve().parents[1])
    )
    import path_settings

    cosinor_df_path, interaction_df_path, interaction_df_null_path, interaction_tree_df_path, exit_pos = path_settings.set_parameters(
        args.year, args.side
    )

    # load null model frame
    df_null = pd.read_pickle(
        os.path.join(path, "null_model_interactions_2019_cam_0.pkl")
    )

    # clean df
    df_null.drop(
        columns=[
            "x_pos_start_bee0",
            "y_pos_start_bee0",
            "theta_start_bee0",
            "x_pos_start_bee1",
            "y_pos_start_bee1",
            "theta_start_bee1",
            "x_pos_end_bee0",
            "y_pos_end_bee0",
            "theta_end_bee0",
            "x_pos_end_bee1",
            "y_pos_end_bee1",
            "theta_end_bee1",
            "age_bee0",
            "age_bee1",
            "amplitude_bee0",
            "p_value_bee0",
            "amplitude_bee1",
            "p_value_bee1",
        ],
        inplace=True,
    )
    df_null.replace({-np.inf: np.nan, np.inf: np.nan}, inplace=True)
    df_null.dropna(inplace=True)

    # combine df so all bees are considered as focal
    df_null = bb_rhythm.interactions.combine_bees_from_interaction_df_to_be_all_focal(
        df_null
    )
    bb_rhythm.interactions.get_start_velocity(df_null)

    # add bins
    binning = bb_rhythm.utils.Binning(
        bin_name="bins_focal", bin_parameter="velocity_start_focal"
    )
    df_null = binning.add_bins_to_df(
        df_null, n_bins=6, step_size=None, bin_max_n=None, bin_labels=None
    )
    binning = bb_rhythm.utils.Binning(
        bin_name="bins_non_focal", bin_parameter="velocity_start_non_focal"
    )
    df_null = binning.add_bins_to_df(
        df_null, n_bins=6, step_size=None, bin_max_n=None, bin_labels=None
    )

    # load interaction frame
    interaction_path = (
        "/scratch/weronik22/data/2019/interactions_no_duplicates_side2.pkl"
    )
    interaction_df = pd.read_pickle(interaction_path)
    # clean df
    interaction_df.drop(
        columns=[
            "bee_id0",
            "bee_id1",
            "interaction_start",
            "interaction_end",
            "x_pos_start_bee0",
            "y_pos_start_bee0",
            "theta_start_bee0",
            "x_pos_start_bee1",
            "y_pos_start_bee1",
            "theta_start_bee1",
            "x_pos_end_bee0",
            "y_pos_end_bee0",
            "theta_end_bee0",
            "x_pos_end_bee1",
            "y_pos_end_bee1",
            "theta_end_bee1",
            "age_bee0",
            "age_bee1",
            "x_trans_focal_bee0",
            "y_trans_focal_bee0",
            "theta_trans_focal_bee0",
            "x_trans_focal_bee1",
            "y_trans_focal_bee1",
            "theta_trans_focal_bee1",
            "is_bursty_bee0",
            "is_bursty_bee1",
            "is_foraging_bee0",
            "is_foraging_bee1",
            "amplitude_bee0",
            "phase_bee0",
            "p_value_bee0",
            "fit_type_x",
            "amplitude_bee1",
            "phase_bee1",
            "p_value_bee1",
            "fit_type_y",
        ],
        inplace=True,
    )
    interaction_df.replace({-np.inf: np.nan, np.inf: np.nan}, inplace=True)
    interaction_df.dropna(inplace=True)

    # filter overlap
    interaction_df = bb_rhythm.interactions.filter_overlap(interaction_df)

    # combine df so all bees are considered as focal
    interaction_df = bb_rhythm.interactions.combine_bees_from_interaction_df_to_be_all_focal(
        interaction_df
    )
    bb_rhythm.interactions.get_start_velocity(interaction_df)

    # add bins
    binning = bb_rhythm.utils.Binning(
        bin_name="bins_focal", bin_parameter="velocity_start_focal"
    )
    interaction_df = binning.add_bins_to_df(
        interaction_df, n_bins=6, step_size=None, bin_max_n=None, bin_labels=None
    )

    binning = bb_rhythm.utils.Binning(
        bin_name="bins_non_focal", bin_parameter="velocity_start_non_focal"
    )
    interaction_df = binning.add_bins_to_df(
        interaction_df, n_bins=6, step_size=None, bin_max_n=None, bin_labels=None
    )

    # test if normally distributed
    normally_distributed_bins_test_null = bb_rhythm.statistics.test_normally_distributed_bins(
        df_null
    )
    normally_distributed_bins_test_interaction = bb_rhythm.statistics.test_normally_distributed_bins(
        interaction_df
    )

    # test if equal variance
    equal_variance_comparison_bins_test = bb_rhythm.statistics.test_for_comparison_bins(
        df_null,
        interaction_df,
        bb_rhythm.statistics.test_bins_have_equal_variance,
        args=(True, "mean"),
    )

    # test if significantly different mean
    unequal_mean_comparison_bins_test = bb_rhythm.statistics.test_for_comparison_bins(
        df_null,
        interaction_df,
        bb_rhythm.statistics.test_bins_have_unequal_mean,
        args=(True, False),
    )

    # plot p-values
    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    bb_rhythm.plotting.plot_p_values_per_bin_from_test(
        normally_distributed_bins_test_null,
        ax=axs[0, 0],
        norm=(0, 0.35),
        pkl_path="~/../../scratch/juliam98/data/2019/statistics/normal_test_p_value_2019_null_start_vel",
    )
    axs[0, 0].set_title("P-values KS-test normally distributed null model")
    bb_rhythm.plotting.plot_p_values_per_bin_from_test(
        normally_distributed_bins_test_interaction,
        ax=axs[0, 1],
        norm=(0, 0.35),
        pkl_path="~/../../scratch/juliam98/data/2019/statistics/normal_test_p_value_2019_start_vel",
    )
    axs[0, 1].set_title("P-values KS-test normally distributed interactions")
    bb_rhythm.plotting.plot_p_values_per_bin_from_test(
        equal_variance_comparison_bins_test,
        ax=axs[1, 0],
        norm=(0, 0.35),
        pkl_path="~/../../scratch/juliam98/data/2019/statistics/variance_test_p_value_2019_start_vel",
    )
    axs[1, 0].set_title("P-values Levene's test equal variance")
    bb_rhythm.plotting.plot_p_values_per_bin_from_test(
        unequal_mean_comparison_bins_test,
        ax=axs[1, 1],
        norm=(0, 0.35),
        pkl_path="~/../../scratch/juliam98/data/2019/statistics/t_test_p_value_2019_start_vel",
    )
    axs[1, 1].set_title("P-values t-test")
    plt.tight_layout()
    plt.savefig("imgs/test_overview_2019_start_vel.png")
    plt.savefig("imgs/test_overview_2019_start_vel.svg")

    import seaborn as sns
    from matplotlib import rcParams

    # plot distribution per bin as histogram
    rcParams.update({"figure.autolayout": True})
    g = sns.FacetGrid(
        df_null[["bins_non_focal", "bins_focal", "vel_change_bee_focal"]],
        col="bins_focal",
        row="bins_non_focal",
        margin_titles=True,
        row_order=sorted(df_null["bins_non_focal"].unique()),
    )
    g.map(sns.histplot, "vel_change_bee_focal", kde=True)
    g.figure.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(
        "~/../../scratch/juliam98/data/2019/statistics/test_overview_2019_start_vel_dist_null.png"
    )

    # plot distribution per bin as histogram
    rcParams.update({"figure.autolayout": True})
    g = sns.FacetGrid(
        interaction_df[["bins_non_focal", "bins_focal", "vel_change_bee_focal"]],
        col="bins_focal",
        row="bins_non_focal",
        margin_titles=True,
        row_order=sorted(interaction_df["bins_non_focal"].unique()),
    )
    g.map(sns.histplot, "vel_change_bee_focal", kde=True)
    g.figure.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(
        "~/../../scratch/juliam98/data/2019/statistics/test_overview_2019_start_vel_dist.png"
    )
