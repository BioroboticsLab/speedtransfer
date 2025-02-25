import numpy as np
import pandas as pd
import os
import pickle
import argparse
from bb_rhythm import interactions, plotting

"""
This script iterates over all interactions, modeling both participating bees
as rectangular masks and comupting the area of overlap between these two masks.
Lastly, this information is also aggregated with the velocity change yielded
by the given interaction.

Example workflow for the analysis of 2019 data of side 0:

    # First compute area of overlap for small batches of interactions in parallel.
    point_of_interaction.py --year 2019 --side 0 --focal 0 --batch ${SLURM_ARRAY_TASK_ID}
    point_of_interaction.py --year 2019 --side 0 --focal 1 --batch ${SLURM_ARRAY_TASK_ID}

    # Then combine results from batches for each focal bee.
    point_of_interaction.py --year 2019 --side 0 --focal 0
    point_of_interaction.py --year 2019 --side 0 --focal 1

    # Lastly combine results from both bees condsidered as focal.
    point_of_interaction.py --year 2019 --side 0
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
)
parser.add_argument(
    "--side", type=int, help="Which side of the hive to analyze the data for. (0 or 1)"
)
parser.add_argument(
    "--focal",
    type=int,
    required=False,
    help="Which focal bee to create overlaps for. (0 or 1)",
)
parser.add_argument(
    "--batch",
    type=int,
    required=False,
    help="Which batch of interactions to create overlaps for. (0-99)",
)
args = parser.parse_args()


def create_overlap_dicts(interaction_df, focal, batch, save_to):
    """Compute area of overlap between two bee masks for the current
    batch of interaction data.

    Args:
        interaction_df (pd.DataFrame): Interaction data containing information about bee's relative position.
        focal (int): Which bee is currently considered focal (0 or 1).
        batch (int): Which batch of interaction data to do the analysis for.
        save_to (str): Directory where the result dict will be saved to.
    """
    # Get current batch of interactions.
    cur_idx = np.array_split(interaction_df.index, 100)[batch]
    interaction_df = interaction_df.loc[cur_idx]
    print(
        "Processing %d interactions in batch %d for focal bee %d."
        % (len(interaction_df), batch, focal)
    )
    
    # Only use necessary columns.
    interaction_df = interaction_df[
        [
            "x_trans_focal_bee%d" % focal,
            "y_trans_focal_bee%d" % focal,
            "theta_trans_focal_bee%d" % focal,
        ]
    ]

    # Calculate overlaps.
    overlap_dict = interactions.create_overlap_dict(interaction_df, focal_id=focal)
    print("Done creating dict.")
    with open(os.path.join(save_to, "batch%02d.pkl" % batch), "wb") as handle:
        pickle.dump(overlap_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved.")


def combine_overlaps(path, dest):
    """Merge pickled overlap dicts for all batches of data into one dict per focal bee.

    Args:
        path (str): Directory where all the batched overlaps are saved.
        dest (str): File path to which the combined dict will be saved to.
    """
    combined_dict = {}
    
    # Combine results from each batch of data.
    for i in range(100):
        with open(os.path.join(path, "batch%02d.pkl" % i), "rb") as handle:
            d = pickle.load(handle)
            combined_dict.update(d)
            
    with open(dest, "wb") as handle:
        pickle.dump(combined_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_dicts_for_both_bees(dict_path, n_interactions):
    """Combines overlap data of bee 0 and bee 1 into one single dict.

    Args:
        dict_path (str): Directory in which both dicts are located.
        n_interactions (int): Number of interactions in dataframe.

    Returns:
        dict: Combined dict.
    """
    overlap_dict = dict()

    with open(
        os.path.join(dict_path, f"overlaps_side{args.side}_bee0.pkl"), "rb"
    ) as handle:
        overlap_dict_0 = pickle.load(handle)
        print("len0")
        print(len(overlap_dict_0))
        for idx in overlap_dict_0:
            overlap_dict[idx] = overlap_dict_0[idx]

    with open(
        os.path.join(dict_path, f"overlaps_side{args.side}_bee1.pkl"), "rb"
    ) as handle:
        overlap_dict_1 = pickle.load(handle)
        print("len1")
        print(len(overlap_dict_1))
        for idx in overlap_dict_1:
            overlap_dict[idx + n_interactions] = overlap_dict_1[idx]
            
    return overlap_dict


def get_vel_change_per_point_of_interaction(
    interaction_df, overlap_dict, whose_change="focal"
):
    """Computes and saves avg. velocity change at each point of interaction on a bee's body.

    Args:
        interaction_df (pd.DataFrame): Interaction data.
        overlap_dict (dict): Area of overlap of bee masks for each interaction.
        whose_change (str, optional): Whose velocity change to aggregate. Either 'focal' or 'non-focal'. Defaults to 'focal'.
    """
    # Keep only velocity change columns.
    interaction_df = interaction_df.loc[:, ["vel_change_bee0", "vel_change_bee1"]]

    # Set both bees to focal/non-focal.
    combined_df = interactions.combine_bees_from_interaction_df_to_be_all_focal(
        interaction_df
    )

    # Reset index so interaction indices correspond to overlap indices.
    combined_df.reset_index(inplace=True, drop=True)
    
    counts, avg_vel = interactions.compute_interaction_points(
        combined_df, overlap_dict, whose_change=whose_change
    )
    df = plotting.transform_interaction_matrix_to_df(
        avg_vel, counts, whose_change=whose_change
    )
    save_to = os.path.join(
        os.pardir,
        "aggregated_results",
        str(args.year),
        f"speed_change_per_body_pos_{whose_change}_side{args.side}.csv",
    )
    df.to_csv(save_to)


if __name__ == "__main__":
    # Specify location of data.
    data_path = os.path.join(os.pardir, "data", args.year)
    interactions_path = os.path.join(data_path, f"interactions_side{args.side}.pkl")
    overlaps_folder = os.path.join(
        data_path, f"overlaps_side{args.side}_bee{args.focal}"
    )
    
    # Read interaction data.
    interaction_df = pd.read_pickle(interactions_path)
    n_interactions = len(interaction_df)
    
    # Create dicts with overlaps in parallel. (requires ca. 40G RAM per job)
    if args.batch != None:

        # Create folder to store the overlap splits.
        if not os.path.exists(overlaps_folder):
            os.makedirs(overlaps_folder)

        create_overlap_dicts(interaction_df, args.focal, args.batch, overlaps_folder)

    # If no batch number, but an id for the focal bee is provided combine results from individual batch jobs.
    elif args.focal != None:
        # Merge overlap dicts and optionally add overlap info to interaction data.
        save_to = os.path.join(
            data_path, f"overlaps_side{args.side}_bee{args.focal}.pkl"
        )
        combine_overlaps(overlaps_folder, save_to)

    # If neither batch number nor focal id is provided combine results for both focal bees.
    else:
        overlap_dict_path = os.path.join(data_path, f"overlaps_side{args.side}.pkl")
        if os.path.exists(overlap_dict_path):
            with open(overlap_dict_path, "rb") as handle:
                overlap_dict = pickle.load(handle)
        else:
            overlap_dict = combine_dicts_for_both_bees(data_path, n_interactions)
            with open(overlap_dict_path, "wb") as handle:
                pickle.dump(overlap_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compute velocity change depending on area of overlap and save to aggregated results.
        get_vel_change_per_point_of_interaction(
            interaction_df, overlap_dict, whose_change="focal"
        )