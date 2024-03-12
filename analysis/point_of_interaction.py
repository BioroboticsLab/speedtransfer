import numpy as np
import pandas as pd
import os
import pickle
import argparse
from bb_rhythm import interactions, plotting

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int,
                    help="Which year to analyze the data for. (2016 or 2019)")
parser.add_argument("--side", type=int,
                    help="Which side of the hive to analyze the data for. (1 or 2)")
parser.add_argument("--focal", type=int, required=False,
                    help="Which focal bee to create overlaps for. (0 or 1)")
parser.add_argument("--batch", type=int, required=False,
                    help="Which batch of interactions to create overlaps for. (0-99)")  
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
    print("Processing %d interactions in batch %d for focal bee %d." % (len(interaction_df), batch, focal))
    
    # Only use necessary columns.
    interaction_df = interaction_df[["focal%d_x_trans" % focal,"focal%d_y_trans" % focal,"focal%d_theta_trans" % focal]]
    
    # Calculate overlaps.
    overlap_dict = interactions.create_overlap_dict(interaction_df, focal_id=focal)
    print("Done creating dict.")
    with open(os.path.join(save_to, "batch%02d.pkl" % batch), "wb") as handle:
        pickle.dump(overlap_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved.")


def combine_overlaps(interaction_df, path, dest):
    """Merge pickled overlap dicts for all batches of data into one dict per focal bee.

    Args:
        interaction_df (pd.DataFrame): Interaction data.
        path (str): Directory where all the batched overlaps are saved.
        dest (str): File path to which the combined dict will be saved to.

    Returns:
        pd.DataFrame: Interaction data with added overlap column.
    """
    
    # Combine results from each batch of data.
    for i in range(100):
        with open(os.path.join(path,"batch%02d.pkl" % i), "rb") as handle:
            d = pickle.load(handle)
    
        with open(dest, "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return interaction_df


def combine_dicts_for_both_bees(dict_path, n_interactions):
    """Combines overlap data of bee 0 and bee 1 into one single dict.

    Args:
        dict_path (str): Directory in which both dicts are located.
        n_interactions (int): Number of interactions in dataframe.

    Returns:
        dict: Combined dict.
    """
    overlap_dict = dict()

    with open(os.path.join(dict_path, f"overlaps_side{args.side}_bee0.pkl"), "rb") as handle:
        overlap_dict_0 = pickle.load(handle)
        for idx in overlap_dict_0:
            overlap_dict[idx] = overlap_dict_0[idx]

    with open(os.path.join(dict_path, f"overlaps_side{args.side}_bee1.pkl"), "rb") as handle:
        overlap_dict_1 = pickle.load(handle)
        for idx in overlap_dict_1:
            overlap_dict[idx+n_interactions] = overlap_dict_1[idx]
            
    return overlap_dict


def get_vel_change_per_point_of_interaction(interaction_df, overlap_dict, whose_change='focal'):
    """Computes and saves avg. velocity change at each point of interaction on a bee's body.

    Args:
        interaction_df (pd.DataFrame): Interaction data.
        overlap_dict (dict): Area of overlap of bee masks for each interaction.
        whose_change (str, optional): Whose velocity change to aggregate. Either 'focal' or 'non-focal'. Defaults to 'focal'.
    """
    # Keep only velocity change columns.
    interaction_df = interaction_df.loc[:,['vel_change_bee0', 'vel_change_bee1']]
    
    # Set both bees to focal/non-focal.
    combined_df = interactions.combine_bees_from_interaction_df_to_be_all_focal(interaction_df)
    counts, avg_vel = interactions.compute_interaction_points(combined_df, overlap_dict, whose_change=whose_change)
    df = plotting.transform_interaction_matrix_to_df(avg_vel, counts, whose_change=whose_change)
    save_to = os.path.join(os.pardir, "aggregated_results", f"vel_change_per_body_pos_{args.year}_{whose_change}.csv")
    df.to_csv(save_to)



if __name__ == "__main__":
    # Specify location of data.
    data_path = os.path.join(os.pardir, "data", args.year)
    interactions_path = os.path.join(data_path, f"interactions_side{args.side}.pkl")
    overlaps_folder = os.path.join(data_path, f"overlaps_side{args.side}_bee{args.focal}")
    
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
        save_to = os.path.join(data_path, f"overlaps_side{args.side}_bee{args.focal}.pkl")
        interaction_df = combine_overlaps(interaction_df, overlaps_folder, save_to)
    
    # If neither batch number nor focal id is provided combine results for both focal bees.
    else:
        overlap_dict = combine_dicts_for_both_bees(data_path, n_interactions)
        # Compute velocity change depending on area of overlap and save to aggregated results.
        get_vel_change_per_point_of_interaction(interaction_df, overlap_dict, whose_change='focal')
    