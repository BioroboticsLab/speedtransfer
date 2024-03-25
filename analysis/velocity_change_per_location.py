import numpy as np
import pandas as pd
import os
import pickle
import argparse
from bb_rhythm import interactions, plotting, rhythm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_settings

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
)
parser.add_argument(
    "--side", type=int, help="Which side of the hive to analyze the data for. (1 or 2)"
)


def read_data(path):
    # Read interaction data.
    df = pd.read_pickle(path)

    # Remove unnecessary columns.
    df = df[['x_pos_start_bee0','y_pos_start_bee0', 'vel_change_bee0', 'rel_change_bee0',
            'x_pos_start_bee1','y_pos_start_bee1', 'vel_change_bee1', 'rel_change_bee1',
            'interaction_start']]
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df


def replace_time_with_hour(df):
    df['hour'] = df['interaction_start'].dt.hour
    df.drop(columns=['interaction_start'], inplace=True)
    return df


def vstack_interactions(h_df, var_ls):
    # Stack interaction data such that both bees are focal.
    combined_df = pd.DataFrame()
    
    # Set spatial resolution to 1mm^2.
    combined_df['x_grid'] = pd.concat([h_df['x_pos_start_bee0'],
                                       h_df['x_pos_start_bee1']]).round(1).astype(int)
    combined_df['y_grid'] = pd.concat([h_df['y_pos_start_bee0'],
                                       h_df['y_pos_start_bee1']]).round(1).astype(int)
    
    # Keep hour column for filtering.
    combined_df['hour'] = pd.concat([h_df['hour'], h_df['hour']])
    
    for var in var_ls:
        combined_df[var] = pd.concat([h_df['%s_bee0' % var], h_df['%s_bee1' % var]])
    
    return combined_df


def swap_focal_bee(df):
    """Sets the bee with the higher increase in speed to be the focal one.

    Args:
        df (pd.DataFrame): DataFrame with speed changes and positions of both focal bees.

    Returns:
        pd.DataFrame: Data for speed change of focal bee.
    """
    new_row_ls = []
    
    df = df.to_dict(orient='records')
    
    spatial_bin_size = 1
    
    for row in df:
        if row['vel_change_bee0'] > row['vel_change_bee1']:
            new_row_ls.append([int(round(row['x_pos_start_bee0'] / spatial_bin_size)),
                               int(round(row['y_pos_start_bee0'] / spatial_bin_size)),
                               row['vel_change_bee0'], row['hour']])
        elif row['vel_change_bee1'] > row['vel_change_bee0']:
            new_row_ls.append([int(round(row['x_pos_start_bee1'] / spatial_bin_size)),
                               int(round(row['y_pos_start_bee1'] / spatial_bin_size)),
                               row['vel_change_bee1'], row['hour']])
    
    res = pd.DataFrame(new_row_ls, columns=['x_grid', 'y_grid', 'vel_change', 'hour'])
    return res
    

def concat_grids_over_time(df, var='vel_change', aggfunc='median', scale=False):
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
    accumulator = np.zeros((24,h,w))
    
    # Create grid for each hour and add to accumulator.
    for hour in range(24):
        subset = df.loc[df.hour == hour]
        subset = subset.drop(columns=['hour'])
        grid = pd.pivot_table(data=subset, index='y_grid', columns='x_grid',
                              values=var, aggfunc=aggfunc)
        grid = grid.reindex(index=y_vals, columns=x_vals)
        grid = grid.to_numpy()
        accumulator[hour] = grid
    
    if scale:
        for i in range(h):
            for j in range(w):
                accumulator[:,i,j] = rhythm.min_max_scaling(accumulator[:,i,j])
        
    return accumulator


if __name__ == "__main__":
    
    var = 'vel_change'
    aggfunc = 'median'
    path = path_settings.INTERACTION_SIDE_0_DF_PATH_2019
    
    df = read_data()
    df = replace_time_with_hour(df)
    df = swap_focal_bee(df)
    grid_3d = concat_grids_over_time(df, var, aggfunc)


