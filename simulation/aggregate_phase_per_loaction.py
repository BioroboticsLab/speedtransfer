import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
from bb_rhythm import rhythm

def sliding_window_fitting(X: np.array, Y: np.array, agent_id: int, n_timesteps: int, period: int, param='phase') -> list:
    """Fit cosinor model in sliding windows and extract parameter of interest.
    
    Args:
        X (np.array): Time indices.
        Y (np.array): Speed values.
        agent_id (int): ID of the agent.
        n_timesteps (int): Total number of timesteps.
        period (int): Period of the rhythm.
        param (str): Parameter to extract ('phase' or 'amplitude').
    
    Returns:
        list: List of [agent_id, day, parameter_value].
    """
    
    fit_params = []
    
    for t in range(int(n_timesteps / period - 2)):

        subset_x = X[period * t : period * (t+3)]
        subset_y = Y[period * t : period * (t+3)]

        # Check if data is non-constant
        if np.ptp(subset_y) != 0:

            fit_data = rhythm.fit_cosinor_per_bee(subset_x, subset_y, period)
            fit_params.append([agent_id, t, fit_data[param]])
            
    return fit_params
                
    
def get_daily_phases_for_agents(speeds: np.array, save_to='agents_phases.pkl') -> pd.DataFrame:
    """Get daily phases for all agents based on their speed timeseries.

    Args:
        speeds (np.array): Array of shape (n_agents, n_timesteps) with speed values.
        save_to (str): Path to save the resulting DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns ['agent_id', 'day', 'phase'].
    """
    
    phase_ls = []
    period = 400
    n_timesteps = speeds.shape[1]
    
    # Create time index where zero corresponds to 12 noon.
    t_index = np.arange(speeds.shape[1]) - period/4
    
    for agent_id in range(speeds.shape[0]):
        
        # Get speeds timeseries of current agent.
        agent_ts = speeds[agent_id]
        
        agent_phases = sliding_window_fitting(t_index, agent_ts, agent_id, n_timesteps, period)
        phase_ls.extend(agent_phases)
    
    result_df = pd.DataFrame(phase_ls, columns=['agent_id', 'day', 'phase'])
    result_df.to_pickle(save_to)
    return result_df


def accumulate_phases_per_position(position_data: np.array, phase_data: pd.DataFrame, save_to='df.pkl') -> pd.DataFrame:
    """Accumulate phases per position based on position and phase data.

    Args:
        position_data (np.array): Array of shape (n_agents, n_timesteps, 2) with position values.
        phase_data (pd.DataFrame): DataFrame with columns ['agent_id', 'day', 'phase'].
        save_to (str): Path to save the resulting DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns ['x', 'y', 'phase'].
    """
    
    row_ls = []
    
    agent_ids = np.arange(position_data.shape[0])
    days = np.unique(phase_data.day)
    timesteps_per_day = 400
    
    for agent_id in agent_ids:
        for day in days:
            # Get current agent's phase for current day
            try:
                current_phase = phase_data.loc[(phase_data.agent_id == agent_id) & (phase_data.day == day), 'phase'].values[0]
            except Exception as e:
                print(e)
                continue
            # Get all locations the agent has been at that day.
            day_start = day * timesteps_per_day
            day_end = (day + 1) * timesteps_per_day
            daily_positions = position_data[agent_id, day_start:day_end, :]
            
            # Add row with x, y, and phase information.
            for pos in daily_positions:
                x = int(round(pos[0]))
                y = int(round(pos[1]))
                row_ls.append([x, y, current_phase])
    
    result_df = pd.DataFrame(row_ls, columns=['x', 'y', 'phase'])
    result_df.to_pickle(save_to)
    return result_df


def plot_grid(grid: np.array, var: str, year: str = '', aggfunc: str = 'median', save_to: str | None = None) -> None:
    """Plot heatmap of the grid.

    Args:
        grid (np.array): 2D array representing the grid.
        var (str): Variable being plotted ('phase', 'amplitude', 'count').
        year (str): Year identifier for saving the plot.
        aggfunc (str): Aggregation function used ('mean', 'median', etc.).
        save_to (str | None): Path to save the plot. If None, saves to default path.
    """
    
    # Convert phase from rad to time of day.
    if var == 'phase':
        period = 400
        grid = - period * grid / (2 * np.pi) * (24 / period) + 12
        
    label_dict = {'phase': 'Hour of peak activity',
                  'amplitude': 'Amplitude [mm/s]',
                  'count': 'Number of samples'}
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(grid, xticklabels=100, yticklabels=100, square=True,
                cmap='viridis_r', cbar=True, cbar_kws={'label':label_dict[var], 'shrink':0.7},
                robust=True, linewidths=0, rasterized=True)
    plt.xlabel('x position [mm]')
    plt.ylabel('y position [mm]')
    plt.gca().invert_yaxis()
    if not save_to:
        plt.savefig(f'ims_{year}/{var}_{year}_{aggfunc}.pdf', bbox_inches='tight')
    else:
        plt.savefig(save_to, bbox_inches='tight')
        

def convert_phase_to_hours(phase: float, period: int) -> float:
    """Convert phase in radians to hours of the day.

    Args:
        phase (float): Phase in radians.
        period (int): Period of the cycle.

    Returns:
        float: Hour of the day corresponding to the phase.
    """
    return - period * phase / (2 * np.pi) * (24 / period) + 12

if __name__ == "__main__":
    for density in [10.0]:
        path = '/Users/weronik22/Documents/circadians/speedtransfer/simulation/output'
        sim_output = loadmat(f'../../CircadianAgents/homing_pull_0.3/OUT_density_{density:.2f}.mat')
        speeds = sim_output['OUT'][0][0][0]
        positions = sim_output['OUT'][0][0][5]
        
        agents_daily_phases = get_daily_phases_for_agents(speeds, save_to=f'agents_phases_{density}.pkl')
        phase_per_loc = accumulate_phases_per_position(positions, agents_daily_phases, save_to=f'phase_per_loc_hp03_{density}.pkl')
        
        phase_per_loc.loc[:,['x', 'y']] = (phase_per_loc.loc[:,['x', 'y']]).astype(int)
        var = 'phase'
        aggfunc = 'mean'
        grid = rhythm.create_grid_from_df(phase_per_loc, var, aggfunc='count')
        np.save('phase_grid_simulated_count.npy', grid)
    