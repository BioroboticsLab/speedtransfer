import datetime
import pytz
import sys
from pathlib import Path
import os
import argparse

"""
This is a script for creating a dataframe of the parameters of a cosinor fit of the velocity per bee for a time window 
of 3 consecutive days for the period 01.08.-25.08.2016 and  20.08-14.09.2019. The resulting cosinor fit parameter 
dataframes per bee are concatenated to one df.

Comment pipeline to create, run and concat results to a pandas DataFrame:
python cosinor_fit_per_bee.py 
"""


def generate_jobs_2016(num_processes: int = 1):

    # job im imports
    import path_settings
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed


    # set dates
    from_dt = datetime.datetime(2016, 8, 1, hour=12, tzinfo=pytz.UTC)
    to_dt = datetime.datetime(2016, 8, 25, hour=12, tzinfo=pytz.UTC)

    # velocity path
    velocity_df_path = path_settings.VELOCITY_DF_PATH_2016

    # get alive bees
    if os.path.exists(velocity_df_path):
        alive_bees = [int(f[:-7]) for f in os.listdir(velocity_df_path) if f.endswith(".pickle")]
    else:
        raise Exception(f"Could not find {velocity_df_path}")

    # set median time window to 1h
    second = 3600

    # iterate through all bees
    result_list = []

    # helper to submit jobs
    def _submit_args(bee_id):
        return dict(
            bee_id=bee_id,
            from_dt=from_dt,
            to_dt=to_dt,
            velocity_df_path=velocity_df_path,
            second=second,
        )

    if num_processes and num_processes > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(run_job, **_submit_args(bee_id)) for bee_id in alive_bees]
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if not isinstance(result, dict):
                        result_list.append(result)
                except Exception as e:
                    # Skip failed jobs but print a note
                    print(f"[generate_jobs_2016] bee job failed: {e}")
    else:
        # Serial fallback
        for bee_id in alive_bees:
            result = run_job(
                bee_id=bee_id,
                from_dt=from_dt,
                to_dt=to_dt,
                velocity_df_path=velocity_df_path,
                second=second,
            )
            if not isinstance(result, dict):
                result_list.append(result)
    df = pd.concat(result_list)

    # save df
    df.to_csv(path_settings.COSINOR_DF_PATH_2016, index=False)


def generate_jobs_2019(num_processes: int = 1):
    import path_settings
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # set dates
    from_dt = datetime.datetime(2019, 8, 20, hour=12, tzinfo=pytz.UTC)
    to_dt = datetime.datetime(2019, 9, 14, hour=12, tzinfo=pytz.UTC)

    # velocity path
    velocity_df_path = path_settings.VELOCITY_DF_PATH_2019

    # get alive bees
    if os.path.exists(velocity_df_path):
        alive_bees = [int(f[:-7]) for f in os.listdir(velocity_df_path) if f.endswith(".pickle")]
    else:
        raise Exception(f"Could not find {velocity_df_path}")

    # set median time window to 1h
    second = 3600

    # iterate through all bees
    result_list = []

    # helper to submit jobs
    def _submit_args(bee_id):
        return dict(
            bee_id=bee_id,
            from_dt=from_dt,
            to_dt=to_dt,
            velocity_df_path=velocity_df_path,
            second=second,
        )

    if num_processes and num_processes > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(run_job, **_submit_args(bee_id)) for bee_id in alive_bees]
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if not isinstance(result, dict):
                        result_list.append(result)
                except Exception as e:
                    # Skip failed jobs but print a note
                    print(f"[generate_jobs_2019] bee job failed: {e}")
    else:
        # Serial fallback
        for bee_id in alive_bees:
            result = run_job(
                bee_id=bee_id,
                from_dt=from_dt,
                to_dt=to_dt,
                velocity_df_path=velocity_df_path,
                second=second,
            )
            if not isinstance(result, dict):
                result_list.append(result)
    df = pd.concat(result_list)

    # save df
    df.to_csv(path_settings.COSINOR_DF_PATH_2019, index=False)


def run_job(
    bee_id=None, from_dt=None, to_dt=None, velocity_df_path=None, second=None
):
    import bb_rhythm.rhythm

    # create dataframe of results of cosinor fit for each day in time period
    import numpy.core.numeric as numeric
    sys.modules['numpy._core.numeric'] = numeric

    cosinor_df = bb_rhythm.rhythm.create_cosinor_df_per_bee_time_period(
        bee_id=bee_id,
        to_dt=to_dt,
        from_dt=from_dt,
        velocity_df_path=velocity_df_path,
        second=second,
    )
    return cosinor_df

# set sys path
sys.path.append(str(Path("cosinor_fit_per_bee.py").resolve().parents[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--year", type=int, help="Which year to analyze the data for. (2016 or 2019)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of worker processes for parallel execution (2019 only)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override data root (sets SPEEDTRANSFER_DATA_PATH for path_settings).",
    )

    args = parser.parse_args()
    if args.data_path:
        os.environ["SPEEDTRANSFER_DATA_PATH"] = args.data_path

    year = args.year

    if int(year) == 2016:
        generate_jobs_2016(num_processes=args.num_processes)
    elif int(year) == 2019:
        generate_jobs_2019(num_processes=args.num_processes)
    else:
        raise Exception(f"Invalid year: {year}")
