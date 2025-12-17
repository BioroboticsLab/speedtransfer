#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.filterwarnings("ignore")

import datetime
import pytz
import argparse
import numpy as np
import os
import glob
import pandas as pd
import scipy
import re
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()

parser.add_argument(
    "--date_from", type=int, nargs="+", help="Start date in the format dd mm yyyy."
)
parser.add_argument(
    "--date_to", type=int, nargs="+", help="End date in the format dd mm yyyy."
)
parser.add_argument(
    "--trajectory_dir", type=str, help="In which directory the trajectory files are located."
)
parser.add_argument(
    "--num-processes",
    type=int,
    default=4,
    help="Number of worker processes to process trajectory files in parallel.",
)
args = parser.parse_args()

# Get dates in UTC.
year = args.date_from[2]
dt_from = pytz.UTC.localize(
    datetime.datetime(year, args.date_from[1], args.date_from[0], 0)
)
dt_to = pytz.UTC.localize(datetime.datetime(year, args.date_to[1], args.date_to[0], 0))

# Create destination directory for velocity data.
subdir = "velocities_%d_%d-%d_%d_%d" % (
    dt_from.day,
    dt_from.month,
    dt_to.day,
    dt_to.month,
    year,
)
cache_directory = os.path.join(os.pardir, "data", subdir)
print("Data will be saved to %s." % cache_directory)
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory, exist_ok=True)


def get_bee_velocity_df(trajectory_df, max_mm_per_second=None, year="2019", bee_id_confidence=0.1):
    trajectory_df = trajectory_df.query("bee_id_confidence > %d" % bee_id_confidence)[
        ["timestamp", "x_hive", "y_hive", "cam_id"]
    ]

    # time delta in seconds
    trajectory_df["time_passed"] = trajectory_df["timestamp"].diff().dt.total_seconds()

    # spatial deltas
    trajectory_df["dx"] = trajectory_df["x_hive"].diff()
    trajectory_df["dy"] = trajectory_df["y_hive"].diff()

    # velocity in mm/s
    trajectory_df["velocity"] = np.sqrt(trajectory_df["dx"] ** 2 + trajectory_df["dy"] ** 2)
    if year == "2016":
        trajectory_df["time_passed"] = np.round(trajectory_df["time_passed"] * float(3)) / 3.0
        trajectory_df.loc[trajectory_df["time_passed"] == 0.0, "time_passed"] = 1.0 / 3
    trajectory_df["velocity"] = trajectory_df["velocity"] / trajectory_df["time_passed"]

    # set velocity to NaN when camera switches
    cam_switch = trajectory_df["cam_id"].diff().fillna(0) != 0
    trajectory_df.loc[cam_switch, "velocity"] = np.nan

    if max_mm_per_second is not None:
        trajectory_df["velocity"].where(trajectory_df["velocity"] > max_mm_per_second, np.nan, inplace=True)

    trajectory_df.rename(columns={"timestamp": "datetime"}, inplace=True)
    trajectory_df["datetime"] = pd.to_datetime(trajectory_df["datetime"], utc=True)
    return trajectory_df[["datetime", "velocity", "time_passed"]]

def parse_filename_timerange(filename: str):
    """
    Extracts start and end datetime from filenames like:
    2019-08-06T12_00_00Z--2019-08-06T13_00_00Z.parquet
    Returns (start_dt, end_dt) as timezone-aware UTC datetimes.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"(.*?)--(.*)", base)
    if not match:
        return None, None
    start_str, end_str = match.groups()
    # Replace underscores with colons in the time part
    start_str = start_str.replace("_", ":")
    end_str = end_str.replace("_", ":")
    try:
        start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        return start_dt, end_dt
    except ValueError:
        return None, None

def _process_file(task):
    file, out_dir, year = task
    trajectory_df = pd.read_parquet(file)
    timestamp = os.path.splitext(os.path.basename(file))[0]

    for bee_id, bee_df in trajectory_df.groupby("bee_id"):
        filepath = os.path.join(out_dir, f"{bee_id}_{timestamp}.parquet")
        if not os.path.exists(filepath):
            df = get_bee_velocity_df(bee_df, year=year)
            df.to_parquet(filepath, index=False)


def save_bee_velocities(in_dir, out_dir, dt_from, dt_to, year="2019", num_processes=4):
    os.makedirs(out_dir, exist_ok=True)

    # Convert dt_from, dt_to to datetime
    dt_from = pd.to_datetime(dt_from)
    dt_to = pd.to_datetime(dt_to)

    # Get all parquet files and filter by timestamp in filename
    files = sorted(glob.glob(os.path.join(in_dir, "*.parquet")))
    filtered_files = []
    for file in files:
        start_dt, end_dt = parse_filename_timerange(file)
        if start_dt is None:
            continue
        # Keep file if it overlaps the requested time window
        if (start_dt <= dt_to) and (end_dt >= dt_from):
            filtered_files.append(file)

    tasks = [(file, out_dir, year) for file in filtered_files]

    # First pass: Process each parquet file & save per bee per file (parallel by file)
    if num_processes and num_processes > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            list(executor.map(_process_file, tasks))
    else:
        for task in tasks:
            _process_file(task)

    # Second pass: Combine all frames per bee into one file (memory efficient)
    per_bee_files = glob.glob(os.path.join(out_dir, "*_*.parquet"))

    # Get unique bee IDs
    bee_ids = sorted({int(os.path.basename(f).split("_")[0]) for f in per_bee_files})

    for bee_id in bee_ids:
        # Collect all chunk files for this bee
        bee_files = [f for f in per_bee_files if f.startswith(os.path.join(out_dir, f"{bee_id}_"))]

        dfs = []
        for f in bee_files:
            dfs.append(pd.read_parquet(f))

        # Concatenate and sort
        combined_df = pd.concat(dfs).sort_values("datetime")

        # Save combined frame
        combined_path = os.path.join(out_dir, f"{bee_id}.pickle")
        combined_df.to_pickle(combined_path)

        # Remove intermediate chunk files
        for f in bee_files:
            os.remove(f)

        # Explicitly free memory
        del dfs, combined_df


if year in (2016, 2019):
    save_bee_velocities(
        args.trajectory_dir,
        cache_directory,
        dt_from,
        dt_to,
        year,
        num_processes=args.num_processes,
    )
else:
    print(f"Unsupported year: {year}")
