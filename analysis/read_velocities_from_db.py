#!/usr/bin/env python
# coding: utf-8

import os
import bb_behavior.db as db
import warnings
warnings.filterwarnings('ignore')
import time
import datetime
import pytz
import argparse
import numpy as np
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import contextlib

parser = argparse.ArgumentParser()

parser.add_argument("--date_from", type=int, nargs="+",
                    help="Start date in the format dd mm yyyy.")
parser.add_argument("--date_to", type=int, nargs="+",
                    help="End date in the format dd mm yyyy.")
parser.add_argument("--batch", type=int,
                    help="Which batch of bees to download the data for. (0-199)")
args = parser.parse_args()

# Get dates in UTC.
year = args.date_from[2]
dt_from = pytz.UTC.localize(datetime.datetime(year,args.date_from[1],args.date_from[0],0))
dt_to = pytz.UTC.localize(datetime.datetime(year,args.date_to[1],args.date_to[0],0))

# Create destination directory for velocity data.
subdir = "velocities_%d_%d-%d_%d_%d" % (dt_from.day, dt_from.month, dt_to.day, dt_to.month, year)
cache_directory = os.path.join(os.pardir, "data", subdir)
print('Data will be saved to %s.' % cache_directory)
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory, exist_ok=True)

# Define which batch of bees to download the data for.
n_batches = 200
batch_number = args.batch
print('Current batch: ', batch_number, '\n')


def save_bee_velocities(cursor):
    # Get IDs of all alive bees.
    alive_bees = list(db.metadata.get_alive_bees(dt_from, dt_to))
    len_bees = len(alive_bees)
    print(f'{len_bees} alive bees.')

    # Split bee IDs into chuncks and select subset.
    current_batch = np.array_split(alive_bees, n_batches)[batch_number]

    cursor_is_prepared = False

    for bee_id in current_batch:
        bee_id = bee_id.item()
        filepath = os.path.join(cache_directory, "%d.pickle" % bee_id)

        if not os.path.exists(filepath):
            start = time.time()
            df = db.trajectory.get_bee_velocities(bee_id, dt_from, dt_to,
                                                additional_columns=['cam_id', 'x_pos_hive', 'y_pos_hive', 'frame_id', 'orientation_hive'],
                                                cursor=cursor, cursor_is_prepared=cursor_is_prepared)
            cursor_is_prepared = True

            if type(df) == pd.DataFrame:
                print('Saving to %s...' % filepath)
                df.to_pickle(filepath)

            end = time.time()
            print('Took %f sec.' % (end - start))

        else:
            print('File already exists, skipping.')


if year == 2019:
    db.base.server_address = 'beequel.imp.fu-berlin.de:5432'
    db.base.set_season_berlin_2019()
    db.base.beesbook_season_config["bb_detections"] = "bb_detections_2019_berlin_orientationfix"
    with contextlib.closing(db.get_database_connection()) as con:
        cursor = con.cursor()
        save_bee_velocities(cursor)

elif year == 2016:
    with SSHTunnelForwarder(
            'bommel.imp.fu-berlin.de',
            ssh_username="dbreader",
            ssh_password="dbreaderpw",
            remote_bind_address=('127.0.0.1', 5432)
    ) as server:
        db.set_season_berlin_2016()
        db.base.server_address = "localhost:{}".format(server.local_bind_port)
        db.base.beesbook_season_config["bb_alive_bees"] = "alive_bees_2016_bayesian"
        with contextlib.closing(db.get_database_connection()) as con:
            cursor = con.cursor()
            save_bee_velocities(cursor)