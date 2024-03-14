from slurmhelper import SLURMJob
import datetime
import pytz
from sshtunnel import SSHTunnelForwarder

import bb_behavior.db.base
import bb_behavior.db

"""
This is a script for creating a dataframe of the parameters of a cosinor fit of the velocity per bee for a time window 
of 3 consecutive days for the period 01.08.-25.08.2016 and  20.08-14.09.2019. For faster computation the job is 
divided that one bee is one node in the slurmhelper job array. The resulting cosinor fit parameter dataframes per bee 
are concatenated to one df.

Comment pipeline to create, run and concat results to a pandas DataFrame:
python cosinor_fit_per_bee.py --create
python cosinor_fit_per_bee.py --autorun
python cosinor_fit_per_bee.py --postprocess
"""


def generate_jobs_2016():

    # job im imports
    from .. import path_settings

    with SSHTunnelForwarder(
        "bommel.imp.fu-berlin.de",
        ssh_username="dbreader",
        ssh_password="dbreaderpw",
        remote_bind_address=("127.0.0.1", 5432),
    ) as server:
        # server settings
        bb_behavior.db.base.server_address = "localhost:{}".format(
            server.local_bind_port
        )
        bb_behavior.db.set_season_berlin_2016()
        bb_behavior.db.base.beesbook_season_config[
            "bb_alive_bees"
        ] = "alive_bees_2016_bayesian"

        # set dates
        from_dt = datetime.datetime(2016, 8, 1, tzinfo=pytz.UTC)
        to_dt = datetime.datetime(2016, 8, 25, tzinfo=pytz.UTC)

        # get alive bees
        alive_bees = bb_behavior.db.get_alive_bees(from_dt, to_dt)

        # velocity path
        velocity_df_path = path_settings.VELOCITY_DF_PATH_2016

        # set median time window to 1h
        second = 3600

        # iterate through all bees
        for bee_id in alive_bees:
            yield dict(
                bee_id=bee_id,
                from_dt=from_dt,
                to_dt=to_dt,
                velocity_df_path=velocity_df_path,
                second=second,
            )


def generate_jobs_2019():

    from .. import path_settings

    # server settings
    bb_behavior.db.base.server_address = "beequel.imp.fu-berlin.de:5432"
    bb_behavior.db.base.set_season_berlin_2019()
    bb_behavior.db.base.beesbook_season_config[
        "bb_detections"
    ] = "bb_detections_2019_berlin_orientationfix"

    # set dates
    from_dt = datetime.datetime(2019, 8, 20, tzinfo=pytz.UTC)
    to_dt = datetime.datetime(2019, 9, 14, tzinfo=pytz.UTC)

    # get alive bees
    alive_bees = bb_behavior.db.get_alive_bees(from_dt, to_dt)

    # velocity path
    velocity_df_path = path_settings.VELOCITY_DF_PATH_2019

    # set median time window to 1h
    second = 3600

    # iterate through all bees
    for bee_id in alive_bees:
        yield dict(
            bee_id=bee_id,
            from_dt=from_dt,
            to_dt=to_dt,
            velocity_df_path=velocity_df_path,
            second=second,
        )


def run_job_2019(
    bee_id=None, from_dt=None, to_dt=None, velocity_df_path=None, second=None
):
    import bb_behavior.db.base
    import bb_behavior.db
    import bb_rhythm.rhythm

    # server settings
    bb_behavior.db.base.server_address = "beequel.imp.fu-berlin.de:5432"
    bb_behavior.db.base.set_season_berlin_2019()
    bb_behavior.db.base.beesbook_season_config[
        "bb_detections"
    ] = "bb_detections_2019_berlin_orientationfix"

    # create dataframe of results of cosinor fit for each day in time period
    cosinor_df = bb_rhythm.rhythm.create_cosinor_df_per_bee_time_period(
        bee_id=bee_id,
        to_dt=to_dt,
        from_dt=from_dt,
        velocity_df_path=velocity_df_path,
        second=second,
    )
    return cosinor_df


def run_job_2016(
    bee_id=None, from_dt=None, to_dt=None, velocity_df_path=None, second=None
):
    from sshtunnel import SSHTunnelForwarder

    import bb_behavior.db.base
    import bb_behavior.db
    import bb_rhythm.rhythm

    with SSHTunnelForwarder(
        "bommel.imp.fu-berlin.de",
        ssh_username="dbreader",
        ssh_password="dbreaderpw",
        remote_bind_address=("127.0.0.1", 5432),
    ) as server:
        # server settings
        bb_behavior.db.base.server_address = "localhost:{}".format(
            server.local_bind_port
        )
        bb_behavior.db.set_season_berlin_2016()
        bb_behavior.db.base.beesbook_season_config[
            "bb_alive_bees"
        ] = "alive_bees_2016_bayesian"

        # create dataframe of results of cosinor fit for each day in time period
        cosinor_df = bb_rhythm.rhythm.create_cosinor_df_per_bee_time_period(
            bee_id=bee_id,
            to_dt=to_dt,
            from_dt=from_dt,
            velocity_df_path=velocity_df_path,
            second=second,
        )
        return cosinor_df


def concat_jobs_2016(job=None):
    import pandas as pd

    from .. import path_settings

    # collect all dfs per bee and combine to one df
    result_list = []
    for kwarg, result in job.items(ignore_open_jobs=True):
        # case of death or non-existent tracking data
        if type(result) is dict:
            continue
        else:
            result_list.append(result)
    df = pd.concat(result_list)

    # save df
    df.to_pickle(path_settings.COSINOR_DF_PATH_2016)


def concat_jobs_2019(job=None):
    import pandas as pd

    from .. import path_settings

    # collect all dfs per bee and combine to one df
    result_list = []
    for kwarg, result in job.items(ignore_open_jobs=True):
        # case of death or non-existent tracking data
        if type(result) is dict:
            continue
        else:
            result_list.append(result)
    df = pd.concat(result_list)

    # save df
    df.to_pickle(path_settings.COSINOR_DF_PATH_2019)


# create job
job = SLURMJob("2016_circadian_velocities_cosinor_median", "/scratch/juliam98/")
job.map(run_job_2016, generate_jobs_2016())

# set job parameters
job.qos = "standard"
job.partition = "main,scavenger"
job.max_memory = "{}GB".format(2)
job.n_cpus = 1
job.max_job_array_size = 5000
job.time_limit = datetime.timedelta(minutes=60)
job.concurrent_job_limit = 100
job.custom_preamble = "#SBATCH --exclude=g[013-015],b[001-004],c[003-004],g[009-015]"
job.exports = "OMP_NUM_THREADS=2,MKL_NUM_THREADS=2"
job.set_postprocess_fun(concat_jobs_2016)

# run job
job()
