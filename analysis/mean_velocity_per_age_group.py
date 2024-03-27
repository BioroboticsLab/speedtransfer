from slurmhelper import SLURMJob
import datetime
import sys
from pathlib import Path

"""
This is a script for creating a per 10min velocity mean pandas dataframe per bee. Thus for each bee and for each age, 
the velocity is averaged over each 10-minute time window for the period 01.08.-25.08.2016 and 20.08-14.09.2019. For 
faster computation the job is divided so that one bee is one node in the slurmhelper job array.

Comment pipeline to create, run and concat results to a pandas DataFrame:
python mean_velocity_per_age_group.py --create
python mean_velocity_per_age_group.py --autorun
python mean_velocity_per_age_group.py --postprocess
"""


def run_job_2019(bee_id=None, dt_from=None, dt_to=None, velocity_df_path=None):
    # imports for run job
    import bb_behavior.db
    import bb_rhythm.rhythm

    import path_settings

    # server settings
    bb_behavior.db.base.server_address = f"{path_settings.SSH_SERVER_ADDRESS_2019}:{path_settings.REMOTE_BIND_ADDRESS_2019}"
    bb_behavior.db.base.set_season_berlin_2019()
    bb_behavior.db.base.beesbook_season_config[
        "bb_detections"
    ] = "bb_detections_2019_berlin_orientationfix"

    with bb_behavior.db.base.get_database_connection(
        application_name="mean_velocities"
    ) as db:
        cursor = db.cursor()
        grouped_velocities = bb_rhythm.rhythm.create_10_min_mean_velocity_df_per_bee(
            bee_id=bee_id,
            dt_from=dt_from,
            dt_to=dt_to,
            velocity_df_path=velocity_df_path,
            cursor=cursor,
        )
        return grouped_velocities


def run_job_2016(bee_id=None, dt_from=None, dt_to=None, velocity_df_path=None):
    from sshtunnel import SSHTunnelForwarder

    import bb_behavior.db.base
    import bb_behavior.db
    import bb_rhythm.rhythm

    # job im imports
    import path_settings

    with SSHTunnelForwarder(
        path_settings.SSH_SERVER_ADDRESS_2016,
        ssh_username=path_settings.SSH_USERNAME_2016,
        ssh_password=path_settings.SSH_PASSWORD_2016,
        remote_bind_address=path_settings.REMOTE_BIND_ADDRESS_2016,
    ) as server:
        # server settings
        bb_behavior.db.base.server_address = "localhost:{}".format(
            server.local_bind_port
        )
        bb_behavior.db.set_season_berlin_2016()
        bb_behavior.db.base.beesbook_season_config[
            "bb_alive_bees"
        ] = "alive_bees_2016_bayesian"

        with bb_behavior.db.base.get_database_connection(
            application_name="mean_velocities"
        ) as db:
            cursor = db.cursor()
            grouped_velocities = bb_rhythm.rhythm.create_10_min_mean_velocity_df_per_bee(
                bee_id=bee_id,
                dt_from=dt_from,
                dt_to=dt_to,
                velocity_df_path=velocity_df_path,
                cursor=cursor,
            )
            return grouped_velocities


def generate_jobs_2019():
    import pytz
    import datetime
    import bb_behavior.db

    import path_settings

    # server settings
    bb_behavior.db.base.server_address = f"{path_settings.SSH_SERVER_ADDRESS_2019}:{path_settings.REMOTE_BIND_ADDRESS_2019}"
    bb_behavior.db.base.set_season_berlin_2019()
    bb_behavior.db.base.beesbook_season_config[
        "bb_detections"
    ] = "bb_detections_2019_berlin_orientationfix"

    # set dates
    from_dt = datetime.datetime(2019, 8, 20, tzinfo=pytz.UTC)
    to_dt = datetime.datetime(2019, 9, 14, tzinfo=pytz.UTC)

    # get alive bees
    alive_bees = bb_behavior.db.get_alive_bees(from_dt, to_dt)

    # iterate through all bees
    for bee_id in alive_bees:
        yield dict(bee_id=bee_id)


def generate_jobs_2016():
    import datetime
    import pytz
    from sshtunnel import SSHTunnelForwarder

    import bb_behavior.db.base
    import bb_behavior.db

    # job im imports
    import path_settings

    with SSHTunnelForwarder(
        path_settings.SSH_SERVER_ADDRESS_2016,
        ssh_username=path_settings.SSH_USERNAME_2016,
        ssh_password=path_settings.SSH_PASSWORD_2016,
        remote_bind_address=path_settings.REMOTE_BIND_ADDRESS_2016,
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

        # iterate through all bees
        for bee_id in alive_bees:
            yield dict(bee_id=bee_id)


def concat_jobs_2016(job=None):
    import pandas as pd

    import path_settings

    # for each 10min mean velocity per bee combine to one
    result_df = pd.DataFrame(columns=["time", "velocity", "age"])
    for kwarg, result in job.items():
        result_df = pd.concat([result_df, result], ignore_index=True)

    # get mean velocity df per age
    result_df = result_df.groupby(["time", "age"])["velocity"].mean().reset_index()

    # save df
    result_df.to_csv(path_settings.MEAN_VELOCITY_DF_PATH_2016, index=False)


def concat_jobs_2019(job=None):
    import pandas as pd

    import path_settings

    # for each 10min mean velocity per bee combine to one
    result_df = pd.DataFrame(columns=["time", "velocity", "age"])
    for kwarg, result in job.items():
        result_df = pd.concat([result_df, result], ignore_index=True)

    # get mean velocity df per age
    result_df = result_df.groupby(["time", "age"])["velocity"].mean().reset_index()

    # save df
    result_df.to_csv(path_settings.MEAN_VELOCITY_DF_PATH_2019, index=False)


# set sys path
sys.path.append(str(Path("mean_velocity_per_age_group.py").resolve().parents[1]))

# create job
job = SLURMJob("BB2019_bayesian_velocity_mean_10min", "2019")
job.map(run_job_2019, generate_jobs_2019())

# set job parameters
job.qos = "standard"
job.partition = "main,scavenger"
job.max_memory = "{}GB".format(6)
job.n_cpus = 1
job.max_job_array_size = 5000
job.time_limit = datetime.timedelta(minutes=60)
job.concurrent_job_limit = 100
job.custom_preamble = "#SBATCH --exclude=g[013-015]"
job.exports = "OMP_NUM_THREADS=2,MKL_NUM_THREADS=2"
job.set_postprocess_fun(concat_jobs_2019)

# run job
job()
