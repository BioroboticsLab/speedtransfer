from slurmhelper import SLURMJob
import datetime

"""
This is a script for creating a per 10min velocity mean pandas dataframe per bee. Thus for each bee, the velocity is 
averaged over each 10-minute time window for the period 01.08.-25.08.2016 and 20.08-14.09.2019. For faster computation
the job is divided so that one bee is one node in the slurmhelper job array.The resulting 10min mean velocity dataframes 
per bee are grouped by age and the mean velocity per 60min was calculated for each age.

Comment pipeline to create, run and concat results to a pandas DataFrame:
python mean_velocity_per_age_group.py --create
python mean_velocity_per_age_group.py --autorun
python mean_velocity_per_age_group.py --postprocess
"""


def run_job_2019(bee_id=None, dt_from=None, dt_to=None, velocity_df_path=None):
    # imports for run job
    import bb_behavior.db
    import bb_rhythm.rhythm

    # server settings
    bb_behavior.db.base.server_address = "beequel.imp.fu-berlin.de:5432"
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

    # iterate through all bees
    for bee_id in alive_bees:
        yield dict(bee_id=bee_id)


def generate_jobs_2016():
    import datetime
    import pytz
    from sshtunnel import SSHTunnelForwarder

    import bb_behavior.db.base
    import bb_behavior.db

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

        # iterate through all bees
        for bee_id in alive_bees:
            yield dict(bee_id=bee_id)


def concat_jobs_2016(job=None):
    import pandas as pd

    from .. import path_settings

    # for each 10min mean velocity per bee combine to one
    result_df = pd.DataFrame(columns=["time", "velocity", "age"])
    for kwarg, result in job.items():
        result_df = pd.concat([result_df, result], ignore_index=True)

    # get mean velocity df per age
    result_df = result_df.groupby(["time", "age"])["velocity"].mean().reset_index()

    # save df
    result_df.to_pickle(path_settings.MEAN_VELOCITY_DF_PATH_2016)


def concat_jobs_2019(job=None):
    import pandas as pd

    from .. import path_settings

    # for each 10min mean velocity per bee combine to one
    result_df = pd.DataFrame(columns=["time", "velocity", "age"])
    for kwarg, result in job.items():
        result_df = pd.concat([result_df, result], ignore_index=True)

    # get mean velocity df per age
    result_df = result_df.groupby(["time", "age"])["velocity"].mean().reset_index()

    # save df
    result_df.to_pickle(path_settings.MEAN_VELOCITY_DF_PATH_2019)


# create job
job = SLURMJob("BB2019_bayesian_velocity_mean_10min", "/home/juliam98/slurm_tryouts")
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
