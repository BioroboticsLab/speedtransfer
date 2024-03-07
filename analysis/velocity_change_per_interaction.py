from slurmhelper import SLURMJob
import datetime

"""
This is a script for creating a dataframe of the bee interactions and their post-interaction velocity change for the 
period 01.08.-25.08.2016 and 20.08-14.09.2019. An interactions between two bees (focal and non-focal bees) is defined 
when two bees are detected simultaneously in the hive with a confidence threshold of 0.25, the distance between the 
markings on their thorax bodies is no more than 14 mm. These interactions are combined into one interaction if the 
same detections occur within a time interval of 1 second or less between them. For faster computation the job is 
divided that the time window of 2 minutes is one node in the slurmhelper job array.

Comment pipeline to create, run and concat results to a pandas DataFrame:
python velocity_change_per_interaction.py --create
python velocity_change_per_interaction.py --autorun
python velocity_change_per_interaction_concat.py
"""

VELOCITY_DF_PATH_2016 = "../data/2016/velocities_1_8-25_8_2016"
VELOCITY_DF_PATH_2019 = "../data/2019/velocities_20_8-14_9_2019"


def run_job_2016(dt_from=None, dt_to=None, cam_ids=None):
    # imports for run job
    from sshtunnel import SSHTunnelForwarder

    import bb_behavior.db
    import bb_rhythm.interactions

    # connect with DB
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

        # open DB connection
        with bb_behavior.db.base.get_database_connection(
            application_name="find_interactions_in_frame"
        ) as db:
            cursor = db.cursor()

            # fetch all interactions in frames of cams in time window
            interactions_per_frame = bb_rhythm.interactions.fetch_interactions_per_frame(
                cam_ids, cursor, dt_from, dt_to
            )
            if not interactions_per_frame:
                return {None: dict(error="No frames fetched..")}

            # cluster per frame detected interactions per time in interaction events
            interactions = bb_rhythm.interactions.get_all_interactions_over_time(
                interactions_per_frame
            )
            if not interactions:
                return {None: dict(error="No events found..")}

            # calculate per interaction post-interaction velocity changes
            bb_rhythm.interactions.add_post_interaction_velocity_change(
                interactions, velocity_df_path=VELOCITY_DF_PATH_2016
            )
            return interactions


def run_job_2019(dt_from=None, dt_to=None, cam_ids=None):
    # imports for run job
    import bb_behavior.db
    import bb_rhythm.interactions

    # server settings
    bb_behavior.db.base.server_address = "beequel.imp.fu-berlin.de:5432"
    bb_behavior.db.base.set_season_berlin_2019()
    bb_behavior.db.base.beesbook_season_config[
        "bb_detections"
    ] = "bb_detections_2019_berlin_orientationfix"

    # connect with DB
    with bb_behavior.db.base.get_database_connection(
        application_name="find_interactions_in_frame"
    ) as db:
        cursor = db.cursor()
        # fetch all interactions in frames of cams in time window
        interactions_per_frame = bb_rhythm.interactions.fetch_interactions_per_frame(
            cam_ids, cursor, dt_from, dt_to
        )
        if not interactions_per_frame:
            return {None: dict(error="No frames fetched..")}

        # cluster per frame detected interactions per time in interaction events
        interactions = bb_rhythm.interactions.get_all_interactions_over_time(
            interactions_per_frame
        )
        if not interactions:
            return {None: dict(error="No events found..")}

        # calculate per interaction post-interaction velocity changes
        bb_rhythm.interactions.add_post_interaction_velocity_change(
            interactions, velocity_df_path=VELOCITY_DF_PATH_2019
        )
        return interactions


def generate_jobs_2016():
    import datetime
    import pytz

    # set time period and interaction time window
    dt_from = pytz.UTC.localize(datetime.datetime(2016, 8, 1))
    dt_to = pytz.UTC.localize(datetime.datetime(2016, 8, 26))
    delta = datetime.timedelta(minutes=2)

    # create jobs
    dt_current = dt_from - delta
    while dt_current <= dt_to:
        dt_current += delta
        yield dict(dt_from=dt_current, dt_to=dt_current + delta, cam_ids=[0, 1])


def generate_jobs_2019():
    import datetime
    import pytz

    # set time period and interaction time window
    dt_from = pytz.UTC.localize(datetime.datetime(2019, 8, 20))
    dt_to = pytz.UTC.localize(datetime.datetime(2019, 9, 15))
    delta = datetime.timedelta(minutes=2)

    # create jobs
    dt_current = dt_from - delta
    while dt_current <= dt_to:
        dt_current += delta
        yield dict(dt_from=dt_current, dt_to=dt_current + delta, cam_ids=[1])


# create job
job = SLURMJob("velocity_change_per_interaction_cam_id_1_2019", "/scratch/juliam98/")
job.map(run_job_2019, generate_jobs_2019())

# set job parameters
job.qos = "standard"
job.partition = "main,scavenger"
job.max_memory = "{}GB".format(1)
job.n_cpus = 1
job.max_job_array_size = 5000
job.time_limit = datetime.timedelta(minutes=440)
job.concurrent_job_limit = 25
job.custom_preamble = "#SBATCH --exclude=g[013-015]"

# run job
job()
