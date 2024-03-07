from slurmhelper import SLURMJob
import datetime
import bb_rhythm.interactions


def run_job_2016(dt_from=None, dt_to=None, cam_ids=None):
    # imports for run job
    from sshtunnel import SSHTunnelForwarder
    import pytz

    import bb_behavior.db
    from bb_rhythm import interactions

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

        with bb_behavior.db.base.get_database_connection(
            application_name="find_interactions_in_frame"
        ) as db:
            cursor = db.cursor()
            interactions_lst = []
            for cam_id in cam_ids:
                # get frames
                frame_data = bb_behavior.db.get_frames(
                    cam_id, dt_from, dt_to, cursor=cursor
                )
                if not frame_data:
                    continue

                # for each frame id
                for dt, frame_id, cam_id in frame_data:
                    # get interactions
                    interactions_detected = bb_behavior.db.find_interactions_in_frame(
                        frame_id=frame_id, cursor=cursor
                    )
                    for i in interactions_detected:
                        interactions_lst.append(
                            {
                                "bee_id0": i[1],
                                "bee_id1": i[2],
                                "timestamp": dt,
                                "location_info_bee0": (i[5], i[6], i[7]),
                                "location_info_bee1": (i[8], i[9], i[10]),
                            }
                        )
            if len(interactions_lst) == 0:
                return {None: dict(error="No frames fetched..")}

            # cluster interactions per time in events
            events = interactions.get_all_interactions_over_time(interactions_lst)
            if not events:
                return {None: dict(error="No events found..")}

            # get interactions and velocity changes
            extracted_interactions_lst = []
            for key in events:
                event_dict = events[key]
                if events[key]:
                    # extract events parameters as interactions
                    interaction_dict = interactions.extract_parameters_from_events(
                        event_dict[0]
                    )
                    # get velocity changes
                    # "focal" bee
                    (
                        interaction_dict["vel_change_bee0"],
                        interaction_dict["relative_change_bee0"],
                    ) = interactions.get_velocity_change_per_bee(
                        bee_id=interaction_dict["bee_id0"],
                        interaction_start=interaction_dict["interaction_start"].replace(
                            tzinfo=pytz.UTC
                        ),
                        interaction_end=interaction_dict["interaction_end"].replace(
                            tzinfo=pytz.UTC
                        ),
                    )
                    # "non-focal" bee
                    (
                        interaction_dict["vel_change_bee1"],
                        interaction_dict["relative_change_bee1"],
                    ) = interactions.get_velocity_change_per_bee(
                        bee_id=interaction_dict["bee_id1"],
                        interaction_start=interaction_dict["interaction_start"].replace(
                            tzinfo=pytz.UTC
                        ),
                        interaction_end=interaction_dict["interaction_end"].replace(
                            tzinfo=pytz.UTC
                        ),
                    )
                    extracted_interactions_lst.append(interaction_dict)
            return extracted_interactions_lst


def run_job_2019(dt_from=None, dt_to=None, cam_ids=None):
    # imports for run job
    import warnings
    import pytz

    warnings.filterwarnings("ignore")

    import bb_behavior.db
    from bb_rhythm import interactions

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
        interactions_lst = []
        for cam_id in cam_ids:
            # get frames
            frame_data = bb_behavior.db.get_frames(
                cam_id, dt_from, dt_to, cursor=cursor
            )
            if not frame_data:
                continue

            # for each frame id
            for dt, frame_id, cam_id in frame_data:
                # get interactions
                interactions_detected = bb_behavior.db.find_interactions_in_frame(
                    frame_id=frame_id, cursor=cursor
                )
                for i in interactions_detected:
                    interactions_lst.append(
                        {
                            "bee_id0": i[1],
                            "bee_id1": i[2],
                            "timestamp": dt,
                            "location_info_bee0": (i[5], i[6], i[7]),
                            "location_info_bee1": (i[8], i[9], i[10]),
                        }
                    )
        if len(interactions_lst) == 0:
            return {None: dict(error="No frames fetched..")}

        # cluster interactions per time in events
        events = interactions.get_all_interactions_over_time(interactions_lst)
        if not events:
            return {None: dict(error="No events found..")}

        # get interactions and velocity changes
        extracted_interactions_lst = []
        for key in events:
            event_dict = events[key]
            if events[key]:
                # extract events parameters as interactions
                interaction_dict = interactions.extract_parameters_from_events(
                    event_dict[0]
                )
                interaction_start = interaction_dict["interaction_start"].replace(
                    tzinfo=pytz.UTC
                )
                interaction_end = interaction_dict["interaction_end"].replace(
                    tzinfo=pytz.UTC
                )
                # get velocity changes
                # "focal" bee
                (
                    interaction_dict["vel_change_bee0"],
                    interaction_dict["rel_change_bee0"],
                ) = interactions.get_velocity_change_per_bee(
                    bee_id=interaction_dict["bee_id0"],
                    interaction_start=interaction_start,
                    interaction_end=interaction_end,
                )
                # "non-focal" bee
                (
                    interaction_dict["vel_change_bee1"],
                    interaction_dict["rel_change_bee1"],
                ) = interactions.get_velocity_change_per_bee(
                    bee_id=interaction_dict["bee_id1"],
                    interaction_start=interaction_start,
                    interaction_end=interaction_end,
                )
                extracted_interactions_lst.append(interaction_dict)
        return extracted_interactions_lst


def generate_jobs_2016():
    import datetime
    import pytz

    dt_from = pytz.UTC.localize(datetime.datetime(2016, 8, 1))
    dt_to = pytz.UTC.localize(datetime.datetime(2016, 8, 26))
    delta = datetime.timedelta(minutes=2)

    dt_current = dt_from - delta
    while dt_current <= dt_to:
        dt_current += delta
        yield dict(dt_from=dt_current, dt_to=dt_current + delta)


def generate_jobs_2019():
    import datetime
    import pytz

    dt_from = pytz.UTC.localize(datetime.datetime(2019, 8, 20))
    dt_to = pytz.UTC.localize(datetime.datetime(2019, 9, 15))
    delta = datetime.timedelta(minutes=2)

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
# job.exports = "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1"

# run job
job()
