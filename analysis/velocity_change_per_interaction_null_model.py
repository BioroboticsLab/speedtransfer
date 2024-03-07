from slurmhelper import SLURMJob
import datetime


def run_job_2019(
    dt_from=None, dt_to=None, interaction_model_path=None, velocities_path=None
):
    # imports for run job
    import datetime
    import pandas as pd

    import bb_behavior.db
    import bb_rhythm.interactions

    # get interactions start and end time current interaction model
    df_all = pd.read_pickle(interaction_model_path)
    df_all.drop(
        columns=df_all.columns.difference(
            ["interaction_start", "interaction_end", "bee_id0", "overlapping"]
        ),
        inplace=True,
    )
    df_all = bb_rhythm.interactions.filter_overlap(df_all)
    df = df_all[
        (df_all["interaction_start"] >= dt_from) & (df_all["interaction_end"] < dt_to)
    ]
    del df_all
    df["interaction_start"] = df["interaction_start"].dt.round("1S")
    df["interaction_end"] = df["interaction_end"].dt.round("1S")
    df_grouped = (
        df.groupby(by=["interaction_start", "interaction_end"])["bee_id0"]
        .count()
        .reset_index()
    )
    del df

    # server settings
    bb_behavior.db.base.server_address = "beequel.imp.fu-berlin.de:5432"
    bb_behavior.db.base.set_season_berlin_2019()
    bb_behavior.db.base.beesbook_season_config[
        "bb_detections"
    ] = "bb_detections_2019_berlin_orientationfix"

    # connect to DB
    with bb_behavior.db.base.get_database_connection(
        application_name="find_detections_in_frame"
    ) as db:
        cursor = db.cursor()

        # for each interaction
        interactions_lst = []
        for i, group in df_grouped.iterrows():
            table_name = bb_behavior.db.base.get_detections_tablename()
            interaction_start = group["interaction_start"]
            interaction_end = group["interaction_end"]
            delta = datetime.timedelta(seconds=1)
            # noinspection SqlDialectInspection
            sql_statement = """SELECT A.bee_id, A.x_pos, A.y_pos, A.orientation, B.x_pos, B.y_pos, B.orientation
                               FROM {} A, {} B
                               WHERE A.timestamp BETWEEN %s AND %s 
                                     AND A.cam_id BETWEEN 2 AND 3 
                                     AND A.bee_id_confidence >= 0.01 
                                     AND B.timestamp BETWEEN %s AND %s 
                                     AND B.cam_id BETWEEN 2 AND 3 
                                     AND B.bee_id_confidence >= 0.01 
                                     AND A.bee_id=B.bee_id
                               ORDER BY random()
                               limit %s;""".format(
                table_name, table_name
            )
            cursor.execute(
                sql_statement,
                (
                    interaction_start,
                    interaction_start + delta,
                    interaction_end,
                    interaction_end + delta,
                    str(2 * group["bee_id0"]),
                ),
            )
            random_sampled_interactions = cursor.fetchall()

            for index in range(0, len(random_sampled_interactions), 2):
                try:
                    interaction_dict = bb_rhythm.interactions.extract_parameters_from_random_samples(
                        random_sampled_interactions[index : index + 2],
                        interaction_start,
                        interaction_end,
                    )
                except IndexError:
                    continue

                # get velocity changes and add to interaction_dict
                # "focal" bee
                (
                    interaction_dict["vel_change_bee0"],
                    interaction_dict["rel_change_bee0"],
                ) = bb_rhythm.interactions.get_velocity_change_per_bee(
                    bee_id=interaction_dict["bee_id0"],
                    interaction_start=interaction_dict["interaction_start"],
                    interaction_end=interaction_dict["interaction_end"],
                    velocities_path=velocities_path,
                )
                # "non-focal" bee
                (
                    interaction_dict["vel_change_bee1"],
                    interaction_dict["rel_change_bee1"],
                ) = bb_rhythm.interactions.get_velocity_change_per_bee(
                    bee_id=interaction_dict["bee_id1"],
                    interaction_start=interaction_dict["interaction_start"],
                    interaction_end=interaction_dict["interaction_end"],
                    velocities_path=velocities_path,
                )

                # append interaction to interaction list
                interactions_lst.append(interaction_dict)
        return interactions_lst


def run_job_2016(
    dt_from=None, dt_to=None, interaction_model_path=None, velocities_path=None
):
    # imports for run job
    import datetime
    from sshtunnel import SSHTunnelForwarder
    import pandas as pd
    import numpy as np

    import bb_behavior.db
    import bb_rhythm.interactions

    # get interactions start and end time current interaction model
    df_all = pd.read_pickle(interaction_model_path)
    df_all.drop(
        columns=df_all.columns.difference(
            ["interaction_start", "interaction_end", "bee_id0", "overlapping"]
        ),
        inplace=True,
    )
    df_all = bb_rhythm.interactions.filter_overlap(df_all)
    df = df_all[
        (df_all["interaction_start"] >= dt_from) & (df_all["interaction_end"] < dt_to)
    ]
    del df_all
    df["interaction_start"] = df["interaction_start"].dt.round("1S")
    df["interaction_end"] = df["interaction_end"].dt.round("1S")
    df_grouped = (
        df.groupby(by=["interaction_start", "interaction_end"])["bee_id0"]
        .count()
        .reset_index()
    )
    del df

    # connect to ssh
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

        # connect to DB
        with bb_behavior.db.base.get_database_connection(
            application_name="find_detections_in_frame"
        ) as db:
            cursor = db.cursor()

            # for each interaction
            interactions_lst = []
            for i, group in df_grouped.iterrows():
                table_name = bb_behavior.db.base.get_detections_tablename()
                interaction_start = group["interaction_start"]
                interaction_end = group["interaction_end"]
                delta = datetime.timedelta(seconds=1)
                sql_statement = """SELECT A.bee_id, A.x_pos, A.y_pos, A.orientation, B.x_pos, B.y_pos, B.orientation
                                   FROM {} A, {} B
                                   WHERE A.timestamp BETWEEN %s AND %s 
                                         AND A.cam_id BETWEEN 2 AND 3 
                                         AND A.bee_id_confidence >= 0.01 
                                         AND B.timestamp BETWEEN %s AND %s 
                                         AND B.cam_id BETWEEN 2 AND 3 
                                         AND B.bee_id_confidence >= 0.01 
                                         AND A.bee_id=B.bee_id
                                   ORDER BY random()
                                   limit %s;""".format(
                    table_name, table_name
                )
                cursor.execute(
                    sql_statement,
                    (
                        interaction_start,
                        interaction_start + delta,
                        interaction_end,
                        interaction_end + delta,
                        str(2 * group["bee_id0"]),
                    ),
                )
                random_sampled_interactions = cursor.fetchall()

                for index in range(0, len(random_sampled_interactions), 2):
                    try:
                        interaction_dict = bb_rhythm.interactions.extract_parameters_from_random_samples(
                            random_sampled_interactions[index : index + 2],
                            interaction_start,
                            interaction_end,
                        )
                    except IndexError:
                        continue

                    # get velocity changes and add to interaction_dict
                    # "focal" bee
                    (
                        interaction_dict["vel_change_bee0"],
                        interaction_dict["rel_change_bee0"],
                    ) = bb_rhythm.interactions.get_velocity_change_per_bee(
                        bee_id=interaction_dict["bee_id0"],
                        interaction_start=interaction_dict["interaction_start"],
                        interaction_end=interaction_dict["interaction_end"],
                        velocities_path=velocities_path,
                    )
                    # "non-focal" bee
                    (
                        interaction_dict["vel_change_bee1"],
                        interaction_dict["rel_change_bee1"],
                    ) = bb_rhythm.interactions.get_velocity_change_per_bee(
                        bee_id=interaction_dict["bee_id1"],
                        interaction_start=interaction_dict["interaction_start"],
                        interaction_end=interaction_dict["interaction_end"],
                        velocities_path=velocities_path,
                    )

                    # append interaction to interaction list
                    interactions_lst.append(interaction_dict)
        return interactions_lst


def generate_jobs_2019():
    import datetime
    import pytz
    import os

    dt_from = pytz.UTC.localize(datetime.datetime(2019, 8, 20))
    dt_to = pytz.UTC.localize(datetime.datetime(2019, 9, 14))
    delta = datetime.timedelta(minutes=4)
    dt_current = dt_from - delta
    path = "~/../../scratch/weronik22/data/2019/"
    while dt_current <= dt_to:
        dt_current += delta
        yield dict(
            dt_from=dt_current,
            dt_to=dt_current + delta,
            interaction_model_path=os.path.join(path, "interactions_side1_final.pkl"),
            velocities_path="~/../../scratch/weronik22/data/velocities_20_8-14_9_2019/",
        )


def generate_jobs_2016():
    import datetime
    import pytz
    import os

    dt_from = pytz.UTC.localize(datetime.datetime(2016, 8, 1))
    dt_to = pytz.UTC.localize(datetime.datetime(2016, 8, 25))
    delta = datetime.timedelta(minutes=4)
    dt_current = dt_from - delta
    path = "~/../../scratch/weronik22/data/2016/"
    while dt_current <= dt_to:
        dt_current += delta
        yield dict(
            dt_from=dt_current,
            dt_to=dt_current + delta,
            interaction_model_path=os.path.join(
                path, "interactions_no_duplicates_side1.pkl"
            ),
            velocities_path="~/../../scratch/weronik22/data/velocities_1_8-25_8_2016/",
        )


# create job
job = SLURMJob("velocity_change_per_interaction_null_2019_side1", "/scratch/juliam98/")
job.map(run_job_2019, generate_jobs_2019())

# set job parameters
job.qos = "standard"
job.partition = "main,scavenger"
job.max_memory = "{}GB".format(50)
job.n_cpus = 1
job.max_job_array_size = 5000
job.time_limit = datetime.timedelta(minutes=60)
job.concurrent_job_limit = 50
job.custom_preamble = "#SBATCH --exclude=g[013-015]"

# run job
job()
