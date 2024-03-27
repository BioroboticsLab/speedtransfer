import datetime
from slurmhelper import SLURMJob
import sys
from pathlib import Path

"""
This is a script for calculating activating interaction cascade tree for the period 01.08.-25.08.2016 and 
20.08-14.09.2019. An interaction tree is calculated by choosing a subset of bees as the roots of our interaction trees
and then go back in time and recursively add a new child node in the trees whenever the bees interact with each other 
and the parent node has a positive speed change ("becomes activated") after this interaction. For faster computation
the job is in multiple slurmhelper job arrays. The resulting interaction trees are collected and each node of all paths 
in the interaction trees are concatenated to one dataframe.

Comment pipeline to create, run and concat results to a pandas DataFrame:
python social_network_interaction_tree.py --create
python social_network_interaction_tree.py --autorun
python social_network_interaction_tree.py --postprocess
"""


def run_job(
    path=None,
    time_threshold=None,
    vel_change_threshold=None,
    query=None,
    start_time=None,
    end_time=None,
    end_delta=None,
):
    import pandas as pd
    import numpy as np

    import bb_rhythm.interactions
    import bb_rhythm.network

    def random_sample_bee(interaction_df, n, query=None):
        random_samples = interaction_df.query(query).reset_index().sample(n=n)
        return random_samples

    # get interaction_df
    interaction_df = pd.read_csv(path)

    # filter overlap
    interaction_df = bb_rhythm.interactions.filter_overlap(interaction_df)

    # combine df so all bees are considered as focal
    interaction_df = bb_rhythm.interactions.combine_bees_from_interaction_df_to_be_all_focal(
        interaction_df
    )

    # map phase to 24h
    interaction_df["phase_focal"] = (
        (-24 * interaction_df["phase_focal"] / (2 * np.pi)) + 12
    ) % 24
    interaction_df["phase_non_focal"] = (
        (-24 * interaction_df["phase_non_focal"] / (2 * np.pi)) + 12
    ) % 24

    # sample random bees
    source_bee_ids = random_sample_bee(
        interaction_df[
            (interaction_df.interaction_start.dt.time <= start_time)
            & (interaction_df.interaction_start.dt.time > end_time)
        ],
        n=1,
        query=query,
    ).head(1)
    tree = bb_rhythm.network.create_interaction_tree(
        source_bee_ids,
        interaction_df,
        time_threshold,
        vel_change_threshold,
        source_bee_ids.interaction_start.iloc[0] - end_delta,
    )
    return tree


def generate_jobs_2019():
    import datetime

    import path_settings

    path = path_settings.INTERACTION_SIDE_0_DF_PATH_2019
    time_threshold = datetime.timedelta(minutes=30)
    vel_change_threshold = 0
    query = "(age_focal < 5) & (phase_focal > 12) & (p_value_focal < 0.05) & (age_focal > 0) & (vel_change_bee_focal > 0)"
    start_time = datetime.time(hour=15)
    end_time = datetime.time(hour=10)
    end_delta = datetime.timedelta(hours=2)
    for index in range(1000):
        yield dict(
            path=path,
            time_threshold=time_threshold,
            vel_change_threshold=vel_change_threshold,
            query=query,
            start_time=start_time,
            end_time=end_time,
            end_delta=end_delta,
        )


def generate_jobs_2016():
    import datetime

    import path_settings

    path = path_settings.INTERACTION_SIDE_0_DF_PATH_2016
    time_threshold = datetime.timedelta(minutes=30)
    vel_change_threshold = 0
    query = "(age_focal < 5) & (phase_focal > 12) & (p_value_focal < 0.05) & (age_focal > 0) & (vel_change_bee_focal > 0)"
    start_time = datetime.time(hour=15)
    end_time = datetime.time(hour=10)
    end_delta = datetime.timedelta(hours=2)
    for index in range(1000):
        yield dict(
            path=path,
            time_threshold=time_threshold,
            vel_change_threshold=vel_change_threshold,
            query=query,
            start_time=start_time,
            end_time=end_time,
            end_delta=end_delta,
        )


def concat_jobs_2016(job=None):
    import bb_rhythm.network

    import path_settings

    path_df = bb_rhythm.network.tree_to_path_df(job)
    path_df.reset_index(inplace=True, drop=True)
    path_df.to_csv(path_settings.INTERACTION_TREE_DF_PATH_2016, index=False)


def concat_jobs_2019(job=None):
    import bb_rhythm.network

    import path_settings

    path_df = bb_rhythm.network.tree_to_path_df(job)
    path_df.reset_index(inplace=True, drop=True)
    path_df.to_csv(path_settings.INTERACTION_TREE_DF_PATH_2019, index=False)


# set sys path
sys.path.append(str(Path("social_network_interaction_tree.py").resolve().parents[1]))

# create job
job = SLURMJob("interaction_model_tree_2016", "2016")
job.map(run_job, generate_jobs_2016())

# set job parameters
job.qos = "standard"
job.partition = "main,scavenger"
job.max_memory = "{}GB".format(35)
job.n_cpus = 1
job.max_job_array_size = 500
job.time_limit = datetime.timedelta(hours=24)
job.concurrent_job_limit = 100
job.custom_preamble = "#SBATCH --exclude=g[013-015]"
job.set_postprocess_fun(concat_jobs_2016)

# run job
job()
