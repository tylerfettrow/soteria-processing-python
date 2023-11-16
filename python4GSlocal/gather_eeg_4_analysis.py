import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helpers
import importlib

sns.set_theme()

importlib.reload(helpers)

helper = helpers.HELP()

crews_to_process = [
    "Crew_01",
    "Crew_02",
    "Crew_03",
    "Crew_04",
    "Crew_05",
    "Crew_06",
    "Crew_07",
    "Crew_08",
    "Crew_09",
    "Crew_10",
    "Crew_11",
    "Crew_13",
]

total_eventEEG_metric_dataframe = pd.DataFrame()
for i_crew in range(len(crews_to_process)):
    this_subject_event_metric_dataframe = pd.DataFrame()
    crew_dir = crews_to_process[i_crew]
    process_dir_name = crew_dir + "/Processing/"
    print("grabbing results from " + crew_dir)

    event_eegTimeSeries_metrics = helper.read_bucket_table(
        process_dir_name, "event_eegTimeSeries_metrics.csv"
    )
    event_eegTimeSeries_df = event_eegTimeSeries_metrics[
        event_eegTimeSeries_metrics.columns[1:]
    ]

    total_eventEEG_metric_dataframe = pd.concat(
        [total_eventEEG_metric_dataframe, event_eegTimeSeries_df]
    )

helper.reset_folder_storage()

total_eventEEG_metric_dataframe.to_csv(
    "Analysis/" + "total_eventEEG_metric_dataframe.csv"
)
helper.sync_crew_folder_storage()

colors = sns.color_palette("tab10", 2)
colors_array = np.array(colors)

total_eventEEG_metric_dataframe.seat[
    total_eventEEG_metric_dataframe.seat == 1
] = "right"
total_eventEEG_metric_dataframe.seat[total_eventEEG_metric_dataframe.seat == 0] = "left"
total_eventEEG_metric_dataframe.event_label[
    total_eventEEG_metric_dataframe.event_label == 0
] = "No Event"
total_eventEEG_metric_dataframe.event_label[
    total_eventEEG_metric_dataframe.event_label == 1
] = "Event"
total_eventEEG_metric_dataframe.crew = total_eventEEG_metric_dataframe.crew.astype(int)

total_eventEEG_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "taskLoad_index_spec"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("taskLoad_index_spec", color="black")
plt.show()

total_eventEEG_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "engagement_index_spec"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("engagement_index_spec", color="black")
plt.show()
