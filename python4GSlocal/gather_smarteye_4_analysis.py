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

total_eventSmarteye_metric_dataframe = pd.DataFrame()
for i_crew in range(len(crews_to_process)):
    this_subject_event_metric_dataframe = pd.DataFrame()
    crew_dir = crews_to_process[i_crew]
    process_dir_name = crew_dir + "/Processing/"
    print("grabbing results from " + crew_dir)

    event_smarteyeGazeTimeSeries_metrics = helper.read_bucket_table(
        process_dir_name, "event_smarteyeGazeTimeSeries_metrics.csv"
    )
    event_smarteyeGazeTimeSeries_df = event_smarteyeGazeTimeSeries_metrics[
        event_smarteyeGazeTimeSeries_metrics.columns[1:]
    ]

    event_smarteyeTimeSeries_metrics = helper.read_bucket_table(
        process_dir_name, "event_smarteyeTimeSeries_metrics.csv"
    )
    event_smarteyeTimeSeries_df = event_smarteyeTimeSeries_metrics[
        event_smarteyeTimeSeries_metrics.columns[6:]
    ]

    this_subject_event_metric_dataframe = event_smarteyeGazeTimeSeries_df.join(
        event_smarteyeTimeSeries_df
    )

    total_eventSmarteye_metric_dataframe = pd.concat(
        [total_eventSmarteye_metric_dataframe, this_subject_event_metric_dataframe]
    )

helper.reset_folder_storage()

total_eventSmarteye_metric_dataframe.to_csv(
    "Analysis/" + "total_eventSmarteye_metric_dataframe.csv"
)

helper.sync_crew_folder_storage()

colors = sns.color_palette("tab10", 2)
colors_array = np.array(colors)

total_eventSmarteye_metric_dataframe.seat[
    total_eventSmarteye_metric_dataframe.seat == 1
] = "right"
total_eventSmarteye_metric_dataframe.seat[
    total_eventSmarteye_metric_dataframe.seat == 0
] = "left"
total_eventSmarteye_metric_dataframe.event_label[
    total_eventSmarteye_metric_dataframe.event_label == 0
] = "No Event"
total_eventSmarteye_metric_dataframe.event_label[
    total_eventSmarteye_metric_dataframe.event_label == 1
] = "Event"
total_eventSmarteye_metric_dataframe.crew = (
    total_eventSmarteye_metric_dataframe.crew.astype(int)
)

total_eventSmarteye_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "gaze_variance"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("gaze_variance", color="black")
plt.show()

total_eventSmarteye_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "gaze_vel_avg"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("gaze_vel_avg", color="black")
plt.show()

total_eventSmarteye_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "gaze_vel_std"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("gaze_vel_std", color="black")
plt.show()

total_eventSmarteye_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "headHeading_avg"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("headHeading_avg", color="black")
plt.show()

total_eventSmarteye_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "headHeading_std"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("headHeading_std", color="black")
plt.show()

total_eventSmarteye_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "pupilD_avg"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("pupilD_avg", color="black")
plt.show()

total_eventSmarteye_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "pupilD_std"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("pupilD_std", color="black")
plt.show()
