import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
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

total_eventEKG_metric_dataframe = pd.DataFrame()
for i_crew in range(len(crews_to_process)):
    this_subject_event_metric_dataframe = pd.DataFrame()
    crew_dir = crews_to_process[i_crew]
    process_dir_name = crew_dir + "/Processing/"
    print("grabbing results from " + crew_dir)

    event_ekgTimeSeries_metrics = helper.read_bucket_table(
        process_dir_name, "event_ekgTimeSeries_metrics.csv"
    )
    event_ekgTimeSeries_df = event_ekgTimeSeries_metrics[
        event_ekgTimeSeries_metrics.columns[1:]
    ]

    total_eventEKG_metric_dataframe = pd.concat(
        [total_eventEKG_metric_dataframe, event_ekgTimeSeries_df]
    )

helper.reset_folder_storage()

total_eventEKG_metric_dataframe.to_csv(
    "Analysis/" + "total_eventEKG_metric_dataframe.csv"
)
helper.sync_crew_folder_storage()

colors = sns.color_palette("tab10", 2)
colors_array = np.array(colors)

total_eventEKG_metric_dataframe.seat[
    total_eventEKG_metric_dataframe.seat == 1
] = "right"
total_eventEKG_metric_dataframe.seat[total_eventEKG_metric_dataframe.seat == 0] = "left"
total_eventEKG_metric_dataframe.event_label[
    total_eventEKG_metric_dataframe.event_label == 0
] = "No Event"
total_eventEKG_metric_dataframe.event_label[
    total_eventEKG_metric_dataframe.event_label == 1
] = "Event"
total_eventEKG_metric_dataframe.crew = total_eventEKG_metric_dataframe.crew.astype(int)

total_eventEKG_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "beats_per_min"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("beats_per_min", color="black")
plt.show()

total_eventEKG_metric_dataframe.groupby(["crew", "seat", "event_label"])[
    "hr_var"
].mean().unstack(["seat", "event_label"]).plot.bar(
    color=[
        colors_array[0],
        colors_array[0] * 0.9,
        colors_array[1],
        colors_array[1] * 0.9,
    ],
    width=0.9,
)
plt.title("hr_var", color="black")
plt.show()
