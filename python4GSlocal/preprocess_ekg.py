import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from google.cloud import storage
import subprocess
import io
import helpers
import importlib
from matplotlib.widgets import Slider

importlib.reload(helpers)

helper = helpers.HELP()

bucket = helper.getBucket()

########### SETTINGS ##################
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
# crews_to_process = ['Crew_02']
file_types = ["abm_leftseat", "abm_rightseat"]
scenarios = ["1", "2", "3", "5", "6", "7"]
number_of_epochs = 250
time_per_epoch_4_analysis = 10  # seconds
plot_qa_figs = 0
#######################################


# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    ax.axis([pos, pos + 10, ax.margins(y=0.1), ax.margins(y=0.1)])
    fig.canvas.draw_idle()

for i_crew in range(len(crews_to_process)):
    helper.reset_folder_storage()

    event_ekgTimeSeries_metrics = pd.DataFrame()
    crew_dir = crews_to_process[i_crew]
    for i_scenario in range(len(scenarios)):
        for i_seat in range(len(file_types)):
            process_dir_name = crew_dir + "/Processing/"
            blob = bucket.blob(
                process_dir_name
                + file_types[i_seat]
                + "_scenario"
                + scenarios[i_scenario]
                + ".csv"
            )

            if blob.exists():
                print(
                    "QA checking ekg: "
                    + process_dir_name
                    + file_types[i_seat]
                    + "_scenario"
                    + scenarios[i_scenario]
                    + ".csv"
                )
                abm_data = helper.read_bucket_table(process_dir_name,  + file_types[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv")

                time_vector = np.array(abm_data.UserTimeStamp[2:])

                this_event_data = helper.read_bucket_table(process_dir_name, "event_vector_scenario.csv")
			    this_event_data = np.array(this_event_data)
			    this_event_data = this_event_data[:, 1:]

                number_of_epochs_this_scenario = np.floor(
                    time_vector[-1] / time_per_epoch_4_analysis
                )
                this_ekgTimeSeries_np = np.zeros(
                    (int(number_of_epochs_this_scenario), 7)
                )
                this_ekgTimeSeries_np[:, 0] = helper.getCrewInt(
                    crews_to_process[i_crew]
                )
                if i_seat == 0:
                    this_ekgTimeSeries_np[:, 1] = 0
                else:
                    this_ekgTimeSeries_np[:, 1] = 1
                this_ekgTimeSeries_np[:, 2] = i_scenario

                peaks, _ = signal.find_peaks(
                    abm_data.ECG, distance=100, prominence=500, width=1
                )

                if plot_qa_figs:
                    fig, ax = plt.subplots()
                    plt.plot(abm_data.UserTimeStamp, abm_data.ECG)
                    plt.plot(abm_data.UserTimeStamp[peaks], abm_data.ekg[peaks], "x")
                    plt.title(
                        "RawData: "
                        + file_types[i_seat]
                        + "_scenario"
                        + scenarios[i_scenario]
                    )
                    slider_color = "White"
                    axis_position = plt.axes(
                        [0.25, 0.15, 0.65, 0.03], facecolor=slider_color
                    )
                    slider_position = Slider(axis_position, "Pos", 0.1, 90.0)
                    slider_position.on_changed(update)
                    plt.show()

                length_this_data = len(abm_data.ECG)

                for this_epoch in range(int(number_of_epochs_this_scenario)):
                    this_epoch_indices_start = (
                        np.floor(length_this_data / number_of_epochs_this_scenario)
                        * this_epoch
                    )
                    this_epoch_indices_end = this_epoch_indices_start + np.floor(
                        length_this_data / number_of_epochs_this_scenario
                    )

                    if (
                        (
                            time_vector[int(this_epoch_indices_start)]
                            > this_event_data[0, i_scenario] - 60
                        )
                        & (
                            time_vector[int(this_epoch_indices_start)]
                            < this_event_data[0, i_scenario] + 60
                        )
                    ) | (
                        (
                            time_vector[int(this_epoch_indices_start)]
                            > this_event_data[1, i_scenario] - 60
                        )
                        & (
                            time_vector[int(this_epoch_indices_start)]
                            < this_event_data[1, i_scenario] + 60
                        )
                    ):
                        this_ekgTimeSeries_np[this_epoch, 3] = 1
                    else:
                        this_ekgTimeSeries_np[this_epoch, 3] = 0

                    this_ekgTimeSeries_np[this_epoch, 4] = this_epoch

                    peaks_to_include = []
                    for i_peak in range(len(peaks)):
                        if (peaks[i_peak] <= this_epoch_indices_end) & (
                            peaks[i_peak] >= this_epoch_indices_start
                        ):
                            peaks_to_include.append(peaks[i_peak])

                    bpm_peaks_this_epoch = 60 / (
                        np.diff(abm_data.UserTimeStamp[peaks_to_include])
                    )

                    this_ekgTimeSeries_np[this_epoch, 5] = np.nanmean(
                        bpm_peaks_this_epoch
                    )
                    this_ekgTimeSeries_np[this_epoch, 6] = np.sqrt(
                        np.nanmean(bpm_peaks_this_epoch**2)
                    )

                this_ekgTimeSeries_df = pd.DataFrame(this_ekgTimeSeries_np)
                this_ekgTimeSeries_df.columns = [
                    "crew",
                    "seat",
                    "scenario",
                    "event_label",
                    "epoch_index",
                    "beats_per_min",
                    "hr_var",
                ]
                event_ekgTimeSeries_metrics = pd.concat(
                    [event_ekgTimeSeries_metrics, this_ekgTimeSeries_df]
                )

    event_ekgTimeSeries_metrics.to_csv(
        "Processing/" + "event_ekgTimeSeries_metrics.csv"
    )

    helper.sync_crew_folder_storage()
