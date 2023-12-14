import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import helpers
import importlib
from matplotlib.widgets import Slider
import math
import warnings

# Don't condone this, but get annoying runtime warnings on nan epochs
warnings.filterwarnings("ignore")

importlib.reload(helpers)

helper = helpers.HELP()

bucket = helper.getBucket()

########### SETTINGS ##################
crews_to_process = [
    "Crew_02",
    "Crew_03",
    "Crew_04",
    "Crew_05",
]

# crews_to_process = [
#     "Crew_02"
#         "Crew_06",
#     "Crew_07",
#     "Crew_08",
#     "Crew_09",
#     "Crew_10",
#     "Crew_11",
#     "Crew_13"
# ]

file_types = ["abm_leftseat", "abm_rightseat"]
scenarios = ["1", "2", "3", "5", "6", "7"]
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

    pct_usable_matrix = np.zeros((len(scenarios), len(file_types)))
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
                abm_data = helper.read_bucket_table(process_dir_name,  file_types[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv")
                length_this_data = len(abm_data.ECG)
                time_vector = np.array(abm_data.UserTimeStamp[2:])

                this_event_data = helper.read_bucket_table(process_dir_name, "event_vector_scenario.csv")
                this_event_data = np.array(this_event_data)
                this_event_data = this_event_data[:, 1:]

                ekg_quality_df = pd.read_excel(
                    "gs://soteria_study_data/Analysis/"
                    + "ekg_quality_vector.xlsx",
                    helper.getSubWorksheet(
                        crews_to_process[i_crew], file_types[i_seat]
                    ),
                )
                ekg_quality_vector = ekg_quality_df.to_numpy()
                good_indices=[]
                if isinstance(ekg_quality_vector[0,i_scenario],str) & (pd.notnull(ekg_quality_vector[0,i_scenario])):
                    sections_to_remove = ekg_quality_vector[0,i_scenario].split(";")
                    good_indices = np.linspace(0,length_this_data, length_this_data+1)
                    for this_section in range(len(sections_to_remove)):
                        percents_to_remove = sections_to_remove[this_section].split(":")
                        percents_to_remove_arr = [int(i) for i in percents_to_remove]
                        indices_to_remove = np.floor((np.array(percents_to_remove_arr)/100)*length_this_data)
                        good_indices = good_indices[(good_indices<=indices_to_remove[0]) | (good_indices>=indices_to_remove[1])]
                else:
                    good_indices = np.linspace(0,length_this_data, length_this_data+1)
                
                pct_usable_matrix[i_scenario, i_seat] = np.rint(
                        (len(good_indices) / length_this_data) * 100
                    )

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

                if (crews_to_process[i_crew] == "Crew_06") & (file_types[i_seat] == "abm_rightseat"):
                    peaks, _ = signal.find_peaks(
                    abm_data.ECG, distance=100, prominence=100, width=[1,100]
                    )	
                else:
                    peaks, _ = signal.find_peaks(
                        abm_data.ECG, distance=100, prominence=500, width=1
                    )

                if plot_qa_figs:
                    fig, ax = plt.subplots()
                    plt.plot(abm_data.UserTimeStamp, abm_data.ECG)
                    plt.plot(abm_data.UserTimeStamp[peaks], abm_data.ECG[peaks], "x")
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


                if (scenarios[i_scenario] != 8) | (scenarios[i_scenario] !=9):
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
	                        if (peaks[i_peak] <= this_epoch_indices_end) & (peaks[i_peak] >= this_epoch_indices_start) & np.any(peaks[i_peak] == good_indices):
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

    pct_usable_matrix_df = pd.DataFrame(pct_usable_matrix)
    pct_usable_matrix_df.to_csv("Processing/" + "ekg_pct_usable_matrix.csv")
    event_ekgTimeSeries_metrics.to_csv(
        "Processing/" + "event_ekgTimeSeries_metrics.csv"
    )

    helper.sync_crew_folder_storage(crews_to_process[i_crew])
