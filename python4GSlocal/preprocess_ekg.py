import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import helpers
import importlib
from matplotlib.widgets import Slider
import warnings
from argparse import ArgumentParser
import sys

# Don't condone this, but get annoying runtime warnings on nan epochs
warnings.filterwarnings("ignore")

importlib.reload(helpers)

helper = helpers.HELP()

###############################################
parser = ArgumentParser(description="Preprocessing Raw Smarteye Data")
parser.add_argument(
    "-c", "--crews", type=str, default=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "13"]
)
parser.add_argument("-s", "--scenarios", type=str, default=["1", "2", "3", "5", "6", "7"])
parser.add_argument("-d", "--directory", type=str)
parser.add_argument("-p1", "--plot_qa_figs", action="store_true", help="Plots QA figs for each participant and scenario. ")
parser.add_argument("--Push2Cloud", type=bool, default=False)
args = parser.parse_args()
crew_arr, scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews, args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory

file_types = ["abm_leftseat", "abm_rightseat"]
time_per_epoch_4_analysis = 10  # seconds
#######################################


# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    ax.axis([pos, pos + 10, ax.margins(y=0.1), ax.margins(y=0.1)])
    fig.canvas.draw_idle()


for i_crew in range(len(crew_arr)):
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"

    pct_usable_matrix = np.zeros((len(scenarios), len(file_types)))
    event_ekgTimeSeries_metrics = pd.DataFrame()
    for i_scenario in range(len(scenarios)):
        for i_seat in range(len(file_types)):
            try:
                abm_data = helper.read_local_table(
                    "Processing/" + file_types[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv"
                )
                abm_data_exists = 1
            except:
                abm_data_exists = 0
                break
            if abm_data_exists:
                length_this_data = len(abm_data.ECG)
                time_vector = np.array(abm_data.UserTimeStamp[2:])

                this_event_data = helper.read_local_table("Processing/event_vector_scenario.csv")
                this_event_data = np.array(this_event_data)

                ekg_quality_df = helper.read_bucket_xlsx(
                    "Analysis/Analysis_ekg_quality_vector.xlsx", helper.getSubWorksheet(file_types[i_seat])
                )
                ekg_quality_vector = ekg_quality_df.to_numpy()
                print(
                    "Processing EKG: " + helper.crew_dir + "Processing/" + file_types[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv"
                )
                good_indices = []
                if isinstance(ekg_quality_vector[0, i_scenario], str) & (pd.notnull(ekg_quality_vector[0, i_scenario])):
                    sections_to_remove = ekg_quality_vector[0, i_scenario].split(";")
                    good_indices = np.linspace(0, length_this_data, length_this_data + 1)
                    for this_section in range(len(sections_to_remove)):
                        percents_to_remove = sections_to_remove[this_section].split(":")
                        percents_to_remove_arr = [int(i) for i in percents_to_remove]
                        indices_to_remove = np.floor((np.array(percents_to_remove_arr) / 100) * length_this_data)
                        good_indices = good_indices[
                            (good_indices <= indices_to_remove[0]) | (good_indices >= indices_to_remove[1])
                        ]
                else:
                    good_indices = np.linspace(0, length_this_data, length_this_data + 1)

                pct_usable_matrix[i_scenario, i_seat] = np.rint((len(good_indices) / length_this_data) * 100)

                number_of_epochs_this_scenario = np.floor(time_vector[-1] / time_per_epoch_4_analysis)
                this_ekgTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 7))
                this_ekgTimeSeries_np[:, 0] = helper.getCrewInt()
                if i_seat == 0:
                    this_ekgTimeSeries_np[:, 1] = 0
                else:
                    this_ekgTimeSeries_np[:, 1] = 1
                this_ekgTimeSeries_np[:, 2] = i_scenario

                if (crew_arr[i_crew] == "Crew_06") & (file_types[i_seat] == "abm_rightseat"):
                    peaks, _ = signal.find_peaks(abm_data.ECG, distance=100, prominence=100, width=[1, 100])
                else:
                    peaks, _ = signal.find_peaks(abm_data.ECG, distance=100, prominence=500, width=1)

                if args.plot_qa_figs:
                    fig, ax = plt.subplots()
                    plt.plot(np.array(abm_data.UserTimeStamp), np.array(abm_data.ECG))
                    plt.plot(np.array(abm_data.UserTimeStamp[peaks]), np.array(abm_data.ECG[peaks]), "x")
                    plt.title("RawData: " + file_types[i_seat] + "_scenario" + scenarios[i_scenario])
                    slider_color = "White"
                    axis_position = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=slider_color)
                    slider_position = Slider(axis_position, "Pos", 0.1, 90.0)
                    slider_position.on_changed(update)
                    plt.show()

                if (scenarios[i_scenario] != 8) | (scenarios[i_scenario] != 9):
                    for this_epoch in range(int(number_of_epochs_this_scenario)):
                        this_epoch_indices_start = np.floor(length_this_data / number_of_epochs_this_scenario) * this_epoch
                        this_epoch_indices_end = this_epoch_indices_start + np.floor(
                            length_this_data / number_of_epochs_this_scenario
                        )

                        if (
                            (time_vector[int(this_epoch_indices_start)] > this_event_data[0, i_scenario] - 60)
                            & (time_vector[int(this_epoch_indices_start)] < this_event_data[0, i_scenario] + 60)
                        ) | (
                            (time_vector[int(this_epoch_indices_start)] > this_event_data[1, i_scenario] - 60)
                            & (time_vector[int(this_epoch_indices_start)] < this_event_data[1, i_scenario] + 60)
                        ):
                            this_ekgTimeSeries_np[this_epoch, 3] = 1
                        else:
                            this_ekgTimeSeries_np[this_epoch, 3] = 0

                        this_ekgTimeSeries_np[this_epoch, 4] = this_epoch

                        peaks_to_include = []
                        for i_peak in range(len(peaks)):
                            if (
                                (peaks[i_peak] <= this_epoch_indices_end)
                                & (peaks[i_peak] >= this_epoch_indices_start)
                                & np.any(peaks[i_peak] == good_indices)
                            ):
                                peaks_to_include.append(peaks[i_peak])

                        bpm_peaks_this_epoch = 60 / (np.diff(abm_data.UserTimeStamp[peaks_to_include]))

                        this_ekgTimeSeries_np[this_epoch, 5] = np.nanmean(bpm_peaks_this_epoch)
                        this_ekgTimeSeries_np[this_epoch, 6] = np.sqrt(np.nanmean(bpm_peaks_this_epoch**2))

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
                event_ekgTimeSeries_metrics = pd.concat([event_ekgTimeSeries_metrics, this_ekgTimeSeries_df])

    pct_usable_matrix_df = pd.DataFrame(pct_usable_matrix)
    pct_usable_matrix_df.to_csv(helper.local_process_dir + helper.crew_dir + "Processing/" + "ekg_pct_usable_matrix.csv")
    event_ekgTimeSeries_metrics.to_csv(
        helper.local_process_dir + helper.crew_dir + "Processing/" + "event_ekgTimeSeries_metrics.csv"
    )

    if args.Push2Cloud:
        helper.sync_crew_folder_storage()
