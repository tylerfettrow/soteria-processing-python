import pandas as pd
import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt
import helpers
import importlib
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
parser.add_argument("-p1", "--plot_raw", action="store_true", help="Plots raw EEG data for every pilot and scenario. '-' and '+' keys zoom plot in and out. 'left' and 'right' arrows scroll plot. ")
parser.add_argument("-p2", "--plot_bandpower", action="store_true", help="Plots EEG power for every pilot and scenario.")
parser.add_argument("--Push2Cloud", type=bool, default=False)
args = parser.parse_args()
crew_arr, scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews, args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory

file_types = ["abm_leftseat", "abm_rightseat"]
scenarios = args.scenarios
number_of_epochs = 1000
#######################################

for i_crew in range(len(crew_arr)):
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"
    event_eegTimeSeries_metrics = pd.DataFrame()
    eeg_freq_storage = np.zeros((5, 9, len(scenarios)))
    eeg_freq_band_storage = np.zeros((5, 9, len(scenarios), number_of_epochs))
    eeg_freqSpec_band_storage = np.zeros((6, 9, 2000, len(scenarios)))
    eeg_freqSpec_band_storage[:] = np.nan
    eeg_timesec_epoch_storage = np.zeros((len(scenarios), number_of_epochs))
    pct_usable_matrix = np.zeros((len(scenarios), len(file_types)))
    process_dir_name = helper.crew_dir + "/Processing/"

    this_event_data = helper.read_local_table("Processing/event_vector_scenario.csv")
    this_event_data = np.array(this_event_data)
    abm_data_exists = 0

    for i_seat in range(len(file_types)):
        for i_scenario in range(len(scenarios)):
            try:
                abm_data = helper.read_local_table(
                    "Processing/" + file_types[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv"
                )
                abm_data_exists = 1
            except:
                abm_data_exists = 0
                break
            if abm_data_exists:
                electrode_vector_df = helper.read_bucket_xlsx(
                    "Analysis/Analysis_eeg_electrode_quality_vector.xlsx", helper.getSubWorksheet(file_types[i_seat])
                )
                electrode_vector = electrode_vector_df.to_numpy()

                time_end = abm_data.UserTimeStamp[abm_data.UserTimeStamp.shape[0] - 1]

                print(" ")
                print(" ")
                print(
                    "Processing EEG: " + process_dir_name + file_types[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv"
                )

                ch_names = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "POz", "P4"]
                sfreq = 256
                info = mne.create_info(
                    ch_names, sfreq, ch_types="eeg"
                )  # WARNING: for some reason 'eeg' multiplies amplitude by 1mill?? not sure why
                # data = {'UserTimeStamp':abm_data.UserTimeStamp, 'POz':abm_data.POz , 'Fz':abm_data.Fz, 'Cz':abm_data.Cz, 'C3':abm_data.C3, 'C4':abm_data.C4, 'F3':abm_data.F3, 'F4':abm_data.F4, 'P3':abm_data.P3, 'P4':abm_data.P4}
                data = {
                    "F3": abm_data.F3,
                    "Fz": abm_data.Fz,
                    "F4": abm_data.F4,
                    "C3": abm_data.C3,
                    "Cz": abm_data.Cz,
                    "C4": abm_data.C4,
                    "P3": abm_data.P3,
                    "POz": abm_data.POz,
                    "P4": abm_data.P4,
                }
                eeg = pd.DataFrame(
                    data=data
                )  # , abm_data.Fz, abm_data.Cz, abm_data.C3, abm_data.C4, abm_data.F3, abm_data.F4, abm_data.P3, abm_data.P4)
                eeg_trans = eeg.T / 1000000  # adjusting for eeg multiplier
                raw = mne.io.RawArray(eeg_trans, info)

                # https://neuraldatascience.io/7-eeg/erp_filtering.html
                low_cut = 0.1
                hi_cut = 50
                raw_data = raw.copy()
                raw_filt = raw.copy().filter(low_cut, hi_cut)
                raw_filt
                # raw_filt.plot_psd(fmax=100)
                if args.plot_raw:
                    raw_filt.plot(duration=100, block=True)

                this_data = raw_filt.get_data()
                this_data = this_data + 0.0005
                length_this_data = this_data.shape[1]

                waveletname = "morl"
                cmap = "seismic"
                dt = 1 / sfreq
                this_data_dwt = this_data.T
                N = this_data_dwt.shape[0]
                time_vector = np.linspace(0, time_end, N)

                f, t, Sxx = signal.spectrogram(this_data_dwt[:, :].T, 256)

                fmin = 1  # Hz
                fmax = 45  # Hz
                freq_slice = np.where((f >= fmin) & (f <= fmax))

                # keep only frequencies of interest
                f = f[freq_slice]
                Sxx = np.squeeze(Sxx[:, freq_slice, :]) * 10000000000000

                eeg_freqSpec_band_storage[0, :, 0 : Sxx.shape[2], i_scenario] = t
                eeg_freqSpec_band_storage[1, :, 0 : Sxx.shape[2], i_scenario] = Sxx[:, 0:3, :].mean(1)  # delta
                eeg_freqSpec_band_storage[2, :, 0 : Sxx.shape[2], i_scenario] = Sxx[:, 4:7, :].mean(1)  # theta
                eeg_freqSpec_band_storage[3, :, 0 : Sxx.shape[2], i_scenario] = Sxx[:, 8:12, :].mean(1)  # alpha
                eeg_freqSpec_band_storage[4, :, 0 : Sxx.shape[2], i_scenario] = Sxx[:, 13:29, :].mean(1)  # beta
                eeg_freqSpec_band_storage[5, :, 0 : Sxx.shape[2], i_scenario] = Sxx[:, 30:44, :].mean(1)  # gamma

                for this_epoch in range(eeg_freqSpec_band_storage.shape[2]):
                    if np.all(np.isnan(eeg_freqSpec_band_storage[:, :, this_epoch, i_scenario])):
                        number_of_time_epochs_spec = this_epoch - 1
                        break
                eeg_timesec_epoch_storage = eeg_freqSpec_band_storage[0, 0, 0:number_of_time_epochs_spec, i_scenario]
                x_axis_vector_spec = np.linspace(0, 100, number_of_time_epochs_spec)
                this_epoch_bandpower_ratio_spec = np.zeros((4, 9, number_of_time_epochs_spec))
                engagement_index_spec = np.zeros((len(scenarios), number_of_time_epochs_spec))
                filtered_engagement_index_spec = np.zeros((len(scenarios), number_of_time_epochs_spec))
                filtered_taskLoad_index_spec = np.zeros((len(scenarios), number_of_time_epochs_spec))
                taskLoad_index_spec = np.zeros((len(scenarios), number_of_time_epochs_spec))

                print("Processing: " + crew_arr[i_crew] + " scenario " + scenarios[i_scenario])
                for idx in range(0, eeg_timesec_epoch_storage.shape[0]):
                    if this_event_data[0, i_scenario] <= eeg_timesec_epoch_storage[idx]:
                        this_event1_epoch = np.floor((idx - 1) / (number_of_time_epochs_spec / 100))
                        break
                for idx in range(0, eeg_timesec_epoch_storage.shape[0]):
                    # print(idx)
                    if this_event_data[1, i_scenario] <= eeg_timesec_epoch_storage[idx]:
                        this_event2_epoch = np.floor((idx - 1) / (number_of_time_epochs_spec / 100))
                        break

                for this_epoch in range(this_epoch_bandpower_ratio_spec.shape[2]):
                    ### WARNING: removing time and delta band power here
                    this_epoch_bandpower_ratio_spec[:, :, this_epoch] = eeg_freqSpec_band_storage[
                        2:, :, this_epoch, i_scenario
                    ] / eeg_freqSpec_band_storage[2:, :, this_epoch, i_scenario].sum(0)
                    theta = 0
                    alpha = 0
                    beta = 0
                    include_electrode_vector = np.asarray(np.where(electrode_vector[i_scenario, :] == 2)) - 1

                    pct_usable_matrix[i_scenario, i_seat] = np.rint((include_electrode_vector.size / 9) * 100)

                    theta = eeg_freqSpec_band_storage[2, include_electrode_vector, this_epoch, i_scenario].sum()  # theta
                    alpha = eeg_freqSpec_band_storage[3, include_electrode_vector, this_epoch, i_scenario].sum()  # alpha
                    beta = eeg_freqSpec_band_storage[4, include_electrode_vector, this_epoch, i_scenario].sum()  # beta
                    total_power = theta + alpha + beta
                    # Percent power; power relative to total power
                    per_power_theta = theta / total_power
                    per_power_alpha = alpha / total_power
                    per_power_beta = beta / total_power
                    engagement_index_spec[i_scenario, this_epoch] = per_power_beta / (per_power_alpha + per_power_theta)
                    # task load index
                    # if (electrode_vector[i_scenario,1]==2) & (electrode_vector[i_scenario,7]==2):
                    # Fz_theta = eeg_freq_band_storage[1,1,i_scenario,this_epoch]
                    # Pz_alpha = eeg_freq_band_storage[2,7,i_scenario,this_epoch]
                    zero_good = np.where(include_electrode_vector == 0)
                    one_good = np.where(include_electrode_vector == 1)
                    two_good = np.where(include_electrode_vector == 2)
                    # [zero_good[1], one_good[1]]
                    # np.squeeze([zero_good[1],nan,two_good[1]]
                    theta_indices = []
                    theta_indices = np.append(theta_indices, zero_good[1])
                    theta_indices = np.append(theta_indices, one_good[1])
                    theta_indices = np.append(theta_indices, two_good[1])
                    theta_indices = theta_indices.astype(int)
                    six_good = np.where(include_electrode_vector == 6)
                    sev_good = np.where(include_electrode_vector == 7)
                    eit_good = np.where(include_electrode_vector == 8)
                    alpha_indices = []
                    alpha_indices = np.append(alpha_indices, six_good[1])
                    alpha_indices = np.append(alpha_indices, sev_good[1])
                    alpha_indices = np.append(alpha_indices, eit_good[1])
                    alpha_indices = alpha_indices.astype(int)
                    Fz_theta = np.nanmean(
                        eeg_freqSpec_band_storage[
                            2,
                            include_electrode_vector[0][theta_indices],
                            this_epoch,
                            i_scenario,
                        ]
                    )
                    Pz_alpha = np.nanmean(
                        eeg_freqSpec_band_storage[
                            3,
                            include_electrode_vector[0][alpha_indices],
                            this_epoch,
                            i_scenario,
                        ]
                    )
                    taskLoad_index_spec[i_scenario, this_epoch] = Fz_theta / Pz_alpha
                    # else:
                    # 	taskLoad_index_spec[i_scenario,this_epoch] = 'nan'
                    ####################################################################

                this_eegTimeSeries_np = np.zeros((int(number_of_time_epochs_spec), 7))
                this_eegTimeSeries_np[:, 0] = helper.getCrewInt()

                if i_seat == 0:
                    this_eegTimeSeries_np[:, 1] = 0
                    this_eegTimeSeries_np[:, 1] = 0
                else:
                    this_eegTimeSeries_np[:, 1] = 1
                    this_eegTimeSeries_np[:, 1] = 1
                this_eegTimeSeries_np[:, 2] = i_scenario
                for this_epoch in range(int(number_of_time_epochs_spec)):
                    if (
                        (eeg_timesec_epoch_storage[this_epoch] > this_event_data[0, i_scenario] - 60)
                        & (eeg_timesec_epoch_storage[this_epoch] < this_event_data[0, i_scenario] + 60)
                    ) | (
                        (eeg_timesec_epoch_storage[this_epoch] > this_event_data[1, i_scenario] - 60)
                        & (eeg_timesec_epoch_storage[this_epoch] < this_event_data[1, i_scenario] + 60)
                    ):
                        this_eegTimeSeries_np[this_epoch, 3] = 1
                    else:
                        this_eegTimeSeries_np[this_epoch, 3] = 0
                    this_eegTimeSeries_np[this_epoch, 4] = this_epoch

                    this_eegTimeSeries_np[this_epoch, 5] = taskLoad_index_spec[i_scenario, this_epoch]
                    this_eegTimeSeries_np[this_epoch, 6] = engagement_index_spec[i_scenario, this_epoch]

                this_eegTimeSeries_df = pd.DataFrame(this_eegTimeSeries_np)
                this_eegTimeSeries_df.columns = [
                    "crew",
                    "seat",
                    "scenario",
                    "event_label",
                    "epoch_index",
                    "taskLoad_index_spec",
                    "engagement_index_spec",
                ]
                event_eegTimeSeries_metrics = pd.concat([event_eegTimeSeries_metrics, this_eegTimeSeries_df])

                if args.plot_bandpower:
                    this_epoch_bandpower_ratio_spec[np.isnan(this_epoch_bandpower_ratio_spec)] = 0

                    # filtering the bandpower sig
                    # b, a = signal.ellip(4, 0.01, 120, 0.125)
                    b, a = signal.butter(8, 0.125)
                    # filtered_this_epoch_bandpower_ratio_spec = this_epoch_bandpower_ratio_spec
                    filtered_this_epoch_bandpower_ratio_spec = signal.filtfilt(
                        b, a, this_epoch_bandpower_ratio_spec, method="gust"
                    )

                    fig, axs = plt.subplots(3, 3)
                    fig.suptitle(crew_arr[i_crew] + file_types[i_seat] + scenarios[i_scenario])

                    std_bandpower = filtered_this_epoch_bandpower_ratio_spec.std(axis=0)

                    axs[0, 0].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 0, :].T * 100,
                        linewidth=1,
                    )
                    # axs[0, 0].plot(x_axis_vector_spec, std_bandpower[0,:].T * 100 , linewidth=2)
                    axs[0, 0].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[0, 0].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[0, 0].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 1] == 2:
                        axs[0, 0].set_title("F3", color="black")
                    elif electrode_vector[i_scenario, 1] == 1:
                        axs[0, 0].set_title("F3", color="blue")
                    elif electrode_vector[i_scenario, 1] == 0:
                        axs[0, 0].set_title("F3", color="red")

                    axs[0, 1].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[0, 1].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[0, 1].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 1, :].T * 100,
                        linewidth=1,
                    )
                    # axs[0, 1].plot(x_axis_vector_spec, std_bandpower[1,:].T * 100 , linewidth=2)
                    axs[0, 1].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 1] == 2:
                        axs[0, 1].set_title("Fz", color="black")
                    elif electrode_vector[i_scenario, 1] == 1:
                        axs[0, 1].set_title("Fz", color="blue")
                    elif electrode_vector[i_scenario, 1] == 0:
                        axs[0, 1].set_title("Fz", color="red")

                    axs[0, 2].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[0, 2].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[0, 2].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 2, :].T * 100,
                        linewidth=1,
                    )
                    # axs[0, 2].plot(x_axis_vector_spec, std_bandpower[2,:].T * 100, linewidth=2 )
                    axs[0, 2].set_ylim([0, 100])

                    if electrode_vector[i_scenario, 2] == 2:
                        axs[0, 2].set_title("F4", color="black")
                    elif electrode_vector[i_scenario, 2] == 1:
                        axs[0, 2].set_title("F4", color="blue")
                    elif electrode_vector[i_scenario, 2] == 0:
                        axs[0, 2].set_title("F4", color="red")

                    axs[1, 0].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[1, 0].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[1, 0].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 3, :].T * 100,
                        linewidth=1,
                    )
                    # axs[1, 0].plot(x_axis_vector_spec, std_bandpower[3,:].T * 100, linewidth=2)
                    axs[1, 0].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 3] == 2:
                        axs[1, 0].set_title("C3", color="black")
                    elif electrode_vector[i_scenario, 3] == 1:
                        axs[1, 0].set_title("C3", color="blue")
                    elif electrode_vector[i_scenario, 3] == 0:
                        axs[1, 0].set_title("C3", color="red")

                    axs[1, 1].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[1, 1].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[1, 1].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 4, :].T * 100,
                        linewidth=1,
                    )
                    # axs[1, 1].plot(x_axis_vector_spec, std_bandpower[4,:].T * 100, linewidth=2 )
                    axs[1, 1].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 4] == 2:
                        axs[1, 1].set_title("Cz", color="black")
                    elif electrode_vector[i_scenario, 4] == 1:
                        axs[1, 1].set_title("Cz", color="blue")
                    elif electrode_vector[i_scenario, 4] == 0:
                        axs[1, 1].set_title("Cz", color="red")

                    axs[1, 2].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[1, 2].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[1, 2].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 5, :].T * 100,
                        linewidth=1,
                    )
                    # axs[1, 2].plot(x_axis_vector_spec, std_bandpower[5,:].T * 100, linewidth=2 )
                    axs[1, 2].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 5] == 2:
                        axs[1, 2].set_title("C4", color="black")
                    elif electrode_vector[i_scenario, 5] == 1:
                        axs[1, 2].set_title("C4", color="blue")
                    elif electrode_vector[i_scenario, 5] == 0:
                        axs[1, 2].set_title("C4", color="red")

                    axs[2, 0].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[2, 0].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[2, 0].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 6, :].T * 100,
                        linewidth=1,
                    )
                    # axs[2, 0].plot(x_axis_vector_spec, std_bandpower[6,:].T * 100 , linewidth=2)
                    axs[2, 0].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 6] == 2:
                        axs[2, 0].set_title("P3", color="black")
                    elif electrode_vector[i_scenario, 6] == 1:
                        axs[2, 0].set_title("P3", color="blue")
                    elif electrode_vector[i_scenario, 6] == 0:
                        axs[2, 0].set_title("P3", color="red")

                    axs[2, 1].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[2, 1].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[2, 1].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 7, :].T * 100,
                        linewidth=1,
                    )
                    # axs[2, 1].plot(x_axis_vector_spec, std_bandpower[7,:].T * 100 , linewidth=2)
                    axs[2, 1].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 7] == 2:
                        axs[2, 1].set_title("POz", color="black")
                    elif electrode_vector[i_scenario, 7] == 1:
                        axs[2, 1].set_title("POz", color="blue")
                    elif electrode_vector[i_scenario, 7] == 0:
                        axs[2, 1].set_title("POz", color="red")

                    axs[2, 2].axvline(
                        x=this_event1_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[2, 2].axvline(
                        x=this_event2_epoch,
                        ymin=0,
                        ymax=100,
                        color="gray",
                        linestyle="--",
                        label="_Hidden label",
                    )
                    axs[2, 2].plot(
                        x_axis_vector_spec,
                        filtered_this_epoch_bandpower_ratio_spec[:, 8, :].T * 100,
                        linewidth=1,
                    )
                    # axs[2, 2].plot(x_axis_vector_spec, std_bandpower[8,:].T * 100, linewidth=2)
                    axs[2, 2].set_ylim([0, 100])
                    if electrode_vector[i_scenario, 8] == 2:
                        axs[2, 2].set_title("P4", color="black")
                    elif electrode_vector[i_scenario, 8] == 1:
                        axs[2, 2].set_title("P4", color="blue")
                    elif electrode_vector[i_scenario, 8] == 0:
                        axs[2, 2].set_title("P4", color="red")

                    for ax in axs.flat:
                        ax.set(
                            xlabel="percent of trial",
                            ylabel="percent of frequency power",
                        )

                    # Hide x labels and tick labels for top plots and y ticks for right plots.
                    for ax in axs.flat:
                        ax.label_outer()

                    leg = axs[0, 2].legend(
                        [
                            "event1",
                            "event2",
                            "theta",
                            "alpha",
                            "beta",
                            "gamma",
                            "std",
                        ]
                    )

                    for line in leg.get_lines():
                        line.set_linewidth(4.0)

                    fig.set_size_inches((22, 11))
                    fig.savefig(
                        helper.local_process_dir + helper.crew_dir + 
                        "Figures/"
                        + "eeg_powerbandsSpec_"
                        + crew_arr[i_crew]
                        + "_"
                        + file_types[i_seat]
                        + "_scenario"
                        + scenarios[i_scenario]
                        + ".tif",
                        bbox_inches="tight",
                        pad_inches=0,
                    )

    pct_usable_matrix_df = pd.DataFrame(pct_usable_matrix)
    pct_usable_matrix_df.to_csv(helper.local_process_dir + helper.crew_dir + "Processing/" + "eeg_pct_usable_matrix.csv")
    event_eegTimeSeries_metrics.to_csv(
        helper.local_process_dir + helper.crew_dir + "Processing/" + "event_eegTimeSeries_metrics.csv"
    )

    if args.Push2Cloud:
        helper.sync_crew_folder_storage()
