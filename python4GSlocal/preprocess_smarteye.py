import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import math
import matplotlib.colors as colors
from collections import Counter
import helpers
import importlib
import warnings
from argparse import ArgumentParser
import sys

# Don't condone this, but get annoying runtime warnings on nan epochs
warnings.filterwarnings("ignore")

importlib.reload(helpers)

helper = helpers.HELP()

sns.set_theme()


###############################################
parser = ArgumentParser(description="Preprocessing Raw Smarteye Data")
parser.add_argument(
    "-c", "--crews", type=str, default=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "13"]
)
parser.add_argument("-s", "--scenarios", type=str, default=["1", "2", "3", "5", "6", "7"])
parser.add_argument("-d", "--directory", type=str)
parser.add_argument("-p1", "--plot_qatable", action='store_true',help="Plots percent data available for every pilot and scenario.")
parser.add_argument("-p2", "--plot_aoi",action='store_true',help="Plots aoi clustered heatmap every pilot and scenario.")
parser.add_argument("--Push2Cloud", type=bool, default=False)
args = parser.parse_args()
crew_arr, scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews, args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory
file_types = ["smarteye_leftseat", "smarteye_rightseat"]
time_per_epoch_4_analysis = 2  # seconds per epoch
###############################################

for i_crew in range(len(crew_arr)):
    pct_usable_matrix = np.zeros((len(scenarios), len(file_types)))
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"

    event_smarteyeGazeTimeSeries_metrics = pd.DataFrame()
    event_smarteyeTimeSeries_metrics = pd.DataFrame()

    # breakpoint()
    this_event_data = helper.read_local_table("Processing/event_vector_scenario.csv")
    this_event_data = np.array(this_event_data)

    for i_scenario in range(len(scenarios)):
        if (helper.getCrewInt() == 13) & (scenarios[i_scenario] == "5"):
            pct_usable_matrix[i_scenario, :] = 0
        else:
            for i_seat in range(len(file_types)):
                smarteye_data = helper.read_local_table(
                    "Processing/" + file_types[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv"
                )

                print(
                    "Processing Crew: "
                    + crew_arr[i_crew]
                    + " Scenario: "
                    + scenarios[i_scenario]
                    + " Seat: "
                    + file_types[i_seat]
                )
                time_vector = np.array(smarteye_data.UserTimeStamp[2:])
                HeadRotationQ = np.array(smarteye_data.HeadRotationQ[2:])
                PupilDiameterQ = np.array(smarteye_data.PupilDiameterQ[2:])

                direction_gaze = np.array(
                    [
                        smarteye_data.GazeDirectionX[2:],
                        smarteye_data.GazeDirectionY[2:],
                        smarteye_data.GazeDirectionZ[2:],
                    ]
                )
                HeadPosition = np.array(
                    [
                        smarteye_data.HeadPosX[2:],
                        smarteye_data.HeadPosY[2:],
                        smarteye_data.HeadPosZ[2:],
                    ]
                )
                magnitude = np.divide(
                    np.sqrt(
                        np.power(direction_gaze[0, :], 2)
                        + np.power(direction_gaze[1, :], 2)
                        + np.power(direction_gaze[2, :], 2)
                    ),
                    1,
                )
                quality_gaze = smarteye_data.GazeDirectionQ[2:]

                # for the indices that are sequential (i.e. no gaps in good_indices),
                # calculate rate of gaze movement, and remove values that are greater than 700°/s ( Fuchs, A. F. (1967-08-01).
                # "Saccadic and smooth pursuit eye movements in the monkey". The Journal of Physiology.
                degree_per_sec_vector = helper.angle_diff(time_vector, direction_gaze)

                good_indices = np.squeeze(
                    np.where(
                        (magnitude != 0)
                        & (quality_gaze * 100 >= 6)
                        & (degree_per_sec_vector <= 700)
                        & (degree_per_sec_vector != np.nan)
                    )
                )

                length_this_trial = smarteye_data.shape[0]
                pct_usable_matrix[i_scenario, i_seat] = np.rint((len(good_indices) / length_this_trial) * 100)

                projected_headPos_coords = helper.sphere_stereograph(np.squeeze(HeadPosition))
                projected_planar_coords = helper.sphere_stereograph(np.squeeze(direction_gaze))
                good_project_planar_coords = projected_planar_coords[:, good_indices]
                good_project_headPos_coords = projected_headPos_coords[:, good_indices]

                good_project_planar_coords = (
                    good_project_planar_coords - good_project_headPos_coords
                )  # correct origin of gaze vector for head position
                good_project_planar_coords[0, :] = good_project_planar_coords[0, :] * -1  # flip the x axis of gaze vector

                good_ObjectIntersection = smarteye_data.ObjectIntersectionName[good_indices]

                total_good_vel_vals = np.squeeze(degree_per_sec_vector[good_indices])
                # exclude the first since that was set to 0 on purpose
                total_average_gaze_velocity = np.average(total_good_vel_vals[1:])
                total_std_gaze_velocity = np.std(total_good_vel_vals[1:])

                total_m = helper.mean(projected_planar_coords[:, good_indices], 2)
                total_gaze_variance = helper.variance(projected_planar_coords[:, good_indices], 2, total_m)

                headheading_good_indices = np.squeeze(np.where(HeadRotationQ > 0.6))
                pupilD_good_indices = np.squeeze(np.where(PupilDiameterQ > 0.4))

                headheading = np.array(smarteye_data.HeadHeading[2:])

                headheadingDeg = headheading * 180 / math.pi
                time_diff = np.diff(time_vector)

                pupilD = np.array(smarteye_data.PupilDiameter[2:])

                headheadingDeg_rate = np.zeros(headheadingDeg.shape[0])
                headheadingDeg_diff = np.diff(headheadingDeg)
                for this_frame in range(headheadingDeg.shape[0] - 1):
                    if time_diff[this_frame] != 0.0:
                        headheadingDeg_rate[this_frame + 1] = headheadingDeg_diff[this_frame] / time_diff[this_frame]
                    else:
                        headheadingDeg_rate[this_frame + 1] = 0

                number_of_epochs_this_scenario = np.floor(time_vector[-1] / time_per_epoch_4_analysis)

                #### this_smarteyeGazeTimeSeries_np create dataframe that contains
                # 1) crewID
                # 2) seat
                # 3) scenario
                # 4) within 60 seconds of stressor (1) or not (0)
                # 5) epoch index
                # 6) headHeading_avg
                # 7) headHeading_std
                # 8) pupilD_avg
                # 9) pupilD_std
                # 10) gaze_variance
                # 11) gaze_vel_avg
                # 12) gaze_vel_std
                # 13) AOI
                
                this_smarteyeGazeTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 13))
                this_smarteyeGazeTimeSeries_np[:, 0] = helper.getCrewInt()
                if i_seat == 0:
                    this_smarteyeGazeTimeSeries_np[:, 1] = 0
                else:
                    this_smarteyeGazeTimeSeries_np[:, 1] = 1

                this_smarteyeGazeTimeSeries_np[:, 2] = i_scenario
                length_this_data = smarteye_data.shape[0]
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
                        this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 1
                    else:
                        this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 0

                    this_smarteyeGazeTimeSeries_np[this_epoch, 4] = this_epoch
                    this_good_indices = np.squeeze(
                        np.where((good_indices > this_epoch_indices_start) & (good_indices < this_epoch_indices_end))
                    )
                    if this_good_indices.size > 1:
                        this_smarteyeGazeTimeSeries_np[this_epoch, 9] = helper.variance(
                            good_project_planar_coords[:, this_good_indices],
                            2,
                            helper.mean(good_project_planar_coords[:, this_good_indices], 1),
                        )
                        this_smarteyeGazeTimeSeries_np[this_epoch, 10] = np.nanmean(
                            np.squeeze(degree_per_sec_vector[this_good_indices]).T
                        )
                        this_smarteyeGazeTimeSeries_np[this_epoch, 11] = np.nanstd(
                            np.squeeze(degree_per_sec_vector[this_good_indices]).T
                        )

                        unique_ObjectIntersection = helper.get_unique(good_ObjectIntersection)
                        # for i in range(len(unique_ObjectIntersection)):
                        #     indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
                        x = Counter(good_ObjectIntersection[good_ObjectIntersection.index[this_good_indices]])

                        this_smarteyeGazeTimeSeries_np[this_epoch, 12] = x["MCP"]
                    else:
                        this_smarteyeGazeTimeSeries_np[this_epoch, 9] = np.nan
                        this_smarteyeGazeTimeSeries_np[this_epoch, 10] = np.nan
                        this_smarteyeGazeTimeSeries_np[this_epoch, 11] = np.nan

                    this_smarteyeGazeTimeSeries_np[this_epoch, 4] = this_epoch
                    this_smarteyeGazeTimeSeries_np[this_epoch, 5] = np.nanmean(
                        headheadingDeg_rate[int(this_epoch_indices_start) : int(this_epoch_indices_end)].T
                    )
                    this_smarteyeGazeTimeSeries_np[this_epoch, 6] = np.nanstd(
                        headheadingDeg_rate[int(this_epoch_indices_start) : int(this_epoch_indices_end)].T
                    )
                    this_smarteyeGazeTimeSeries_np[this_epoch, 7] = np.nanmean(
                        pupilD[int(this_epoch_indices_start) : int(this_epoch_indices_end)].T
                    )
                    this_smarteyeGazeTimeSeries_np[this_epoch, 8] = np.nanstd(
                        pupilD[int(this_epoch_indices_start) : int(this_epoch_indices_end)].T
                    )

                this_smarteyeGazeTimeSeries_df = pd.DataFrame(this_smarteyeGazeTimeSeries_np)
                this_smarteyeGazeTimeSeries_df.columns = [
                    "crew",
                    "seat",
                    "scenario",
                    "event_label",
                    "epoch_index",
                    "headHeading_avg",
                    "headHeading_std",
                    "pupilD_avg",
                    "pupilD_std",
                    "gaze_variance",
                    "gaze_vel_avg",
                    "gaze_vel_std",
                    "AOI",
                ]
                event_smarteyeGazeTimeSeries_metrics = pd.concat(
                    [
                        event_smarteyeGazeTimeSeries_metrics,
                        this_smarteyeGazeTimeSeries_df,
                    ]
                )

            
                # assign colors to each possible object (i.e. make a map for each object)
                # for each object, determine what indices are labeled with it, then plot those indices with the mapped color

                # for i_obj in range(len(unique_ObjectIntersection)):
                # 	"Inst Panel" in good_ObjectIntersection

                if args.plot_aoi:
                    NUM_COLORS = len(unique_ObjectIntersection)
                    cm = plt.get_cmap("gist_rainbow")
                    cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)

                    fig1 = plt.figure(1)
                    ax = plt.gca()
                    # color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

                    if i_seat == 0:
                        for i in range(NUM_COLORS):
                            indices_this_object = np.squeeze(
                                np.where(good_ObjectIntersection == unique_ObjectIntersection[i])
                            )
                            col = cm(1.0 * i / NUM_COLORS)
                            ax.scatter(
                                good_project_planar_coords[0, indices_this_object],
                                good_project_planar_coords[1, indices_this_object],
                                color=col,
                                alpha=0.5,
                                marker=".",
                            )
                            ax.legend(unique_ObjectIntersection)
                            ax.set_title("Gaze vector left seat / Capt")

                        fig1.set_size_inches((22, 11))
                        ax = plt.gca()
                        ax.get_xaxis().set_visible(False)
                        ax.axis("off")
                        plt.savefig(helper.local_process_dir + helper.crew_dir + "Figures/smarteyeGaze_" + scenarios[i_scenario] + "_leftseat.jpg")
                        plt.close()
                    else:
                        for i in range(NUM_COLORS):
                            indices_this_object = np.squeeze(
                                np.where(good_ObjectIntersection == unique_ObjectIntersection[i])
                            )
                            col = cm(1.0 * i / NUM_COLORS)
                            ax.scatter(
                                good_project_planar_coords[0, indices_this_object],
                                good_project_planar_coords[1, indices_this_object],
                                color=col,
                                alpha=0.5,
                                marker=".",
                            )
                            ax.legend(unique_ObjectIntersection)
                            ax.set_title("Gaze vector right seat / FO")

                        fig1.set_size_inches((22, 11))
                        ax = plt.gca()
                        ax.get_xaxis().set_visible(False)
                        ax.axis("off")
                        plt.savefig(helper.local_process_dir + helper.crew_dir + "Figures/smarteyeGaze_" + scenarios[i_scenario] + "_rightseat.jpg")
                        plt.close()

    if args.plot_qatable:
        fig, ax = plt.subplots()
        cbar_kws = {"ticks": [0, 100]}
        ax = sns.heatmap(pct_usable_matrix, linewidths=0.5, cbar_kws=cbar_kws, annot=True, fmt=".3g")
        ax.set_xticklabels(file_types)
        ax.set_yticklabels(scenarios)
        ax.set(xlabel="pilot", ylabel="scenarios")
        plt.yticks(rotation=0)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        fig.tight_layout()
        # plt.show()
        plt.savefig(helper.local_process_dir + helper.crew_dir + "Figures/" + "smarteye_pct_usable.jpg")
        matplotlib.pyplot.close()
        np.save(helper.local_process_dir + helper.crew_dir + "Processing/" + "smarteye_pct_usable_matrix", pct_usable_matrix)

    pct_usable_matrix_df = pd.DataFrame(pct_usable_matrix)
    pct_usable_matrix_df.to_csv(
        helper.local_process_dir + helper.crew_dir + "Processing/" + "smarteye_pct_usable_matrix.csv"
    )
    event_smarteyeGazeTimeSeries_metrics.to_csv(
        helper.local_process_dir + helper.crew_dir + "Processing/" + "event_smarteyeGazeTimeSeries_metrics.csv"
    )

    if args.Push2Cloud:
        helper.sync_crew_folder_storage()
