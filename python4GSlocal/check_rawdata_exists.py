import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import helpers
from argparse import ArgumentParser
import sys

sns.set_theme()

importlib.reload(helpers)

helper = helpers.HELP()

###############################################
parser = ArgumentParser(description="Export Raw Data")
parser.add_argument(
    "-c", "--crews", type=str, default=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "13"]
)
parser.add_argument("-s", "--scenarios", type=str, default=["1", "2", "3", "5", "6", "7"])
parser.add_argument("-d", "--directory", type=str)
parser.add_argument("--Push2Cloud", type=bool, default=False)
args = parser.parse_args()
crew_arr, scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews, args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory
file_types = [
    "ifd_cockpit",
    "smarteye_leftseat",
    "smarteye_rightseat",
    "abm_leftseat",
    "abm_rightseat",
    "emp_acc_leftseat",
    "emp_acc_rightseat",
    "emp_bvp_leftseat",
    "emp_bvp_rightseat",
    "emp_gsr_leftseat",
    "emp_gsr_rightseat",
    "emp_ibi_leftseat",
    "emp_ibi_rightseat",
    "emp_temp_leftseat",
    "emp_temp_rightseat",
]
# file_types = ["emp_acc_leftseat","emp_acc_rightseat"]
# file_types = [ "IFD_COCKPIT.log", "SmartEye_Left_Seat_ET.log" , "SmartEye_Right_Seat_ET.log", "ABM.log", "ABM-1.log", "Emp_Emp_Device_Acc.log" ,"Emp_Emp_Device2_Acc.log" , "Emp_Emp_Device_Bvp.log","Emp_Emp_Device2_Bvp.log" ,"Emp_Emp_Device_Gsr.log","Emp_Emp_Device2_Gsr.log","Emp_Emp_Device_Ibi.log","Emp_Emp_Device2_Ibi.log" ,"Emp_Emp_Device_Temp.log", "Emp_Emp_Device2_Temp.log"]
###############################################

for i_crew in range(len(crew_arr)):
    file_existence_matrix = np.zeros((len(file_types), len(scenarios)))
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"

    print("checking files exists: " + helper.crew_dir)

    for i_scenario in range(len(scenarios)):
        for i_devicefile in range(len(file_types)):
            process_dir_name = helper.crew_dir + "/Processing/"
            try:
                this_table = helper.read_local_table(
                    "Processing/" + file_types[i_devicefile] + "_scenario" + scenarios[i_scenario] + ".csv"
                )
                print(
                    "file exists: "
                    + process_dir_name
                    + file_types[i_devicefile]
                    + "_scenario"
                    + scenarios[i_scenario]
                    + ".csv"
                )
                file_existence_matrix[i_devicefile, i_scenario] = 1
            except:
                print(
                    "missing: " + process_dir_name + file_types[i_devicefile] + "_scenario" + scenarios[i_scenario] + ".csv"
                )

    fig, ax = plt.subplots()
    cbar_kws = {"ticks": [0, 1]}
    ax = sns.heatmap(file_existence_matrix, linewidths=0.5, cbar_kws=cbar_kws)
    ax.set_xticklabels(scenarios)
    ax.set_yticklabels(file_types)
    ax.set(xlabel="scenarios", ylabel="devices")
    plt.yticks(rotation=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    fig.tight_layout()
    # plt.show()
    os.makedirs(helper.local_process_dir + helper.crew_dir + "Figures", exist_ok=True)
    plt.savefig(helper.local_process_dir + helper.crew_dir + "Figures/file_existence.jpg")
    matplotlib.pyplot.close()

    file_existence_matrix_df = pd.DataFrame(file_existence_matrix)
    file_existence_matrix_df.to_csv(helper.local_process_dir + helper.crew_dir + "Processing/" + "file_existence_matrix.csv")

    if args.Push2Cloud:
        helper.sync_crew_folder_storage()
