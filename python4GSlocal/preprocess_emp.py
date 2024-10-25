import numpy as np
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
parser.add_argument("-p1", "--plot_qa_figs", action="store_true", help="Plots QA figs for each participant and scenario. ")
parser.add_argument("--Push2Cloud", type=bool, default=False)
args = parser.parse_args()
crew_arr, scenarios = helper.ParseCrewAndScenario(sys.argv, args.crews, args.scenarios)
if args.directory:
    helper.local_process_dir = args.directory

file_types = ["emp_acc", "emp_bvp", "emp_gsr", "emp_ibi", "emp_temp"]
seat = ["leftseat", "rightseat"]
#######################################


for i_crew in range(len(crew_arr)):
    pct_usable_matrix = np.zeros((len(scenarios), len(file_types)))
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"
    process_dir_name = helper.crew_dir + "/Processing/"

    emp_data_exists = 0

    for i_seat in range(len(seat)):
        for i_file in range(len(file_types)):
            for i_scenario in range(len(scenarios)):
                process_dir_name = helper.crew_dir + "Processing/"
                try:
                    print(
                        "trying to read: " + process_dir_name,
                        file_types[i_file] + "_" + seat[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv",
                    )
                    emp_data = helper.read_local_table(
                        "Processing/" + file_types[i_file] + "_" + seat[i_seat] + "_scenario" + scenarios[i_scenario] + ".csv",
                    )
                    emp_data_exists = 1
                except:
                    print("failed")
                    emp_data_exists = 0
                    break
                if emp_data_exists:
                    continue
                    # plt.plot(emp_data)
                    # plt.show()
