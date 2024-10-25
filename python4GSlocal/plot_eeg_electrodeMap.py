import numpy as np
import seaborn as sns
sns.set_theme()
import importlib
import helpers
import matplotlib
import proplot as pplt
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys

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
file_types = ["abm_leftseat", "abm_rightseat"]
if args.directory:
    helper.local_process_dir = args.directory


def get_electrode_pos_x(electrode_num):
    if electrode_num == 0:
        return -1
    if electrode_num == 1:
        return 0
    if electrode_num == 2:
        return 1
    if electrode_num == 3:
        return -1
    if electrode_num == 4:
        return 0
    if electrode_num == 5:
        return 1
    if electrode_num == 6:
        return -1
    if electrode_num == 7:
        return 0
    if electrode_num == 8:
        return 1


def get_electrode_pos_y(electrode_num):
    if electrode_num == 0:
        return 1
    if electrode_num == 1:
        return 1
    if electrode_num == 2:
        return 1
    if electrode_num == 3:
        return 0
    if electrode_num == 4:
        return 0
    if electrode_num == 5:
        return 0
    if electrode_num == 6:
        return -1
    if electrode_num == 7:
        return -1
    if electrode_num == 8:
        return -1


fig = pplt.figure(space=0, refwidth=0.7)
axs = fig.subplots(nrows=len(scenarios), ncols=len(crew_arr) * 2)
axs.format(
    abc=False,
    xticks=[],
    yticks=[],
)

u = 0  # x-position of the center
v = 0  # y-position of the center
a = 1.75  # radius on the x-axis
b = 2  # radius on the y-axis

t = np.linspace(0, 2 * 3.14, 100)
head_x = u + a * np.cos(t)
head_y = v + b * np.sin(t)

# plt.close("all")
axs_idx = 0

for i_scenario in range(len(scenarios)):
    for i_crew in range(len(crew_arr)):
        helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"
        for i_seat in range(len(file_types)):
            electrode_vector_df = helper.read_bucket_xlsx(
                "Analysis/Analysis_eeg_electrode_quality_vector.xlsx", helper.getSubWorksheet(file_types[i_seat])
            )
            electrode_vector = electrode_vector_df.to_numpy()

            for this_electrode_idx in range(9):
                if electrode_vector[i_scenario, this_electrode_idx + 1] == 0:
                    axs[axs_idx].plot(
                        get_electrode_pos_x(this_electrode_idx), get_electrode_pos_y(this_electrode_idx), "o", color="r"
                    )
                elif electrode_vector[i_scenario, this_electrode_idx + 1] == 1:
                    axs[axs_idx].plot(
                        get_electrode_pos_x(this_electrode_idx), get_electrode_pos_y(this_electrode_idx), "o", color="y"
                    )
                elif electrode_vector[i_scenario, this_electrode_idx + 1] == 2:
                    axs[axs_idx].plot(
                        get_electrode_pos_x(this_electrode_idx), get_electrode_pos_y(this_electrode_idx), "o", color="g"
                    )
            axs[axs_idx].plot(head_x, head_y, color="k")
            axs[axs_idx].axis(xmin=-2, xmax=2.5)

            print(
                "plotting crew " + crew_arr[i_crew] + ", seat " + str(i_seat),
                ", scenario " + str(i_scenario),
                ", electrode " + str(this_electrode_idx),
                ", axs " + str(axs_idx),
            )
            axs_idx += 1

# plt.show()
plt.savefig(helper.local_process_dir + "Figures/" + "eeg_electrodeMap.tif")
matplotlib.pyplot.close()
