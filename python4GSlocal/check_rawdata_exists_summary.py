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
###############################################

total_file_existence_matrix = np.zeros((len(file_types), len(scenarios), len(crew_arr)))

for i_crew in range(len(crew_arr)):
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"

    this_matrix = helper.read_local_table("Processing/" + "file_existence_matrix.csv")
    this_matrix = np.array(this_matrix)

    total_file_existence_matrix[:, :, i_crew] = this_matrix[:, 1:]

total_file_existence_matrix_squeezed = np.squeeze(np.sum(total_file_existence_matrix, axis=2))
total_file_existence_matrix_squeezed_averaged = np.multiply(
    np.divide(total_file_existence_matrix_squeezed, len(crew_arr)), 100
)

fig, ax = plt.subplots()
cbar_kws = {"ticks": [0, 100]}
ax = sns.heatmap(
    np.rint(np.round(total_file_existence_matrix_squeezed_averaged)),
    linewidths=0.5,
    cbar_kws=cbar_kws,
    annot=True,
    fmt=".3g",
)
ax.set_xticklabels(scenarios)
ax.set_yticklabels(file_types)
ax.set(xlabel="scenarios", ylabel="devices")
plt.yticks(rotation=0)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

os.makedirs(helper.local_process_dir + "Figures", exist_ok=True)

fig.tight_layout()
# plt.show()
plt.savefig(helper.local_process_dir + "Figures/" + "total_file_existence.jpg")
matplotlib.pyplot.close()

fig, ax = plt.subplots()
cbar_kws = {"ticks": [0, 100]}
ax = sns.heatmap(
    np.rint(np.round(total_file_existence_matrix_squeezed_averaged)),
    linewidths=0.5,
    cbar=False,
    cbar_kws=cbar_kws,
    annot=True,
    fmt=".3g",
)
# ax.set_xticklabels(scenarios)
# ax.set_yticklabels(file_types)
# ax.set(xlabel='scenarios', ylabel='devices')
plt.axis("off")
plt.yticks(rotation=0)
# ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

fig.tight_layout()
# plt.show()

plt.savefig(helper.local_process_dir + "Figures/" + "total_file_existence_nolabels.jpg")
matplotlib.pyplot.close()

if args.Push2Cloud:
    helper.sync_crew_folder_storage()
