import numpy as np
import matplotlib.pyplot as plt
import helpers
import importlib
import seaborn as sns
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
###############################################


total_available_matrix = np.zeros((len(scenarios), 2, len(crew_arr)))

for i_crew in range(len(crew_arr)):
    helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"
    print(helper.crew_dir)
    this_matrix = helper.read_local_table("Processing/ekg_pct_usable_matrix.csv")
    this_matrix = np.array(this_matrix)
    total_available_matrix[:, :, i_crew] = this_matrix[:, 1:]
total_data_available_matrix_reshaped = total_available_matrix.transpose(0, 2, 1).reshape(
    6, total_available_matrix.shape[1] * total_available_matrix.shape[2]
)

# fig, ax = plt.subplots()
# cbar_kws = { 'ticks' : [0, 100] }
# ax = sns.heatmap(np.rint(total_data_available_matrix_reshaped), linewidths=.5, cbar_kws = cbar_kws,annot=True,fmt='.3g')
# # ax.set_xticklabels(scenarios)
# # ax.set_yticklabels(file_types)
# # ax.set(xlabel='scenarios', ylabel='devices')
# # plt.yticks(rotation=0)
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()

# # fig.tight_layout()
# # plt.show()
# plt.savefig("Figures/" + 'total_ekg_available_matrix.jpg')
# # matplotlib.pyplot.close()

fig, ax = plt.subplots()
cbar_kws = {"ticks": [0, 100]}
ax = sns.heatmap(
    np.rint(total_data_available_matrix_reshaped), linewidths=0.5, cbar=False, cbar_kws=cbar_kws, annot=True, fmt=".3g"
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
plt.savefig(helper.local_process_dir + "Figures/" + "total_ekg_available_matrix.jpg")
# matplotlib.pyplot.close()
