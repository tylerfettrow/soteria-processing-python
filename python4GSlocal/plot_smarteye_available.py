import pandas as pd
import os
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from scipy import signal
import statistics
import subprocess
import time
import io
import seaborn as sns; sns.set_theme()
import matplotlib
import matplotlib as mpl
import helpers
import importlib
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
if args.directory:
    helper.local_process_dir = args.directory
###############################################

total_smarteye_available_matrix = np.zeros((len(scenarios),2,len(crew_arr)))

for i_crew in range(len(crew_arr)):
	helper.crew_dir = "Crew_" + crew_arr[i_crew] + "/"
	print(helper.crew_dir)
	this_matrix = helper.read_local_table('Processing/smarteye_pct_usable_matrix.csv')
	this_matrix = np.array(this_matrix)
	total_smarteye_available_matrix[:,:,i_crew] = this_matrix[:,1:]

total_data_available_matrix_reshaped = total_smarteye_available_matrix.transpose(0,2,1).reshape(6,total_smarteye_available_matrix.shape[1]*total_smarteye_available_matrix.shape[2])


fig, ax = plt.subplots()
cbar_kws = { 'ticks' : [0, 100] }
ax = sns.heatmap(np.rint(total_data_available_matrix_reshaped), linewidths=.5,cbar = False, cbar_kws = cbar_kws,annot=True,fmt='.3g')
plt.axis('off')
plt.yticks(rotation=0)
ax.xaxis.tick_top()

fig.tight_layout()
plt.savefig(helper.local_process_dir + "Figures/" + 'total_smarteye_available_matrix.jpg')
matplotlib.pyplot.close()
