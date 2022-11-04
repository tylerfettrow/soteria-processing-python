import pandas as pd
import numpy as np
import os
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import find_peaks

# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    ax.axis([pos, pos+10, ax.margins(y=.1), ax.margins(y=.1)])
    fig.canvas.draw_idle()
slider_color = 'White'				 


# crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
crews_to_process = ['Crew_01']
# sample_Rate = 260
file_types = ["abm_leftseat","abm_rightseat"]
scenarios = ["1","2","3","6","7","8","9"]
# file_types = ["abm_leftseat"]
# scenarios = ["1"]
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'

for i_crew in range(len(crews_to_process)):
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	for i_scenario in range(len(scenarios)):
		for i_seat in range(len(file_types)):
			process_dir_name = crew_dir + "/Processing/"
			if exists(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'):
				print("QA checking ECG: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				abm_data = pd.read_table((process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')

				ecg = pd.DataFrame(abm_data.UserTimeStamp, abm_data.ECG)
				ecg = abm_data[['UserTimeStamp','ECG']].set_index('UserTimeStamp')
				peaks, _ = find_peaks(abm_data.ECG, distance=100,prominence=500, width=1)

				fig, ax = plt.subplots()
				plt.plot(abm_data.UserTimeStamp, abm_data.ECG)
				plt.plot(abm_data.UserTimeStamp[peaks],abm_data.ECG[peaks],"x")
				# ecg.plot(ax=ax)
				plt.title('RawData: ' + file_types[i_seat] + '_scenario' + scenarios[i_scenario])
				axis_position = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor = slider_color)
				slider_position = Slider(axis_position, 'Pos', 0.1, 90.0)				 
				slider_position.on_changed(update) 
				plt.show()