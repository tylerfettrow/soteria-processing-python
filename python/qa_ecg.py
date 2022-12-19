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


crews_to_process = ['Crew_07','Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_01']
# crews_to_process = ['Crew_02','Crew_03', 'Crew_04', 'Crew_05', 'Crew_06']
# sample_Rate = 260
file_types = ["abm_leftseat","abm_rightseat"]
scenarios = ["1","2","3","5","6","7","8","9"]
# scenarios = ["5","6","7","8","9"]
# scenarios = ["8","9"]
# file_types = ["abm_leftseat"]
# scenarios = ["1"]
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'

number_of_bands = 250

plot_qa_figs = 0

for i_crew in range(len(crews_to_process)):
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	ecg_peaks_bpm_leftseat = np.zeros((number_of_bands,len(scenarios)))
	ecg_peaks_bpm_rightseat = np.zeros((number_of_bands,len(scenarios)))
	# ecg_peaks_rightseat = np.zeros((number_of_bands,len(scenarios)))
	ecg_timesec_band_storage = np.zeros((len(scenarios),number_of_bands))
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

				if plot_qa_figs:
					fig, ax = plt.subplots()
					plt.plot(abm_data.UserTimeStamp, abm_data.ECG)
					plt.plot(abm_data.UserTimeStamp[peaks],abm_data.ECG[peaks],"x")
					# ecg.plot(ax=ax)
					plt.title('RawData: ' + file_types[i_seat] + '_scenario' + scenarios[i_scenario])
					axis_position = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor = slider_color)
					slider_position = Slider(axis_position, 'Pos', 0.1, 90.0)				 
					slider_position.on_changed(update) 
					plt.show()

				length_this_data = len(abm_data.ECG)
				for this_band in range(number_of_bands):
					this_band_indices_start = np.floor(length_this_data/number_of_bands) * this_band
					this_band_indices_end = this_band_indices_start + np.floor(length_this_data/number_of_bands)
					ecg_timesec_band_storage[i_scenario,this_band] = abm_data.UserTimeStamp[this_band_indices_start]

					this_band_start_time = abm_data.UserTimeStamp[this_band_indices_start]
					if this_band == number_of_bands - 1:
						this_band_end_time = abm_data.UserTimeStamp[this_band_indices_end-1]
					else:
						this_band_end_time = abm_data.UserTimeStamp[this_band_indices_end]



					# find the start and stop (seconds) for each band
					# then average the peaks that fall after (start) and before (end) the band indices
					# store that value in ecg_peaks_leftseat(this_band, i_scenario)
					peaks_to_include = []
					for i_peak in range(len(peaks)):
						if (peaks[i_peak] <= this_band_indices_end) & (peaks[i_peak] >= this_band_indices_start):
							peaks_to_include.append(peaks[i_peak])

					bpm_peaks_this_band = 60/(np.diff(abm_data.UserTimeStamp[peaks_to_include]))

					if i_seat == 0:
						ecg_peaks_bpm_leftseat[this_band, i_scenario] = bpm_peaks_this_band.mean()
					elif i_seat == 1:
						ecg_peaks_bpm_rightseat[this_band, i_scenario] = bpm_peaks_this_band.mean()
				
	np.save(crew_dir + "/Processing/" + 'ecg_bpm_leftseat',ecg_peaks_bpm_leftseat)
	np.save(crew_dir + "/Processing/" + 'ecg_timesec_band_storage', ecg_timesec_band_storage)
	np.save(crew_dir + "/Processing/" + 'ecg_bpm_rightseat',ecg_peaks_bpm_rightseat)
	# np.save(crew_dir + "/Processing/" + 'ecg_timesec_band_storage_leftseat', ecg_timesec_band_storage)