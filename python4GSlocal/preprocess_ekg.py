import pandas as pd
import numpy as np
import os
from os.path import exists
import mne
import matplotlib.pyplot as plt
from scipy import signal
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch, periodogram
from google.cloud import storage
from numpy import linalg as la
from os.path import exists
import subprocess
import time
import matplotlib.pyplot as plt

# update() function to change the graph when the
# slider is in use
def update(val):
    pos = slider_position.val
    ax.axis([pos, pos+10, ax.margins(y=.1), ax.margins(y=.1)])
    fig.canvas.draw_idle()
slider_color = 'White'				 


crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_02']
file_types = ["abm_leftseat","abm_rightseat"]
scenarios = ["1","2","3","5","6","7"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

number_of_epochs = 250

plot_qa_figs = 0

for i_crew in range(len(crews_to_process)):
	if exists("Figures"):
		subprocess.Popen('rm -rf Figures', shell=True)
		time.sleep(5)
		os.mkdir("Figures")
	else:
		os.mkdir("Figures")
	if exists("Processing"):
		subprocess.Popen('rm -rf Processing', shell=True)
		time.sleep(5)
		os.mkdir("Processing")
	else:
		os.mkdir("Processing")	
		
	pct_usable_matrix = np.zeros((len(scenarios),len(file_types)))
	ekg_peaks_bpm_leftseat = np.zeros((number_of_epochs,len(scenarios)))
	ekg_peaks_bpm_rightseat = np.zeros((number_of_epochs,len(scenarios)))
	# ekg_peaks_rightseat = np.zeros((number_of_bands,len(scenarios)))
	ekg_timesec_epoch_storage = np.zeros((len(scenarios),number_of_epochs))
	crew_dir = crews_to_process[i_crew]
	for i_scenario in range(len(scenarios)):
		for i_seat in range(len(file_types)):
			process_dir_name = crew_dir + "/Processing/"
			blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
			if blob.exists():
				print("QA checking ekg: " + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				abm_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')

				# ekg = pd.DataFrame(abm_data.UserTimeStamp, abm_data.ECG)
				# ekg = abm_data[['UserTimeStamp','ECG']].set_index('UserTimeStamp')
				peaks, _ = signal.find_peaks(abm_data.ECG, distance=100,prominence=500, width=1)

				if plot_qa_figs:
					fig, ax = plt.subplots()
					plt.plot(abm_data.UserTimeStamp, abm_data.ECG)
					plt.plot(abm_data.UserTimeStamp[peaks],abm_data.ekg[peaks],"x")
					plt.title('RawData: ' + file_types[i_seat] + '_scenario' + scenarios[i_scenario])
					axis_position = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor = slider_color)
					slider_position = Slider(axis_position, 'Pos', 0.1, 90.0)
					slider_position.on_changed(update)
					plt.show()

				length_this_data = len(abm_data.ECG)
				for this_epoch in range(number_of_epochs):
					this_epoch_indices_start = np.floor(length_this_data/number_of_epochs) * this_epoch
					this_epoch_indices_end = this_epoch_indices_start + np.floor(length_this_data/number_of_epochs)
					ekg_timesec_epoch_storage[i_scenario,this_epoch] = abm_data.UserTimeStamp[this_epoch_indices_start]

					this_epoch_start_time = abm_data.UserTimeStamp[this_epoch_indices_start]
					if this_epoch == number_of_epochs - 1:
						this_epoch_end_time = abm_data.UserTimeStamp[this_epoch_indices_end-1]
					else:
						this_epoch_end_time = abm_data.UserTimeStamp[this_epoch_indices_end]

					peaks_to_include = []
					for i_peak in range(len(peaks)):
						if (peaks[i_peak] <= this_epoch_indices_end) & (peaks[i_peak] >= this_epoch_indices_start):
							peaks_to_include.append(peaks[i_peak])

					bpm_peaks_this_epoch = 60/(np.diff(abm_data.UserTimeStamp[peaks_to_include]))

					if i_seat == 0:
						ekg_peaks_bpm_leftseat[this_epoch, i_scenario] = bpm_peaks_this_epoch.mean()
					elif i_seat == 1:
						ekg_peaks_bpm_rightseat[this_epoch, i_scenario] = bpm_peaks_this_epoch.mean()
	
	np.save("Processing/" + 'ekg_bpm_leftseat',ekg_peaks_bpm_leftseat)
	np.save("Processing/" + 'ekg_timesec_epoch_storage', ekg_timesec_epoch_storage)
	np.save("Processing/" + 'ekg_bpm_rightseat',ekg_peaks_bpm_rightseat)
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)			
	# np.save(crew_dir + "/Processing/" + 'ekg_timesec_band_storage_leftseat', ekg_timesec_band_storage)