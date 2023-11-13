import pandas as pd
import numpy as np
import os
from os.path import exists
import mne
from scipy import signal
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
import io
import shutil

def getCrewInt(crewID):
	if (crewID == 'Crew_01'):
		b = 1
	elif (crewID == 'Crew_02'):
		b = 2
	elif (crewID == 'Crew_03'):
		b = 3
	elif (crewID == 'Crew_04'):
		b = 4
	elif (crewID == 'Crew_05'):
		b = 5
	elif (crewID == 'Crew_06'):
		b = 6
	elif (crewID == 'Crew_07'):
		b = 7
	elif (crewID == 'Crew_08'):
		b = 8
	elif (crewID == 'Crew_09'):
		b = 9
	elif (crewID == 'Crew_10'):
		b = 10
	elif (crewID == 'Crew_11'):
		b = 11
	elif (crewID == 'Crew_12'):
		b = 12
	elif (crewID == 'Crew_13'):
		b = 13
	return b

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
crews_to_process = ['Crew_01']
electrodes = [ 'F3','Fz', 'F4', 'C3','Cz', 'C4', 'P3','POz', 'P4']
file_types = ["leftseat","rightseat"]
scenarios = ["1","2","3","5","6","7"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

plot_engagement = 1
plot_workload = 0

# if plot_workload:
# 	rta_df = pd.read_excel(('gs://soteria_study_data/' + 'SOTERIA_Survey_Spreadsheet.xlsx'))
# 	blob = bucket.blob('SOTERIA_Survey_Spreadsheet.xlsx')
# 	data_bytes = blob.download_as_bytes()
# 	rta_df = pd.read_excel(data_bytes)
# 	rta_array = rta_df.to_numpy()

# 	xls = pd.ExcelFile(path_to_project+'/SOTERIA_Survey_Spreadsheet.xlsx')
# 	rta_df = pd.read_excel(xls, 'Day2_RTA_Workload')
# 	rta_array = rta_df.to_numpy()

for i_crew in range(len(crews_to_process)):

	if exists("Figures"):
		# subprocess.Popen('rm -rf Figures', shell=True)
		shutil.rmtree('Figures', ignore_errors=True)
		time.sleep(5)
		os.mkdir("Figures")
	else:
		os.mkdir("Figures")
	if exists("Processing"):
		# subprocess.Popen('rm -rf Processing', shell=True)
		shutil.rmtree('Processing', ignore_errors=True)
		time.sleep(5)
		os.mkdir("Processing")
	else:
		os.mkdir("Processing")

	crew_dir = crews_to_process[i_crew]
	process_dir_name = crew_dir + '/Processing/'

	# f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	# this_event_data = np.load(io.BytesIO(f_stream.read()))
	this_event_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + 'event_vector_scenario.csv'),delimiter=',')
	this_event_data = np.array(this_event_data)
	this_event_data = this_event_data[:,1:]

	event_eegTimeSeries_metrics = pd.read_table(('gs://soteria_study_data/' + process_dir_name + 'event_eegTimeSeries_metrics.csv'),delimiter=',')				

	for i_seat in range(len(file_types)):
		if file_types[i_seat] == "leftseat":			
			########## left seat #######################
			# f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'eeg_freqSpec_band_storage_leftseat.npy', 'rb')
			# eeg_freqSpec_band_storage = np.load(io.BytesIO(f_stream.read()))
			eeg_freqSpec_band_storage = pd.read_table(('gs://soteria_study_data/' + process_dir_name + 'eeg_freqSpec_band_storage_leftseat.csv'),delimiter=',')
			eeg_freqSpec_band_storage = np.array(eeg_freqSpec_band_storage)
			eeg_freqSpec_band_storage = eeg_freqSpec_band_storage[:,1:]

			# 5 (freq bands) x 9 (electrodes) x 6 (scenarios) x 200 (epochs)
			# delta # theta # alpha # beta # gamma
			
			############# TO DO.. need to load event_vector #######################
			
			# create a vector 9(electrodes) x scenarios of interest(6) to classify whether electrode is 2 = good 1=questionable  0=bad
			# electrodes = [ 'F3','Fz', 'F4', 'C3','Cz', 'C4', 'P3','POz', 'P4']
			# TO DO: need to make this able to be read directly from xlsx or something of that sort
			if crews_to_process[i_crew] == 'Crew_01':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[0,3:]
			if crews_to_process[i_crew] == 'Crew_02':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[2,3:]
			if crews_to_process[i_crew] == 'Crew_03':
				electrode_vector = np.array([[ 2, 1, 2, 2, 2, 2, 0, 2, 2], [2, 0, 2, 2, 2, 1, 0, 2, 2], [2, 0, 2, 2, 2, 1, 0, 2, 2], [2, 0, 2, 2, 2, 0, 0, 2, 2], [2, 0, 2, 2, 2, 0, 0, 2, 2], [2, 2, 2, 2, 2, 2, 0, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[4,3:]
			if crews_to_process[i_crew] == 'Crew_04':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[6,3:]	
			if crews_to_process[i_crew] == 'Crew_05':
				electrode_vector = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 0, 2, 2, 2, 2, 0], [2, 2, 2, 2, 2, 2, 2, 2, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[8,3:]	
			if crews_to_process[i_crew] == 'Crew_06':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[10,3:]	
			if crews_to_process[i_crew] == 'Crew_07':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[12,3:]	
			if crews_to_process[i_crew] == 'Crew_08':
				electrode_vector = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
				if plot_workload:
					rta_vector = rta_array[14,3:]	
			if crews_to_process[i_crew] == 'Crew_09':
				electrode_vector = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
				if plot_workload:
					rta_vector = rta_array[16,3:]	
			if crews_to_process[i_crew] == 'Crew_10':
				electrode_vector = np.array([[ 2, 0, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[18,3:]	
			if crews_to_process[i_crew] == 'Crew_11':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[20,3:]	
			if crews_to_process[i_crew] == 'Crew_12':
				electrode_vector = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[22,3:]
			if crews_to_process[i_crew] == 'Crew_13':
				electrode_vector = np.array([[ 0, 0, 2, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[24,3:]

		elif file_types[i_seat] == "rightseat":
			########## right seat #######################
			# f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'eeg_freqSpec_band_storage_rightseat.npy', 'rb')
			# eeg_freqSpec_band_storage = np.load(io.BytesIO(f_stream.read()))
			eeg_freqSpec_band_storage = pd.read_table(('gs://soteria_study_data/' + process_dir_name + 'eeg_freqSpec_band_storage_leftseat.csv'),delimiter=',')
			eeg_freqSpec_band_storage = np.array(eeg_freqSpec_band_storage)
			eeg_freqSpec_band_storage = eeg_freqSpec_band_storage[:,1:]
			# eeg_freq_band_storage = np.load(crew_dir + "/Processing/" + 'eeg_freq_band_storage_rightseat.npy')

			if crews_to_process[i_crew] == 'Crew_01':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[1,3:]
			if crews_to_process[i_crew] == 'Crew_02':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[3,3:]
			if crews_to_process[i_crew] == 'Crew_03':
				electrode_vector = np.array([[ 2, 0, 2, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2, 2], [0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 0, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[5,3:]
			if crews_to_process[i_crew] == 'Crew_04':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
				if plot_workload:
					rta_vector = rta_array[7,3:]
			if crews_to_process[i_crew] == 'Crew_05':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[9,3:]
			if crews_to_process[i_crew] == 'Crew_06':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
				if plot_workload:
					rta_vector = rta_array[11,3:]
			if crews_to_process[i_crew] == 'Crew_07':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[13,3:]
			if crews_to_process[i_crew] == 'Crew_08':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[15,3:]
			if crews_to_process[i_crew] == 'Crew_09':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
				if plot_workload:
					rta_vector = rta_array[17,3:]
			if crews_to_process[i_crew] == 'Crew_10':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 0, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 0, 2], [1, 1, 1, 1, 1, 1, 1, 0, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[19,3:]
			if crews_to_process[i_crew] == 'Crew_11':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[21,3:]
			if crews_to_process[i_crew] == 'Crew_12':
				electrode_vector = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 2, 2, 2, 2, 2, 2], [0, 2, 0, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[23,3:]
			if crews_to_process[i_crew] == 'Crew_13':
				electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
				if plot_workload:
					rta_vector = rta_array[25,3:]

		for i_scenario in range(len(scenarios)):	
			for this_epoch in range(eeg_freqSpec_band_storage.shape[2]):	
				if np.all(np.isnan(eeg_freqSpec_band_storage[:,:,this_epoch,i_scenario])):
					number_of_time_epochs_spec = this_epoch -1
					break
			eeg_timesec_epoch_storage = eeg_freqSpec_band_storage[0,0,0:number_of_time_epochs_spec,i_scenario]
			x_axis_vector_spec = np.linspace(0,100,number_of_time_epochs_spec)
			this_epoch_bandpower_ratio_spec = np.zeros((4,9,number_of_time_epochs_spec))
			engagement_index_spec = np.zeros((len(scenarios),number_of_time_epochs_spec))
			filtered_engagement_index_spec = np.zeros((len(scenarios),number_of_time_epochs_spec))
			filtered_taskLoad_index_spec = np.zeros((len(scenarios),number_of_time_epochs_spec))
			taskLoad_index_spec = np.zeros((len(scenarios),number_of_time_epochs_spec))

			print("Plotting: " + crew_dir + ' scenario' + scenarios[i_scenario])
			for idx in range(0,eeg_timesec_epoch_storage.shape[0]):
				# print(idx)
				if this_event_data[0, i_scenario] <= eeg_timesec_epoch_storage[ idx]:
					this_event1_epoch = np.floor((idx - 1)/(number_of_time_epochs_spec/100)) 
					break
			for idx in range(0,eeg_timesec_epoch_storage.shape[0]):
				# print(idx)
				if this_event_data[1, i_scenario] <= eeg_timesec_epoch_storage[idx]:
					this_event2_epoch = np.floor((idx - 1)/(number_of_time_epochs_spec/100))
					break

			if plot_engagement:
				##### ENGAGE AND TASK INDEX SPEC ############
				if (i_seat == 0) & (i_scenario == 0):
					fig_eng_spec, axs_eng_spec = plt.subplots(len(scenarios), 1)
					fig_eng_spec.suptitle(crews_to_process[i_crew]+'_engagement')
	
				# b, a = signal.ellip(4, 0.01, 120, 0.125)
				b, a = signal.butter(8, 0.125)
				# filtered_this_epoch_bandpower_ratio = this_epoch_bandpower_ratio
				# if (np.isnan(engagement_index_spec[i_scenario,:]).any()) | (~np.isfinite(engagement_index_spec[i_scenario,:])).any():
				filtered_engagement_index_spec[i_scenario,:] = engagement_index_spec[i_scenario,:]
				# else:
				# 	# print('plotting engagement_index')
				# 	filtered_engagement_index_spec[i_scenario,:] = signal.filtfilt(b, a, engagement_index_spec[i_scenario,:], method="gust")
					# filtered_engagement_index[i_scenario,:] = engagement_index[i_scenario,:]

				if file_types[i_seat] == "leftseat":
					axs_eng_spec[i_scenario].plot(x_axis_vector_spec, filtered_engagement_index_spec[i_scenario,:], linewidth=3, color='blue')
				else:
					axs_eng_spec[i_scenario].plot(x_axis_vector_spec, filtered_engagement_index_spec[i_scenario,:], linewidth=3, color='red')
					axs_eng_spec[i_scenario].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
					axs_eng_spec[i_scenario].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs_eng_spec[i_scenario].set_ylabel(scenarios[i_scenario])
				axs_eng_spec[0].legend(["left engagement_index","right engagement_index"])
				axs_eng_spec[i_scenario].set_xlim((0, 100))
				axs_eng_spec[i_scenario].set_ylim((0, 1))

				if (i_seat == 0) & (i_scenario == 0):
					fig_task_spec, axs_task_spec = plt.subplots(len(scenarios), 1)
					fig_task_spec.suptitle(crews_to_process[i_crew]+'_taskLoad')
	
				# b, a = signal.ellip(4, 0.01, 120, 0.125)
				b, a = signal.butter(8, 0.125)
				# filtered_this_epoch_bandpower_ratio = this_epoch_bandpower_ratio

				# if (np.isnan(taskLoad_index_spec[i_scenario,:]).any()) | (~np.isfinite(taskLoad_index_spec[i_scenario,:])).any():
				filtered_taskLoad_index_spec[i_scenario,:] = taskLoad_index_spec[i_scenario,:]
				# else:
				# 	filtered_taskLoad_index_spec[i_scenario,:] = signal.filtfilt(b, a, taskLoad_index_spec[i_scenario,:], method="gust")
					# filtered_taskLoad_index[i_scenario,:] = taskLoad_index[i_scenario,:]

				if file_types[i_seat] == "leftseat":
					axs_task_spec[i_scenario].plot(x_axis_vector_spec, filtered_taskLoad_index_spec[i_scenario,:], linewidth=3, color="blue")
				else:
					axs_task_spec[i_scenario].plot(x_axis_vector_spec, filtered_taskLoad_index_spec[i_scenario,:], linewidth=3, color="red")
					axs_task_spec[i_scenario].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
					axs_task_spec[i_scenario].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs_task_spec[i_scenario].set_ylabel(scenarios[i_scenario])
				axs_task_spec[0].legend(["left task_index","right task_index"])
				axs_task_spec[i_scenario].set_xlim((0, 100))
				# axs_task_spec[i_scenario].set_ylim((0, 50))

	if plot_engagement:
		fig_eng_spec.set_size_inches((22, 11))
		fig_eng_spec.savefig("Figures/" + 'eeg_engagement_spec_'+crews_to_process[i_crew]+'.tif',bbox_inches='tight',pad_inches=0)
		fig_task_spec.set_size_inches((22, 11))
		fig_task_spec.savefig("Figures/" + 'eeg_taskLoad_spec_'+crews_to_process[i_crew]+'.tif',bbox_inches='tight',pad_inches=0)
		# fig_eng.close()

	subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Figures"', shell=True)
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)
	
	# plt.close()
	# plt.show()
				#############################################################