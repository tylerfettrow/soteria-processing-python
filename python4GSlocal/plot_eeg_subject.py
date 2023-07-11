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
from tensorflow.python.lib.io import file_io
import io

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_02']
electrodes = [ 'F3','Fz', 'F4', 'C3','Cz', 'C4', 'P3','POz', 'P4']
file_types = ["leftseat","rightseat"]
scenarios = ["1","2","3","5","6","7"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")


plot_individual = 1
plot_engagement = 1
plot_workload = 0

# if plot_workload:
	# rta_df = pd.read_excel(('gs://soteria_study_data/' + 'SOTERIA_Survey_Spreadsheet.xlsx'))
	# blob = bucket.blob('SOTERIA_Survey_Spreadsheet.xlsx')
	# data_bytes = blob.download_as_bytes()
	# rta_df = pd.read_excel(data_bytes)
	# rta_array = rta_df.to_numpy()

	# xls = pd.ExcelFile(path_to_project+'/SOTERIA_Survey_Spreadsheet.xlsx')
	# rta_df = pd.read_excel(xls, 'Day2_RTA_Workload')
	# rta_array = rta_df.to_numpy()

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

	crew_dir = crews_to_process[i_crew]
	process_dir_name = crew_dir + '/Processing/'

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	this_event_data = np.load(io.BytesIO(f_stream.read()))

	for i_seat in range(len(file_types)):
		if file_types[i_seat] == "leftseat":
			f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'eeg_timesec_epoch_storage_leftseat.npy', 'rb')
			eeg_timesec_epoch_storage = np.load(io.BytesIO(f_stream.read()))
			
			########## left seat #######################
			f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'eeg_freq_band_storage_leftseat.npy', 'rb')
			eeg_freq_band_storage = np.load(io.BytesIO(f_stream.read()))
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
			f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'eeg_freq_band_storage_rightseat.npy', 'rb')
			eeg_freq_band_storage = np.load(io.BytesIO(f_stream.read()))
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

		number_of_time_epochs = eeg_timesec_epoch_storage.shape[1]
		x_axis_vector = np.linspace(0,100,number_of_time_epochs)
		engagement_index = np.zeros((len(scenarios),number_of_time_epochs))
		filtered_engagement_index = np.zeros((len(scenarios),number_of_time_epochs))
		filtered_taskLoad_index = np.zeros((len(scenarios),number_of_time_epochs))
		taskLoad_index = np.zeros((len(scenarios),number_of_time_epochs))
		for i_scenario in range(len(scenarios)):
			### WARNING: need to adjust for inclusion of freq bands 
			this_epoch_bandpower_ratio = np.zeros((4,9,number_of_time_epochs))
			print("Plotting: " + crew_dir + '_'+ file_types[i_seat]+ '_scenario' + scenarios[i_scenario] + '.csv')

			this_event1_epoch = 0
			this_event2_epoch = 0
			# convert events (seconds) to band number ... assuming left and right seat for same crew are similar enough
			for idx in range(0,eeg_timesec_epoch_storage.shape[1]):
				# print(idx)
				if this_event_data[0, i_scenario] <= eeg_timesec_epoch_storage[i_scenario, idx]:
					this_event1_epoch = np.floor((idx - 1)/(number_of_time_epochs/100)) 
					break
			for idx in range(0,eeg_timesec_epoch_storage.shape[1]):
				# print(idx)
				if this_event_data[1, i_scenario] <= eeg_timesec_epoch_storage[i_scenario, idx]:
					this_event2_epoch = np.floor((idx - 1)/(number_of_time_epochs/100))
					break

			##### ENGAGE AND TASK INDEX ############
			for this_epoch in range(eeg_freq_band_storage.shape[3]):
				### WARNING: removing delta band power here
				eeg_freq_band_storage[eeg_freq_band_storage < 0] = 0
				this_epoch_bandpower_ratio[:,:,this_epoch] = eeg_freq_band_storage[1:,:,i_scenario,this_epoch] / eeg_freq_band_storage[1:,:,i_scenario,this_epoch].sum(0)
				theta = 0
				alpha = 0
				beta = 0

				include_electrode_vector = np.where(electrode_vector[i_scenario,:]==2)
										
				theta =  eeg_freq_band_storage[1,include_electrode_vector,i_scenario,this_epoch].sum() # theta
				alpha = eeg_freq_band_storage[2,include_electrode_vector,i_scenario,this_epoch].sum() # alpha
				beta = eeg_freq_band_storage[3,include_electrode_vector,i_scenario,this_epoch].sum() # beta
				total_power = theta + alpha + beta
				# Percent power; power relative to total power
				per_power_theta = theta / total_power
				per_power_alpha = alpha / total_power
				per_power_beta = beta / total_power

				engagement_index[i_scenario,this_epoch] = per_power_beta / (per_power_alpha + per_power_theta)
				
				# task load index
				if (electrode_vector[i_scenario,1]==2) & (electrode_vector[i_scenario,7]==2):
					# Fz_theta = eeg_freq_band_storage[1,1,i_scenario,this_epoch]
					# Pz_alpha = eeg_freq_band_storage[2,7,i_scenario,this_epoch]
					Fz_theta = np.nanmean(eeg_freq_band_storage[1,0:2,i_scenario,this_epoch])
					Pz_alpha = np.nanmean(eeg_freq_band_storage[2,6:8,i_scenario,this_epoch])
					taskLoad_index[i_scenario,this_epoch] = Fz_theta / Pz_alpha
				else:
					taskLoad_index[i_scenario,this_epoch] = 'nan'
			########################################################################			
			
			if plot_individual:
				this_epoch_bandpower_ratio[np.isnan(this_epoch_bandpower_ratio)] = 0

				# filtering the bandpower sig
				# b, a = signal.ellip(4, 0.01, 120, 0.125)
				b, a = signal.butter(8, 0.125)
				# filtered_this_epoch_bandpower_ratio = this_epoch_bandpower_ratio
				filtered_this_epoch_bandpower_ratio = signal.filtfilt(b, a, this_epoch_bandpower_ratio, method="gust")

				fig, axs = plt.subplots(3, 3)
				fig.suptitle(crews_to_process[i_crew]+file_types[i_seat]+scenarios[i_scenario])
				
				std_bandpower = filtered_this_epoch_bandpower_ratio.std(axis=0)

				if plot_workload:
					rta_vector = rta_vector[~pd.isna(rta_vector)]
					if (crews_to_process[i_crew] == 'Crew_12') & (scenarios[i_scenario] == '5'):
						# check to see if the number of rta inputs match the eeg_timesec_storage
						expected_length_of_data = eeg_timesec_epoch_storage[3,-1]/60
						if len(rta_vector) != int(np.floor(expected_length_of_data)):
							print('WARNING: RTA workload does not match trial length')
							print('rta = '+ str(len(rta_vector)) + ' eeg: ' + str(int(np.floor(expected_length_of_data))))
					elif scenarios[i_scenario] == '2':
						expected_length_of_data = eeg_timesec_epoch_storage[1,-1]/60
						if len(rta_vector) != int(np.floor(expected_length_of_data)):
							print('WARNING: RTA workload does not match trial length')
							print('rta = '+ str(len(rta_vector)) + ' eeg: ' + str(int(np.floor(expected_length_of_data))))
						rta_vector_timewarped = np.repeat(rta_vector,int(np.floor(number_of_time_epochs/len(rta_vector))))
						if len(rta_vector_timewarped) < number_of_time_epochs:
							diff = number_of_time_epochs - len(rta_vector_timewarped)
							rta_vector_timewarped = np.append(rta_vector_timewarped,np.ones(diff)*rta_vector_timewarped[-1])
						axs[0,0].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[0,1].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[0,2].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[1,0].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[1,1].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[1,2].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[2,0].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[2,1].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
						axs[2,2].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)

				axs[0, 0].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,0,:].T * 100, linewidth=1)
				axs[0, 0].plot(x_axis_vector, std_bandpower[0,:].T * 100 , linewidth=2)
				axs[0, 0].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[0, 0].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[0, 0].set_ylim([0, 100])
				if electrode_vector[i_scenario,1] == 2:
					axs[0, 0].set_title('F3', color='black')
				elif electrode_vector[i_scenario,1] == 1:
					axs[0, 0].set_title('F3', color='blue')
				elif electrode_vector[i_scenario,1] == 0:
					axs[0, 0].set_title('F3', color='red')

				axs[0, 1].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[0, 1].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[0, 1].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,1,:].T * 100, linewidth=1)
				axs[0, 1].plot(x_axis_vector, std_bandpower[1,:].T * 100 , linewidth=2)
				axs[0, 1].set_ylim([0, 100])
				if electrode_vector[i_scenario,1] == 2:
					axs[0, 1].set_title('Fz', color='black')
				elif electrode_vector[i_scenario,1] == 1:
					axs[0, 1].set_title('Fz', color='blue')
				elif electrode_vector[i_scenario,1] == 0:
					axs[0, 1].set_title('Fz', color='red')

				axs[0, 2].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[0, 2].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[0, 2].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,2,:].T * 100, linewidth=1)
				axs[0, 2].plot(x_axis_vector, std_bandpower[2,:].T * 100, linewidth=2 )
				axs[0, 2].set_ylim([0, 100])

				
				if electrode_vector[i_scenario,2] == 2:
					axs[0, 2].set_title('F4', color='black')
				elif electrode_vector[i_scenario,2] == 1:
					axs[0, 2].set_title('F4', color='blue')
				elif electrode_vector[i_scenario,2] == 0:
					axs[0, 2].set_title('F4', color='red')

				axs[1, 0].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[1, 0].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[1, 0].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,3,:].T * 100, linewidth=1)
				axs[1, 0].plot(x_axis_vector, std_bandpower[3,:].T * 100, linewidth=2)
				axs[1, 0].set_ylim([0, 100])
				if electrode_vector[i_scenario,3] == 2:
					axs[1, 0].set_title('C3', color='black')
				elif electrode_vector[i_scenario,3] == 1:
					axs[1, 0].set_title('C3', color='blue')
				elif electrode_vector[i_scenario,3] == 0:
					axs[1, 0].set_title('C3', color='red')

				axs[1, 1].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[1, 1].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[1, 1].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,4,:].T * 100, linewidth=1)
				axs[1, 1].plot(x_axis_vector, std_bandpower[4,:].T * 100, linewidth=2 )
				axs[1, 1].set_ylim([0, 100])
				if electrode_vector[i_scenario,4] == 2:
					axs[1, 1].set_title('Cz', color='black')
				elif electrode_vector[i_scenario,4] == 1:
					axs[1, 1].set_title('Cz', color='blue')
				elif electrode_vector[i_scenario,4] == 0:
					axs[1, 1].set_title('Cz', color='red')

				axs[1, 2].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[1, 2].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[1, 2].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,5,:].T * 100, linewidth=1)
				axs[1, 2].plot(x_axis_vector, std_bandpower[5,:].T * 100, linewidth=2 )
				axs[1, 2].set_ylim([0, 100])
				if electrode_vector[i_scenario,5] == 2:
					axs[1, 2].set_title('C4', color='black')
				elif electrode_vector[i_scenario,5] == 1:
					axs[1, 2].set_title('C4', color='blue')
				elif electrode_vector[i_scenario,5] == 0:
					axs[1, 2].set_title('C4', color='red')

				axs[2, 0].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[2, 0].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')				
				axs[2, 0].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,6,:].T * 100, linewidth=1)
				axs[2, 0].plot(x_axis_vector, std_bandpower[6,:].T * 100 , linewidth=2)
				axs[2, 0].set_ylim([0, 100])
				if electrode_vector[i_scenario,6] == 2:
					axs[2, 0].set_title('P3', color='black')
				elif electrode_vector[i_scenario,6] == 1:
					axs[2, 0].set_title('P3', color='blue')
				elif electrode_vector[i_scenario,6] == 0:
					axs[2, 0].set_title('P3', color='red')

				axs[2, 1].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[2, 1].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[2, 1].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,7,:].T * 100, linewidth=1)
				axs[2, 1].plot(x_axis_vector, std_bandpower[7,:].T * 100 , linewidth=2)
				axs[2, 1].set_ylim([0, 100])
				if electrode_vector[i_scenario,7] == 2:
					axs[2, 1].set_title('POz', color='black')
				elif electrode_vector[i_scenario,7] == 1:
					axs[2, 1].set_title('POz', color='blue')
				elif electrode_vector[i_scenario,7] == 0:
					axs[2, 1].set_title('POz', color='red')

				axs[2, 2].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[2, 2].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--', label='_Hidden label')
				axs[2, 2].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,8,:].T * 100, linewidth=1)
				axs[2, 2].plot(x_axis_vector, std_bandpower[8,:].T * 100, linewidth=2)
				axs[2, 2].set_ylim([0, 100])
				if electrode_vector[i_scenario,8] == 2:
					axs[2, 2].set_title('P4', color='black')
				elif electrode_vector[i_scenario,8] == 1:
					axs[2, 2].set_title('P4', color='blue')
				elif electrode_vector[i_scenario,8] == 0:
					axs[2, 2].set_title('P4', color='red')
				
				for ax in axs.flat:
				    ax.set(xlabel="percent of trial", ylabel='percent of frequency power')

				# Hide x labels and tick labels for top plots and y ticks for right plots.
				for ax in axs.flat:
				    ax.label_outer()

				if plot_workload:
					if (crews_to_process[i_crew] == 'Crew_12') & (scenarios[i_scenario] == '5'):
						leg = axs[0, 2].legend(["workload", "event1","event2","theta","alpha","beta","gamma", "std"])
					elif scenarios[i_scenario] == '2':
						leg = axs[0, 2].legend(["workload","event1","event2", "theta","alpha","beta","gamma", "std"])
					else:
						leg = axs[0, 2].legend(["theta","alpha","beta","gamma", "std"])
					for line in leg.get_lines():
						line.set_linewidth(4.0)
				else:
					leg = axs[0, 2].legend(["event1","event2","theta","alpha","beta","gamma", "std"])
				for line in leg.get_lines():
					line.set_linewidth(4.0)
				
				fig.set_size_inches((22, 11))
				# plt.savefig(crew_dir + "/Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_leftseat_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0, dpi=500)
				fig.savefig("Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_'+file_types[i_seat]+'_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0)

			if plot_engagement:
				##### ENGAGE AND TASK INDEX ############
				if (i_seat == 0) & (i_scenario == 0):
					fig_eng, axs_eng = plt.subplots(len(scenarios), 1)
					fig_eng.suptitle(crews_to_process[i_crew]+'_engagement')
	
				# b, a = signal.ellip(4, 0.01, 120, 0.125)
				b, a = signal.butter(8, 0.125)
				# filtered_this_epoch_bandpower_ratio = this_epoch_bandpower_ratio
				if (np.isnan(engagement_index[i_scenario,:]).any()) | (~np.isfinite(engagement_index[i_scenario,:])).any():
					filtered_engagement_index[i_scenario,:] = engagement_index[i_scenario,:]
				else:
					# print('plotting engagement_index')
					filtered_engagement_index[i_scenario,:] = signal.filtfilt(b, a, engagement_index[i_scenario,:], method="gust")
					# filtered_engagement_index[i_scenario,:] = engagement_index[i_scenario,:]

				if file_types[i_seat] == "leftseat":
					axs_eng[i_scenario].plot(x_axis_vector, filtered_engagement_index[i_scenario,:], linewidth=3, color='blue')
				else:
					axs_eng[i_scenario].plot(x_axis_vector, filtered_engagement_index[i_scenario,:], linewidth=3, color='red')
				axs_eng[i_scenario].set_ylabel(scenarios[i_scenario])
				axs_eng[0].legend(["left engagement_index","left task_index","right engagement_index", "right task_index"])
				axs_eng[i_scenario].set_xlim((0, 100))
				axs_eng[i_scenario].set_ylim((0, 1))

				if (i_seat == 0) & (i_scenario == 0):
					fig_task, axs_task = plt.subplots(len(scenarios), 1)
					fig_task.suptitle(crews_to_process[i_crew]+'_taskLoad')
	
				# b, a = signal.ellip(4, 0.01, 120, 0.125)
				b, a = signal.butter(8, 0.125)
				# filtered_this_epoch_bandpower_ratio = this_epoch_bandpower_ratio

				if (np.isnan(taskLoad_index[i_scenario,:]).any()) | (~np.isfinite(taskLoad_index[i_scenario,:])).any():
					filtered_taskLoad_index[i_scenario,:] = taskLoad_index[i_scenario,:]
				else:
					filtered_taskLoad_index[i_scenario,:] = signal.filtfilt(b, a, taskLoad_index[i_scenario,:], method="gust")
					# filtered_taskLoad_index[i_scenario,:] = taskLoad_index[i_scenario,:]

				if file_types[i_seat] == "leftseat":
					axs_task[i_scenario].plot(x_axis_vector, filtered_taskLoad_index[i_scenario,:], linewidth=3, color="blue")
				else:
					axs_task[i_scenario].plot(x_axis_vector, filtered_taskLoad_index[i_scenario,:], linewidth=3, color="red")
				axs_task[i_scenario].set_ylabel(scenarios[i_scenario])
				axs_task[0].legend(["left task_index","right task_index"])
				axs_task[i_scenario].set_xlim((0, 100))
				axs_task[i_scenario].set_ylim((0, 50))

	if plot_engagement:
		fig_eng.set_size_inches((22, 11))
		fig_eng.savefig("Figures/" + 'eeg_engagement_'+crews_to_process[i_crew]+'_.tif',bbox_inches='tight',pad_inches=0)
		fig_task.set_size_inches((22, 11))
		fig_task.savefig("Figures/" + 'eeg_taskLoad_'+crews_to_process[i_crew]+'_.tif',bbox_inches='tight',pad_inches=0)
		# fig_eng.close()

	subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Figures"', shell=True)
	
	# plt.close()
	# plt.show()
				#############################################################