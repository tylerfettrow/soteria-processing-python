import numpy as np
import pandas as pd
import os
from os.path import exists
import mne
import matplotlib.pyplot as plt
from scipy import signal
import statistics
import numpy.matlib

# crews_to_process = [ 'Crew_12', 'Crew_13']
crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04','Crew_05', 'Crew_06','Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = [ 'Crew_12', 'Crew_13']
crews_to_process = ['Crew_02']
scenarios = ["1","2","3","5","6","7"]
# scenarios = ["1"]
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'
electrodes = [ 'F3','Fz', 'F4', 'C3','Cz', 'C4', 'P3','POz', 'P4']
file_types = ["leftseat","rightseat"]
# x_axis_vector = np.linspace(0,100,200)

plot_individual = 1
plot_workload = 1


if plot_workload:
	xls = pd.ExcelFile(path_to_project+'/SOTERIA_Survey_Spreadsheet.xlsx')
	rta_df = pd.read_excel(xls, 'Day2_RTA_Workload')
	rta_array = rta_df.to_numpy()

for i_crew in range(len(crews_to_process)):
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	this_event_data = np.load(crew_dir + "/Processing/" + 'event_vector_scenario.npy')

	for i_seat in range(len(file_types)):
		if file_types[i_seat] == "leftseat":
			eeg_timesec_epoch_storage = np.load(crew_dir + "/Processing/" + 'eeg_timesec_epoch_storage_leftseat.npy')

			########## left seat #######################
			eeg_freq_band_storage= np.load(crew_dir + "/Processing/" + 'eeg_freq_band_storage_leftseat.npy')
			# 5 (freq bands) x 9 (electrodes) x 6 (scenarios) x 200 (epochs)
			# delta # theta # alpha # beta # gamma
			
			############# TO DO.. need to load event_vector #######################
			
			# create a vector 9(electrodes) x scenarios of interest(6) to classify whether electrode is 2 = good 1=questionable  0=bad
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
			eeg_freq_band_storage = np.load(crew_dir + "/Processing/" + 'eeg_freq_band_storage_rightseat.npy')

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
			
			for this_epoch in range(eeg_freq_band_storage.shape[3]):
				### WARNING: removing delta band power here
				eeg_freq_band_storage[eeg_freq_band_storage < 0] = 0
				this_epoch_bandpower_ratio[:,:,this_epoch] = eeg_freq_band_storage[1:,:,i_scenario,this_epoch] / eeg_freq_band_storage[1:,:,i_scenario,this_epoch].sum(0)
				theta = 0
				alpha = 0
				beta = 0
				for this_electrode in range(eeg_freq_band_storage.shape[1]):    # delta # theta # alpha # beta # gamma
					theta = theta + eeg_freq_band_storage[1,:,i_scenario,this_epoch].sum(0)
					alpha = alpha + eeg_freq_band_storage[2,:,i_scenario,this_epoch].sum(0)
					beta = beta + eeg_freq_band_storage[3,:,i_scenario,this_epoch].sum(0)
				total_power = theta + alpha + beta
				# Percent power; power relative to total power
				per_power_theta = theta / total_power
				per_power_alpha = alpha / total_power
				per_power_beta = beta / total_power
				engagement_index[i_scenario,this_epoch] = per_power_beta / (per_power_alpha + per_power_theta)

			this_epoch_bandpower_ratio[np.isnan(this_epoch_bandpower_ratio)] = 0

			# filtering the bandpower sig
			b, a = signal.ellip(4, 0.01, 120, 0.125)
			# filtered_this_epoch_bandpower_ratio = this_epoch_bandpower_ratio
			filtered_this_epoch_bandpower_ratio = signal.filtfilt(b, a, this_epoch_bandpower_ratio, method="gust")

			fig, axs = plt.subplots(3, 3)
			fig.suptitle(crews_to_process[i_crew]+'_leftseat_'+scenarios[i_scenario])
			
			std_bandpower = filtered_this_epoch_bandpower_ratio.std(axis=0)

			# task load index

			    # Fz_theta = np.sum(channel_power[Fz].theta)
			    # Pz_alpha = np.sum(channel_power[Pz].alpha)
			    # return Fz_theta / Pz_alpha

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
			axs[0, 0].plot(x_axis_vector, std_bandpower[1,:].T * 100 , linewidth=2)
			'gray'
			# axs[0, 0].plot(x_axis_vector, engagement_index[1,:].T * 100 , linewidth=2)
			# axs[0, 0].plot(x_axis_vector, taskload_index[1,:].T * 100 , linewidth=2)
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
			axs[0, 1].plot(x_axis_vector, std_bandpower[0,:].T * 100 , linewidth=2)
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

			leg = axs[0,2].legend()
			if plot_workload:
				if (crews_to_process[i_crew] == 'Crew_12') & (scenarios[i_scenario] == '5'):
					leg = axs[0, 2].legend(["workload", "theta","alpha","beta","gamma", "std"])
				elif scenarios[i_scenario] == '2':
					leg = axs[0, 2].legend(["workload", "theta","alpha","beta","gamma", "std"])
				else:
					leg = axs[0, 2].legend(["theta","alpha","beta","gamma", "std"])
				for line in leg.get_lines():
					line.set_linewidth(4.0)
			else:
				leg = axs[0, 2].legend(["theta","alpha","beta","gamma", "std"])
			for line in leg.get_lines():
				line.set_linewidth(4.0)
			
			fig.set_size_inches((22, 11))
			# plt.savefig(crew_dir + "/Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_leftseat_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0, dpi=500)
			plt.savefig(crew_dir + "/Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_'+file_types[i_seat]+'_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0)
		




#####################################################################


	# for i_scenario in range(len(scenarios)):
	# 	this_epoch_bandpower_ratio = np.zeros((4,9,number_of_time_epochs))
	# 	print("Plotting: " + crew_dir + '_rightseat_scenario' + scenarios[i_scenario] + '.csv')
	# 	for this_epoch in range(eeg_freq_band_storage.shape[3]):
	# 		eeg_freq_band_storage[eeg_freq_band_storage < 0] = 0
	# 		# total_bandpower = eeg_freq_band_storage[:,:,i_scenario,this_band].sum(1)
	# 		this_epoch_bandpower_ratio[:,:,this_epoch] = eeg_freq_band_storage[1:,:,i_scenario,this_epoch] / eeg_freq_band_storage[1:,:,i_scenario,this_epoch].sum(0)
	# 		# print(i_scenario)

	# 	this_epoch_bandpower_ratio[np.isnan(this_epoch_bandpower_ratio)] = 0

	# 	# filtering the bandpower sig
	# 	b, a = signal.ellip(4, 0.01, 120, 0.125)
	# 	filtered_this_epoch_bandpower_ratio = signal.filtfilt(b, a, this_epoch_bandpower_ratio, method="gust")

	# 	fig, axs = plt.subplots(3, 3)
	# 	fig.suptitle(crews_to_process[i_crew]+'_rightseat_'+scenarios[i_scenario])
		
	# 	# for this_electrode in range(this_band_bandpower_ratio.shape[1]):
	# 	# 	plt.plot(this_band_bandpower_ratio[:,this_electrode,:].T)
	# 	# 	plt.title(electrodes[this_electrode])
	# 	# 	plt.legend(['delta', 'theta', 'alpha', 'beta', 'gamma'])
	# 	# 	plt.show()

	# 	this_event1_epoch = 0
	# 	this_event2_epoch = 0
	# 	# convert events (seconds) to band number ... assuming left and right seat for same crew are similar enough
	# 	for idx in range(0,eeg_timesec_epoch_storage.shape[1]):
	# 		# print(idx)
	# 		if this_event_data[0, i_scenario] <= eeg_timesec_epoch_storage[i_scenario, idx]:
	# 			this_event1_epoch = np.floor((idx - 1)/(number_of_time_epochs/100)) 
	# 			break
	# 	for idx in range(0,eeg_timesec_epoch_storage.shape[1]):
	# 		# print(idx)
	# 		if this_event_data[1, i_scenario] <= eeg_timesec_epoch_storage[i_scenario, idx]:
	# 			this_event2_epoch = np.floor((idx - 1)/(number_of_time_epochs/100))
	# 			break

	# 	std_bandpower = filtered_this_epoch_bandpower_ratio.std(axis=0)

	# 	if plot_workload:
	# 		rta_vector = rta_vector[~pd.isna(rta_vector)]
	# 		if (crews_to_process[i_crew] == 'Crew_12') & (scenarios[i_scenario] == '5'):
	# 			# check to see if the number of rta inputs match the eeg_timesec_storage
	# 			expected_length_of_data = eeg_timesec_epoch_storage[3,-1]/60
	# 			if len(rta_vector) != int(np.floor(expected_length_of_data)):
	# 				print('WARNING: RTA workload does not match trial length')
	# 				print('rta = '+ str(len(rta_vector)) + ' eeg: ' + str(int(np.floor(expected_length_of_data))))
	# 			# axs[0,0].plot(, linewidth=1)
	# 		elif scenarios[i_scenario] == '2':
	# 			expected_length_of_data = eeg_timesec_epoch_storage[1,-1]/60
	# 			if len(rta_vector) != int(np.floor(expected_length_of_data)):
	# 				print('WARNING: RTA workload does not match trial length')
	# 				print('rta = '+ str(len(rta_vector)) + ' eeg: ' + str(np.floor(expected_length_of_data)))
	# 			rta_vector_timewarped = np.repeat(rta_vector,int(number_of_time_epochs/len(rta_vector)))
	# 			if len(rta_vector_timewarped) < number_of_time_epochs:
	# 				diff = number_of_time_epochs - len(rta_vector_timewarped)
	# 				rta_vector_timewarped = np.append(rta_vector_timewarped,np.ones(diff)*rta_vector_timewarped[-1])
	# 			axs[0,0].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[0,1].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[0,2].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[1,0].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[1,1].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[1,2].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[2,0].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[2,1].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)
	# 			axs[2,2].plot(x_axis_vector, rta_vector_timewarped * 10, linewidth=1)

	# 	axs[0, 0].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,0,:].T * 100, linewidth=1)
	# 	axs[0, 0].plot(x_axis_vector, std_bandpower[0,:].T * 100, linewidth=2 )
	# 	axs[0, 0].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[0, 0].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[0, 0].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,0] == 2:
	# 		axs[0, 0].set_title('F3', color='black')
	# 	elif electrode_vector[i_scenario,0] == 1:
	# 		axs[0, 0].set_title('F3', color='blue')
	# 	elif electrode_vector[i_scenario,0] == 0:
	# 		axs[0, 0].set_title('F3', color='red')

	# 	axs[0, 1].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,1,:].T * 100, linewidth=1)
	# 	axs[0, 1].plot(x_axis_vector, std_bandpower[1,:].T * 100, linewidth=2 )
	# 	axs[0, 1].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[0, 1].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[0, 1].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,1] == 2:
	# 		axs[0, 1].set_title('Fz', color='black')
	# 	elif electrode_vector[i_scenario,1] == 1:
	# 		axs[0, 1].set_title('Fz', color='blue')
	# 	elif electrode_vector[i_scenario,1] == 0:
	# 		axs[0, 1].set_title('Fz', color='red')

	# 	axs[0, 2].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,2,:].T * 100, linewidth=1)
	# 	axs[0, 2].plot(x_axis_vector, std_bandpower[2,:].T * 100, linewidth=2 )
	# 	axs[0, 2].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[0, 2].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[0, 2].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,2] == 2:
	# 		axs[0, 2].set_title('F4', color='black')
	# 	elif electrode_vector[i_scenario,2] == 1:
	# 		axs[0, 2].set_title('F4', color='blue')
	# 	elif electrode_vector[i_scenario,2] == 0:
	# 		axs[0, 2].set_title('F4', color='red')

	# 	if plot_workload:
	# 		leg = axs[0, 2].legend(["workload", "theta","alpha","beta","gamma", "std"])
	# 	else
	# 		leg = axs[0, 2].legend(["theta","alpha","beta","gamma", "std"])
	# 	for line in leg.get_lines():
	# 		line.set_linewidth(4.0)

	# 	axs[1, 0].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,3,:].T * 100, linewidth=1)
	# 	axs[1, 0].plot(x_axis_vector, std_bandpower[3,:].T * 100, linewidth=2 )
	# 	axs[1, 0].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[1, 0].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[1, 0].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,3] == 2:
	# 		axs[1, 0].set_title('C3', color='black')
	# 	elif electrode_vector[i_scenario,3] == 1:
	# 		axs[1, 0].set_title('C3', color='blue')
	# 	elif electrode_vector[i_scenario,3] == 0:
	# 		axs[1, 0].set_title('C3', color='red')
	# 	axs[1, 1].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,4,:].T * 100, linewidth=1)
	# 	axs[1, 1].plot(x_axis_vector, std_bandpower[4,:].T * 100, linewidth=2 )
	# 	axs[1, 1].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[1, 1].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[1, 1].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,4] == 2:
	# 		axs[1, 1].set_title('Cz', color='black')
	# 	elif electrode_vector[i_scenario,4] == 1:
	# 		axs[1, 1].set_title('Cz', color='blue')
	# 	elif electrode_vector[i_scenario,4] == 0:
	# 		axs[1, 1].set_title('Cz', color='red')

	# 	axs[1, 2].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,5,:].T * 100, linewidth=1)
	# 	axs[1, 2].plot(x_axis_vector, std_bandpower[5,:].T * 100, linewidth=2)
	# 	axs[1, 2].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[1, 2].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[1, 2].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,5] == 2:
	# 		axs[1, 2].set_title('C4', color='black')
	# 	elif electrode_vector[i_scenario,5] == 1:
	# 		axs[1, 2].set_title('C4', color='blue')
	# 	elif electrode_vector[i_scenario,5] == 0:
	# 		axs[1, 2].set_title('C4', color='red')

	# 	axs[2, 0].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,6,:].T * 100, linewidth=1)
	# 	axs[2, 0].plot(x_axis_vector, std_bandpower[6,:].T * 100, linewidth=2 )
	# 	axs[2, 0].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[2, 0].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[2, 0].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,6] == 2:
	# 		axs[2, 0].set_title('P3', color='black')
	# 	elif electrode_vector[i_scenario,6] == 1:
	# 		axs[2, 0].set_title('P3', color='blue')
	# 	elif electrode_vector[i_scenario,6] == 0:
	# 		axs[2, 0].set_title('P3', color='red')

	# 	axs[2, 1].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,7,:].T * 100, linewidth=1)
	# 	axs[2, 1].plot(x_axis_vector, std_bandpower[7,:].T * 100, linewidth=2 )
	# 	axs[2, 1].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[2, 1].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[2, 1].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,7] == 2:
	# 		axs[2, 1].set_title('POz', color='black')
	# 	elif electrode_vector[i_scenario,7] == 1:
	# 		axs[2, 1].set_title('POz', color='blue')
	# 	elif electrode_vector[i_scenario,7] == 0:
	# 		axs[2, 1].set_title('POz', color='red')

	# 	axs[2, 2].plot(x_axis_vector, filtered_this_epoch_bandpower_ratio[:,8,:].T * 100, linewidth=1)
	# 	axs[2, 2].plot(x_axis_vector, std_bandpower[8,:].T * 100, linewidth=2 )
	# 	axs[2, 2].axvline(x = this_event1_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[2, 2].axvline(x = this_event2_epoch, ymin = 0, ymax = 100, color = 'gray',linestyle='--')
	# 	axs[2, 2].set_ylim([0, 100])
	# 	if electrode_vector[i_scenario,8] == 2:
	# 		axs[2, 2].set_title('P4', color='black')
	# 	elif electrode_vector[i_scenario,8] == 1:
	# 		axs[2, 2].set_title('P4', color='blue')
	# 	elif electrode_vector[i_scenario,8] == 0:
	# 		axs[2, 2].set_title('P4', color='red')

	# 	for ax in axs.flat:
	# 	    ax.set(xlabel="percent of trial", ylabel='percent of frequency power')

	# 	# Hide x labels and tick labels for top plots and y ticks for right plots.
	# 	for ax in axs.flat:
	# 	    ax.label_outer()
	# 	# fig.set_size_inches((22, 11))
	# 	# plt.savefig(crew_dir + "/Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_rightseat_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0, dpi=500)
	# 	plt.savefig(crew_dir + "/Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_rightseat_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0)
	# 	# Instead of plotting individual tables (embed percent text in subplot or in title of subplot?)
	# 	#matplotlib.pyplot.close()    
	# 	# plt.show()