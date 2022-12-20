import numpy as np
import os
from os.path import exists
import mne
import matplotlib.pyplot as plt
from scipy import signal
import statistics

# crews_to_process = [ 'Crew_12', 'Crew_13']
crews_to_process = ['Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = [ 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_06']
scenarios = ["1","2","3","5","6","7"]
# scenarios = ["1"]
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'
electrodes = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'POz', 'P3', 'P4']
x_axis_vector = np.linspace(0,100,200)

plot_individual = 1
plot_group = 0


for i_crew in range(len(crews_to_process)):
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	this_event_data = np.load(crew_dir + "/Processing/" + 'event_vector_scenario.npy')

	eeg_timesec_band_storage = np.load(crew_dir + "/Processing/" + 'eeg_timesec_band_storage_leftseat.npy')
	#time_vector = pd.read_csv((crew_dir + "/Processing/" + 'ifd_cockpit_scenario' + scenarios[i_scenario]+ '.csv'), usecols = ['UserTimeStamp'])

	########## left seat #######################
	this_data_left = np.load(crew_dir + "/Processing/" + 'eeg_freq_band_storage_leftseat.npy')



	############# TO DO.. need to load event_vector #######################
	
	# create a vector 9(electrodes) x scenarios of interest(6) to classify whether electrode is 2 = good 1=questionable  0=bad
	if crews_to_process[i_crew] == 'Crew_01':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_02':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_03':
		electrode_vector = np.array([[ 1, 2, 2, 2, 2, 2, 2, 0, 2], [0, 2, 2, 2, 2, 1, 2, 0, 2], [0, 2, 2, 2, 2, 1, 2, 0, 2], [0, 2, 2, 2, 2, 0, 2, 0, 2], [0, 2, 2, 2, 2, 0, 2, 0, 2], [2, 2, 2, 2, 2, 2, 2, 0, 2]])
	if crews_to_process[i_crew] == 'Crew_04':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_05':
		electrode_vector = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 0, 2, 2, 2, 0], [2, 2, 2, 2, 2, 2, 2, 2, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_06':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_07':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_08':
		electrode_vector = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
	if crews_to_process[i_crew] == 'Crew_09':
		electrode_vector = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
	if crews_to_process[i_crew] == 'Crew_10':
		electrode_vector = np.array([[ 0, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_11':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_12':
		electrode_vector = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_13':
		electrode_vector = np.array([[ 0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])


	for i_scenario in range(len(scenarios)):
		this_band_bandpower_ratio = np.zeros((5,9,200))
		print("Plotting: " + crew_dir + '_leftseat_scenario' + scenarios[i_scenario] + '.csv')

		this_event1_band = 0
		this_event2_band = 0
		# convert events (seconds) to band number ... assuming left and right seat for same crew are similar enough
		for idx in range(0,eeg_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[0, i_scenario] <= eeg_timesec_band_storage[i_scenario, idx]:
				this_event1_band = np.floor((idx - 1)/2) 
				break
		for idx in range(0,eeg_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[1, i_scenario] <= eeg_timesec_band_storage[i_scenario, idx]:
				this_event2_band = np.floor((idx - 1)/2)
				break

		
		for this_band in range(this_data_left.shape[3]):
			this_data_left[this_data_left < 0] = 0
			# total_bandpower = this_data_left[:,:,i_scenario,this_band].sum(1)
			this_band_bandpower_ratio[:,:,this_band] = this_data_left[:,:,i_scenario,this_band] / this_data_left[:,:,i_scenario,this_band].sum(0)
			# print(i_scenario)

		this_band_bandpower_ratio[np.isnan(this_band_bandpower_ratio)] = 0

		# filtering the bandpower sig
		b, a = signal.ellip(4, 0.01, 120, 0.125)
		filtered_this_band_bandpower_ratio = signal.filtfilt(b, a, this_band_bandpower_ratio, method="gust")

		fig, axs = plt.subplots(3, 3)
		fig.suptitle(crews_to_process[i_crew]+'_leftseat_'+scenarios[i_scenario])
		
		# for this_electrode in range(this_band_bandpower_ratio.shape[1]):
		# 	plt.plot(this_band_bandpower_ratio[:,this_electrode,:].T)
		# 	plt.title(electrodes[this_electrode])
		# 	plt.legend(['delta', 'theta', 'alpha', 'beta', 'gamma'])
		# 	plt.show()

		std_bandpower = filtered_this_band_bandpower_ratio.std(axis=0)

		axs[0, 0].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,0,:].T * 100)
		axs[0, 0].plot(x_axis_vector, std_bandpower[0,:].T * 100 , linewidth=4)
		axs[0, 0].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 0].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 0].set_ylim([0, 100])

		if electrode_vector[i_scenario,0] == 2:
			axs[0, 0].set_title('Fz', color='black')
		elif electrode_vector[i_scenario,0] == 1:
			axs[0, 0].set_title('Fz', color='blue')
		elif electrode_vector[i_scenario,0] == 0:
			axs[0, 0].set_title('Fz', color='red')
		axs[0, 1].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,1,:].T * 100)
		axs[0, 1].plot(x_axis_vector, std_bandpower[1,:].T * 100 , linewidth=4)
		axs[0, 1].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 1].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 1].set_ylim([0, 100])
		if electrode_vector[i_scenario,1] == 2:
			axs[0, 1].set_title('F3', color='black')
		elif electrode_vector[i_scenario,1] == 1:
			axs[0, 1].set_title('F3', color='blue')
		elif electrode_vector[i_scenario,1] == 0:
			axs[0, 1].set_title('F3', color='red')
		axs[0, 2].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,2,:].T * 100)
		axs[0, 2].plot(x_axis_vector, std_bandpower[2,:].T * 100, linewidth=4 )
		axs[0, 2].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 2].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 2].set_ylim([0, 100])
		
		leg = axs[0, 2].legend(["delta","theta","alpha","beta","gamma"])
		for line in leg.get_lines():
			line.set_linewidth(4.0)
		
		if electrode_vector[i_scenario,2] == 2:
			axs[0, 2].set_title('F4', color='black')
		elif electrode_vector[i_scenario,2] == 1:
			axs[0, 2].set_title('F4', color='blue')
		elif electrode_vector[i_scenario,2] == 0:
			axs[0, 2].set_title('F4', color='red')
		axs[1, 0].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,3,:].T * 100)
		axs[1, 0].plot(x_axis_vector, std_bandpower[3,:].T * 100, linewidth=4)
		axs[1, 0].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 0].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 0].set_ylim([0, 100])
		if electrode_vector[i_scenario,3] == 2:
			axs[1, 0].set_title('Cz', color='black')
		elif electrode_vector[i_scenario,3] == 1:
			axs[1, 0].set_title('Cz', color='blue')
		elif electrode_vector[i_scenario,3] == 0:
			axs[1, 0].set_title('Cz', color='red')
		axs[1, 1].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,4,:].T * 100)
		axs[1, 1].plot(x_axis_vector, std_bandpower[4,:].T * 100, linewidth=4 )
		axs[1, 1].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 1].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 1].set_ylim([0, 100])
		if electrode_vector[i_scenario,4] == 2:
			axs[1, 1].set_title('C3', color='black')
		elif electrode_vector[i_scenario,4] == 1:
			axs[1, 1].set_title('C3', color='blue')
		elif electrode_vector[i_scenario,4] == 0:
			axs[1, 1].set_title('C3', color='red')
		axs[1, 2].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,5,:].T * 100)
		axs[1, 2].plot(x_axis_vector, std_bandpower[5,:].T * 100, linewidth=4 )
		axs[1, 2].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 2].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 2].set_ylim([0, 100])
		if electrode_vector[i_scenario,5] == 2:
			axs[1, 2].set_title('C4', color='black')
		elif electrode_vector[i_scenario,5] == 1:
			axs[1, 2].set_title('C4', color='blue')
		elif electrode_vector[i_scenario,5] == 0:
			axs[1, 2].set_title('C4', color='red')
		axs[2, 0].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,6,:].T * 100)
		axs[2, 0].plot(x_axis_vector, std_bandpower[6,:].T * 100 , linewidth=4)
		axs[2, 0].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 0].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 0].set_ylim([0, 100])
		if electrode_vector[i_scenario,6] == 2:
			axs[2, 0].set_title('POz', color='black')
		elif electrode_vector[i_scenario,6] == 1:
			axs[2, 0].set_title('POz', color='blue')
		elif electrode_vector[i_scenario,6] == 0:
			axs[2, 0].set_title('POz', color='red')
		axs[2, 1].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,7,:].T * 100)
		axs[2, 1].plot(x_axis_vector, std_bandpower[7,:].T * 100 , linewidth=4)
		axs[2, 1].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 1].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 1].set_ylim([0, 100])
		if electrode_vector[i_scenario,7] == 2:
			axs[2, 1].set_title('P3', color='black')
		elif electrode_vector[i_scenario,7] == 1:
			axs[2, 1].set_title('P3', color='blue')
		elif electrode_vector[i_scenario,7] == 0:
			axs[2, 1].set_title('P3', color='red')
		axs[2, 2].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,8,:].T * 100)
		axs[2, 2].plot(x_axis_vector, std_bandpower[8,:].T * 100, linewidth=4)
		axs[2, 2].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 2].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
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
		fig.set_size_inches((22, 11))
		plt.savefig(crew_dir + "/Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_leftseat_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0, dpi=500)
		# Instead of plotting individual tables (embed percent text in subplot or in title of subplot?)
		#matplotlib.pyplot.close()    
		# plt.show()



	########## right seat #######################
	this_data_right = np.load(crew_dir + "/Processing/" + 'eeg_freq_band_storage_rightseat.npy')

	if crews_to_process[i_crew] == 'Crew_01':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_02':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_03':
		electrode_vector = np.array([[ 0, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 2, 2, 2, 2, 2, 2, 2], [0, 0, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_04':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
	if crews_to_process[i_crew] == 'Crew_05':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_06':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
	if crews_to_process[i_crew] == 'Crew_07':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_08':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_09':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
	if crews_to_process[i_crew] == 'Crew_10':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 0, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 0, 2, 2], [1, 1, 1, 1, 1, 1, 0, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_11':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_12':
		electrode_vector = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 2, 2, 2, 2, 2, 2], [2, 0, 0, 2, 2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 2, 2, 2, 2, 2, 2]])
	if crews_to_process[i_crew] == 'Crew_13':
		electrode_vector = np.array([[ 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]])

	for i_scenario in range(len(scenarios)):
		this_band_bandpower_ratio = np.zeros((5,9,200))
		print("Plotting: " + crew_dir + '_rightseat_scenario' + scenarios[i_scenario] + '.csv')
		for this_band in range(this_data_left.shape[3]):
			this_data_right[this_data_right < 0] = 0
			# total_bandpower = this_data_left[:,:,i_scenario,this_band].sum(1)
			this_band_bandpower_ratio[:,:,this_band] = this_data_right[:,:,i_scenario,this_band] / this_data_right[:,:,i_scenario,this_band].sum(0)
			# print(i_scenario)

		this_band_bandpower_ratio[np.isnan(this_band_bandpower_ratio)] = 0

		# filtering the bandpower sig
		b, a = signal.ellip(4, 0.01, 120, 0.125)
		filtered_this_band_bandpower_ratio = signal.filtfilt(b, a, this_band_bandpower_ratio, method="gust")

		fig, axs = plt.subplots(3, 3)
		fig.suptitle(crews_to_process[i_crew]+'_rightseat_'+scenarios[i_scenario])
		
		# for this_electrode in range(this_band_bandpower_ratio.shape[1]):
		# 	plt.plot(this_band_bandpower_ratio[:,this_electrode,:].T)
		# 	plt.title(electrodes[this_electrode])
		# 	plt.legend(['delta', 'theta', 'alpha', 'beta', 'gamma'])
		# 	plt.show()

		this_event1_band = 0
		this_event2_band = 0
		# convert events (seconds) to band number ... assuming left and right seat for same crew are similar enough
		for idx in range(0,eeg_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[0, i_scenario] <= eeg_timesec_band_storage[i_scenario, idx]:
				this_event1_band = np.floor((idx - 1)/2) 
				break
		for idx in range(0,eeg_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[1, i_scenario] <= eeg_timesec_band_storage[i_scenario, idx]:
				this_event2_band = np.floor((idx - 1)/2)
				break

		std_bandpower = filtered_this_band_bandpower_ratio.std(axis=0)

		axs[0, 0].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,0,:].T * 100)
		axs[0, 0].plot(x_axis_vector, std_bandpower[0,:].T * 100, linewidth=4 )
		axs[0, 0].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 0].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 0].set_ylim([0, 100])
		if electrode_vector[i_scenario,0] == 2:
			axs[0, 0].set_title('Fz', color='black')
		elif electrode_vector[i_scenario,0] == 1:
			axs[0, 0].set_title('Fz', color='blue')
		elif electrode_vector[i_scenario,0] == 0:
			axs[0, 0].set_title('Fz', color='red')
		axs[0, 1].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,1,:].T * 100)
		axs[0, 1].plot(x_axis_vector, std_bandpower[1,:].T * 100, linewidth=4 )
		axs[0, 1].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 1].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 1].set_ylim([0, 100])
		if electrode_vector[i_scenario,1] == 2:
			axs[0, 1].set_title('F3', color='black')
		elif electrode_vector[i_scenario,1] == 1:
			axs[0, 1].set_title('F3', color='blue')
		elif electrode_vector[i_scenario,1] == 0:
			axs[0, 1].set_title('F3', color='red')
		axs[0, 2].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,2,:].T * 100)
		axs[0, 2].plot(x_axis_vector, std_bandpower[2,:].T * 100, linewidth=4 )
		axs[0, 2].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 2].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[0, 2].set_ylim([0, 100])
		if electrode_vector[i_scenario,2] == 2:
			axs[0, 2].set_title('F4', color='black')
		elif electrode_vector[i_scenario,2] == 1:
			axs[0, 2].set_title('F4', color='blue')
		elif electrode_vector[i_scenario,2] == 0:
			axs[0, 2].set_title('F4', color='red')
		axs[1, 0].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,3,:].T * 100)
		axs[1, 0].plot(x_axis_vector, std_bandpower[3,:].T * 100, linewidth=4 )
		axs[1, 0].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 0].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 0].set_ylim([0, 100])
		if electrode_vector[i_scenario,3] == 2:
			axs[1, 0].set_title('Cz', color='black')
		elif electrode_vector[i_scenario,3] == 1:
			axs[1, 0].set_title('Cz', color='blue')
		elif electrode_vector[i_scenario,3] == 0:
			axs[1, 0].set_title('Cz', color='red')
		axs[1, 1].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,4,:].T * 100)
		axs[1, 1].plot(x_axis_vector, std_bandpower[4,:].T * 100, linewidth=4 )
		axs[1, 1].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 1].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 1].set_ylim([0, 100])
		if electrode_vector[i_scenario,4] == 2:
			axs[1, 1].set_title('C3', color='black')
		elif electrode_vector[i_scenario,4] == 1:
			axs[1, 1].set_title('C3', color='blue')
		elif electrode_vector[i_scenario,4] == 0:
			axs[1, 1].set_title('C3', color='red')
		axs[1, 2].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,5,:].T * 100)
		axs[1, 2].plot(x_axis_vector, std_bandpower[5,:].T * 100, linewidth=4 )
		axs[1, 2].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 2].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[1, 2].set_ylim([0, 100])
		if electrode_vector[i_scenario,5] == 2:
			axs[1, 2].set_title('C4', color='black')
		elif electrode_vector[i_scenario,5] == 1:
			axs[1, 2].set_title('C4', color='blue')
		elif electrode_vector[i_scenario,5] == 0:
			axs[1, 2].set_title('C4', color='red')
		axs[2, 0].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,6,:].T * 100)
		axs[2, 0].plot(x_axis_vector, std_bandpower[6,:].T * 100, linewidth=4 )
		axs[2, 0].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 0].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 0].set_ylim([0, 100])
		if electrode_vector[i_scenario,6] == 2:
			axs[2, 0].set_title('POz', color='black')
		elif electrode_vector[i_scenario,6] == 1:
			axs[2, 0].set_title('POz', color='blue')
		elif electrode_vector[i_scenario,6] == 0:
			axs[2, 0].set_title('POz', color='red')
		axs[2, 1].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,7,:].T * 100)
		axs[2, 1].plot(x_axis_vector, std_bandpower[7,:].T * 100, linewidth=4 )
		axs[2, 1].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 1].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 1].set_ylim([0, 100])
		if electrode_vector[i_scenario,7] == 2:
			axs[2, 1].set_title('P3', color='black')
		elif electrode_vector[i_scenario,7] == 1:
			axs[2, 1].set_title('P3', color='blue')
		elif electrode_vector[i_scenario,7] == 0:
			axs[2, 1].set_title('P3', color='red')
		axs[2, 2].plot(x_axis_vector, filtered_this_band_bandpower_ratio[:,8,:].T * 100)
		axs[2, 2].plot(x_axis_vector, std_bandpower[8,:].T * 100, linewidth=4 )
		axs[2, 2].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[2, 2].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
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
		fig.set_size_inches((22, 11))
		plt.savefig(crew_dir + "/Figures/" + 'eeg_powerbands_'+crews_to_process[i_crew]+'_rightseat_scenario'+scenarios[i_scenario]+'.tif',bbox_inches='tight',pad_inches=0, dpi=500)
		# Instead of plotting individual tables (embed percent text in subplot or in title of subplot?)
		#matplotlib.pyplot.close()    
		# plt.show()