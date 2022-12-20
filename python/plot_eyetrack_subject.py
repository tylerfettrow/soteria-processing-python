import numpy as np
import os
from os.path import exists
import mne
import matplotlib.pyplot as plt
from scipy import signal
import statistics

# crews_to_process = ['Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_01', 'Crew_02', 'Crew_13']
crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04']
scenarios = ["1","2","3","5","6","7"]

path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'
x_axis_vector = np.linspace(0,100,1000)

for i_crew in range(len(crews_to_process)):
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	process_dir_name = crew_dir + "/Processing/"

	pupild_leftseat = np.load(crew_dir + "/Processing/" + 'smarteye_pupild_leftseat.npy')
	pupild_rightseat = np.load(crew_dir + "/Processing/" + 'smarteye_pupild_rightseat.npy')

	headHeading_leftseat = np.load(crew_dir + "/Processing/" + 'smarteye_headHeading_leftseat.npy')
	headHeading_rightseat = np.load(crew_dir + "/Processing/" + 'smarteye_headHeading_rightseat.npy')

	this_event_data = np.load(crew_dir + "/Processing/" + 'event_vector_scenario.npy')

	smarteye_timesec_band_storage = np.load(crew_dir + "/Processing/" + 'smarteye_timesec_band_storage.npy')
	
	plt.style.use('classic')

	fig, axs = plt.subplots(len(scenarios),1)
	fig.suptitle(crews_to_process[i_crew])
	for i_scenario in range(len(scenarios)):
		# convert events (seconds) to band number ... assuming left and right seat for same crew are similar enough
		for idx in range(0,smarteye_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[0, i_scenario] <= smarteye_timesec_band_storage[i_scenario, idx]:
				this_event1_band = np.floor((idx - 1)/10) 
				break
		for idx in range(0,smarteye_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[1, i_scenario] <= smarteye_timesec_band_storage[i_scenario, idx]:
				this_event2_band = np.floor((idx - 1)/10)
				break

		axs[i_scenario].plot(x_axis_vector, pupild_leftseat[:,i_scenario])
		axs[i_scenario].plot(x_axis_vector,  pupild_rightseat[:,i_scenario])
		axs[i_scenario].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[i_scenario].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[i_scenario].set_ylim([0, .02])
		axs[i_scenario].set_xlim([0, 100])
		axs[i_scenario].set_title('scenario ' + scenarios[i_scenario])
		axs[i_scenario].text(5, 0.018,  np.floor(smarteye_timesec_band_storage[i_scenario, 999]), fontsize='small')
	leg = axs[0].legend(["left seat","right seat"])
	for line in leg.get_lines():
		line.set_linewidth(4.0)
	for ax in axs.flat:
	    ax.set(xlabel="percent of trial", ylabel='pupil diameter')
	for ax in axs.flat:
	    ax.label_outer()
		
	fig.set_size_inches((8.5, 11))
	plt.savefig(crew_dir + "/Figures/" + 'smarteye_pupild_' + crews_to_process[i_crew] + '.tif',bbox_inches='tight',pad_inches=0, dpi=500)



	fig, axs = plt.subplots(len(scenarios),1)
	fig.suptitle(crews_to_process[i_crew])
	for i_scenario in range(len(scenarios)):
		# convert events (seconds) to band number ... assuming left and right seat for same crew are similar enough
		for idx in range(0,smarteye_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[0, i_scenario] <= smarteye_timesec_band_storage[i_scenario, idx]:
				this_event1_band = np.floor((idx - 1)/10) 
				break
		for idx in range(0,smarteye_timesec_band_storage.shape[1]):
			# print(idx)
			if this_event_data[1, i_scenario] <= smarteye_timesec_band_storage[i_scenario, idx]:
				this_event2_band = np.floor((idx - 1)/10)
				break

		axs[i_scenario].plot(x_axis_vector, headHeading_leftseat[:,i_scenario])
		axs[i_scenario].plot(x_axis_vector, headHeading_rightseat[:,i_scenario])
		axs[i_scenario].axvline(x = this_event1_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[i_scenario].axvline(x = this_event2_band, ymin = 0, ymax = 100, color = 'b',linestyle='--')
		axs[i_scenario].set_ylim([-2, 2])
		axs[i_scenario].set_xlim([0, 100])
		axs[i_scenario].set_title('scenario ' + scenarios[i_scenario])
		axs[i_scenario].text(5, 1.5, np.floor(smarteye_timesec_band_storage[i_scenario, 999]), fontsize='small')
	leg = axs[0].legend(["left seat","right seat"])
	for line in leg.get_lines():
		line.set_linewidth(4.0)
	for ax in axs.flat:
	    ax.set(xlabel="percent of trial", ylabel='Heading Heading')
	for ax in axs.flat:
	    ax.label_outer()
		
	fig.set_size_inches((8.5, 11))
	plt.savefig(crew_dir + "/Figures/" + 'smarteye_headHeading_' + crews_to_process[i_crew] + '.tif',bbox_inches='tight',pad_inches=0, dpi=500)
		