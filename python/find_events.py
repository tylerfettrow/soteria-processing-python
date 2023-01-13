import numpy as np
import os
from os.path import exists
import mne
import matplotlib.pyplot as plt
import pandas as pd
import math 

crews_to_process = ['Crew_01', 'Crew_02', 'Crew_03', 'Crew_04', 'Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_12', 'Crew_13']
# crews_to_process = ['Crew_01']
scenarios = ["1","2","3","5","6","7"]
path_to_project = 'C:/Users/tfettrow/Box/SOTERIA'

for i_crew in range(len(crews_to_process)):
	event_vector_timesec = np.zeros((2,len(scenarios)))
	crew_dir = path_to_project + '/' + crews_to_process[i_crew]
	process_dir_name = crew_dir + "/Processing/"

	for i_scenario in range(len(scenarios)):

		print("finding events: " + process_dir_name + 'scenario'+scenarios[i_scenario])

		if scenarios[i_scenario] == "1":
			event_vector_timesec[:,i_scenario] = [130, 865]
		if scenarios[i_scenario] == "2":
			event_vector_timesec[:,i_scenario] = [135, 600]
		if scenarios[i_scenario] == "3":
			# - tailwind (speed)
			# - autopilot A disengage
			# event_vector_timesec[:,i_scenario] = np.array([
			event_1_idx = 0
			if exists(process_dir_name + 'ifd_cockpit_scenario' + scenarios[i_scenario] + '.csv'):
				ifd_data = pd.read_table((process_dir_name + 'ifd_cockpit_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
				diff_alt = np.diff(ifd_data.altitude)
				for idx in range(0, len(diff_alt)):
					if diff_alt[idx] <= -.3:
						event_1_idx = idx
						break

				event_vector_timesec[:,i_scenario] = [ifd_data.UserTimeStamp[event_1_idx], 540]
		if scenarios[i_scenario] == "5":
			event_1_idx = 0
			event_2_idx = 0
			if exists(process_dir_name +'ifd_cockpit_scenario'+scenarios[i_scenario] + '.csv'):
				ifd_data = pd.read_table((process_dir_name + 'ifd_cockpit_scenario'+scenarios[i_scenario] + '.csv'),delimiter=',')
				roll_dps = ifd_data.roll_angle_rate_dps
				for idx in range(0,len(roll_dps)):
					if roll_dps[idx] <= -10:
						event_1_idx = idx
						break

				lat = ifd_data.latitude
				lon = ifd_data.longitude
				lat_t = 34.89083
				lon_t = -80.4755
				lat_diff = (lat - lat_t) * 60 #deg->NM #WARNING: not accounting for sphere
				lon_diff = (lon - lon_t) * 60 #deg->NM #WARNING: not accounting for sphere
				rad_vector = np.zeros((len(lat_diff)))
				for idx in range(0,len(lat_diff)):
					this_rad = math.sqrt(pow(lat_diff[idx], 2) + pow(lon_diff[idx], 2))
					# rad_vector[idx] = math.sqrt(pow(lat_diff[idx], 2) + pow(lon_diff[idx], 2))
					if this_rad <= 10:
						event_2_idx = idx
						break

				# plt.plot(lat_diff, lon_diff)


				# - roll_dps
				# - lat/lon from mangy
				# tru 1 fail lat: 34.89083
				# tru 1 fail lon: -80.4755
				# tru 1 fail radius: 10.0
				if crews_to_process[i_crew] == 'Crew_04':
					event_vector_timesec[:,i_scenario] = [ifd_data.UserTimeStamp[event_1_idx], 0]
				elif crews_to_process[i_crew] == 'Crew_05':
					event_vector_timesec[:,i_scenario] = [0, 0]
				elif crews_to_process[i_crew] == 'Crew_13':
					event_vector_timesec[:,i_scenario] = [0, 0]
				else:
					event_vector_timesec[:,i_scenario] = [ifd_data.UserTimeStamp[event_1_idx], ifd_data.UserTimeStamp[event_2_idx]]
				
		if scenarios[i_scenario] == "6":
			event_vector_timesec[:,i_scenario] = [30, 370]
		
		np.save(process_dir_name + 'event_vector_scenario', event_vector_timesec)
