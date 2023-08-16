import os
import numpy as np
import numpy.matlib
import pandas as pd
from os.path import exists
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import seaborn as sns; sns.set_theme()
import matplotlib.gridspec as gridspec
import subprocess
import time
from google.cloud import storage
from numpy import linalg as la
from os.path import exists
from distinctipy import distinctipy
from tensorflow.python.lib.io import file_io
import io
import math
import statistics
import matplotlib.colors as colors

def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

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

def sphere_stereograph(p):
	# p = np.squeeze(direction_gaze[:,good_indices])

	[m,n]=p.shape

	s = np.divide(2.0, ( 1.0 + p[m-1,0:n] ))
	ss = numpy.matlib.repmat( s, m, 1 )

	f = np.zeros((m,1))
	f[m-1] = -1.0
	ff = numpy.matlib.repmat( f, 1, n )

	q = np.multiply(ss, p) + np.multiply(( 1.0 - ss ), ff);

	b = q[0:2,:]

	return b


def angle_diff(time_vector,direction_gaze):
	degree_per_sec_vector = np.zeros(direction_gaze.shape[1])

	time_diff = np.diff(time_vector)
	for this_frame in range(direction_gaze.shape[1]-1):
		if time_diff[this_frame] != 0.0:
			degree_diff_vector = math.degrees(2 * math.atan(la.norm(np.multiply(direction_gaze[:,this_frame],la.norm(direction_gaze[:,this_frame+1])) - np.multiply(la.norm(direction_gaze[:,this_frame]),direction_gaze[:,this_frame+1])) / la.norm(np.multiply(direction_gaze[:,this_frame], la.norm(direction_gaze[:,this_frame+1])) + np.multiply(la.norm(direction_gaze[:,this_frame]), direction_gaze[:,this_frame+1]))))
			degree_per_sec_vector[this_frame+1] = degree_diff_vector / time_diff[this_frame]
		else:
			degree_per_sec_vector[this_frame+1] = 0
	return degree_per_sec_vector


def mean(a, n):
 
    # Calculating sum
    sum = 0;
    for i in range(n):
        for j in range(n):
            sum += a[i][j];
     
    # Returning mean
    return sum / (n * n);
 
# Function for calculating variance
def variance(a, n, m):
    sum = 0;
    for i in range(n):
        for j in range(n):
 
            # subtracting mean
            # from elements
            a[i][j] -= m;
 
            # a[i][j] = fabs(a[i][j]);
            # squaring each terms
            a[i][j] *= a[i][j];
 
    # taking sum
    for i in range(n):
        for j in range(n):
            sum += a[i][j];
 
    return sum / (n * n);	

crews_to_process = ['Crew_01','Crew_02','Crew_03', 'Crew_04','Crew_05', 'Crew_06', 'Crew_07', 'Crew_08', 'Crew_09', 'Crew_10', 'Crew_11', 'Crew_13']
# crews_to_process = ['Crew_02']
file_types = ["smarteye_leftseat","smarteye_rightseat"]
scenarios = ["1","2","3","5","6","7"]
# scenarios = ["1"]
storage_client = storage.Client(project="soteria-fa59")
bucket = storage.Bucket(storage_client, "soteria_study_data", user_project="soteria-fa59")

leftseat_heatmap = np.zeros((100,100,len(scenarios)))
rightseat_heatmap = np.zeros((100,100,len(scenarios)))

plot_heatmap_and_qatable = 0 # embeds pct_usable value too
number_of_epochs = 1000
time_per_epoch_4_analysis = 10

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
	crew_dir = crews_to_process[i_crew]
	process_dir_name = crew_dir + "/Processing/"

	event_smarteyeGazeTimeSeries_metrics = pd.DataFrame()
	total_gaze_variance_matrix = np.zeros((len(file_types),len(scenarios)))
	total_gaze_velocity_avg_matrix = np.zeros((len(file_types),len(scenarios)))
	total_gaze_velocity_std_matrix = np.zeros((len(file_types),len(scenarios)))
	# event_smarteyeGaze_metrics = np.zeros((3,3,len(file_types),len(scenarios)))

	event_smarteyeGaze_metrics = np.zeros((len(scenarios)*2,12))
	event_smarteyeGaze_metrics[:, 0] = getCrewInt(crews_to_process[i_crew])

	pupild_leftseat = np.zeros((number_of_epochs,len(scenarios)))
	pupild_rightseat = np.zeros((number_of_epochs,len(scenarios)))
	headHeading_leftseat = np.zeros((number_of_epochs,len(scenarios)))
	headHeading_rightseat = np.zeros((number_of_epochs,len(scenarios)))
	smarteye_timesec_epoch_storage = np.zeros((len(scenarios),number_of_epochs))
	# event_smarteyeTime_metrics = np.zeros((4,3,len(file_types),len(scenarios)))

	event_smarteyeTimeSeries_metrics = pd.DataFrame()
	event_smarteyeTime_metrics = np.zeros((len(scenarios)*2,15))
	event_smarteyeTime_metrics[:, 0] = getCrewInt(crews_to_process[i_crew])
	event_smarteyeTime_column_values = ['crew', 'seat', 'scenario', 'headHeading_avg', 'headHeading_std', 'pupilD_avg', 'pupilD_std']

	f_stream = file_io.FileIO('gs://soteria_study_data/'+ process_dir_name + 'event_vector_scenario.npy', 'rb')
	this_event_data = np.load(io.BytesIO(f_stream.read()))

	for i_scenario in range(len(scenarios)):
		if ((getCrewInt(crews_to_process[i_crew]) == 13) & (scenarios[i_scenario] == '5')):
			break
		else:
			print("Processing Crew: " + crews_to_process[i_crew] + " Scenario: "+scenarios[i_scenario])
			for i_seat in range(len(file_types)):
				if (i_seat == 0):
					event_smarteyeTime_metrics[i_scenario*2, 1] = 0
					event_smarteyeTime_metrics[i_scenario*2, 2] = scenarios[i_scenario]
					event_smarteyeGaze_metrics[i_scenario*2, 1] = 0
					event_smarteyeGaze_metrics[i_scenario*2, 2] = scenarios[i_scenario]
				else:
					event_smarteyeTime_metrics[i_scenario*2+1, 1] = 1
					event_smarteyeTime_metrics[i_scenario*2+1, 2] = scenarios[i_scenario]
					event_smarteyeGaze_metrics[i_scenario*2+1, 1] = 1
					event_smarteyeGaze_metrics[i_scenario*2+1, 2] = scenarios[i_scenario]

				blob = bucket.blob(process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv')
				if blob.exists():
					smarteye_data = pd.read_table(('gs://soteria_study_data/' + process_dir_name + file_types[i_seat] + '_scenario' + scenarios[i_scenario] + '.csv'),delimiter=',')
					print("Processing Crew: " + crews_to_process[i_crew] + " Scenario: "+scenarios[i_scenario])
					time_vector = np.array(smarteye_data.UserTimeStamp[2:])
					HeadRotationQ = np.array(smarteye_data.HeadRotationQ[2:])
					PupilDiameterQ = np.array(smarteye_data.PupilDiameterQ[2:])

					direction_gaze = np.array([smarteye_data.GazeDirectionX[2:], smarteye_data.GazeDirectionY[2:], smarteye_data.GazeDirectionZ[2:]])
					HeadPosition = np.array([smarteye_data.HeadPosX[2:], smarteye_data.HeadPosY[2:], smarteye_data.HeadPosZ[2:]])
					magnitude = np.divide(np.sqrt(np.power(direction_gaze[0,:],2) + np.power(direction_gaze[1,:],2) + np.power(direction_gaze[2,:],2)),1)
					quality_gaze = smarteye_data.GazeDirectionQ[2:]
					degree_per_sec_vector = angle_diff(time_vector, direction_gaze)

					# for the indices that are sequential (i.e. no gaps in good_indices), calculate rate of gaze movement, and remove values that are greater than 700°/s ( Fuchs, A. F. (1967-08-01). "Saccadic and smooth pursuit eye movements in the monkey". The Journal of Physiology. 191 (3): 609–631. doi:10.1113/jphysiol.1967.sp008271. ISSN 1469-7793. PMC 1365495. PMID 4963872.)
					# unit circle 1 -> -1
					# WARNING: not quite what I intended ^^
					good_indices = np.squeeze(np.where((magnitude!=0) & (quality_gaze*100 >= 6) & (degree_per_sec_vector<=700) & (degree_per_sec_vector!=np.nan)))

					# projected_planar_coords = sphere_stereograph(np.squeeze(direction_gaze[:,good_indices]))
					projected_headPos_coords = sphere_stereograph(np.squeeze(HeadPosition))
					projected_planar_coords = sphere_stereograph(np.squeeze(direction_gaze))
					good_project_planar_coords = projected_planar_coords[:,good_indices]
					good_project_headPos_coords = projected_headPos_coords[:,good_indices]
					
					good_project_planar_coords = good_project_planar_coords - good_project_headPos_coords # correct origin of gaze vector for head position
					good_project_planar_coords[0,:] = good_project_planar_coords[0,:] * -1 # flip the x axis of gaze vector

					good_ObjectIntersection = smarteye_data.ObjectIntersectionName[good_indices]
					# object_data_good = np.squeeze(object_data[good_indices])
					# unique_objects = np.unique(object_data_good.tolist())

					total_good_vel_vals = np.squeeze(degree_per_sec_vector[good_indices])
					# exclude the first since that was set to 0 on purpose
					total_average_gaze_velocity = np.average(total_good_vel_vals[1:])
					total_std_gaze_velocity = np.std(total_good_vel_vals[1:])

					# https://www.geeksforgeeks.org/variance-standard-deviation-matrix/
					total_m = mean(projected_planar_coords[:,good_indices],2)
					total_gaze_variance = variance(projected_planar_coords[:,good_indices], 2, total_m)

					headheading_good_indices = np.squeeze(np.where(HeadRotationQ > .6))
					pupilD_good_indices = np.squeeze(np.where(PupilDiameterQ > .4))

					headheading = np.array(smarteye_data.HeadHeading[2:])

					headheadingDeg = headheading * 180/math.pi
					time_diff = np.diff(time_vector)
					
					pupilD = np.array(smarteye_data.PupilDiameter[2:])

					headheadingDeg_rate = np.zeros(headheadingDeg.shape[0])
					headheadingDeg_diff = np.diff(headheadingDeg)
					for this_frame in range(headheadingDeg.shape[0]-1):
						if time_diff[this_frame] != 0.0:
							headheadingDeg_rate[this_frame+1] = headheadingDeg_diff[this_frame] / time_diff[this_frame]
						else:
							headheadingDeg_rate[this_frame+1] = 0

					if (scenarios[i_scenario] != '7'):
						# need to find the indices 1 minute before and 1 minute after (start and end of event epoch)
						event1_epoch_start = this_event_data[0, i_scenario] - 60
						difference_array = np.absolute(time_vector-event1_epoch_start)
						event1_start_index = difference_array.argmin()
						event1_epoch_end = this_event_data[0, i_scenario] + 60
						difference_array = np.absolute(time_vector-event1_epoch_end)
						event1_end_index = difference_array.argmin()
						event2_epoch_start = this_event_data[1, i_scenario] - 60
						difference_array = np.absolute(time_vector-event2_epoch_start)
						event2_start_index = difference_array.argmin()
						event2_epoch_end = this_event_data[1, i_scenario] + 60
						difference_array = np.absolute(time_vector-event2_epoch_end)
						event2_end_index = difference_array.argmin()
						difference_array = np.absolute(time_vector-120)
						two_min_index = difference_array.argmin()

						headheading_event1_good_indices = np.squeeze(np.where((headheading_good_indices > event1_start_index) & (headheading_good_indices < event1_end_index)))
						headheading_event2_good_indices = np.squeeze(np.where((headheading_good_indices > event2_start_index) & (headheading_good_indices < event2_end_index)))
						headheading_twomin_good_indices = np.squeeze(np.where((headheading_good_indices < two_min_index)))

						pupilD_event1_good_indices = np.squeeze(np.where((pupilD_good_indices > event1_start_index) & (pupilD_good_indices < event1_end_index)))
						pupilD_event2_good_indices = np.squeeze(np.where((pupilD_good_indices > event2_start_index) & (pupilD_good_indices < event2_end_index)))
						pupilD_twomin_good_indices = np.squeeze(np.where((pupilD_good_indices < two_min_index)))

						event1_good_indices = np.squeeze(np.where((good_indices > event1_start_index) & (good_indices < event1_end_index)))
						event2_good_indices = np.squeeze(np.where((good_indices > event2_start_index) & (good_indices < event2_end_index)))
						twomin_good_indices = np.squeeze(np.where((good_indices < two_min_index)))

						twomin_mean = mean(projected_planar_coords[:,twomin_good_indices],2)

						event1_mean = mean(projected_planar_coords[:,event1_good_indices],2)
						
						event2_mean = mean(projected_planar_coords[:,event2_good_indices],2)
						
						twomin_good_vel_vals = np.squeeze(degree_per_sec_vector[twomin_good_indices])
						event1_good_vel_vals = np.squeeze(degree_per_sec_vector[event1_good_indices])
						event2_good_vel_vals = np.squeeze(degree_per_sec_vector[event2_good_indices])

						if (i_seat == 0):
							event_smarteyeTime_metrics[i_scenario*2, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 4] = statistics.mean(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 5] = statistics.mean(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 7] = statistics.stdev(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 8] = statistics.stdev(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 10] = statistics.mean(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 11] = statistics.mean(pupilD[pupilD_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 13] = statistics.stdev(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 14] = statistics.stdev(pupilD[pupilD_event2_good_indices])
							event_smarteyeGaze_metrics[i_scenario*2, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, twomin_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 4] = variance(projected_planar_coords[:,event1_good_indices], 2, event1_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 5] = variance(projected_planar_coords[:,event2_good_indices], 2, event2_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 6] = np.nanmean(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 7] = np.nanmean(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 8] = np.nanmean(event2_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 9] = np.nanstd(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 10] = np.nanstd(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 11] = np.nanstd(event2_good_vel_vals[1:])
						else:
							event_smarteyeTime_metrics[i_scenario*2+1, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 4] = statistics.mean(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 5] = statistics.mean(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 7] = statistics.stdev(headheadingDeg_rate[headheading_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 8] = statistics.stdev(headheadingDeg_rate[headheading_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 10] = statistics.mean(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 11] = statistics.mean(pupilD[pupilD_event2_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 13] = statistics.stdev(pupilD[pupilD_event1_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 14] = statistics.stdev(pupilD[pupilD_event2_good_indices])
							event_smarteyeGaze_metrics[i_scenario*2+1, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, twomin_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 4] = variance(projected_planar_coords[:,event1_good_indices], 2, event1_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 5] = variance(projected_planar_coords[:,event2_good_indices], 2, event2_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 6] = np.nanmean(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 7] = np.nanmean(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 8] = np.nanmean(event2_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 9] = np.nanstd(twomin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 10] = np.nanstd(event1_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 11] = np.nanstd(event2_good_vel_vals[1:])

						number_of_epochs_this_scenario = np.floor(time_vector[-1]/time_per_epoch_4_analysis)
						this_smarteyeTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 9))
						this_smarteyeTimeSeries_np[:,0] = getCrewInt(crews_to_process[i_crew])
						this_smarteyeGazeTimeSeries_np = np.zeros((int(number_of_epochs_this_scenario), 8))
						this_smarteyeGazeTimeSeries_np[:,0] = getCrewInt(crews_to_process[i_crew])
						if (i_seat == 0):
							this_smarteyeTimeSeries_np[:,1] = 0
							this_smarteyeGazeTimeSeries_np[:,1] = 0
						else:
							this_smarteyeTimeSeries_np[:,1] = 1
							this_smarteyeGazeTimeSeries_np[:,1] = 1
						this_smarteyeTimeSeries_np[:,2] = i_scenario
						length_this_data = smarteye_data.shape[0]
						for this_epoch in range(int(number_of_epochs_this_scenario)):
							this_epoch_indices_start = np.floor(length_this_data/number_of_epochs_this_scenario) * this_epoch
							this_epoch_indices_end = this_epoch_indices_start + np.floor(length_this_data/number_of_epochs_this_scenario)
							smarteye_timesec_epoch_storage[i_scenario,this_epoch] = smarteye_data.UserTimeStamp[this_epoch_indices_start]
							if file_types[i_seat]  == "smarteye_leftseat":
								pupild_leftseat[this_epoch,i_scenario] = smarteye_data.PupilDiameter[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()
								headHeading_leftseat[this_epoch,i_scenario] = smarteye_data.HeadHeading[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()
								
							elif file_types[i_seat]  == "smarteye_rightseat":
								pupild_rightseat[this_epoch,i_scenario] = smarteye_data.PupilDiameter[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()
								headHeading_rightseat[this_epoch,i_scenario] = smarteye_data.HeadHeading[int(this_epoch_indices_start):int(this_epoch_indices_end)].mean()

							if ((time_vector[int(this_epoch_indices_start)] > this_event_data[0, i_scenario] - 60) & (time_vector[int(this_epoch_indices_start)] < this_event_data[0, i_scenario] + 60)) | ((time_vector[int(this_epoch_indices_start)] > this_event_data[1, i_scenario] - 60) & (time_vector[int(this_epoch_indices_start)] < this_event_data[1, i_scenario] + 60)):
								this_smarteyeTimeSeries_np[this_epoch, 3] = 1
								this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 1
							else:
								this_smarteyeTimeSeries_np[this_epoch, 3] = 0
								this_smarteyeGazeTimeSeries_np[this_epoch, 3] = 0

							this_smarteyeGazeTimeSeries_np[this_epoch, 4] = this_epoch
							this_good_indices = np.squeeze(np.where((good_indices > this_epoch_indices_start) & (good_indices < this_epoch_indices_end)))
							if (this_good_indices.size > 1):
								this_smarteyeGazeTimeSeries_np[this_epoch, 5] = variance(projected_planar_coords[:,this_good_indices], 2, mean(projected_planar_coords[:,this_good_indices],1))
								this_smarteyeGazeTimeSeries_np[this_epoch, 6] = np.nanmean(np.squeeze(degree_per_sec_vector[this_good_indices]))
								this_smarteyeGazeTimeSeries_np[this_epoch, 7] = np.nanstd(np.squeeze(degree_per_sec_vector[this_good_indices]))
							else:
								this_smarteyeGazeTimeSeries_np[this_epoch, 5] = np.nan
								this_smarteyeGazeTimeSeries_np[this_epoch, 6] = np.nan
								this_smarteyeGazeTimeSeries_np[this_epoch, 7] = np.nan

							this_smarteyeTimeSeries_np[this_epoch, 4] = this_epoch
							this_smarteyeTimeSeries_np[this_epoch, 5] = np.nanmean(headheadingDeg_rate[int(this_epoch_indices_start):int(this_epoch_indices_end)])
							this_smarteyeTimeSeries_np[this_epoch, 6] = np.nanstd(headheadingDeg_rate[int(this_epoch_indices_start):int(this_epoch_indices_end)])
							this_smarteyeTimeSeries_np[this_epoch, 7] = np.nanmean(pupilD[int(this_epoch_indices_start):int(this_epoch_indices_end)])
							this_smarteyeTimeSeries_np[this_epoch, 8] = np.nanstd(pupilD[int(this_epoch_indices_start):int(this_epoch_indices_end)])

						this_smarteyeGazeTimeSeries_df = pd.DataFrame(this_smarteyeGazeTimeSeries_np)
						this_smarteyeGazeTimeSeries_df.columns = ['crew', 'seat', 'scenario', 'event_label', 'epoch_index', 'gaze_variance', 'gaze_vel_avg', 'gaze_vel_std']
						event_smarteyeGazeTimeSeries_metrics = pd.concat([event_smarteyeGazeTimeSeries_metrics,this_smarteyeGazeTimeSeries_df])

						this_smarteyeTimeSeries_df = pd.DataFrame(this_smarteyeTimeSeries_np)
						this_smarteyeTimeSeries_df.columns = ['crew', 'seat', 'scenario', 'event_label', 'epoch_index','headHeading_avg', 'headHeading_std', 'pupilD_avg', 'pupilD_std']
						event_smarteyeTimeSeries_metrics = pd.concat([event_smarteyeTimeSeries_metrics,this_smarteyeTimeSeries_df])
					else:
						difference_array = np.absolute(time_vector - ((time_vector[-1]/2) - 150))
						fivemin_start_index = difference_array.argmin()
						difference_array = np.absolute(time_vector - ((time_vector[-1]/2) + 150))
						fivemin_stop_index = difference_array.argmin()

						fivemin_good_indices = np.squeeze(np.where((good_indices > fivemin_start_index) & (good_indices < fivemin_stop_index)))
						fivemin_mean = mean(projected_planar_coords[:,fivemin_good_indices],2)
						fivemin_good_vel_vals = np.squeeze(degree_per_sec_vector[fivemin_good_indices])
						headheading_fivmin_good_indices = np.squeeze(np.where((headheading_good_indices > fivemin_start_index) & (headheading_good_indices < fivemin_stop_index)))
						pupilD_fivmin_good_indices = np.squeeze(np.where((pupilD_good_indices > fivemin_start_index) & (pupilD_good_indices < fivemin_stop_index)))

						if (i_seat == 0):
							event_smarteyeTime_metrics[i_scenario*2, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
							event_smarteyeGaze_metrics[i_scenario*2, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, fivemin_mean)
							event_smarteyeGaze_metrics[i_scenario*2, 6] = np.nanmean(fivemin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2, 9] = np.nanstd(fivemin_good_vel_vals[1:])
						else:
							event_smarteyeTime_metrics[i_scenario*2+1, 3] = statistics.mean(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 6] = statistics.stdev(headheadingDeg_rate[headheading_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 9] = statistics.mean(pupilD[pupilD_twomin_good_indices])
							event_smarteyeTime_metrics[i_scenario*2+1, 12] = statistics.stdev(pupilD[pupilD_twomin_good_indices])
							event_smarteyeGaze_metrics[i_scenario*2+1, 3] = variance(projected_planar_coords[:,twomin_good_indices], 2, fivemin_mean)
							event_smarteyeGaze_metrics[i_scenario*2+1, 6] = np.nanmean(fivemin_good_vel_vals[1:])
							event_smarteyeGaze_metrics[i_scenario*2+1, 9] = np.nanstd(fivemin_good_vel_vals[1:])
					
					if plot_heatmap_and_qatable:
						x = smarteye_data.ObjectIntersectionX * 1000 # m to mm?
						y = smarteye_data.ObjectIntersectionY * 1000 # m to mm? where is the origin?
						length_this_trial = smarteye_data.shape[0]
						data_quality_vector = np.ones(length_this_trial)
						for i_index in range(smarteye_data.shape[0]):
							if smarteye_data.IntersectionIndex[i_index] == 0 or smarteye_data.GazeDirectionQ[i_index] < .50:
								x[i_index] = 0
								y[i_index] = 0
								data_quality_vector[i_index] = 0
						pct_usable_matrix[i_scenario,i_seat] = np.rint((np.sum(data_quality_vector)/length_this_trial) * 100)
						x[ x==0 ] = np.nan
						y[ y==0 ] = np.nan
						x = x[~np.isnan(x)]
						y = y[~np.isnan(y)]

					else:
						pct_usable_matrix[i_scenario,i_seat] = np.nan
						
						if plot_heatmap_and_qatable:
							empty_heatmap = np.zeros((100,100))
							if file_types[i_seat] == "smarteye_leftseat":
								leftseat_heatmap[:,:,i_scenario] = empty_heatmap
							elif file_types[i_seat] == "smarteye_rightseat":
								rightseat_heatmap[:,:,i_scenario] = empty_heatmap

					unique_ObjectIntersection  = unique(good_ObjectIntersection)
					# assign colors to each possible object (i.e. make a map for each object)
					# for 
					# for each object, determine what indices are labeled with it, then plot those indices with the mapped color

					# for i_obj in range(len(unique_ObjectIntersection)):
					# 	"Inst Panel" in good_ObjectIntersection

					NUM_COLORS = len(unique_ObjectIntersection)
					cm = plt.get_cmap('gist_rainbow')
					cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)

					fig1 = plt.figure(1)
					ax = plt.gca()
					# color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
					
					if (i_seat == 0):
						for i in range(NUM_COLORS):
						    indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
						    col= cm(1.*i/NUM_COLORS)
						    ax.scatter(good_project_planar_coords[0,indices_this_object], good_project_planar_coords[1,indices_this_object], c = col, alpha=.5,marker='.' )
						    ax.legend(unique_ObjectIntersection)
						    ax.set_title('Gaze vector left seat / Capt')
						# plt.show()
						# print("should be creating fig")
						# plt.xlim((-1.5, 1.5))
						# plt.ylim((-1, 0))
						fig1.set_size_inches((22, 11))
						ax = plt.gca()
						ax.get_xaxis().set_visible(False)
						ax.axis('off')
						plt.savefig('Figures/smarteyeGaze_'+scenarios[i_scenario]+'_leftseat.jpg')
						plt.close()
					else:
						for i in range(NUM_COLORS):
						    indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
						    col= cm(1.*i/NUM_COLORS)
						    ax.scatter(good_project_planar_coords[0,indices_this_object], good_project_planar_coords[1,indices_this_object], c = col, alpha=.5,marker='.' )
						    ax.legend(unique_ObjectIntersection)
						    ax.set_title('Gaze vector right seat / FO')
						# plt.show()
						# plt.xlim((-1.5, 1.5))
						# plt.ylim((-1, 0))
						fig1.set_size_inches((22, 11))
						ax = plt.gca()
						ax.get_xaxis().set_visible(False)
						ax.axis('off')
						plt.savefig('Figures/smarteyeGaze_'+scenarios[i_scenario]+'_rightseat.jpg')
						plt.close()

					# need to create an array that is preallocated based on all the potential AOIs (get from helper files) x scenario
					# 

						# plt.text(-.9,-.05,"Gaze Variance="+ str(np.round(total_gaze_variance,3)),c='c', fontsize='large')
						# plt.text(-.9,-.10,"Gaze Velocity=" + str(int(np.round(total_average_gaze_velocity))) + "$\pm$" + str(int(np.round(total_std_gaze_velocity)))+"$^\circ$/sec",c='c', fontsize='large')
					# else:
					# 	plt.scatter(good_project_planar_coords[0,:], good_project_planar_coords[1,:], c = 'r', alpha=.05,marker='.' )
						# plt.text(.65,-.05,"Gaze Variance="+ str(np.round(total_gaze_variance,3)),c='r', fontsize='large')
						# plt.text(.65,-.10,"Gaze Velocity=" + str(int(np.round(total_average_gaze_velocity))) + "$\pm$" + str(int(np.round(total_std_gaze_velocity)))+"$^\circ$/sec",c='r', fontsize='large')
					# del projected_planar_coords
					# del good_indices
					# del direction_gaze
					# del quality_gaze
					# del magnitude
				
					# fig2 = plt.figure(2)
					# ax = plt.gca()
					# if (i_seat == 0):
					# 	for i in range(NUM_COLORS):
					# 	    indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
					# 	    col= cm(1.*i/NUM_COLORS)
					# 	    plt.scatter(good_project_headPos_coords[0,indices_this_object], good_project_headPos_coords[1,indices_this_object], c = col, alpha=.5,marker='.' )
					# 	    ax.legend(unique_ObjectIntersection)
					# 	    ax.set_title('Head position left seat / Capt')
					# 	plt.show()
					# else:
					# 	for i in range(NUM_COLORS):
					# 	    indices_this_object = np.squeeze(np.where(good_ObjectIntersection == unique_ObjectIntersection[i]))
					# 	    col= cm(1.*i/NUM_COLORS)
					# 	    plt.scatter(good_project_headPos_coords[0,indices_this_object], good_project_headPos_coords[1,indices_this_object], c = col, alpha=.5,marker='.' )
					# 	    ax.legend(unique_ObjectIntersection)
					# 	    ax.set_title('Head position right seat / FO')
					# 	plt.show()
					total_gaze_variance_matrix[i_seat, i_scenario] = total_gaze_variance
					total_gaze_velocity_avg_matrix[i_seat, i_scenario] = total_average_gaze_velocity
					total_gaze_velocity_std_matrix[i_seat, i_scenario] = total_std_gaze_velocity

					# fig2 = plt.figure(2)
					# if (i_seat == 0):
					# 	plt.scatter(good_project_headPos_coords[0,:], good_project_headPos_coords[1,:], c = 'c', alpha=.05,marker='.' )
					# else:
					# 	plt.scatter(good_project_headPos_coords[0,:], good_project_headPos_coords[1,:], c = 'r', alpha=.05,marker='.' )
					

					total_gaze_variance_matrix[i_seat, i_scenario] = total_gaze_variance
					total_gaze_velocity_avg_matrix[i_seat, i_scenario] = total_average_gaze_velocity
					total_gaze_velocity_std_matrix[i_seat, i_scenario] = total_std_gaze_velocity


			

			# print("should be creating fig")
			# plt.xlim((-1, 1))
			# plt.ylim((-1, 0))
			# fig1.set_size_inches((22, 11))
			# ax = plt.gca()
			# ax.get_xaxis().set_visible(False)
			# ax.axis('off')
			# plt.savefig('Figures/smarteyeGaze_'+scenarios[i_scenario]+'.jpg')
			# plt.show()

			# matplotlib.pyplot.close()
			# plt.close()


	if plot_heatmap_and_qatable:
		fig, ax = plt.subplots()
		cbar_kws = { 'ticks' : [0, 100] }
		ax = sns.heatmap(pct_usable_matrix, linewidths=.5, cbar_kws = cbar_kws,annot=True,fmt='.3g')
		ax.set_xticklabels(file_types)
		ax.set_yticklabels(scenarios)
		ax.set(xlabel='pilot', ylabel='scenarios')
		plt.yticks(rotation=0)
		ax.xaxis.set_label_position('top')
		ax.xaxis.tick_top()
		fig.tight_layout()
		# plt.show()
		plt.savefig("Figures/" + 'smarteye_pct_usable.jpg')
		matplotlib.pyplot.close()
		np.save("Processing/" + 'pct_usable_matrix',pct_usable_matrix)


	np.save("Processing/" + 'smarteye_pupild_leftseat',pupild_leftseat)
	np.save("Processing/" + 'smarteye_pupild_rightseat',pupild_rightseat)
	np.save("Processing/" + 'smarteye_headHeading_leftseat',headHeading_leftseat)
	np.save("Processing/" + 'smarteye_headHeading_rightseat',headHeading_rightseat)
	np.save("Processing/" + 'smarteye_timesec_epoch_storage',smarteye_timesec_epoch_storage)
	np.save("Processing/" + 'event_smarteyeTime_metrics', event_smarteyeTime_metrics)
	event_smarteyeGazeTimeSeries_metrics.info()
	event_smarteyeGazeTimeSeries_metrics.to_csv("Processing/" + 'event_smarteyeGazeTimeSeries_metrics.csv')
	event_smarteyeTimeSeries_metrics.info()
	event_smarteyeTimeSeries_metrics.to_csv("Processing/" + 'event_smarteyeTimeSeries_metrics.csv')
	subprocess.call('gsutil -m rsync -r Figures/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Figures"', shell=True)
	subprocess.call('gsutil -m rsync -r Processing/ "gs://soteria_study_data/"'+ crews_to_process[i_crew] + '"/Processing"', shell=True)

	# print('should have saved')